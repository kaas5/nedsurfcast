"""
Forecast dataset: GFS-Wave wave systems matched to buoy wave systems
======================================================================

Archived GFS-Wave forecasts (wind sea + swell 1-3) are compared with wave systems measured by
the RWS buoys (partitions of the measured spectra, common/partition.py):

    1. buoy partitions at every GFS valid time (spectra averaged over +-BUOY_SPECTRA_WINDOW)
    2. GFS partitions at the station's nearest sea cell, merged to the buoy's resolution
       (merge_unresolvable: systems a 1D buoy cannot separate become one)
    3. Hungarian matching on peak frequency and direction (match_partitions)
    4. every GFS system becomes one sample: its matched buoy system is the target, or the
       target is "absent" (false alarm). Buoy systems without a GFS partner are misses.

One sample per GFS system removes the slot-order problem: there is no "swell 1 vs swell 1".

Zarr groups (one store, fcfg.PARTITION_ZARR):
    grid           (time, lead_hours, [swell_partition,] lat, lon)  GFS fields, unmerged, + sea_mask
    stations       (station,)        position, nearest sea cell, validation group
    station_times  (time, lead_hours, station)   bulk context at the station: HTSGW, wind (bilinear),
                                                 PERPW/DIRPW (nearest), buoy Hm0, whether a spectrum exists
    samples        (sample,)         one row per merged GFS system at a station
    misses         (miss,)           buoy systems that no GFS system matched
    spectra        (time, lead_hours, station, freq)   averaged buoy spectrum (energy, direction, spread)
                                     and the GFS spectrum rebuilt from its partitions (JONSWAP per partition,
                                     common.partition.synthesize_spectrum) on the same frequency bins

`time` is the issue time; valid_time = time + lead_hours. Samples and misses index the other
groups with time_idx, lead_idx and station. Residuals are not stored: compute them in the model
code (e.g. log Hs ratio, wrapped direction difference) from the gfs_* and buoy_* values.

Run from anywhere with: python -m nedsurfcast.forecast.preprocess
"""

import json
import shutil

import numpy as np
import pandas as pd
import xarray as xr

from nedsurfcast import config as cfg
from nedsurfcast.common.buoys import match_to_sea_cell
from nedsurfcast.common.grouping import group_points
from nedsurfcast.common.partition import (average_spectra, load_spectra, match_partitions, merge_unresolvable,
                                          partition_dataset, partition_distance, synthesize_spectrum)
from nedsurfcast.forecast import config as fcfg

GRID_STEP = 0.25            # deg, GFS global.0p25
HS_SMALL = 0.3              # m; only used to split the report into small and large systems
CFGRIB_NAMES = {"WVHGT": "shww", "WVPER": "mpww", "WVDIR": "wvdir", "SWELL": "shts", "SWPER": "mpts",
                "SWDIR": "swdir", "HTSGW": "swh", "PERPW": "perpw", "DIRPW": "dirpw", "UGRD": "u", "VGRD": "v"}


# ---------------------------------------------------------------------------
# 1. GFS FORECAST RUNS
# ---------------------------------------------------------------------------

def gfs_field_names() -> list[str]:
    # "HTSGW:surface" -> "HTSGW"; "SWELL:1/2/3 in sequence" -> one "SWELL" with a swell_partition dimension
    return list(dict.fromkeys(field.split(":")[0] for field in fcfg.GFSWAVE_FIELDS))


def gfs_files() -> list:
    """Downloaded runs (gfswave_YYYYMMDD_HHz.nc) with an issue time in cfg.DATE_RANGE."""
    files = sorted(fcfg.GFSWAVE_DIR.glob("gfswave_[0-9]*_[0-9][0-9]z.nc"))
    start, end = pd.Timestamp(cfg.START_DATE), pd.Timestamp(cfg.END_DATE)
    files = [f for f in files if start <= pd.to_datetime(f.stem.split("_")[1]) < end]
    if not files:
        raise FileNotFoundError(f"No GFS-Wave runs in {fcfg.GFSWAVE_DIR} for {start:%Y-%m-%d}..{end:%Y-%m-%d}; "
                                "run python -m nedsurfcast.download.gfs")
    return files


def load_gfs_runs() -> xr.Dataset:
    """All runs in cfg.DATE_RANGE stacked along issue time -> (time, lead_hours, [swell_partition,] lat, lon)."""
    runs = [xr.open_dataset(f, chunks={}) for f in gfs_files()]
    ds = xr.concat(runs, dim="time", join="outer")   # outer: a run missing some lead times gets NaN there
    ds = ds.drop_vars([c for c in ("orderedSequenceData",) if c in ds.coords])
    ds = ds.rename({k: v for k, v in fcfg.GFSWAVE_VAR_NAMES.items() if k in ds.data_vars})
    ds = ds.sortby("time")

    missing = [v for v in gfs_field_names() if v not in ds.data_vars]
    if missing:
        raise ValueError(f"{missing} are in GFSWAVE_FIELDS but not in the downloaded runs; download again with overwrite=True")
    return ds[gfs_field_names()]


# ---------------------------------------------------------------------------
# 2. STATIONS, THEIR GRID CELLS AND GFS VALUES THERE
# ---------------------------------------------------------------------------

def station_cells(sea_mask: xr.DataArray, lat: float, lon: float) -> dict:
    """Nearest sea cell + the 4 grid points around the station with bilinear weights (land points dropped)."""
    near_lat, near_lon = match_to_sea_cell(sea_mask.where(sea_mask).to_dataset(name="sea"), lat, lon, var="sea")
    lat0, lon0 = np.floor(lat / GRID_STEP) * GRID_STEP, np.floor(lon / GRID_STEP) * GRID_STEP
    fy, fx = (lat - lat0) / GRID_STEP, (lon - lon0) / GRID_STEP
    corners = [(lat0, lon0, (1 - fy) * (1 - fx)), (lat0, lon0 + GRID_STEP, (1 - fy) * fx),
               (lat0 + GRID_STEP, lon0, fy * (1 - fx)), (lat0 + GRID_STEP, lon0 + GRID_STEP, fy * fx)]
    corners = [(round(a, 4), round(o, 4), w) for a, o, w in corners
               if bool(sea_mask.sel(lat=round(a, 4), lon=round(o, 4), method="nearest"))]
    w = np.array([c[2] for c in corners])
    return {"nearest": (near_lat, near_lon),
            "corners": [(a, o) for a, o, _ in corners],
            "weights": w / w.sum() if len(w) else w}


def extract_gfs_cells(files, cells: list[tuple[float, float]]) -> xr.Dataset:
    """GFS values at the given cells for every run -> (time, lead_hours, cell[, partition]).

    Partitions: index 0 = wind sea, 1..3 = swell 1..3, as hs/tp/dir. Also HTSGW, PERPW, DIRPW and wind.
    Reads the run files one by one, which is much faster than selecting points from the dask dataset.
    """
    lats = xr.DataArray([c[0] for c in cells], dims="cell")
    lons = xr.DataArray([c[1] for c in cells], dims="cell")
    n = CFGRIB_NAMES
    runs = []
    for i, f in enumerate(files):
        with xr.open_dataset(f) as ds:
            pt = ds.sel(lat=lats, lon=lons).load()

        def parts(wind_sea, swell):
            return np.stack([pt[n[wind_sea]].values] + [pt[n[swell]].isel(swell_partition=k).values for k in range(3)], -1)

        dims, dims_p = ("lead_hours", "cell"), ("lead_hours", "cell", "partition")
        runs.append(xr.Dataset(
            {"hs": (dims_p, parts("WVHGT", "SWELL")), "tp": (dims_p, parts("WVPER", "SWPER")),
             "dir": (dims_p, parts("WVDIR", "SWDIR")),
             "htsgw": (dims, pt[n["HTSGW"]].values), "perpw": (dims, pt[n["PERPW"]].values),
             "dirpw": (dims, pt[n["DIRPW"]].values), "u10": (dims, pt[n["UGRD"]].values),
             "v10": (dims, pt[n["VGRD"]].values), "valid_time": ("lead_hours", pt["valid_time"].values)},
            coords={"lead_hours": pt.lead_hours.values, "time": pt.time.values}))
        if (i + 1) % 50 == 0:
            print(f"  GFS: {i + 1}/{len(files)} runs read")
    return xr.concat(runs, "time")


def gfs_parts(hs, tp, dr) -> np.ndarray:
    """(n_partitions,) arrays -> (n, 3) with (hs, fp, dir); absent partitions (NaN or 0 m) -> NaN rows."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.column_stack([hs, 1 / tp, dr]).astype(float)
    out[~(np.isfinite(hs) & (hs > 0))] = np.nan
    return out


def _distance_km(lat1, lon1, lat2, lon2):
    # Equirectangular, accurate enough over the North Sea
    dx = (np.asarray(lon2) - lon1) * 111.32 * np.cos(np.deg2rad(lat1))
    dy = (np.asarray(lat2) - lat1) * 110.57
    return np.hypot(dx, dy)


# ---------------------------------------------------------------------------
# 3. MATCHED WAVE SYSTEMS
# ---------------------------------------------------------------------------

def buoy_partitions(spectra: xr.Dataset, valid_times: np.ndarray) -> tuple[xr.Dataset, xr.Dataset]:
    """Averaged buoy spectra at the GFS valid times (time, freq) and their wave systems (time, partition)
    + hs_total (Hm0 in the partition range)."""
    averaged = average_spectra(spectra, window=fcfg.BUOY_SPECTRA_WINDOW).reindex(time=valid_times)
    return averaged, partition_dataset(averaged, max_partitions=6, **fcfg.BUOY_PARTITION_KWARGS)


def spectrum_freqs() -> tuple[np.ndarray, np.ndarray]:
    """Buoy frequency bin centres and the mask of the bins inside the partitioning range."""
    f = np.arange(fcfg.RWS_SPECTRA_NBINS) * fcfg.RWS_SPECTRA_DF + fcfg.RWS_SPECTRA_DF / 2
    lo, hi = fcfg.BUOY_PARTITION_KWARGS["f_range"]
    return f, (f >= lo) & (f <= hi)


def _other_systems(parts: np.ndarray, i: int) -> tuple[float, float, float]:
    """Order-invariant context of the systems other than i: count, combined Hs, energy-weighted direction."""
    others = np.delete(parts, i, axis=0)
    others = others[np.isfinite(others[:, 0])]
    if len(others) == 0:
        return 0, 0.0, np.nan
    e = others[:, 0] ** 2 / 16
    rad = np.deg2rad(others[:, 2])
    direction = np.rad2deg(np.arctan2(np.sum(e * np.sin(rad)), np.sum(e * np.cos(rad)))) % 360
    return len(others), float(4 * np.sqrt(e.sum())), float(direction)


def build_partition_dataset() -> dict[str, xr.Dataset]:
    files = gfs_files()
    gfs = load_gfs_runs()
    sea_mask = gfs["HTSGW"].isel(time=0, lead_hours=0).notnull().compute()
    sea_mask = sea_mask.drop_vars([c for c in sea_mask.coords if c not in ("lat", "lon")])
    print(f"GFS: {gfs.sizes['time']} runs x {gfs.sizes['lead_hours']} lead times, fields {list(gfs.data_vars)}")

    # stations with spectra in the period
    stations = {}
    for st in fcfg.RWS_SPECTRA_STATIONS:
        try:
            spectra = load_spectra(fcfg.RWS_SPECTRA_DIR, st, cfg.START_DATE, cfg.END_DATE)
        except FileNotFoundError:
            print(f"{st}: no spectra downloaded, skipping")
            continue
        if spectra.sizes["time"] == 0:
            print(f"{st}: no spectra in {cfg.START_DATE:%Y-%m-%d}..{cfg.END_DATE:%Y-%m-%d}, skipping")
            continue
        lat, lon = float(spectra.attrs["lat"]), float(spectra.attrs["lon"])
        stations[st] = dict(spectra=spectra, lat=lat, lon=lon, **station_cells(sea_mask, lat, lon))
    names = list(stations)

    cells = sorted({c for s in stations.values() for c in [s["nearest"], *s["corners"]]})
    cell_index = {c: i for i, c in enumerate(cells)}
    print(f"{len(names)} stations, {len(cells)} grid cells; reading GFS at those cells")
    pts = extract_gfs_cells(files, cells)
    valid = pts["valid_time"].values                                    # (time, lead)
    valid_times = np.unique(valid.ravel())
    n_time, n_lead = valid.shape
    hs, tp, dr = pts["hs"].values, pts["tp"].values, pts["dir"].values

    # merged GFS systems per (run, lead, cell), computed once and shared by stations with the same cell
    f_buoy, f_sel = spectrum_freqs()
    f_spec = f_buoy[f_sel]
    cache = {}
    gfs_spec_cache = {}

    def cell_spectrum(it, il, c):
        # rebuilt from the raw (unmerged) partitions: the energies add up, so merging is irrelevant here
        if (it, il, c) not in gfs_spec_cache:
            gfs_spec_cache[(it, il, c)] = synthesize_spectrum(gfs_parts(hs[it, il, c], tp[it, il, c], dr[it, il, c]), f_spec)
        return gfs_spec_cache[(it, il, c)]

    def cell_systems(it, il, c):
        if (it, il, c) not in cache:
            p = gfs_parts(hs[it, il, c], tp[it, il, c], dr[it, il, c])
            if fcfg.MERGE_GFS:
                p = merge_unresolvable(p, f_buoy, **fcfg.BUOY_PARTITION_KWARGS)
            cache[(it, il, c)] = p[np.isfinite(p[:, 0])] if len(p) else p
        return cache[(it, il, c)]

    station_times = {v: np.full((n_time, n_lead, len(names)), np.nan)
                     for v in ("gfs_htsgw", "gfs_u10", "gfs_v10", "gfs_perpw", "gfs_dirpw", "buoy_hm0")}
    spec_shape = (n_time, n_lead, len(names), len(f_spec))
    spectra_out = {v: np.full(spec_shape, np.nan, dtype=np.float32)
                   for v in ("buoy_energy", "buoy_direction", "buoy_spread", "gfs_energy", "gfs_direction")}
    valid_pos = np.searchsorted(valid_times, valid)                      # (time, lead) -> index in valid_times
    samples, misses = [], []
    for k, st in enumerate(names):
        s = stations[st]
        averaged, buoy = buoy_partitions(s["spectra"], valid_times)
        for v, src in (("buoy_energy", "energy_density"), ("buoy_direction", "direction"), ("buoy_spread", "spread")):
            spectra_out[v][:, :, k, :] = averaged[src].values[:, f_sel][valid_pos]
        b_all = np.stack([buoy.hs.values, buoy.fp.values, buoy.dir.values, buoy.spread.values], -1)
        b_hm0 = buoy["hs_total"].values
        tpos = {t: i for i, t in enumerate(valid_times)}
        near = cell_index[s["nearest"]]
        corners = [cell_index[c] for c in s["corners"]] or [near]
        w = s["weights"] if len(s["corners"]) else np.array([1.0])

        # bulk context: bilinear for the smooth fields, nearest for peak period/direction
        for v, src in (("gfs_htsgw", "htsgw"), ("gfs_u10", "u10"), ("gfs_v10", "v10")):
            station_times[v][:, :, k] = np.einsum("tlc,c->tl", pts[src].values[:, :, corners], w)
        station_times["gfs_perpw"][:, :, k] = pts["perpw"].values[:, :, near]
        station_times["gfs_dirpw"][:, :, k] = pts["dirpw"].values[:, :, near]

        for it in range(n_time):
            for il in range(n_lead):
                iv = tpos[valid[it, il]]
                if not np.isfinite(b_hm0[iv]):
                    continue                                    # no spectrum at this valid time
                station_times["buoy_hm0"][it, il, k] = b_hm0[iv]
                spectra_out["gfs_energy"][it, il, k], spectra_out["gfs_direction"][it, il, k] = cell_spectrum(it, il, near)
                g = cell_systems(it, il, near)
                b = b_all[iv][np.isfinite(b_all[iv][:, 0])]
                pairs, unmatched_g, unmatched_b = match_partitions(g, b, **fcfg.MATCH_KWARGS)
                partner = dict(pairs)
                key = dict(time_idx=it, lead_idx=il, station=st)
                for i in range(len(g)):
                    n_other, other_hs, other_dir = _other_systems(g, i)
                    j = partner.get(i)
                    row = dict(key, gfs_hs=g[i, 0], gfs_fp=g[i, 1], gfs_dir=g[i, 2], gfs_rank=i,
                               n_gfs_systems=len(g), other_hs=other_hs, other_dir=other_dir, matched=j is not None,
                               buoy_hs=np.nan, buoy_fp=np.nan, buoy_dir=np.nan, buoy_spread=np.nan, match_distance=np.nan)
                    if j is not None:
                        row.update(buoy_hs=b[j, 0], buoy_fp=b[j, 1], buoy_dir=b[j, 2], buoy_spread=b[j, 3],
                                   match_distance=float(partition_distance(g[i, 1], g[i, 2], b[j, 1], b[j, 2],
                                                                           fcfg.MATCH_KWARGS["sigma_f"],
                                                                           fcfg.MATCH_KWARGS["sigma_dir"])))
                    samples.append(row)
                for j in unmatched_b:
                    misses.append(dict(key, buoy_hs=b[j, 0], buoy_fp=b[j, 1], buoy_dir=b[j, 2], buoy_spread=b[j, 3],
                                       n_gfs_systems=len(g)))
        print(f"{st}: {sum(1 for r in samples if r['station'] == st)} GFS systems, "
              f"{sum(1 for r in misses if r['station'] == st)} missed buoy systems")

    coords_tl = {"time": pts.time.values, "lead_hours": pts.lead_hours.values}
    out = {}

    grid = gfs.copy()
    grid["sea_mask"] = sea_mask
    out["grid"] = grid

    near_lat = np.array([stations[st]["nearest"][0] for st in names])
    near_lon = np.array([stations[st]["nearest"][1] for st in names])
    lat = np.array([stations[st]["lat"] for st in names])
    lon = np.array([stations[st]["lon"] for st in names])
    out["stations"] = xr.Dataset({
        "lat": ("station", lat), "lon": ("station", lon),
        "nearest_lat": ("station", near_lat), "nearest_lon": ("station", near_lon),
        "nearest_km": ("station", _distance_km(lat, lon, near_lat, near_lon)),
        "n_sea_corners": ("station", np.array([len(stations[st]["corners"]) for st in names])),
        "station_group": ("station", group_points(near_lat, near_lon, cfg.BUOY_GROUP_DISTANCE_KM)),
        "station_name": ("station", np.array([stations[st]["spectra"].attrs.get("station_name", "") for st in names])),
    }, coords={"station": names})

    out["station_times"] = xr.Dataset(
        {v: (("time", "lead_hours", "station"), a) for v, a in station_times.items()} |
        {"valid_time": (("time", "lead_hours"), valid), "has_buoy": (("time", "lead_hours", "station"),
                                                                     np.isfinite(station_times["buoy_hm0"]))},
        coords=coords_tl | {"station": names})

    out["spectra"] = xr.Dataset(
        {v: (("time", "lead_hours", "station", "freq"), a) for v, a in spectra_out.items()},
        coords=coords_tl | {"station": names, "freq": f_spec})
    out["spectra"]["freq"].attrs = {"units": "Hz", "long_name": "centre of the 10 mHz buoy frequency bin"}
    for v in ("buoy_energy", "gfs_energy"):
        out["spectra"][v].attrs = {"units": "m2/Hz"}
    for v in ("buoy_direction", "gfs_direction", "buoy_spread"):
        out["spectra"][v].attrs = {"units": "degree"}
    out["spectra"]["gfs_energy"].attrs["long_name"] = "JONSWAP (gamma 3.3) per GFS partition, summed; only where a buoy spectrum exists"
    out["samples"] = _table_to_dataset(pd.DataFrame(samples), "sample", pts, valid)
    out["misses"] = _table_to_dataset(pd.DataFrame(misses), "miss", pts, valid)
    return out


def _table_to_dataset(df: pd.DataFrame, dim: str, pts: xr.Dataset, valid: np.ndarray) -> xr.Dataset:
    """Rows -> Dataset along dim, with issue time, lead and valid time added for convenience."""
    ds = xr.Dataset({c: (dim, df[c].values) for c in df.columns})
    it, il = df["time_idx"].values, df["lead_idx"].values
    ds["time"] = (dim, pts.time.values[it])
    ds["lead_hours"] = (dim, pts.lead_hours.values[il])
    ds["valid_time"] = (dim, valid[it, il])
    ds["station"] = ds["station"].astype(str)
    return ds


# ---------------------------------------------------------------------------
# 4. WRITE AND REPORT
# ---------------------------------------------------------------------------

def save(groups: dict[str, xr.Dataset]):
    attrs = {
        "description": "GFS-Wave wave systems (merged to buoy resolution) matched to RWS buoy spectrum partitions",
        "time": "issue time of the forecast run; valid_time = time + lead_hours",
        "merge_gfs": str(fcfg.MERGE_GFS),
        "match_kwargs": json.dumps(fcfg.MATCH_KWARGS),
        "buoy_partition_kwargs": json.dumps(fcfg.BUOY_PARTITION_KWARGS),
        "buoy_spectra_window": fcfg.BUOY_SPECTRA_WINDOW,
        "period": f"{cfg.START_DATE:%Y-%m-%d}..{cfg.END_DATE:%Y-%m-%d}",
    }
    if fcfg.PARTITION_ZARR.exists():
        shutil.rmtree(fcfg.PARTITION_ZARR)
    for name, ds in groups.items():
        ds = ds.copy()
        for v in ds.variables:
            ds[v].encoding = {}            # netCDF encodings are not valid for Zarr
        if name == "grid":
            ds = ds.chunk({"time": 8, **{d: -1 for d in ds.dims if d != "time"}})
        ds.attrs = attrs
        ds.to_zarr(str(fcfg.PARTITION_ZARR), group=name, mode="a" if fcfg.PARTITION_ZARR.exists() else "w")
        print(f"  wrote group {name}: {dict(ds.sizes)}")
    print(f"Saved to {fcfg.PARTITION_ZARR}")


def load(group: str) -> xr.Dataset:
    return xr.open_zarr(str(fcfg.PARTITION_ZARR), group=group)


def report(samples: xr.Dataset, misses: xr.Dataset, station_times: xr.Dataset):
    s = samples[["station", "lead_hours", "gfs_hs", "matched"]].to_dataframe()
    m = misses[["station", "lead_hours", "buoy_hs"]].to_dataframe()
    s["size"] = np.where(s.gfs_hs >= HS_SMALL, f">= {HS_SMALL} m", f"< {HS_SMALL} m")
    m["size"] = np.where(m.buoy_hs >= HS_SMALL, f">= {HS_SMALL} m", f"< {HS_SMALL} m")

    def counts(by):
        c = pd.DataFrame({
            "matched": s[s.matched].groupby(by).size(),
            "unmatched_gfs": s[~s.matched].groupby(by).size(),
            "unmatched_buoy": m.groupby(by).size(),
        }).fillna(0).astype(int)
        c["false_alarm_rate"] = (c.unmatched_gfs / (c.matched + c.unmatched_gfs)).round(3)
        c["hit_rate"] = (c.matched / (c.matched + c.unmatched_buoy)).round(3)
        return c

    has = station_times["has_buoy"]
    print(f"\nRun/lead/station combinations with a buoy spectrum: {int(has.sum())} of {has.size} "
          f"({float(has.mean()):.0%}); without one there are no samples")
    total = counts(lambda _: "all")
    print("\nAll systems:"); print(total.to_string())
    print("\nBy size (GFS Hs for samples, buoy Hs for misses):"); print(counts("size").to_string())
    print("\nBy station:"); print(counts("station").to_string())
    by_lead = counts("lead_hours")
    print("\nBy lead time:"); print(by_lead.loc[[l for l in (0, 24, 48, 72) if l in by_lead.index]].to_string())


# ---------------------------------------------------------------------------
# 5. GFS POINT SPECTRA IN THE BUOY FORMAT
# ---------------------------------------------------------------------------

def _rebin_cumulative(density: np.ndarray, edges_from: np.ndarray, edges_to: np.ndarray) -> np.ndarray:
    """Re-bin densities (..., n_from) per Hz to other bins, preserving energy: the cumulative integral is
    interpolated at the new edges (uniform density within an old bin) and differenced again."""
    cum = np.concatenate([np.zeros(density.shape[:-1] + (1,)), np.cumsum(density * np.diff(edges_from), -1)], -1)
    idx = np.clip(np.searchsorted(edges_from, edges_to) - 1, 0, len(edges_from) - 2)
    w = np.clip((edges_to - edges_from[idx]) / (edges_from[idx + 1] - edges_from[idx]), 0, 1)
    cum_to = cum[..., idx] * (1 - w) + cum[..., idx + 1] * w
    return np.diff(cum_to, axis=-1) / np.diff(edges_to)


def gfs_spectrum_to_buoy_bins(energy: np.ndarray, freq: np.ndarray, direction_to: np.ndarray):
    """GFS point spectra E(f, theta) -> the buoy format.

    energy (..., direction, frequency) in m2/Hz/rad on GFS's log-spaced frequencies, directions nautical
    "going to" (degrees). Returns energy density (..., n_bins) in m2/Hz and the mean direction per bin
    (degrees, "coming from") on the buoy's 10 mHz bins within the partition range. The mean direction is
    the energy-weighted vector average, the same definition as the buoy's mean direction per frequency.
    """
    f_buoy, f_sel = spectrum_freqs()
    centres = f_buoy[f_sel]
    edges_to = np.r_[centres - fcfg.RWS_SPECTRA_DF / 2, centres[-1] + fcfg.RWS_SPECTRA_DF / 2]
    ratio = freq[1] / freq[0]                                       # log-spaced: geometric bin edges
    edges_from = np.r_[freq[0] / np.sqrt(ratio), np.sqrt(freq[1:] * freq[:-1]), freq[-1] * np.sqrt(ratio)]

    dtheta = 2 * np.pi / len(direction_to)
    rad_from = np.deg2rad((np.asarray(direction_to) + 180) % 360)[:, None]
    missing = ~np.isfinite(energy).any(axis=(-2, -1))
    e = np.nan_to_num(energy)
    e_f = _rebin_cumulative(e.sum(-2) * dtheta, edges_from, edges_to)
    e_sin = _rebin_cumulative((e * np.sin(rad_from)).sum(-2) * dtheta, edges_from, edges_to)
    e_cos = _rebin_cumulative((e * np.cos(rad_from)).sum(-2) * dtheta, edges_from, edges_to)
    direction = np.where(e_f > 0, np.rad2deg(np.arctan2(e_sin, e_cos)) % 360, np.nan)
    e_f[missing] = np.nan
    direction[missing] = np.nan
    return e_f, direction


def load_gfs_station_spectra(times: np.ndarray, leads: np.ndarray) -> xr.Dataset:
    """Downloaded point spectra (download/gfs.py --spectra) in the buoy format
    -> (time, lead_hours, station, freq), stations named after the RWS station at the same spot."""
    f_buoy, f_sel = spectrum_freqs()
    names = list(fcfg.GFSWAVE_SPEC_STATIONS.values())
    shape = (len(times), len(leads), len(names), int(f_sel.sum()))
    energy = np.full(shape, np.nan, dtype=np.float32)
    direction = np.full(shape, np.nan, dtype=np.float32)
    n_found = 0
    for it, t in enumerate(pd.to_datetime(times)):
        path = fcfg.GFSWAVE_SPEC_DIR / f"gfswave_spec_{t:%Y%m%d}_{t:%H}z.nc"
        if not path.exists():
            continue
        with xr.open_dataset(path) as ds:
            ds = ds.load()
        n_found += 1
        e, d = gfs_spectrum_to_buoy_bins(ds["energy"].values, ds["frequency"].values, ds["direction"].values)
        for gfs_name, rws_name in fcfg.GFSWAVE_SPEC_STATIONS.items():
            if gfs_name not in ds.station.values:
                continue
            k = names.index(rws_name)
            s = list(ds.station.values).index(gfs_name)
            il = [list(leads).index(h) for h in ds.lead_hours.values if h in leads]
            src = [i for i, h in enumerate(ds.lead_hours.values) if h in leads]
            energy[it, il, k] = e[s, src]
            direction[it, il, k] = d[s, src]
    print(f"GFS point spectra: {n_found} of {len(times)} runs found in {fcfg.GFSWAVE_SPEC_DIR}")
    out = xr.Dataset(
        {"gfs_spec_energy": (("time", "lead_hours", "station", "freq"), energy, {"units": "m2/Hz"}),
         "gfs_spec_direction": (("time", "lead_hours", "station", "freq"), direction,
                                {"units": "degree", "long_name": "mean direction per bin, coming from"})},
        coords={"time": times, "lead_hours": leads, "station": names, "freq": f_buoy[f_sel]},
        attrs={"description": "GFS-Wave point spectra (station output) re-binned to the buoy frequency bins",
               "gfs_stations": json.dumps(fcfg.GFSWAVE_SPEC_STATIONS)})
    return out


def append_gfs_station_spectra():
    """Add (or replace) the group gfs_station_spectra in fcfg.PARTITION_ZARR, on the runs and lead
    times of the existing dataset, without rebuilding the other groups."""
    st = load("station_times")
    ds = load_gfs_station_spectra(st["time"].values, st["lead_hours"].values)
    group_dir = fcfg.PARTITION_ZARR / "gfs_station_spectra"
    if group_dir.exists():
        shutil.rmtree(group_dir)
    ds.to_zarr(str(fcfg.PARTITION_ZARR), group="gfs_station_spectra", mode="a")
    print(f"  wrote group gfs_station_spectra: {dict(ds.sizes)}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build the forecast dataset")
    ap.add_argument("--station-spectra", action="store_true",
                    help="only (re)build the gfs_station_spectra group in the existing dataset")
    args = ap.parse_args()
    if not args.station_spectra:
        groups = build_partition_dataset()
        save(groups)
        report(groups["samples"], groups["misses"], groups["station_times"])
    append_gfs_station_spectra()
