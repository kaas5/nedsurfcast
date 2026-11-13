"""
How large is the coarse-grid (gradient) bias compared with the GFS error?
==========================================================================

For every spectrum station, compares wave systems (partitions) of:

    GFS nearest  vs buoy      the GFS error at the nearest sea cell (what the pipeline uses now)
    GFS interp   vs buoy      the GFS error after interpolating the partitions to the station
    neighbour    vs nearest   difference between the 4 grid points around the station
    GFS interp   vs nearest   what interpolation changes = the gradient bias it would remove

Partitions are matched per sample with common.partition.match_partitions (peak frequency
and direction), so the slot order of GFS and buoy plays no role. The same comparisons are
made for the bulk wave height (HTSGW vs the buoy's Hm0) as a reference.

GFS partitions: wind sea (WVHGT/WVPER/WVDIR) + swell 1-3 (SWELL/SWPER/SWDIR); the periods are
peak periods and the directions "coming from", like the buoy (checked at K13).

MERGE_GFS: first merge GFS partitions that the 1D buoy partitioning could not separate
(common.partition.merge_unresolvable), so both sides have the same resolution.

Reading: if "interp vs nearest" is much smaller than "nearest vs buoy", the gradient bias is
negligible and the nearest sea cell is good enough. If "interp vs buoy" is clearly lower than
"nearest vs buoy", interpolating the partitions is worth it.

Run from anywhere with: python -m nedsurfcast.forecast.partition_differences
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from nedsurfcast.common.partition import (average_spectra, interpolate_partitions, load_spectra,
                                          match_partitions, merge_unresolvable, partition_dataset, _angle_diff)
from nedsurfcast.forecast import config as fcfg
from nedsurfcast.forecast.preprocess import extract_gfs_cells, gfs_files, gfs_parts, station_cells
from nedsurfcast import config as cfg

HS_MIN = 0.3            # m; systems below this count neither as hit nor as miss (but can still be matched)
MERGE_GFS = True        # merge GFS partitions the buoy cannot resolve before comparing

COMPARISONS = ["GFS nearest vs buoy", "GFS interp vs buoy", "neighbour vs nearest", "GFS interp vs nearest"]


# ---------------------------------------------------------------------------
# 3. COMPARE TWO PARTITION SETS
# ---------------------------------------------------------------------------

def compare(a: np.ndarray, b: np.ndarray, records: list, **meta):
    """Match source a against reference b; store matched pairs and hit/false-alarm counts."""
    pairs, unmatched_a, unmatched_b = match_partitions(a, b)
    for i, j in pairs:
        if max(a[i, 0], b[j, 0]) < HS_MIN:
            continue
        records.append(dict(meta, kind="pair", d_hs=a[i, 0] - b[j, 0], d_tp=1 / a[i, 1] - 1 / b[j, 1],
                            d_dir=float(_angle_diff(a[i, 2], b[j, 2])), ref_hs=b[j, 0]))
    big_b = [j for j in range(len(b)) if np.isfinite(b[j, 0]) and b[j, 0] >= HS_MIN]
    big_a = [i for i in range(len(a)) if np.isfinite(a[i, 0]) and a[i, 0] >= HS_MIN]
    records.append(dict(meta, kind="count", n_ref=len(big_b), n_hit=sum(j not in unmatched_b for j in big_b),
                        n_src=len(big_a), n_false=sum(i in unmatched_a for i in big_a)))


def summarize(records: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    pairs = records[records.kind == "pair"]
    counts = records[records.kind == "count"]
    rmse = lambda x: float(np.sqrt(np.mean(np.square(x))))
    stats = pairs.groupby(by).agg(n_pairs=("d_hs", "size"), rmse_hs=("d_hs", rmse), bias_hs=("d_hs", "mean"),
                                  rmse_tp=("d_tp", rmse), mae_dir=("d_dir", "mean"))
    c = counts.groupby(by)[["n_ref", "n_hit", "n_src", "n_false"]].sum()
    stats["hit_rate"] = c["n_hit"] / c["n_ref"]
    stats["false_alarm"] = c["n_false"] / c["n_src"]
    return stats.round(3)


# ---------------------------------------------------------------------------
# 4. RUN
# ---------------------------------------------------------------------------

def run():
    files = gfs_files()
    with xr.open_dataset(files[0]) as first:
        grid = first[["swh"]].isel(lead_hours=0).load()

    stations = {}
    for st in fcfg.RWS_SPECTRA_STATIONS:
        try:
            spec = load_spectra(fcfg.RWS_SPECTRA_DIR, st, cfg.START_DATE, cfg.END_DATE)
        except FileNotFoundError:
            print(f"{st}: no spectra downloaded, skipping")
            continue
        lat, lon = float(spec.attrs["lat"]), float(spec.attrs["lon"])
        stations[st] = dict(spec=spec, lat=lat, lon=lon, **station_cells(grid["swh"].notnull(), lat, lon))

    cells = sorted({c for s in stations.values() for c in [s["nearest"], *s["corners"]]})
    cell_index = {c: i for i, c in enumerate(cells)}
    print(f"{len(stations)} stations, {len(cells)} grid cells, reading {len(files)} GFS runs")
    gfs = extract_gfs_cells(files, cells)
    valid = gfs["valid_time"].values                                   # (time, lead)

    # GFS partitions per (run, lead, cell), merged to the buoy's resolution once and shared by all stations
    f_buoy = np.arange(fcfg.RWS_SPECTRA_NBINS) * fcfg.RWS_SPECTRA_DF + fcfg.RWS_SPECTRA_DF / 2
    cache = {}

    def cell_parts(it, il, c):
        if (it, il, c) not in cache:
            p = gfs_parts(hs[it, il, c], tp[it, il, c], dr[it, il, c])
            cache[(it, il, c)] = merge_unresolvable(p, f_buoy, **fcfg.BUOY_PARTITION_KWARGS) if MERGE_GFS else p
        return cache[(it, il, c)]

    hs, tp, dr, ht = (gfs[v].values for v in ("hs", "tp", "dir", "htsgw"))
    records, bulk, info = [], [], []
    for st, s in stations.items():
        spectra = average_spectra(s["spec"], window=fcfg.BUOY_SPECTRA_WINDOW)
        times = np.unique(valid.ravel())
        spectra = spectra.reindex(time=times)
        buoy = partition_dataset(spectra, max_partitions=6, **fcfg.BUOY_PARTITION_KWARGS)
        buoy_arr = np.stack([buoy.hs.values, buoy.fp.values, buoy.dir.values], -1)   # (valid, partition, 3)
        buoy_hm0 = buoy["hs_total"].values
        tpos = {t: i for i, t in enumerate(times)}

        near = cell_index[s["nearest"]]
        corners = [cell_index[c] for c in s["corners"]]
        w = s["weights"]
        ref_corner = corners[int(np.argmax(w))] if corners else None
        dist = lambda c: np.hypot((c[0] - s["lat"]) * 110.57, (c[1] - s["lon"]) * 111.32 * np.cos(np.deg2rad(s["lat"])))
        info.append(dict(station=st, lat=s["lat"], lon=s["lon"], nearest_km=round(dist(s["nearest"]), 1),
                         sea_corners=len(corners)))

        for it in range(gfs.sizes["time"]):
            for il, lead in enumerate(gfs.lead_hours.values):
                meta = dict(station=st, lead=int(lead))
                parts = {c: cell_parts(it, il, c) for c in set([near, *corners])}
                interp = interpolate_partitions([parts[c] for c in corners], w) if len(corners) >= 2 else None
                b = buoy_arr[tpos[valid[it, il]]]
                has_buoy = np.isfinite(buoy_hm0[tpos[valid[it, il]]])

                if has_buoy:
                    compare(parts[near], b, records, comparison="GFS nearest vs buoy", **meta)
                    if interp is not None:
                        compare(interp, b, records, comparison="GFS interp vs buoy", **meta)
                if interp is not None:
                    compare(interp, parts[near], records, comparison="GFS interp vs nearest", **meta)
                    for c in corners:
                        if c != ref_corner:
                            compare(parts[c], parts[ref_corner], records, comparison="neighbour vs nearest", **meta)

                # bulk wave height as reference
                h_near = ht[it, il, near]
                h_interp = float(np.sum(w * ht[it, il, corners])) if len(corners) >= 2 else np.nan
                bulk.append(dict(station=st, lead=int(lead), near=h_near, interp=h_interp,
                                 buoy=buoy_hm0[tpos[valid[it, il]]] if has_buoy else np.nan,
                                 neighbour=np.nanmax(np.abs(ht[it, il, corners] - ht[it, il, ref_corner])) if len(corners) >= 2 else np.nan))
        print(f"{st}: done")

    return pd.DataFrame(records), pd.DataFrame(bulk), pd.DataFrame(info).set_index("station")


def bulk_summary(bulk: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame({
        "station": bulk.station,
        "rmse_near_vs_buoy": (bulk.near - bulk.buoy) ** 2,
        "rmse_interp_vs_buoy": (bulk.interp - bulk.buoy) ** 2,
        "bias_near_vs_buoy": bulk.near - bulk.buoy,
        "rms_interp_vs_near": (bulk.interp - bulk.near) ** 2,
        "rms_max_neighbour_diff": bulk.neighbour ** 2,
    })
    out = d.groupby("station").mean()                 # NaN-skipping means
    squared = [c for c in out.columns if c != "bias_near_vs_buoy"]
    out[squared] = np.sqrt(out[squared])
    return out.round(3)


def plot(stats: pd.DataFrame, info: pd.DataFrame):
    order = info.sort_values("lat").index
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    width = 0.2
    x = np.arange(len(order))
    for ax, col, label in zip(axes, ["rmse_hs", "rmse_tp", "mae_dir"], ["RMSE Hs (m)", "RMSE Tp (s)", "mean |Δdir| (°)"]):
        for k, comp in enumerate(COMPARISONS):
            vals = [stats[col].get((st, comp), np.nan) for st in order]
            ax.bar(x + (k - 1.5) * width, vals, width, label=comp)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(ncols=4, fontsize=8, loc="upper left")
    axes[-1].set_xticks(x, [f"{st}\n{info.loc[st, 'nearest_km']:.0f} km, {info.loc[st, 'sea_corners']} sea" for st in order], fontsize=8)
    fig.suptitle(f"Matched wave systems (Hs >= {HS_MIN} m, GFS merged to buoy resolution: {MERGE_GFS}): "
                 "GFS error vs grid-cell differences, all lead times")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    records, bulk, info = run()

    print("\nStations (nearest_km = distance to the nearest sea cell, sea_corners = sea points among the 4 around it)")
    print(info.to_string())

    stats = summarize(records, ["station", "comparison"])
    print(f"\nMatched wave systems with Hs >= {HS_MIN} m, all lead times")
    print(stats.to_string())

    print("\nAll stations together")
    print(summarize(records, ["comparison"]).to_string())

    print("\nGFS vs buoy per lead time (all stations)")
    lead_stats = summarize(records[records.comparison.isin(COMPARISONS[:2])], ["comparison", "lead"])
    print(lead_stats.loc[(slice(None), [0, 24, 48, 72]), :].to_string())

    print("\nBulk wave height: HTSGW vs buoy Hm0 (m)")
    print(bulk_summary(bulk).to_string())

    plot(stats, info)
    plt.show()
