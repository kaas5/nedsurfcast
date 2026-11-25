"""
Live forecasts of the corrected spectrum at the buoy stations
===============================================================

From one GFS-Wave run (00z, as in training) the inputs are built exactly as forecast/preprocess.py and
forecast/load_data.py build them for training, the saved model (fcfg.FORECAST_MODEL, forecast/checkpoints.py)
predicts the correction, and the result is written as JSON.

Inputs (download/gfs.py --live, in fcfg.GFSWAVE_LIVE_DIR; the same file format as the archive):
    run at the issue time T, f000-f072        GFS spectrum (rebuilt from the partitions) and bulk context
    run at T - 24 h, f000-f021                nowcast at 00/03z of the previous day; history fallback
    06/12/18z runs of the previous day, f000/f003   nowcast: history features before T

Per station (preprocess section 2 and 6): the nearest sea cell gives the partitions -> rebuilt spectrum
(common.partition.synthesize_spectrum for the model input, preprocess._spectra_from_parts for the history,
both stored as float32 like the zarr), HTSGW and wind are bilinear over the sea corners, peak period and
direction nearest. load_data.history_pairs_from gives the same history pairs as in training. Location and
fetch features come from the zarr group station_location (static per station).

Output (fcfg.FORECAST_OUTPUT_DIR/forecast_YYYYMMDD_HHz.json and forecast_latest.json):
    issue_time, model, model_timestamp, created, freq (Hz), units, period_bands (Hz), stations: list of
        id, name, lat, lon, evaluated (False for fcfg.TRAIN_ONLY_STATIONS: trained on, never tested)
        forecast: per lead time  lead_hours, valid_time, and for "corrected" (model) and "gfs" (the rebuilt
            GFS spectrum, uncorrected):
                energy      m2/Hz per frequency bin
                direction   deg (coming from), energy-weighted mean per bin; null in empty bins
                hm0         m: total and per period band (as in the training scores)
                partitions  common.partition.partition_spectrum with fcfg.BUOY_PARTITION_KWARGS (as for the
                            buoys): hs, tp, fp, dir, f_lo, f_hi; spread is null (the model has no spread)
NaN is written as null.

Run from anywhere with: python -m nedsurfcast.forecast.predict [--issue YYYYMMDDHH] [--model NAME] [--no-download]
"""

import datetime as dt
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from nedsurfcast.common.partition import partition_spectrum, synthesize_spectrum
from nedsurfcast.forecast import config as fcfg
from nedsurfcast.forecast.checkpoints import load_model
from nedsurfcast.forecast.load_data import PERIOD_BANDS, corrected_spectrum, features, history_pairs_from
from nedsurfcast.forecast.preprocess import (_spectra_from_parts, extract_gfs_cells, gfs_parts, load,
                                             spectrum_freqs, station_cells)
from nedsurfcast.forecast.scoring import band_hm0

PREVIOUS_LEADS = list(range(0, 24, 3))     # of the previous day's run (download.gfs.LIVE_PREVIOUS_LEADS)
DECIMALS = dict(energy=5, direction=1, hm0=3, partition=4)


# ---------------------------------------------------------------------------
# 1. INPUTS, AS IN TRAINING
# ---------------------------------------------------------------------------

def _run_file(run_dir: Path, issue: dt.datetime) -> Path:
    path = Path(run_dir) / f"gfswave_{issue:%Y%m%d}_{issue:%H}z.nc"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: python -m nedsurfcast.download.gfs --live --issue {issue:%Y%m%d%H}")
    return path


def station_setup() -> dict:
    """Stations of the dataset (fcfg.RWS_SPECTRA_STATIONS) with their nearest sea cell and bilinear corners."""
    st = load("stations").load()
    sea_mask = load("grid")["sea_mask"].load()
    names = [s for s in st["station"].values if s in fcfg.RWS_SPECTRA_STATIONS]
    out = {"names": names, "lat": [], "lon": [], "label": [], "nearest": [], "corners": [], "weights": []}
    for s in names:
        lat, lon = float(st["lat"].sel(station=s)), float(st["lon"].sel(station=s))
        c = station_cells(sea_mask, lat, lon)
        stored = (float(st["nearest_lat"].sel(station=s)), float(st["nearest_lon"].sel(station=s)))
        assert np.allclose(c["nearest"], stored), f"{s}: nearest cell {c['nearest']} != dataset {stored}"
        out["lat"].append(lat)
        out["lon"].append(lon)
        out["label"].append(str(st["station_name"].sel(station=s).values))
        out["nearest"].append(stored)
        out["corners"].append(c["corners"] or [stored])
        out["weights"].append(c["weights"] if len(c["corners"]) else np.array([1.0]))
    return out


def _cells(stations: dict) -> tuple[list, dict]:
    cells = sorted({c for k in range(len(stations["names"])) for c in [stations["nearest"][k], *stations["corners"][k]]})
    return cells, {c: i for i, c in enumerate(cells)}


def _parts(pts: xr.Dataset, cell_idx: list[int]) -> np.ndarray:
    """extract_gfs_cells output of one run -> partitions (lead, station, 4, 3) with (hs, tp, dir)."""
    return np.stack([pts[v].values[0][:, cell_idx] for v in ("hs", "tp", "dir")], -1)


def _rebuilt(parts: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(..., 4, 3) -> history spectra (..., freq) as in the zarr groups gfs_rebuilt / gfs_nowcast (float32)."""
    e, d = _spectra_from_parts(parts.reshape(-1, 4, 3), f)
    shape = parts.shape[:-2] + (len(f),)
    return (e.reshape(shape).astype(np.float32).astype(float), d.reshape(shape).astype(np.float32).astype(float))


def build_inputs(issue: dt.datetime, run_dir: Path = None, nowcast_dir: Path = None, stations: dict = None) -> dict:
    """The `data` dict of load_data.load_data for every (lead, station) of run `issue`, without buoy data:
    gfs_energy, gfs_direction, freq, index, meta, history_pairs (so load_data.features works unchanged).
    run_dir / nowcast_dir default to the live folders; the archive folders give the training inputs."""
    run_dir = Path(run_dir or fcfg.GFSWAVE_LIVE_DIR)
    nowcast_dir = Path(nowcast_dir or fcfg.GFSWAVE_LIVE_DIR / "nowcast")
    stations = stations or station_setup()
    names = stations["names"]
    f_buoy, f_sel = spectrum_freqs()
    f = f_buoy[f_sel]
    cells, cell_index = _cells(stations)
    near = [cell_index[c] for c in stations["nearest"]]
    leads = np.array(fcfg.GFSWAVE_LEAD_HOURS)
    previous = issue - dt.timedelta(days=1)

    # the run itself
    pts = extract_gfs_cells([_run_file(run_dir, issue)], cells)
    if list(pts["lead_hours"].values) != list(leads):
        raise ValueError(f"run {issue:%Y%m%d %H}z has lead times {pts['lead_hours'].values}, expected {leads}")
    parts = _parts(pts, near)                                              # (lead, station, 4, 3)
    n_lead, n_st = parts.shape[:2]
    gfs_e = np.full((n_lead, n_st, len(f)), np.nan, dtype=np.float32)
    gfs_d = np.full_like(gfs_e, np.nan)
    for il in range(n_lead):
        for k in range(n_st):
            gfs_e[il, k], gfs_d[il, k] = synthesize_spectrum(gfs_parts(*parts[il, k].T), f, fcfg.GFS_SPECTRUM_SHAPE)
    bulk = {}
    for v, src in (("gfs_htsgw", "htsgw"), ("gfs_u10", "u10"), ("gfs_v10", "v10")):
        a = pts[src].values[0]                                             # (lead, cell)
        bulk[v] = np.stack([a[:, [cell_index[c] for c in stations["corners"][k]]] @ stations["weights"][k]
                            for k in range(n_st)], 1)
    bulk["gfs_perpw"] = pts["perpw"].values[0][:, near].astype(float)     # float64, as station_times
    bulk["gfs_dirpw"] = pts["dirpw"].values[0][:, near].astype(float)

    # rebuilt spectra of this run and of the previous day's run (time, lead, station, freq)
    e_all = np.full((2, n_lead, n_st, len(f)), np.nan)
    d_all = np.full_like(e_all, np.nan)
    e_all[1], d_all[1] = _rebuilt(parts, f)
    prev = extract_gfs_cells([_run_file(run_dir, previous)], cells)
    prev_leads = list(prev["lead_hours"].values)
    if not set(PREVIOUS_LEADS) <= set(prev_leads):
        raise ValueError(f"run {previous:%Y%m%d %H}z lacks lead times {sorted(set(PREVIOUS_LEADS) - set(prev_leads))}")
    pe, pd_ = _rebuilt(_parts(prev, near), f)
    for j, h in enumerate(prev_leads):
        if h in leads:
            e_all[0, list(leads).index(h)], d_all[0, list(leads).index(h)] = pe[j], pd_[j]

    # nowcast: f000/f003 of the previous day's 00z run and of its 06/12/18z runs (rebuild_gfs_nowcast)
    nc_valid, nc_parts = [], []
    for run in [previous] + [previous.replace(hour=h) for h in fcfg.GFSWAVE_NOWCAST_CYCLES]:
        src = prev if run == previous else extract_gfs_cells([_run_file(nowcast_dir, run)], cells)
        p = _parts(src, near)
        for j, h in enumerate(src["lead_hours"].values):
            if int(h) in fcfg.GFSWAVE_NOWCAST_LEAD_HOURS:
                nc_valid.append(np.datetime64(run + dt.timedelta(hours=int(h)), "ns"))
                nc_parts.append(p[j])
    order = np.argsort(nc_valid)
    nc_times = np.array(nc_valid)[order]
    e_nc, d_nc = _rebuilt(np.stack(nc_parts)[order], f)

    # rows ordered like load_data: (time, lead, station)
    il, ist = (a.ravel() for a in np.meshgrid(np.arange(n_lead), np.arange(n_st), indexing="ij"))
    it = np.ones_like(il)
    t_issue = np.datetime64(issue, "ns")
    meta = pd.DataFrame({"time": np.full(len(il), t_issue), "station": np.array(names)[ist],
                         "lead_hours": leads[il]} | {v: a[il, ist] for v, a in bulk.items()})
    data = {"gfs_energy": gfs_e[il, ist].astype(float), "gfs_direction": gfs_d[il, ist].astype(float),
            "freq": f, "index": (it, il, ist), "meta": meta}
    data["history_pairs"] = history_pairs_from(e_all, d_all, np.array([np.datetime64(previous, "ns"), t_issue]),
                                               leads, e_nc, d_nc, nc_times, data["index"])
    return data


# ---------------------------------------------------------------------------
# 2. FORECAST AND JSON
# ---------------------------------------------------------------------------

def _round(a, decimals):
    """Array -> nested list with NaN as None."""
    a = np.round(np.asarray(a, float), decimals)
    return [None if not np.isfinite(v) else float(v) for v in a]


def spectrum_summary(energy: np.ndarray, direction: np.ndarray, f: np.ndarray, spread: np.ndarray = None) -> dict:
    """One spectrum -> {energy, direction, hm0, partitions} as in the forecast JSON (module docstring).
    spread (deg per bin, measured spectra only) gives the partitions a spread; without it spread is null."""
    e = energy[None]
    hm0 = {"total": band_hm0(e, f)[0]} | {band: band_hm0(e, f, lo, hi)[0] for band, (lo, hi) in PERIOD_BANDS.items()}
    parts = partition_spectrum(f, energy, direction, spread, **fcfg.BUOY_PARTITION_KWARGS)
    keys = ("hs", "tp", "fp", "dir", "spread", "f_lo", "f_hi")
    return {"energy": _round(energy, DECIMALS["energy"]), "direction": _round(direction, DECIMALS["direction"]),
            "hm0": dict(zip(hm0, _round(list(hm0.values()), DECIMALS["hm0"]))),
            "partitions": [dict(zip(keys, _round([p[k] for k in keys], DECIMALS["partition"]))) for p in parts]}


def forecast(issue: dt.datetime, model_name: str = None, run_dir: Path = None, nowcast_dir: Path = None) -> dict:
    """Corrected spectra of run `issue` at every station and lead time -> the JSON structure (module docstring)."""
    model = load_model(model_name or fcfg.FORECAST_MODEL)
    stations = station_setup()
    data = build_inputs(issue, run_dir, nowcast_dir, stations)
    f = data["freq"]
    if not np.allclose(model.freq, f):
        raise ValueError("the model was trained on other frequency bins")
    energy, direction = corrected_spectrum(data, model.predict(features(data, model.feature_set)))
    m = data["meta"]
    out = {"issue_time": f"{issue:%Y-%m-%dT%H:%MZ}", "model": model.name, "model_timestamp": model.timestamp,
           "created": f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%MZ}", "freq": _round(f, 4),
           "units": {"energy": "m2/Hz", "direction": "degree, coming from", "hm0": "m", "hs": "m", "tp": "s",
                     "fp": "Hz", "dir": "degree, coming from", "f_lo": "Hz", "f_hi": "Hz"},
           "period_bands": {band: [lo, hi] for band, (lo, hi) in PERIOD_BANDS.items()}, "stations": []}
    for k, name in enumerate(stations["names"]):
        rows = np.flatnonzero(m["station"].to_numpy() == name)
        rows = rows[np.argsort(m["lead_hours"].to_numpy()[rows])]
        out["stations"].append({
            "id": name, "name": stations["label"][k], "lat": stations["lat"][k], "lon": stations["lon"][k],
            "evaluated": name not in fcfg.TRAIN_ONLY_STATIONS,
            "forecast": [{"lead_hours": int(m["lead_hours"].iat[r]),
                          "valid_time": f"{issue + dt.timedelta(hours=int(m['lead_hours'].iat[r])):%Y-%m-%dT%H:%MZ}",
                          "corrected": spectrum_summary(energy[r], direction[r], f),
                          "gfs": spectrum_summary(data["gfs_energy"][r], data["gfs_direction"][r], f)}
                         for r in rows]})
    return out


def write(result: dict, out_dir: Path = None) -> Path:
    out_dir = Path(out_dir or fcfg.FORECAST_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    issue = dt.datetime.strptime(result["issue_time"], "%Y-%m-%dT%H:%MZ")
    path = out_dir / f"forecast_{issue:%Y%m%d}_{issue:%H}z.json"
    path.write_text(json.dumps(result, separators=(",", ":")), encoding="utf8")
    shutil.copyfile(path, out_dir / "forecast_latest.json")
    return path


if __name__ == "__main__":
    import argparse

    from nedsurfcast.download.gfs import download_live, latest_issue

    ap = argparse.ArgumentParser(description="Live forecast of the corrected spectrum at the buoy stations")
    ap.add_argument("--issue", help="issue time YYYYMMDDHH (default: the newest complete 00z run)")
    ap.add_argument("--model", help=f"model name in MODELS (default {fcfg.FORECAST_MODEL})")
    ap.add_argument("--no-download", action="store_true", help="use the files already in GFSWAVE_LIVE_DIR")
    args = ap.parse_args()
    issue = dt.datetime.strptime(args.issue, "%Y%m%d%H") if args.issue else None
    if not args.no_download:
        issue = download_live(issue)
    elif issue is None:
        issue = latest_issue()
    path = write(forecast(issue, args.model))
    print(f"forecast of the {issue:%Y-%m-%d %H}z run written to {path} (and forecast_latest.json)")
