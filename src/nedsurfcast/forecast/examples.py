"""
Historical forecasts with the best and the worst corrections, as JSON
=======================================================================

Corrects every historical forecast in the dataset with the forecast model (fcfg.FORECAST_MODEL) and saves the
runs where the correction helped most and where it hurt most, in the JSON format of forecast/predict.py plus
the measured buoy spectrum.

Out of sample: the saved final model was trained on all of these forecasts, so its corrections here would be
too good. Instead every forecast is corrected by the cross-validation model that never saw it
(scoring.station_group_folds: trained without the station's group and without its test months), as in
train_vs_buoys.py.

A case is one run at one station (all its lead times). Score over lead times >= scoring.EVAL_MIN_LEAD:

    skill = 1 - RMSE(corrected sqrt(E)) / RMSE(GFS sqrt(E))     against the buoy, over all bins and lead times

Only cases with at least MIN_LEADS scored lead times and a mean measured Hm0 of at least MIN_HM0 (calm seas
give tiny errors and arbitrary ratios), not at fcfg.TRAIN_ONLY_STATIONS (never tested). Best: the
N_CASES highest skills, worst: the N_CASES lowest, at most one case per run (no five stations of one storm).

Output in fcfg.EXAMPLES_DIR:
    best_<k>_<station>_<YYYYMMDD>_<HH>z.json, worst_<k>_...json   the forecast JSON of predict.py with one
        station; per lead time also "measured" (the averaged buoy spectrum, partitions with spread), and
        "case" with the scores. Only lead times with a buoy spectrum (the rows of the dataset).
    index.json      every case: file, category, rank, station, issue time and the scores

Run from anywhere with: python -m nedsurfcast.forecast.examples
"""

import datetime as dt
import json

import numpy as np
import pandas as pd

from nedsurfcast.forecast import config as fcfg
from nedsurfcast.forecast.load_data import PERIOD_BANDS, corrected_spectrum, load_data
from nedsurfcast.forecast.models import MODELS, predict_folds
from nedsurfcast.forecast.predict import _round, spectrum_summary
from nedsurfcast.forecast.preprocess import load
from nedsurfcast.forecast.scoring import EVAL_MIN_LEAD, band_hm0, station_group_folds

N_CASES = 5
MIN_LEADS = 20          # of the 24 lead times >= EVAL_MIN_LEAD
MIN_HM0 = 0.5           # m, mean measured Hm0 over the scored lead times


def case_scores(data: dict, corrected: tuple, pred_ok: np.ndarray) -> pd.DataFrame:
    """One row per (run, station): skill, RMSE of sqrt(E) and Hm0 of the correction and of GFS."""
    m = data["meta"]
    f = data["freq"]
    eb, eg, ec = data["buoy_energy"], data["gfs_energy"], corrected[0]
    rows = pred_ok & (m["lead_hours"].to_numpy() >= EVAL_MIN_LEAD) & np.isfinite(eb).all(1)
    hb, hg, hc = band_hm0(eb, f), band_hm0(eg, f), band_hm0(ec, f)
    df = pd.DataFrame({
        "time": m["time"], "station": m["station"],
        "se_corr": ((np.sqrt(ec) - np.sqrt(eb)) ** 2).sum(1), "se_gfs": ((np.sqrt(eg) - np.sqrt(eb)) ** 2).sum(1),
        "he_corr": (hc - hb) ** 2, "he_gfs": (hg - hb) ** 2, "hm0": hb, "n": 1}).loc[rows]
    g = df.groupby(["time", "station"]).agg(se_corr=("se_corr", "sum"), se_gfs=("se_gfs", "sum"),
                                             he_corr=("he_corr", "mean"), he_gfs=("he_gfs", "mean"),
                                             hm0=("hm0", "mean"), n=("n", "sum"))
    nf = len(f)
    out = pd.DataFrame({
        "skill_sqrt_e": 1 - np.sqrt(g["se_corr"] / g["se_gfs"]),
        "rmse_sqrt_e_corrected": np.sqrt(g["se_corr"] / (g["n"] * nf)), "rmse_sqrt_e_gfs": np.sqrt(g["se_gfs"] / (g["n"] * nf)),
        "rmse_hm0_corrected": np.sqrt(g["he_corr"]), "rmse_hm0_gfs": np.sqrt(g["he_gfs"]),
        "mean_measured_hm0": g["hm0"], "n_leads": g["n"]})
    return out.reset_index()


def pick(scores: pd.DataFrame, best: bool) -> pd.DataFrame:
    ok = (scores["n_leads"] >= MIN_LEADS) & (scores["mean_measured_hm0"] >= MIN_HM0) & \
         ~scores["station"].isin(fcfg.TRAIN_ONLY_STATIONS)
    s = scores[ok].sort_values("skill_sqrt_e", ascending=not best)
    return s.drop_duplicates("time").head(N_CASES)              # at most one case per run


def case_json(data, corrected, spread, fold_of_row, case: pd.Series, category: str, rank: int, model_name: str) -> dict:
    m = data["meta"]
    f = data["freq"]
    st = load("stations").load()
    name = case["station"]
    issue = pd.Timestamp(case["time"]).to_pydatetime()
    rows = np.flatnonzero((m["time"].to_numpy() == np.datetime64(case["time"])) & (m["station"].to_numpy() == name))
    rows = rows[np.argsort(m["lead_hours"].to_numpy()[rows])]
    scores = {k: (float(round(v, 4)) if isinstance(v, float) else int(v)) for k, v in case.items()
              if k not in ("time", "station")}
    return {
        "issue_time": f"{issue:%Y-%m-%dT%H:%MZ}", "model": model_name,
        "model_timestamp": None, "evaluation": "out of sample: cross-validation model trained without this "
                                               "station group and without these months",
        "created": f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%MZ}", "freq": _round(f, 4),
        "units": {"energy": "m2/Hz", "direction": "degree, coming from", "hm0": "m", "hs": "m", "tp": "s",
                  "fp": "Hz", "dir": "degree, coming from", "spread": "degree", "f_lo": "Hz", "f_hi": "Hz"},
        "period_bands": {band: [lo, hi] for band, (lo, hi) in PERIOD_BANDS.items()},
        "case": {"category": category, "rank": rank, "station": name, "fold": fold_of_row[rows[0]],
                 "score": "skill_sqrt_e = 1 - RMSE(corrected sqrt E) / RMSE(GFS sqrt E) vs the buoy, "
                          f"lead times >= {EVAL_MIN_LEAD} h"} | scores,
        "stations": [{
            "id": name, "name": str(st["station_name"].sel(station=name).values),
            "lat": float(st["lat"].sel(station=name)), "lon": float(st["lon"].sel(station=name)),
            "evaluated": name not in fcfg.TRAIN_ONLY_STATIONS,
            "forecast": [{"lead_hours": int(m["lead_hours"].iat[r]),
                          "valid_time": f"{issue + dt.timedelta(hours=int(m['lead_hours'].iat[r])):%Y-%m-%dT%H:%MZ}",
                          "corrected": spectrum_summary(corrected[0][r], corrected[1][r], f),
                          "gfs": spectrum_summary(data["gfs_energy"][r], data["gfs_direction"][r], f),
                          "measured": spectrum_summary(data["buoy_energy"][r], data["buoy_direction"][r], f, spread[r])}
                         for r in rows]}]}


def run(model_name: str = None, n_cases: int = None):
    global N_CASES
    N_CASES = n_cases or N_CASES
    model_name = model_name or fcfg.FORECAST_MODEL
    data = load_data()
    folds = station_group_folds(data)
    print(f"{len(data['meta'])} forecasts (run, lead, station); out-of-sample {model_name} over {len(folds)} folds")
    pred = predict_folds(data, folds, {model_name: MODELS[model_name]})[model_name]
    ok = np.isfinite(pred).all(1)
    corrected = corrected_spectrum(data, np.nan_to_num(pred))
    fold_of_row = np.empty(len(ok), object)
    for label, _, is_test in folds:
        fold_of_row[is_test] = label
    spec = load("spectra")
    it, il, ist = data["index"]
    spread = spec["buoy_spread"].values[it, il, ist].astype(float)

    scores = case_scores(data, corrected, ok)
    out = fcfg.EXAMPLES_DIR
    out.mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("best_*.json")) + list(out.glob("worst_*.json")):
        old.unlink()
    index = []
    for category, best in (("best", True), ("worst", False)):
        for rank, (_, case) in enumerate(pick(scores, best).iterrows(), start=1):
            issue = pd.Timestamp(case["time"])
            path = out / f"{category}_{rank}_{case['station']}_{issue:%Y%m%d}_{issue:%H}z.json"
            path.write_text(json.dumps(case_json(data, corrected, spread, fold_of_row, case, category, rank,
                                                 model_name), separators=(",", ":")), encoding="utf8")
            index.append({"file": path.name, "category": category, "rank": rank, "station": case["station"],
                          "issue_time": f"{issue:%Y-%m-%dT%H:%MZ}"} |
                         {k: float(round(case[k], 4)) for k in ("skill_sqrt_e", "rmse_hm0_corrected", "rmse_hm0_gfs",
                                                                "mean_measured_hm0")})
            print(f"  {category} {rank}: {case['station']:6s} {issue:%Y-%m-%d %H}z  skill {case['skill_sqrt_e']:+.1%}  "
                  f"Hm0 RMSE {case['rmse_hm0_corrected']:.2f} m (GFS {case['rmse_hm0_gfs']:.2f} m), "
                  f"mean Hm0 {case['mean_measured_hm0']:.2f} m, {int(case['n_leads'])} lead times")
    eligible = scores[(scores["n_leads"] >= MIN_LEADS) & (scores["mean_measured_hm0"] >= MIN_HM0)
                      & ~scores["station"].isin(fcfg.TRAIN_ONLY_STATIONS)]
    (out / "index.json").write_text(json.dumps({
        "model": model_name, "n_eligible_cases": int(len(eligible)),
        "selection": f">= {MIN_LEADS} scored lead times, mean measured Hm0 >= {MIN_HM0} m, one case per run",
        "skill_distribution": {q: float(round(eligible["skill_sqrt_e"].quantile(p), 4))
                               for q, p in (("p10", 0.1), ("median", 0.5), ("p90", 0.9))},
        "cases": index}, indent=1), encoding="utf8")
    print(f"{len(eligible)} eligible cases; skill median {eligible['skill_sqrt_e'].median():+.1%}, "
          f"10-90% {eligible['skill_sqrt_e'].quantile(0.1):+.1%} .. {eligible['skill_sqrt_e'].quantile(0.9):+.1%}")
    print(f"written to {out}")
    return scores


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Best and worst corrected historical forecasts as JSON")
    ap.add_argument("--model", choices=list(MODELS), help=f"default {fcfg.FORECAST_MODEL}")
    ap.add_argument("-n", type=int, help=f"cases per category (default {N_CASES})")
    args = ap.parse_args()
    run(args.model, args.n)
