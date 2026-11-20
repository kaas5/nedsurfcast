"""
Corrected spectra against the buoy measurements at ALL stations (alternating months)
======================================================================================

forecast/compare_gfs_spectra.py can only test at K133 and EPL3, where GFS publishes a point spectrum. Here
every station is a test station, scored against its own measurements, so the effect of location-dependent
features (depth, exposure, fetch mismatch) shows up at the coastal stations where it matters.

Folds: for every station group, two folds (even and odd ~monthly blocks of compare_gfs_spectra.test_block).
Each fold leaves out the group AND its test blocks (with TEST_GAP_HOURS on both sides) from training, and
predicts that group inside its blocks: an unseen station in unseen months. Together every station is tested
over the whole year.

Compared with the measured (averaged buoy) spectrum, lead times >= EVAL_MIN_LEAD:
    GFS reconstruction    GFS spectrum rebuilt from its partitions, scaled to GFS's HTSGW of the predicted
                          time step (train_spectrum.SCALE_TO_HTSGW): the model input and the reference for
                          the skill, available at every station
    GFS reconstruction (unscaled)   the same before the scaling
    GFS point spectrum    GFS's own spectrum, only at K133/EPL3 (NaN elsewhere)
    models                MODEL_VARIANTS (only GFS-derived input)
and raw GFS bulk output: GFS's significant wave height HTSGW (bilinear at the station) against the measured Hm0.
At K133/EPL3 the raw GFS output (its point spectrum) is the comparison: on the rows where it exists, every
spectrum's skill is also given relative to it.
Note: the buoy Hm0 covers the buoy's frequency range (fcfg.BUOY_PARTITION_KWARGS f_range), HTSGW all
frequencies, so a small positive HTSGW bias is expected.

Only tables are written (fcfg.SPECTRUM_MODEL_DIR/buoy_eval_*.csv).

Run from anywhere with: python -m nedsurfcast.forecast.evaluate_buoys [--models NAME ...]
"""

import numpy as np
import pandas as pd

from nedsurfcast.forecast import config as fcfg
from nedsurfcast.forecast.compare_gfs_spectra import (TEST_GAP_HOURS, attach_point_spectra, bootstrap_skill,
                                                      by_lead, score_table, skill, test_block)
from nedsurfcast.forecast.train_spectrum import (EVAL_MIN_LEAD, MODELS, _mlp, band_hm0, corrected_spectrum,
                                                 features, load_data, targets)

REFERENCE = "GFS reconstruction"
UNSCALED = "GFS reconstruction (unscaled)"
POINT = "GFS point spectrum"
MODEL_VARIANTS = {                      # name -> (model factory, feature set)
    "ridge (GFS)": (MODELS["ridge"][0], "gfs"),
    "ridge (GFS + location)": (MODELS["ridge"][0], "gfs+location"),
    "ridge (GFS + location + fetch)": (MODELS["ridge"][0], "gfs+location+fetch"),
    "ridge (GFS + history per band + location + fetch)": (MODELS["ridge"][0], "gfs+history-bands+location+fetch"),
    "mlp (GFS)": (_mlp, "gfs"),
    "mlp (GFS + location + fetch)": (_mlp, "gfs+location+fetch"),
}
STATION_METRICS = ["rmse sqrt(E)", "rmse Hm0", "rmse Hm0 Tp < 6 s", "rmse Hm0 6-10 s", "rmse Hm0 Tp > 10 s"]


def predict_all_stations(data: dict, variants: dict) -> dict[str, np.ndarray]:
    """Out-of-sample residuals for every row: per station group two folds (even / odd blocks)."""
    m = data["meta"]
    Y = targets(data)
    X = {fs: features(data, fs) for _, fs in variants.values()}
    pred = {name: np.full(Y.shape, np.nan) for name in variants}
    groups = m["buoy_group"].to_numpy()
    blocks = [test_block(m, k, 2)[:2] for k in range(2)]
    for g in np.unique(groups):
        for k, (in_block, near_block) in enumerate(blocks):
            is_train = (groups != g) & ~near_block
            is_test = (groups == g) & in_block
            if not is_test.any():
                continue
            for name, (make, fs) in variants.items():
                model = make().fit(X[fs][is_train], Y[is_train])
                pred[name][is_test] = np.clip(model.predict(X[fs][is_test]), Y[is_train].min(0), Y[is_train].max(0))
        names = sorted(m.loc[groups == g, "station"].unique())
        print(f"  group {g} {names}: done", flush=True)
    return pred


def raw_hm0_table(data: dict, sets: dict, rows: np.ndarray) -> pd.DataFrame:
    """RMSE and bias of total Hm0 per station: raw GFS HTSGW, the reconstruction, the point spectrum and models."""
    f = data["freq"]
    m = data["meta"]
    station = m["station"].to_numpy()
    hb = band_hm0(data["buoy_energy"], f, 0, 1)
    est = {"GFS HTSGW (raw)": m["gfs_htsgw"].to_numpy()}
    est |= {name: band_hm0(e, f, 0, 1) for name, (e, _) in sets.items()}
    records = []
    for st in ["all"] + sorted(np.unique(station[rows])):
        r = rows if st == "all" else rows & (station == st)
        for name, h in est.items():
            ok = r & np.isfinite(h)
            if ok.sum() == 0:
                continue
            err = h[ok] - hb[ok]
            records.append({"station": st, "spectrum": name, "n": int(ok.sum()),
                            "rmse Hm0": float(np.sqrt(np.mean(err ** 2))), "bias Hm0": float(err.mean())})
    return pd.DataFrame(records).set_index(["station", "spectrum"])


def run(models: list[str] | None = None):
    variants = {k: v for k, v in MODEL_VARIANTS.items() if models is None or k in models}
    data = load_data()
    has_point = attach_point_spectra(data)
    m = data["meta"]
    print(f"{len(m)} spectra at {m['station'].nunique()} stations; models: {list(variants)}")
    print(f"  folds: every station group x 2 alternating ~monthly block sets, {TEST_GAP_HOURS} h gaps")
    pred = predict_all_stations(data, variants)

    rows = m["lead_hours"].to_numpy() >= EVAL_MIN_LEAD
    for p in pred.values():
        rows &= np.isfinite(p).all(1)
    print(f"\nScoring {int(rows.sum())} spectra (lead times >= {EVAL_MIN_LEAD} h, {m.loc[rows, 'time'].nunique()} runs)\n")

    sets = {REFERENCE: (data["gfs_energy"], data["gfs_direction"]),
            UNSCALED: (data["gfs_energy_unscaled"], data["gfs_direction"]),
            POINT: (data["point_energy"], data["point_direction"])}
    for name, y in pred.items():
        sets[name] = corrected_spectrum(data, np.nan_to_num(y))
    no_point = {k: v for k, v in sets.items() if k != POINT}

    station = m["station"].to_numpy()
    overall = score_table(no_point, data, rows)
    # the point spectrum only where it exists (part of K133/EPL3); everything else on all rows
    per_station = score_table(no_point, data, rows, by=station)
    # K133/EPL3: every spectrum on the rows with a GFS point spectrum, skill relative to that raw GFS output
    point_rows = score_table(sets, data, rows & has_point, by=station)
    point_skill = skill(point_rows, POINT)
    leads = by_lead(no_point, data, rows)
    boot = bootstrap_skill(no_point, data, rows, baseline=REFERENCE,
                           metrics=("rmse sqrt(E)", "rmse Hm0", "rmse Hm0 Tp < 6 s", "rmse Hm0 Tp > 10 s"))
    raw = raw_hm0_table(data, sets, rows)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", 500)
    pct = lambda t: t.map(lambda v: f"{v:+.1%}" if np.isfinite(v) else "")
    print("Errors against the measured spectra, all stations (lower is better):")
    print(overall.droplevel("group").round(3).to_string())
    print(f"\nSkill vs the {REFERENCE} (1 - RMSE / RMSE_reconstruction):")
    print(pct(skill(overall, REFERENCE).droplevel("group")).to_string())
    print(f"\nSkill with 95% bootstrap interval over forecast runs:")
    print(boot.to_string())
    sk = skill(per_station, REFERENCE)
    for metric in STATION_METRICS[:3]:
        print(f"\nSkill vs the {REFERENCE} per station, {metric}:")
        print(pct(sk[metric].unstack("spectrum")[list(no_point)].drop(columns=REFERENCE)).to_string())
    print("\nK133/EPL3 on the spectra with a GFS point spectrum: errors against the buoy (RMSE)")
    print(point_rows.round(3).to_string())
    print("\nK133/EPL3: skill vs the raw GFS point spectrum (1 - RMSE / RMSE_point):")
    print(pct(point_skill).to_string())
    print("\nTotal Hm0 against the buoy per station, including raw GFS HTSGW (m; bias = estimate - buoy):")
    print(raw.round(3).to_string())
    print("\nPer lead time:")
    print(leads.round(3).to_string())

    out = fcfg.SPECTRUM_MODEL_DIR
    out.mkdir(parents=True, exist_ok=True)
    overall.droplevel("group").to_csv(out / "buoy_eval_overall.csv")
    per_station.to_csv(out / "buoy_eval_per_station.csv")
    point_rows.to_csv(out / "buoy_eval_point_stations.csv")
    leads.to_csv(out / "buoy_eval_by_lead.csv")
    boot.to_csv(out / "buoy_eval_bootstrap.csv")
    raw.to_csv(out / "buoy_eval_hm0_raw_gfs.csv")
    print(f"\nTables saved to {out} (buoy_eval_*.csv)")
    return overall, per_station, raw


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Score the corrected spectra against the buoys at all stations")
    ap.add_argument("--models", nargs="+", metavar="NAME", help=f"subset of {list(MODEL_VARIANTS)}")
    args = ap.parse_args()
    unknown = set(args.models or []) - set(MODEL_VARIANTS)
    if unknown:
        ap.error(f"unknown models {sorted(unknown)}; choose from {list(MODEL_VARIANTS)}")
    run(args.models)
