"""
Corrected spectra against the buoy measurements at every station
==================================================================

Every station is a test station. Folds (scoring.station_group_folds): per station group two folds, on the even
and on the odd ~monthly blocks; each trains without the group and without its test blocks (+-
scoring.TEST_GAP_HOURS) and predicts the group inside its blocks: unseen stations in unseen months. Together
every station is tested over the whole year.

Score against the measurements: the normalised error (scoring, "nrmse")

    nrmse = sqrt(sum (prediction - measured)^2 / sum measured^2)      (0 = perfect; lower is better)

pooled over the test rows of all folds, lead times >= scoring.EVAL_MIN_LEAD. The rebuilt GFS spectrum (the
model input) is scored the same way as the reference.

Printed: nrmse per model with 95% bootstrap intervals over forecast runs, per lead time, and the RMSE.
Tables in fcfg.SPECTRUM_MODEL_DIR/buoys_*.csv; buoys_per_fold_station.csv has the nrmse of every model per
fold and station (to find problems at individual stations).

Run from anywhere with: python -m nedsurfcast.forecast.train_vs_buoys [--models NAME ...]
"""

import numpy as np
import pandas as pd

from nedsurfcast.forecast import config as fcfg
from nedsurfcast.forecast.load_data import corrected_spectrum, load_data
from nedsurfcast.forecast.models import MODELS, predict_folds
from nedsurfcast.forecast.scoring import (EVAL_MIN_LEAD, bootstrap, by_lead, formatted, score_table, setup_pandas,
                                          station_group_folds)

RECONSTRUCTION = "GFS reconstruction"


def run(models: dict = MODELS):
    data = load_data()
    m = data["meta"]
    station = m["station"].to_numpy()
    folds = station_group_folds(data)
    print(f"{len(m)} spectra at {len(np.unique(station))} stations; {len(folds)} folds; models: {list(models)}")
    pred = predict_folds(data, folds, models)

    scored = m["lead_hours"].to_numpy() >= EVAL_MIN_LEAD
    for p in pred.values():
        scored &= np.isfinite(p).all(1)
    print(f"\nScoring {int(scored.sum())} spectra (lead times >= {EVAL_MIN_LEAD} h, {m.loc[scored, 'time'].nunique()} runs)")

    sets = {RECONSTRUCTION: (data["gfs_energy"], data["gfs_direction"])}
    sets |= {name: corrected_spectrum(data, np.nan_to_num(y)) for name, y in pred.items()}

    nrmse = score_table(sets, data, scored, "nrmse").droplevel("group")
    rmse = score_table(sets, data, scored, "rmse").droplevel("group")
    boot = bootstrap(sets, data, scored, "nrmse")
    leads = by_lead(sets, data, scored, "nrmse")
    per_fold = []
    for label, _, is_test in folds:
        t = score_table(sets, data, scored & is_test, "nrmse", by=station).reset_index()
        per_fold.append(t.rename(columns={"group": "station"}).assign(fold=label))
    per_fold = pd.concat(per_fold).set_index(["fold", "station", "spectrum"])

    setup_pandas()
    print("\nRMSE against the buoy (lower is better):")
    print(formatted(rmse, "rmse").to_string())
    print("\nNormalised error against the buoy per lead time (lower is better):")
    print(leads.map(lambda v: f"{v:.1%}").to_string())
    print("\nNormalised error against the buoy, 95% bootstrap interval over forecast runs:")
    print(boot.to_string())
    print("\nNormalised error against the buoy, sqrt(sum err^2 / sum measured^2), all folds pooled (lower is better):")
    print(formatted(nrmse, "nrmse").to_string())

    out = fcfg.SPECTRUM_MODEL_DIR
    out.mkdir(parents=True, exist_ok=True)
    nrmse.to_csv(out / "buoys_nrmse.csv")
    rmse.to_csv(out / "buoys_rmse.csv")
    boot.to_csv(out / "buoys_bootstrap.csv")
    leads.to_csv(out / "buoys_by_lead.csv")
    per_fold.to_csv(out / "buoys_per_fold_station.csv")
    print(f"\nTables saved to {out} (buoys_*.csv)")
    return nrmse, per_fold


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Normalised error of the corrected spectra against the buoys")
    ap.add_argument("--models", nargs="+", choices=list(MODELS), help="subset of the models (default all)")
    args = ap.parse_args()
    run({k: v for k, v in MODELS.items() if args.models is None or k in args.models})
