"""
Corrected spectra against GFS's own forecast spectrum (K133 / EPL3)
=====================================================================

The models correct a GFS spectrum that is rebuilt from the GFS partitions. GFS also publishes its real forecast
spectrum at a few points; at the stations where such a point spectrum exists (fcfg.GFSWAVE_SPEC_STATIONS) the
fair question is whether the corrected spectrum beats GFS's own forecast:

    skill = 1 - RMSE(model vs buoy) / RMSE(GFS point spectrum vs buoy)      (> 0: better than GFS)

Folds (scoring.point_station_folds): station k is tested on the alternating ~monthly blocks k, k + n, ...,
trained without its station group and without its test blocks (+- scoring.TEST_GAP_HOURS). Scored over lead
times >= scoring.EVAL_MIN_LEAD on the spectra where the point spectrum exists. The rebuilt GFS spectrum (the
model input) is shown as well.

Printed: skill per model (both stations pooled, and per station), with 95% bootstrap intervals over forecast
runs, and per lead time; plus the RMSE against the buoy. Tables in fcfg.SPECTRUM_MODEL_DIR/gfs_point_*.csv.

Run from anywhere with: python -m nedsurfcast.forecast.train_vs_gfs_point [--models NAME ...]
"""

import numpy as np

from nedsurfcast.forecast import config as fcfg
from nedsurfcast.forecast.load_data import corrected_spectrum, load_data
from nedsurfcast.forecast.models import MODELS, predict_folds
from nedsurfcast.forecast.scoring import (EVAL_MIN_LEAD, bootstrap, by_lead, formatted, point_station_folds,
                                          score_table, setup_pandas)

BASELINE = "GFS point spectrum"
RECONSTRUCTION = "GFS reconstruction"


def run(models: dict = MODELS):
    data = load_data()
    m = data["meta"]
    station = m["station"].to_numpy()
    print(f"{int(data['has_point'].sum())} spectra with a GFS point spectrum at "
          f"{sorted(np.unique(station[data['has_point']]))}; models: {list(models)}")
    pred = predict_folds(data, point_station_folds(data), models)

    rows = data["has_point"] & (m["lead_hours"].to_numpy() >= EVAL_MIN_LEAD)
    for p in pred.values():
        rows &= np.isfinite(p).all(1)
    print(f"\nScoring {int(rows.sum())} spectra (lead times >= {EVAL_MIN_LEAD} h, {m.loc[rows, 'time'].nunique()} runs)")

    sets = {BASELINE: (data["point_energy"], data["point_direction"]),
            RECONSTRUCTION: (data["gfs_energy"], data["gfs_direction"])}
    sets |= {name: corrected_spectrum(data, np.nan_to_num(y)) for name, y in pred.items()}

    skill = score_table(sets, data, rows, "skill", BASELINE)
    skill_st = score_table(sets, data, rows, "skill", BASELINE, by=station)
    rmse = score_table(sets, data, rows, "rmse")
    boot = bootstrap(sets, data, rows, "skill", BASELINE)
    leads = by_lead(sets, data, rows, "skill", BASELINE)

    setup_pandas()
    print(f"\nRMSE against the buoy, stations pooled (lower is better):")
    print(formatted(rmse.droplevel("group"), "rmse").to_string())
    print(f"\nSkill per station vs the {BASELINE}:")
    print(formatted(skill_st.drop(index=BASELINE, level="spectrum"), "skill").to_string())
    print(f"\nSkill vs the {BASELINE} per lead time:")
    print(leads.drop(index=BASELINE).map(lambda v: f"{v:+.1%}").to_string())
    print(f"\nSkill vs the {BASELINE}, 95% bootstrap interval over forecast runs:")
    print(boot.to_string())
    print(f"\nSkill vs the {BASELINE} (1 - RMSE / RMSE_point), stations pooled:")
    print(formatted(skill.droplevel("group").drop(index=BASELINE), "skill").to_string())

    out = fcfg.SPECTRUM_MODEL_DIR
    out.mkdir(parents=True, exist_ok=True)
    skill.droplevel("group").to_csv(out / "gfs_point_skill.csv")
    skill_st.to_csv(out / "gfs_point_skill_per_station.csv")
    rmse.droplevel("group").to_csv(out / "gfs_point_rmse.csv")
    boot.to_csv(out / "gfs_point_bootstrap.csv")
    leads.to_csv(out / "gfs_point_by_lead.csv")
    print(f"\nTables saved to {out} (gfs_point_*.csv)")
    return skill, skill_st


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Skill of the corrected spectra vs GFS's point spectra")
    ap.add_argument("--models", nargs="+", choices=list(MODELS), help="subset of the models (default all)")
    args = ap.parse_args()
    run({k: v for k, v in MODELS.items() if args.models is None or k in args.models})
