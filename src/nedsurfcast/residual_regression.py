"""
Pooled point-wise regression: can local conditions predict WW3's error?
=========================================================================

A go/no-go test before investing in a grid-to-grid network. Every row is
one (buoy, time) pair:

    features = WW3 output + ERA5 wind/pressure at the buoy's matched grid cell
    target   = buoy observation - WW3 value at that cell   (the residual)

All buoys are pooled into ONE table and ONE model. No coordinates or buoy
ids are used as features, so whatever the model learns has to be a
relationship between local conditions and WW3 error, which is exactly
what would let it generalize to grid cells that have no buoy.

Two validation schemes answer two different questions:
  - leave-one-group-out: does the correction transfer to an unseen LOCATION?
                         (nearby buoys are held out together, see load_data.buoy_groups)
  - blocked-time       : does it transfer to unseen WEATHER (other time periods)?

Data loading/features live in load_data.py, models and validation schemes
in models.py, figures in plot.py.

Prerequisites (pip install):
    scikit-learn scipy pandas xarray zarr numpy matplotlib

IMPORTANT: build the gridded Zarr with lead_time_hours=0 for this test.
With a lead-time offset, the grid time axis is shifted relative to real
time, and the buoy series (resampled onto that axis) would be compared
against WW3 values from a different real time, corrupting every residual.
"""

import numpy as np
import pandas as pd

from load_data import build_table, feature_columns, load
from models import MODELS, SCHEMES
from plot import plot_pred_vs_actual


# ---------------------------------------------------------------------------
# 1. REPORTING
# ---------------------------------------------------------------------------

def rmse(x) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def summarize(name: str, scheme: str, df: pd.DataFrame, pred: pd.Series):
    ok = pred.notna()
    raw_err = df.loc[ok, "target"]        # error of uncorrected WW3
    new_err = raw_err - pred[ok]          # error of WW3 + predicted correction
    r_raw, r_new = rmse(raw_err), rmse(new_err)
    print(
        f"{scheme:<15} {name:<10} n={int(ok.sum()):>6}  "
        f"RMSE raw={r_raw:.3f} corrected={r_new:.3f}  skill={1 - r_new / r_raw:+.3f}  "
        f"bias raw={raw_err.mean():+.3f} corrected={new_err.mean():+.3f}"
    )


def _rmse_table(d: pd.DataFrame, grouper) -> pd.DataFrame:
    g = d.assign(sq_raw=d["err_raw"] ** 2, sq_new=d["err_new"] ** 2).groupby(grouper, observed=True)
    out = pd.DataFrame({"n": g.size(), "rmse_raw": np.sqrt(g["sq_raw"].mean()), "rmse_new": np.sqrt(g["sq_new"].mean())})
    out["skill"] = 1 - out["rmse_new"] / out["rmse_raw"]
    return out


def breakdown(df: pd.DataFrame, pred: pd.Series, label: str):
    ok = pred.notna()
    d = df.loc[ok].copy()
    d["err_raw"] = d["target"]
    d["err_new"] = d["target"] - pred[ok]

    print(f"\n--- {label}: per buoy ---")
    print(_rmse_table(d, d.index.get_level_values("buoy_id")).round(3))

    # Bin by the WW3 value (known at inference time), not by the observation
    d["ww3_sea_state"] = pd.qcut(d["ww3_target"], [0, 0.5, 0.8, 0.95, 1.0], duplicates="drop")
    print(f"\n--- {label}: by WW3 sea state (quantile bins, last bin = most extreme 5%) ---")
    print(_rmse_table(d, d["ww3_sea_state"]).round(3))


# ---------------------------------------------------------------------------
# 2. RUN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ds = load()
    df = build_table(ds)
    cols = feature_columns(df)

    print("Rows per buoy (after dropping missing targets):")
    print(df.groupby(level="buoy_id").size().to_string())
    print("\nBuoy groups (held out together in leave-one-group):")
    print(df.reset_index().groupby("buoy_group")["buoy_id"].unique().apply(list).to_string())
    print(f"\nFeatures ({len(cols)}): {cols}\n")

    results = {}
    for scheme, evaluate in SCHEMES.items():
        for name, make_model in MODELS.items():
            pred = evaluate(df, cols, make_model)
            results[(scheme, name)] = pred
            summarize(name, scheme, df, pred)
        print()

    chosen = ("leave-one-group", "hist-gbr")
    breakdown(df, results[chosen], "leave-one-group / hist-gbr")
    plot_pred_vs_actual(df, results[chosen], "Leave-one-group-out: predicted vs. actual WW3 residual")
