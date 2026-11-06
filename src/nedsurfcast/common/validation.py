"""
Validation schemes and skill scores for residual models
=========================================================

Every scheme returns out-of-sample predictions for every row of a pooled
table with a "target" column (the residual), a "buoy_group" column and a
"time" index level.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

N_TIME_BLOCKS = 5
TIME_GAP_HOURS = 24                              # buffer around each test block to limit autocorrelation leakage


# ---------------------------------------------------------------------------
# 1. VALIDATION SCHEMES
# ---------------------------------------------------------------------------

# Nearby buoys see (almost) the same conditions, so a whole group is held out at once
def evaluate_leave_one_group_out(df, cols, make_model) -> pd.Series:
    groups = df["buoy_group"]
    pred = pd.Series(np.nan, index=df.index)
    for train_idx, test_idx in LeaveOneGroupOut().split(df, groups=groups):
        model = make_model()
        model.fit(df.iloc[train_idx][cols], df.iloc[train_idx]["target"])
        pred.iloc[test_idx] = model.predict(df.iloc[test_idx][cols])
    return pred


def evaluate_blocked_time(df, cols, make_model, n_blocks=N_TIME_BLOCKS, gap_hours=TIME_GAP_HOURS) -> pd.Series:
    times = df.index.get_level_values("time")
    blocks = np.array_split(np.sort(times.unique()), n_blocks)
    gap = pd.Timedelta(hours=gap_hours)
    pred = pd.Series(np.nan, index=df.index)

    for block in blocks:
        t0, t1 = pd.Timestamp(block[0]), pd.Timestamp(block[-1])
        is_test = (times >= t0) & (times <= t1)
        is_train = (times < t0 - gap) | (times > t1 + gap)  # gap rows are used by neither side
        if is_test.sum() == 0 or is_train.sum() == 0:
            continue
        model = make_model()
        model.fit(df.loc[is_train, cols], df.loc[is_train, "target"])
        pred.loc[is_test] = model.predict(df.loc[is_test, cols])
    return pred


SCHEMES = {
    "leave-one-group": evaluate_leave_one_group_out,
    "blocked-time": evaluate_blocked_time,
}


# ---------------------------------------------------------------------------
# 2. SCORES
# ---------------------------------------------------------------------------

def rmse(x) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def summarize(name: str, scheme: str, df: pd.DataFrame, pred: pd.Series):
    ok = pred.notna()
    raw_err = df.loc[ok, "target"]        # error of the uncorrected model
    new_err = raw_err - pred[ok]          # error of model + predicted correction
    r_raw, r_new = rmse(raw_err), rmse(new_err)
    print(
        f"{scheme:<15} {name:<10} n={int(ok.sum()):>6}  "
        f"RMSE raw={r_raw:.3f} corrected={r_new:.3f}  skill={1 - r_new / r_raw:+.3f}  "
        f"bias raw={raw_err.mean():+.3f} corrected={new_err.mean():+.3f}"
    )
