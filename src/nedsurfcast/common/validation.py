"""
Validation schemes and skill scores for residual models
=========================================================

Every scheme returns out-of-sample predictions for every row of a pooled
table with a "target" column (the residual), a "buoy_group" column and a
"time" index level.
"""

import numpy as np
import pandas as pd

N_TIME_BLOCKS = 5
TIME_GAP_HOURS = 24                              # buffer around each test block to limit autocorrelation leakage


# ---------------------------------------------------------------------------
# 1. VALIDATION SCHEMES
# ---------------------------------------------------------------------------

# Nearby buoys see (almost) the same conditions, so a whole group is held out at once
def leave_one_group_folds(df):
    """Yield (train, test) boolean masks: every buoy group is the test set once."""
    groups = df["buoy_group"].to_numpy()
    for g in np.unique(groups):
        yield groups != g, groups == g


def blocked_time_folds(df, n_blocks=N_TIME_BLOCKS, gap_hours=TIME_GAP_HOURS):
    """Yield (train, test) boolean masks: contiguous time blocks, with a gap on both sides used by neither."""
    times = df.index.get_level_values("time")
    gap = pd.Timedelta(hours=gap_hours)
    for block in np.array_split(np.sort(times.unique()), n_blocks):
        t0, t1 = pd.Timestamp(block[0]), pd.Timestamp(block[-1])
        is_test = np.asarray((times >= t0) & (times <= t1))
        is_train = np.asarray((times < t0 - gap) | (times > t1 + gap))
        if is_test.any() and is_train.any():
            yield is_train, is_test


FOLDS = {
    "leave-one-group": leave_one_group_folds,
    "blocked-time": blocked_time_folds,
}


def _evaluate(folds, df, cols, make_model) -> pd.Series:
    pred = pd.Series(np.nan, index=df.index)
    for is_train, is_test in folds:
        model = make_model()
        model.fit(df.loc[is_train, cols], df.loc[is_train, "target"])
        pred.loc[is_test] = model.predict(df.loc[is_test, cols])
    return pred


def evaluate_leave_one_group_out(df, cols, make_model) -> pd.Series:
    return _evaluate(leave_one_group_folds(df), df, cols, make_model)


def evaluate_blocked_time(df, cols, make_model, n_blocks=N_TIME_BLOCKS, gap_hours=TIME_GAP_HOURS) -> pd.Series:
    return _evaluate(blocked_time_folds(df, n_blocks, gap_hours), df, cols, make_model)


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
