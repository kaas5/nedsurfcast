"""
Residual models and validation schemes
========================================

MODELS maps a name to a factory that returns a fresh, unfitted model, so
every fold starts from scratch. SCHEMES maps a name to a function that
returns out-of-sample predictions for every row of the pooled table.
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_TIME_BLOCKS = 5
TIME_GAP_HOURS = 24                              # buffer around each test block to limit autocorrelation leakage


# ---------------------------------------------------------------------------
# 1. MODELS (all start from the same two baselines)
# ---------------------------------------------------------------------------

MODELS = {
    # Constant offset: how much of the error is plain average bias? Any real model must beat this.
    "mean-bias": lambda: DummyRegressor(strategy="mean"),
    "ridge": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)),
    # Small, heavily regularized trees; handles NaN (lags at the start, gaps) natively
    "hist-gbr": lambda: HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=50, l2_regularization=1.0
    ),
}


# ---------------------------------------------------------------------------
# 2. VALIDATION SCHEMES
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
