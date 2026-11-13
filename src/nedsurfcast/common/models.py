"""
Residual models
=================

MODELS maps a name to a factory that returns a fresh, unfitted model, so
every fold starts from scratch. The validation schemes that use them live
in validation.py; shared by the nowcast and forecast pipelines.
"""

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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
