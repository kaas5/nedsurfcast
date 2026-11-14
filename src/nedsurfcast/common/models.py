"""
Residual models
=================

MODELS (regressors) and CLASSIFIERS map a name to a factory that returns a fresh,
unfitted model, so every fold starts from scratch. The validation schemes that use them live
in validation.py; shared by the nowcast and forecast pipelines.
"""

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
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
        max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=50, l2_regularization=1.0,
        random_state=0,  # early stopping uses a random validation split above 10k rows
    ),
}

CLASSIFIERS = {
    # Base rate only: the climatological probability. Any real classifier must beat this.
    "prior": lambda: DummyClassifier(strategy="prior"),
    "logistic": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                      LogisticRegression(C=1.0, max_iter=1000)),
    "hist-gbc": lambda: HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=50, l2_regularization=1.0,
        random_state=0,  # early stopping uses a random validation split above 10k rows
    ),
}
