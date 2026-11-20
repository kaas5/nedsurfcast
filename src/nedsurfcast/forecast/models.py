"""
Models of the spectrum correction
===================================

Every model predicts the target of forecast/load_data.py (per-bin residual of the rebuilt GFS spectrum) from
one feature set. MODELS maps a short name to (estimator factory, feature set):

    ridge       ridge regression on the GFS spectrum and bulk context
    ridge-hlf   the same + history per period band + location + fetch mismatch
    mlp         MLP (common/torch_mlp.py, GPU if available) on the GFS spectrum and bulk context
    mlp-hlf     the same + history per period band + location + fetch mismatch

Both estimators are pipelines with median imputation (empty bands give NaN features) and standardisation.
"""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nedsurfcast.common.torch_mlp import TorchMLPRegressor
from nedsurfcast.forecast.load_data import features, targets

RIDGE_ALPHA = 10.0
MLP_KWARGS = dict(hidden_layer_sizes=(128, 64), alpha=1e-3, learning_rate_init=1e-3, batch_size=256,
                  max_iter=40, early_stopping=True, n_iter_no_change=5, random_state=0)
FULL = "gfs+history-bands+location+fetch"


def ridge():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=RIDGE_ALPHA))


def mlp():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), TorchMLPRegressor(**MLP_KWARGS))


MODELS = {
    "ridge": (ridge, "gfs"),
    "ridge-hlf": (ridge, FULL),
    "mlp": (mlp, "gfs"),
    "mlp-hlf": (mlp, FULL),
}


def fit_predict(make, X_train, Y_train, X_test) -> np.ndarray:
    """Fit a fresh estimator and predict, clipped to the range of the training targets (no extrapolation)."""
    model = make().fit(X_train, Y_train)
    return np.clip(model.predict(X_test), Y_train.min(0), Y_train.max(0))


def predict_folds(data: dict, folds: list, models: dict = MODELS) -> dict[str, np.ndarray]:
    """folds: (label, is_train, is_test) boolean masks over the rows. Returns per model the out-of-sample
    residuals (NaN in rows that no fold tests)."""
    Y = targets(data)
    X = {fs: features(data, fs) for _, fs in models.values()}
    pred = {name: np.full(Y.shape, np.nan) for name in models}
    for label, is_train, is_test in folds:
        for name, (make, fs) in models.items():
            pred[name][is_test] = fit_predict(make, X[fs][is_train], Y[is_train], X[fs][is_test])
        print(f"  {label}: trained on {int(is_train.sum())}, predicted {int(is_test.sum())} spectra", flush=True)
    return pred
