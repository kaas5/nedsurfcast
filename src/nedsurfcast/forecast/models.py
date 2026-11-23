"""
Models of the spectrum correction
===================================

Every model predicts the target of forecast/load_data.py (per-bin residual of the rebuilt GFS spectrum) from
one feature set. MODELS maps a short name to (estimator factory, feature set):

    ridge       ridge regression on the GFS spectrum and bulk context
    ridge-hlf   the same + history per period band + location + fetch mismatch
    mlp         MLP (common/torch_mlp.py, GPU if available) on the GFS spectrum and bulk context: one joint
                network for the energy and the direction outputs (EnergyDirectionMLP)
    mlp-hlf     the same + history per period band + location + fetch mismatch
    cnn         1D CNN over the frequency bins (common/torch_cnn.py): per bin the GFS spectrum variables, the
                bin frequency and the bulk context as channels; stacked convolutions widen the window
                (SpectrumCNN)
    cnn-hlf     the same + history per period band + location + fetch mismatch (as extra bulk channels)

One joint network, not one per block: separate networks were worse on energy (the direction outputs help the
shared layers). Weighting the loss towards energy also made energy worse, so the loss is a plain mean squared
error. Dropout 0.3 after every hidden layer: the best of a sweep (0.1-0.5, and L2 alpha 1e-3 to 100 without
dropout) on the all-station folds. Early stopping validates on time blocks of the training issue times
(scoring.TEST_GAP_HOURS left out around them), not on random rows that share days with the training rows.
The CNN trains the same way (same loss, dropout, early stopping); dropout 0.5 and 5 layers of kernel 5 were
the best of a sweep (3-10 layers, kernel 3-7, dropout 0.1-0.5). SpectrumCNN(n_seeds=n) averages n networks
that differ only in their random seed (initialisation, batch order, dropout).

All estimators include median imputation (empty bands give NaN features) and standardisation (for the CNN per
spectral variable over all bins, torch_cnn.BlockScaler).
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nedsurfcast.common.torch_cnn import BlockScaler, TorchCNNRegressor
from nedsurfcast.common.torch_mlp import TorchMLPRegressor
from nedsurfcast.forecast.load_data import features, targets
from nedsurfcast.forecast.scoring import TEST_GAP_HOURS

RIDGE_ALPHA = 10.0
MLP_KWARGS = dict(hidden_layer_sizes=(128, 64), alpha=1e-3, learning_rate_init=1e-3, batch_size=256,
                  max_iter=40, early_stopping=True, n_iter_no_change=5, random_state=0, dropout=0.3,
                  validation_blocks=2, n_time_blocks=20, validation_gap=np.timedelta64(TEST_GAP_HOURS, "h"))
CNN_KWARGS = {k: v for k, v in MLP_KWARGS.items() if k != "hidden_layer_sizes"} | dict(
    channels=64, kernel_size=5, n_layers=5, dropout=0.5)       # window of 1 + 5 * 4 = 21 bins (0.21 Hz)
N_SPECTRAL = 4                                                 # per-bin blocks at the front of the gfs features
FULL = "gfs+history-bands+location+fetch"


def ridge():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=RIDGE_ALPHA))


class EnergyDirectionMLP(RegressorMixin, BaseEstimator):
    """MLP pipeline for the target of load_data.targets (first third: sqrt(E) per bin = energy, the rest:
    direction vectors).
        split=False     one network for all outputs (default)
        split=True      separate networks for the energy and the direction block
    fit(X, Y, groups=times) passes the issue times to early stopping (time-block validation)."""

    def __init__(self, split=False, mlp_kwargs=None):
        self.split = split
        self.mlp_kwargs = mlp_kwargs

    def fit(self, X, Y, groups=None):
        kw = MLP_KWARGS if self.mlp_kwargs is None else self.mlp_kwargs
        nf = Y.shape[1] // 3
        self.blocks_ = [slice(0, nf), slice(nf, None)] if self.split else [slice(None)]
        self.parts_ = []
        for sl in self.blocks_:
            net = TorchMLPRegressor(**kw)
            pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), net)
            self.parts_.append(pipe.fit(X, Y[:, sl], torchmlpregressor__groups=groups))
        return self

    def predict(self, X):
        return np.hstack([p.predict(X) for p in self.parts_])


def mlp():
    return EnergyDirectionMLP()


class SpectrumCNN(RegressorMixin, BaseEstimator):
    """CNN pipeline for the target of load_data.targets: X from load_data.features (the gfs block starts with
    N_SPECTRAL per-bin blocks: log10 E, sqrt E, direction sin, cos; every other column is a bulk channel).
    fit(X, Y, groups=times, freq=bin frequencies): time-block early stopping, frequency as position channel.
    n_seeds > 1: an ensemble of networks with random_state, random_state + 1, ...; predict averages them."""

    def __init__(self, cnn_kwargs=None, n_seeds=1):
        self.cnn_kwargs = cnn_kwargs
        self.n_seeds = n_seeds

    def fit(self, X, Y, groups=None, freq=None):
        kw = CNN_KWARGS if self.cnn_kwargs is None else self.cnn_kwargs
        nf = Y.shape[1] // 3
        self.prep_ = make_pipeline(SimpleImputer(strategy="median"), BlockScaler(N_SPECTRAL, nf))
        Xs = self.prep_.fit_transform(X)
        seed = kw.get("random_state", 0)
        self.nets_ = [TorchCNNRegressor(n_bins=nf, n_spectral=N_SPECTRAL, positions=freq,
                                        **(kw | {"random_state": seed + k})).fit(Xs, Y, groups=groups)
                      for k in range(self.n_seeds)]
        return self

    def predict_members(self, X) -> np.ndarray:
        """(n_seeds, n, n_outputs): the prediction of every network."""
        Xs = self.prep_.transform(X)
        return np.stack([net.predict(Xs) for net in self.nets_])

    def predict(self, X):
        return self.predict_members(X).mean(0)


def cnn():
    return SpectrumCNN()


N_SEEDS = 5


def cnn_ensemble():
    return SpectrumCNN(n_seeds=N_SEEDS)


MODELS = {
    "ridge": (ridge, "gfs"),
    "ridge-hlf": (ridge, FULL),
    "mlp": (mlp, "gfs"),
    "mlp-hlf": (mlp, FULL),
    "cnn": (cnn, "gfs"),
    "cnn-hlf": (cnn, FULL),
}


def fit_predict(make, X_train, Y_train, X_test, times_train=None, freq=None) -> np.ndarray:
    """Fit a fresh estimator and predict, clipped to the range of the training targets (no extrapolation).
    times_train (issue times of the training rows) goes to estimators that validate on time blocks, freq (bin
    frequencies) to the CNN."""
    model = make()
    if isinstance(model, SpectrumCNN):
        model.fit(X_train, Y_train, groups=times_train, freq=freq)
    elif isinstance(model, EnergyDirectionMLP):
        model.fit(X_train, Y_train, groups=times_train)
    else:
        model.fit(X_train, Y_train)
    return np.clip(model.predict(X_test), Y_train.min(0), Y_train.max(0))


def predict_folds(data: dict, folds: list, models: dict = MODELS) -> dict[str, np.ndarray]:
    """folds: (label, is_train, is_test) boolean masks over the rows. Returns per model the out-of-sample
    residuals (NaN in rows that no fold tests)."""
    Y = targets(data)
    X = {fs: features(data, fs) for _, fs in models.values()}
    times = data["meta"]["time"].to_numpy()
    pred = {name: np.full(Y.shape, np.nan) for name in models}
    for label, is_train, is_test in folds:
        for name, (make, fs) in models.items():
            pred[name][is_test] = fit_predict(make, X[fs][is_train], Y[is_train], X[fs][is_test],
                                              times[is_train], data["freq"])
        print(f"  {label}: trained on {int(is_train.sum())}, predicted {int(is_test.sum())} spectra", flush=True)
    return pred
