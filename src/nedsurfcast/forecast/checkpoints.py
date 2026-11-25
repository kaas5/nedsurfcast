"""
Saving and loading the trained PyTorch models of the spectrum correction
==========================================================================

Only the PyTorch models of forecast/models.py (EnergyDirectionMLP, SpectrumCNN) are saved; the sklearn models
(ridge) are not. Files in fcfg.CHECKPOINT_DIR, named after the model in models.MODELS:

    <name>_<YYYYmmdd-HHMMSS>.pt         a single network
    <name>_<YYYYmmdd-HHMMSS>_<n>.pt     member n (1, 2, ...) of an ensemble; all members share the timestamp

Every file holds everything needed to predict with that network, as tensors and plain Python values (loads
with torch.load(..., weights_only=True), no sklearn objects; params and trained_on as JSON text, which
torch 1.13 needs for floats):
    model, timestamp, member, n_members, feature_set    which model and which input (load_data.features)
    estimator, params, n_in, n_out, state_dict          the network: class, constructor parameters (JSON), weights
    impute, scale_mean, scale_std                       input preprocessing: NaN -> impute, then
                                                        (x - scale_mean) / scale_std
    y_min, y_max                                        range of the training targets: predictions are clipped
                                                        to it (as models.fit_predict does)
    freq                                                bin frequencies (Hz) of the target layout
    trained_on                                          rows, stations and issue-time range of the training data
                                                        (JSON)
    torch_version

load_model(name, timestamp=None) loads a model (the newest timestamp by default; every member of an ensemble)
and returns a CheckpointModel whose predict(X) gives the residuals of load_data.targets, averaged over the
members and clipped: corrected spectrum = load_data.corrected_spectrum(data, predict(features(data, fs))).
"""

import json
import re
from pathlib import Path

import numpy as np
import torch

from nedsurfcast.common.torch_cnn import TorchCNNRegressor
from nedsurfcast.common.torch_mlp import TorchMLPRegressor
from nedsurfcast.forecast import config as fcfg
from nedsurfcast.forecast.models import EnergyDirectionMLP, SpectrumCNN

ESTIMATORS = {"TorchMLPRegressor": TorchMLPRegressor, "TorchCNNRegressor": TorchCNNRegressor}
TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"


def is_torch_model(model) -> bool:
    return isinstance(model, (EnergyDirectionMLP, SpectrumCNN))


def _members(model) -> list[tuple]:
    """Per network of a fitted model: (net, imputer medians, scaler mean, scaler std)."""
    if isinstance(model, SpectrumCNN):
        imputer, scaler = model.prep_[0], model.prep_[1]
        return [(net, imputer.statistics_, scaler.mean_, scaler.scale_) for net in model.nets_]
    if isinstance(model, EnergyDirectionMLP):
        if len(model.parts_) != 1:
            raise NotImplementedError("saving EnergyDirectionMLP(split=True) is not supported")
        imputer, scaler, net = model.parts_[0]
        return [(net, imputer.statistics_, scaler.mean_, scaler.scale_)]
    raise TypeError(f"not a PyTorch model of forecast/models.py: {type(model).__name__}")


def _plain_params(net) -> dict:
    """Constructor parameters as plain Python values (arrays -> lists, the validation gap -> hours)."""
    out = {}
    for k, v in net.get_params().items():
        if isinstance(v, np.timedelta64):
            v = float(v / np.timedelta64(1, "h"))
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        elif isinstance(v, np.generic):
            v = v.item()
        out[k] = v
    return out


def _tensor(a) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a, dtype=np.float64))


def save_model(model, name: str, feature_set: str, timestamp: str, Y_train: np.ndarray, freq: np.ndarray,
               trained_on: dict, out_dir: Path = None) -> list[Path]:
    """Save every network of a fitted PyTorch model from MODELS; returns the written paths."""
    out_dir = Path(out_dir or fcfg.CHECKPOINT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    members = _members(model)
    paths = []
    for n, (net, medians, mean, std) in enumerate(members, start=1):
        ckpt = {
            "model": name, "timestamp": timestamp, "member": n if len(members) > 1 else None,
            "n_members": len(members), "feature_set": feature_set,
            "estimator": type(net).__name__, "params": json.dumps(_plain_params(net)),
            "n_in": len(medians), "n_out": int(Y_train.shape[1]),
            "state_dict": {k: v.detach().cpu() for k, v in net.model_.state_dict().items()},
            "impute": _tensor(medians), "scale_mean": _tensor(mean), "scale_std": _tensor(std),
            "y_min": _tensor(Y_train.min(0)), "y_max": _tensor(Y_train.max(0)), "freq": _tensor(freq),
            "trained_on": json.dumps(trained_on), "torch_version": str(torch.__version__),
        }
        suffix = f"_{n}" if len(members) > 1 else ""
        path = out_dir / f"{name}_{timestamp}{suffix}.pt"
        torch.save(ckpt, path)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _pattern(name: str) -> re.Pattern:
    return re.compile(rf"^{re.escape(name)}_(\d{{8}}-\d{{6}})(?:_(\d+))?\.pt$")


def list_checkpoints(name: str, checkpoint_dir: Path = None) -> dict[str, list[Path]]:
    """Timestamp -> the files of that model (members sorted by n), oldest timestamp first."""
    pat, found = _pattern(name), {}
    for path in Path(checkpoint_dir or fcfg.CHECKPOINT_DIR).glob(f"{name}_*.pt"):
        if mt := pat.match(path.name):
            found.setdefault(mt.group(1), []).append((int(mt.group(2) or 0), path))
    return {ts: [p for _, p in sorted(v)] for ts, v in sorted(found.items())}


class CheckpointModel:
    """A saved model: predict(X) = mean over the members of the network outputs, clipped to the training range.
    X: load_data.features(data, self.feature_set)."""

    def __init__(self, ckpts: list[dict], device=None):
        first = ckpts[0]
        self.name, self.timestamp, self.feature_set = first["model"], first["timestamp"], first["feature_set"]
        self.freq = first["freq"].numpy()
        self.trained_on = json.loads(first["trained_on"])
        self.y_min, self.y_max = first["y_min"].numpy(), first["y_max"].numpy()
        self.members = []
        for c in ckpts:
            params = dict(json.loads(c["params"]), validation_gap=None, device=device)
            net = ESTIMATORS[c["estimator"]](**params)
            dev = net._device()
            net.model_ = net._build(c["n_in"], c["n_out"], torch.Generator()).to(dev)
            net.model_.load_state_dict(c["state_dict"])
            net._y_1d = False
            self.members.append((net, c["impute"].numpy(), c["scale_mean"].numpy(), c["scale_std"].numpy()))

    def predict_members(self, X) -> np.ndarray:
        """(n_members, n, n_outputs), unclipped."""
        X = np.asarray(X, float)
        out = []
        for net, impute, mean, std in self.members:
            Xi = np.where(np.isnan(X), impute, X)
            out.append(net.predict((Xi - mean) / std))
        return np.stack(out)

    def predict(self, X) -> np.ndarray:
        return np.clip(self.predict_members(X).mean(0), self.y_min, self.y_max)


def load_model(name: str, timestamp: str = None, checkpoint_dir: Path = None, device=None) -> CheckpointModel:
    """Load model `name` (as in models.MODELS) saved at `timestamp` (default: the newest), all its members."""
    found = list_checkpoints(name, checkpoint_dir)
    if not found:
        raise FileNotFoundError(f"no checkpoints of {name!r} in {checkpoint_dir or fcfg.CHECKPOINT_DIR}")
    ts = timestamp or list(found)[-1]
    if ts not in found:
        raise FileNotFoundError(f"no checkpoint {name}_{ts} (available: {list(found)})")
    ckpts = [torch.load(p, map_location="cpu", weights_only=True) for p in found[ts]]
    if len(ckpts) != ckpts[0]["n_members"]:
        raise ValueError(f"{name}_{ts}: {len(ckpts)} files, but the model has {ckpts[0]['n_members']} members")
    return CheckpointModel(ckpts, device=device)
