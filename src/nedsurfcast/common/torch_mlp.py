"""
PyTorch MLP regressor with a sklearn estimator interface
==========================================================

Trains a fully connected ReLU network on the GPU (if available) and can be dropped into a sklearn Pipeline
(fit / predict / get_params).

Training:
    loss            0.5 * mean squared error over samples and outputs
                    + 0.5 * alpha * sum(weights^2) / batch size   (L2 on weights, not on biases; applied as
                    Adam weight_decay = alpha / batch size, which gives the same gradient)
    optimiser       Adam (beta 0.9 / 0.999, eps 1e-8), constant learning rate
    dropout         dropout probability after every hidden ReLU layer (0 = none; off in predict)
    initialisation  Glorot uniform (factor 6 for relu layers, 2 for the linear output layer)
    early stopping  stop when the validation R^2 (mean over outputs) has not improved by tol for
                    n_iter_no_change epochs; the weights of the best epoch are restored. Validation rows:
                      fit(X, y)                  a random validation_fraction of the rows
                      fit(X, y, groups=times)    with validation_blocks set: time blocks. The unique values of
                                                 `groups` (e.g. issue times) are split in n_time_blocks
                                                 contiguous blocks; validation_blocks of them, spread evenly,
                                                 validate, and rows within validation_gap of them (same unit
                                                 as groups, e.g. np.timedelta64) are left out of training.
                                                 Random rows share days with the training rows and keep
                                                 improving while the model overfits to those days.
"""

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin


class TorchMLPRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, hidden_layer_sizes=(128, 64), alpha=1e-3, learning_rate_init=1e-3, batch_size=256,
                 max_iter=40, early_stopping=True, validation_fraction=0.1, n_iter_no_change=5, tol=1e-4,
                 validation_blocks=None, n_time_blocks=20, validation_gap=None, dropout=0.0,
                 random_state=0, device=None, verbose=False):
        self.dropout = dropout
        self.validation_blocks = validation_blocks
        self.n_time_blocks = n_time_blocks
        self.validation_gap = validation_gap
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    # -----------------------------------------------------------------------
    def _device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build(self, n_in: int, n_out: int, gen: torch.Generator) -> torch.nn.Sequential:
        sizes = [n_in, *self.hidden_layer_sizes, n_out]
        layers = []
        for k, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            lin = torch.nn.Linear(a, b)
            factor = 2.0 if k == len(sizes) - 2 else 6.0               # 2 for the identity output
            bound = np.sqrt(factor / (a + b))
            with torch.no_grad():
                lin.weight.uniform_(-bound, bound, generator=gen)
                lin.bias.uniform_(-bound, bound, generator=gen)
            layers.append(lin)
            if k < len(sizes) - 2:
                layers.append(torch.nn.ReLU())
                if self.dropout:
                    layers.append(torch.nn.Dropout(self.dropout))
        return torch.nn.Sequential(*layers)

    @staticmethod
    def _r2(pred: torch.Tensor, y: torch.Tensor) -> float:
        """R^2 averaged over outputs (constant outputs count as 1)."""
        ss_res = ((y - pred) ** 2).sum(0)
        ss_tot = ((y - y.mean(0)) ** 2).sum(0)
        r2 = torch.where(ss_tot > 0, 1 - ss_res / ss_tot.clamp_min(1e-30),
                         torch.where(ss_res > 0, torch.zeros_like(ss_res), torch.ones_like(ss_res)))
        return float(r2.mean())

    def _time_split(self, groups) -> tuple[np.ndarray, np.ndarray]:
        """(validation, training) row indices from time blocks of `groups` (see the module docstring)."""
        g = np.asarray(groups)
        blocks = np.array_split(np.unique(g), self.n_time_blocks)
        picks = np.linspace(0, len(blocks) - 1, self.validation_blocks + 2)[1:-1].round().astype(int)
        val, near = np.zeros(len(g), bool), np.zeros(len(g), bool)
        for k in picks:
            lo, hi = blocks[k][0], blocks[k][-1]
            val |= (g >= lo) & (g <= hi)
            gap = self.validation_gap if self.validation_gap is not None else 0 * (hi - lo)
            near |= (g >= lo - gap) & (g <= hi + gap)
        return np.flatnonzero(val), np.flatnonzero(~near)

    # -----------------------------------------------------------------------
    def fit(self, X, y, groups=None):
        dev = self._device()
        X = torch.as_tensor(np.asarray(X, dtype=np.float32), device=dev)
        y = np.asarray(y, dtype=np.float32)
        self._y_1d = y.ndim == 1
        y = torch.as_tensor(y.reshape(len(y), -1), device=dev)
        torch.manual_seed(self.random_state)
        gen_cpu = torch.Generator().manual_seed(self.random_state)

        n = len(X)
        if self.early_stopping and self.validation_blocks and groups is not None:
            val_idx, tr_idx = (torch.as_tensor(i, device=dev) for i in self._time_split(groups))
            X_val, y_val, X_tr, y_tr = X[val_idx], y[val_idx], X[tr_idx], y[tr_idx]
        elif self.early_stopping:
            perm = torch.randperm(n, generator=gen_cpu)
            n_val = max(1, int(round(self.validation_fraction * n)))
            val_idx, tr_idx = perm[:n_val].to(dev), perm[n_val:].to(dev)
            X_val, y_val, X_tr, y_tr = X[val_idx], y[val_idx], X[tr_idx], y[tr_idx]
        else:
            X_tr, y_tr = X, y

        self.model_ = self._build(X.shape[1], y.shape[1], gen_cpu).to(dev)
        bs = min(self.batch_size, len(X_tr))
        weights = [p for name, p in self.model_.named_parameters() if name.endswith("weight")]
        biases = [p for name, p in self.model_.named_parameters() if not name.endswith("weight")]
        # the L2 term 0.5 alpha |W|^2 / batch adds alpha W / batch to the weight gradients: Adam's (coupled)
        # weight_decay does exactly that, without an extra pass over the weights (last short batch: ~same)
        opt = torch.optim.Adam([{"params": weights, "weight_decay": self.alpha / bs},
                                {"params": biases, "weight_decay": 0.0}],
                               lr=self.learning_rate_init, betas=(0.9, 0.999), eps=1e-8, foreach=True)
        best_score, best_state, no_improve = -np.inf, None, 0
        self.loss_curve_, self.validation_scores_ = [], []

        for epoch in range(self.max_iter):
            self.model_.train()
            order = torch.randperm(len(X_tr), generator=gen_cpu).to(dev)
            total = torch.zeros((), device=dev)                    # summed on the device: no sync per batch
            for start in range(0, len(X_tr), bs):
                idx = order[start:start + bs]
                loss = 0.5 * ((self.model_(X_tr[idx]) - y_tr[idx]) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += loss.detach() * len(idx)
            # data loss only, without the L2 term
            self.loss_curve_.append(float(total) / len(X_tr))
            self.n_iter_ = epoch + 1

            if self.early_stopping:
                score = self._r2(self._predict_tensor(X_val), y_val)
                self.validation_scores_.append(score)
                if score > best_score + self.tol:
                    no_improve = 0
                else:
                    no_improve += 1
                if score > best_score:
                    best_score = score
                    best_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
                if self.verbose:
                    print(f"  epoch {epoch + 1}: loss {self.loss_curve_[-1]:.5f}, validation R2 {score:.4f}")
                if no_improve >= self.n_iter_no_change:
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
            self.best_validation_score_ = best_score
        return self

    def _predict_tensor(self, X: torch.Tensor, chunk: int = 65536) -> torch.Tensor:
        self.model_.eval()
        with torch.no_grad():
            return torch.cat([self.model_(X[i:i + chunk]) for i in range(0, len(X), chunk)])

    def predict(self, X) -> np.ndarray:
        dev = next(self.model_.parameters()).device
        out = self._predict_tensor(torch.as_tensor(np.asarray(X, dtype=np.float32), device=dev)).cpu().numpy()
        return out.ravel() if self._y_1d else out
