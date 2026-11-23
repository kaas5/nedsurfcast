"""
1D CNN over the frequency axis with a sklearn estimator interface
===================================================================

For inputs and targets that are spectra: the same small network is applied along the frequency bins, so the
correction at one bin is learned from every bin (shared weights) and neighbouring bins get similar
corrections.

Layouts (flat 2D arrays, as for every sklearn estimator):
    X   n_spectral blocks of n_bins columns (one per-bin variable each, e.g. log10 E, sqrt E, direction
        sin, cos), then any number of bulk columns (one value per sample, e.g. wind, Hs, lead time)
    y   n_out blocks of n_bins columns (e.g. sqrt E residual, direction sin residual, cos residual)

Network, per sample: input channels per bin = the n_spectral per-bin variables + the bin position
(`positions`, e.g. the frequency in Hz, standardised) + every bulk column copied to all bins. Then

    1x1 conv (in -> channels)
    n_layers residual blocks:  x + Dropout(ReLU(Conv1d(x, kernel_size, padding "same")))
    1x1 conv (channels -> n_out)

Each block widens the window a bin sees by kernel_size - 1, so the output at a bin depends on
1 + n_layers * (kernel_size - 1) neighbouring bins (receptive_field()). Zero padding at the ends of the
spectrum; the position channel tells the network where it is.

BlockScaler standardises X for this layout: one mean and std per spectral block (pooled over its bins, so the
bins keep their relative size) and per bulk column. Per-column scaling would give every bin its own scale and
undo the weight sharing.

Training (loss, Adam, L2 via weight decay, early stopping incl. time-block validation, dropout) is that of
TorchMLPRegressor (common/torch_mlp.py); only the network differs.
"""

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin

from nedsurfcast.common.torch_mlp import TorchMLPRegressor


class BlockScaler(TransformerMixin, BaseEstimator):
    """Standardise the first n_spectral blocks of n_bins columns per block, the remaining columns per column."""

    def __init__(self, n_spectral=4, n_bins=47):
        self.n_spectral = n_spectral
        self.n_bins = n_bins

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        ns = self.n_spectral * self.n_bins
        spec = X[:, :ns].reshape(len(X), self.n_spectral, self.n_bins)
        mean = np.repeat(np.nanmean(spec, axis=(0, 2)), self.n_bins)
        std = np.repeat(np.nanstd(spec, axis=(0, 2)), self.n_bins)
        self.mean_ = np.concatenate([mean, np.nanmean(X[:, ns:], 0)])
        self.scale_ = np.concatenate([std, np.nanstd(X[:, ns:], 0)])
        self.scale_[~(self.scale_ > 0)] = 1.0
        return self

    def transform(self, X):
        return (np.asarray(X, float) - self.mean_) / self.scale_


class _SpectrumCNN(torch.nn.Module):
    def __init__(self, n_spectral, n_bins, n_bulk, n_out, positions, channels, kernel_size, n_layers, dropout):
        super().__init__()
        self.n_spectral, self.n_bins, self.n_out = n_spectral, n_bins, n_out
        pos = np.asarray(positions, np.float32)
        pos = (pos - pos.mean()) / (pos.std() if pos.std() > 0 else 1.0)
        self.register_buffer("pos", torch.as_tensor(pos).view(1, 1, n_bins))
        self.inp = torch.nn.Conv1d(n_spectral + 1 + n_bulk, channels, 1)
        self.convs = torch.nn.ModuleList(
            torch.nn.Conv1d(channels, channels, kernel_size, padding="same") for _ in range(n_layers))
        self.drop = torch.nn.Dropout(dropout) if dropout else torch.nn.Identity()
        self.out = torch.nn.Conv1d(channels, n_out, 1)

    def forward(self, x):
        n, nb = len(x), self.n_bins
        spec = x[:, :self.n_spectral * nb].view(n, self.n_spectral, nb)
        bulk = x[:, self.n_spectral * nb:].unsqueeze(2).expand(-1, -1, nb)
        h = self.inp(torch.cat([spec, self.pos.expand(n, -1, -1), bulk], 1))
        for conv in self.convs:
            h = h + self.drop(torch.relu(conv(h)))
        return self.out(h).reshape(n, self.n_out * nb)              # block per output channel, as y


class TorchCNNRegressor(TorchMLPRegressor):
    """TorchMLPRegressor with a 1D CNN over the bins instead of fully connected layers (module docstring).
    Extra parameters: n_bins, n_spectral, positions (n_bins values, e.g. frequencies), channels, kernel_size,
    n_layers. hidden_layer_sizes is not used."""

    def __init__(self, n_bins=47, n_spectral=4, positions=None, channels=64, kernel_size=5, n_layers=5,
                 alpha=1e-3, learning_rate_init=1e-3, batch_size=256, max_iter=40, early_stopping=True,
                 validation_fraction=0.1, n_iter_no_change=5, tol=1e-4, validation_blocks=None, n_time_blocks=20,
                 validation_gap=None, dropout=0.0, random_state=0, device=None, verbose=False):
        super().__init__(alpha=alpha, learning_rate_init=learning_rate_init, batch_size=batch_size,
                         max_iter=max_iter, early_stopping=early_stopping, validation_fraction=validation_fraction,
                         n_iter_no_change=n_iter_no_change, tol=tol, validation_blocks=validation_blocks,
                         n_time_blocks=n_time_blocks, validation_gap=validation_gap, dropout=dropout,
                         random_state=random_state, device=device, verbose=verbose)
        self.n_bins = n_bins
        self.n_spectral = n_spectral
        self.positions = positions
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers

    def receptive_field(self) -> int:
        """Number of bins the output at one bin depends on."""
        return 1 + self.n_layers * (self.kernel_size - 1)

    def _build(self, n_in: int, n_out: int, gen: torch.Generator) -> torch.nn.Module:
        n_bulk = n_in - self.n_spectral * self.n_bins
        assert n_bulk >= 0 and n_out % self.n_bins == 0, "X / y do not match n_spectral and n_bins"
        positions = np.arange(self.n_bins) if self.positions is None else self.positions
        return _SpectrumCNN(self.n_spectral, self.n_bins, n_bulk, n_out // self.n_bins, positions,
                            self.channels, self.kernel_size, self.n_layers, self.dropout)
