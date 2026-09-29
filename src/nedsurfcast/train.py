"""
North Sea swell prediction — PyTorch training pipeline skeleton
==================================================================
 
Reads the aligned, coarsened Zarr store produced by
`northsea_swell_pipeline.py`, wraps it in a Dataset/DataLoader, and trains
a placeholder CNN. The model architecture here is intentionally minimal —
swap `PlaceholderSwellCNN` for a real architecture (ConvLSTM, U-Net,
Fourier neural operator, etc.) once the data pipeline and training loop
are validated end-to-end.
 
Prerequisites (pip install):
    torch xarray zarr numpy
"""
 
from pathlib import Path
 
import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from torch.utils.data import Dataset, DataLoader

import config as cfg
 
# ---------------------------------------------------------------------------
# 0. CONFIG
# ---------------------------------------------------------------------------

 
BATCH_SIZE = 4            # keep small given limited memory; full grids per sample add up fast
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)
 
# ---------------------------------------------------------------------------
# 1. DATASET
# ---------------------------------------------------------------------------
 
class NorthSeaSwellDataset(Dataset):
    """
    Each sample:
        X: wind variables over the past HISTORY_LEN timesteps, stacked as
           channels -> shape (HISTORY_LEN * len(WIND_VARS), lat, lon)
        y: wave variables at the current (already lead-time-aligned)
           timestep -> shape (len(WAVE_VARS), lat, lon)
 
    The lead-time offset was already baked into the data during the
    pipeline's `align_time()` step, so index i here lines up wind history
    ending at time t with wave labels at time t + lead_time_hours. This
    dataset does not need to know the lead time itself.
    """
 
    def __init__(self, zarr_path: Path, time_slice: slice, history_len: int = cfg.HISTORY_LEN):
        ds = xr.open_zarr(str(zarr_path), consolidated=False)
        ds = ds.sel(time=time_slice)
 
        # Load into memory as numpy arrays up front. This is only safe
        # because we've already coarsened the grid — for a much larger
        # domain/resolution, keep this lazy (dask-backed) and slice inside
        # __getitem__ instead.
        self.wind = ds[cfg.WIND_VARS].to_array().transpose("time", "variable", "lat", "lon").values.astype("float32")
        self.wave = ds[cfg.MODEL_OUTPUT_VARS].to_array().transpose("time", "variable", "lat", "lon").values.astype("float32")
 
        self.history_len = history_len
        self.n_wind_vars = len(cfg.WIND_VARS)
        self.n_wave_vars = len(cfg.MODEL_OUTPUT_VARS)
 
        # Valid indices: need `history_len` past steps available before t
        self.valid_indices = list(range(history_len - 1, self.wind.shape[0]))
 
        if len(self.valid_indices) == 0:
            raise ValueError(
                f"No valid samples: dataset has {self.wind.shape[0]} timesteps "
                f"but history_len={history_len} requires at least that many."
            )
 
    def __len__(self):
        return len(self.valid_indices)
 
    def __getitem__(self, idx):
        t = self.valid_indices[idx]
        window = self.wind[t - self.history_len + 1: t + 1]      # (history_len, n_wind_vars, lat, lon)
        X = window.reshape(-1, window.shape[-2], window.shape[-1])  # (history_len * n_wind_vars, lat, lon)
        y = self.wave[t]                                          # (n_wave_vars, lat, lon)
        return torch.from_numpy(X), torch.from_numpy(y)
 
 
# ---------------------------------------------------------------------------
# 2. PLACEHOLDER MODEL
# ---------------------------------------------------------------------------
 
class PlaceholderSwellCNN(nn.Module):
    """
    Minimal placeholder: a few stacked conv blocks that preserve spatial
    resolution (padding='same'), mapping stacked wind-history channels to
    wave variable channels. This is intentionally simple — it exists to
    validate the data pipeline and training loop shapes/gradients flow
    correctly, not to be a serious swell model. Replace with a real
    architecture (ConvLSTM for temporal structure, U-Net for multi-scale
    spatial features, etc.) once this trains without errors.
    """
 
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding="same", padding_mode="replicate"),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding="same", padding_mode="replicate"),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding="same", padding_mode="replicate"),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )
 
    def forward(self, x):
        #x = (x - x.mean(dim=(0, 2, 3), keepdim=True)) / (x.std(dim=(0, 2, 3), keepdim=True) + 1e-6)
        return self.net(x)
 
 
# ---------------------------------------------------------------------------
# 3. TRAIN / VALIDATE LOOP
# ---------------------------------------------------------------------------
 
def run_epoch(model, loader, optimizer, criterion, train: bool):
    model.train(mode=train)
    total_loss = 0.0
    n_batches = 0
 
    with torch.set_grad_enabled(train):
        for X, y in loader:
            X, y = X.to(cfg.DEVICE), y.to(cfg.DEVICE)            
 
            if train:
                optimizer.zero_grad()
 
            pred = model(X)
            loss = criterion(pred, y)

            #print(pred)
            #print(loss)
 
            if train:
                loss.backward()
                optimizer.step()
 
            total_loss += loss.item()
            n_batches += 1
 
    return total_loss / max(n_batches, 1)

class MaskedMSELoss(nn.Module):
    """
    Drop-in replacement for nn.MSELoss() that ignores NaN target cells
    (e.g. land, where wave height/period/direction are physically
    undefined). Used exactly like nn.MSELoss(): criterion(pred, target).
 
    If a batch happens to be all-NaN (fully masked), returns a zero loss
    with gradient rather than NaN, so a single degenerate batch doesn't
    poison training.
    """
 
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(target)
        if not mask.any():
            return pred.sum() * 0.0
        return ((pred[mask] - target[mask]) ** 2).mean()
 
def main():
    train_ds = NorthSeaSwellDataset(cfg.OUTPUT_ZARR, time_slice=slice(None, cfg.TRAIN_END))
    val_ds = NorthSeaSwellDataset(cfg.OUTPUT_ZARR, time_slice=slice(cfg.TRAIN_END, cfg.VAL_END))
 
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
 
    in_channels = cfg.HISTORY_LEN * len(cfg.WIND_VARS)
    out_channels = len(cfg.MODEL_OUTPUT_VARS)
 
    model = PlaceholderSwellCNN(in_channels, out_channels).to(cfg.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = MaskedMSELoss()
 
    best_val_loss = float("inf")

    print(f'training on device: {cfg.DEVICE}')
 
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, criterion, train=False)
 
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")
 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_DIR / cfg.BEST_MODEL_NAME)
 
    print(f"Training complete. Best val_loss={best_val_loss:.4f}, checkpoint saved to {CHECKPOINT_DIR / cfg.BEST_MODEL_NAME}")
 
if __name__ == "__main__":
    main()