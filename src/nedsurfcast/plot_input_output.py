"""
North Sea swell prediction — minimal input/output visualization
==================================================================
 
Plots one timestep of the aligned Zarr training data: wind (input) on
the left, a wave variable (output/label) on the right, both with a
land/sea outline so the North Sea coastline is visible.
 
The land/sea outline is derived from where the wave label data is NaN
(wave height/period/direction are physically undefined over land) —
no extra coastline dataset or dependency (e.g. cartopy) required.
 
Prerequisites (pip install):
    matplotlib xarray zarr numpy
"""
 
from pathlib import Path
 
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

import config as cfg
 
# NOTE: adjust this import to match your actual project layout
# (e.g. `from nedsurfcast.train import PlaceholderSwellCNN, HISTORY_LEN`)
from train import PlaceholderSwellCNN
 
#ZARR_PATH = Path("data/processed/northsea_training_data.zarr")
TIME_INDEX = 2000
CHECKPOINT_PATH = Path(f"checkpoints/{cfg.BEST_MODEL_NAME}")
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
WIND_VARS = ["u10", "v10"]
WAVE_VAR_FOR_PLOT = "VHM0"          # significant wave height, shown as the "output"
WAVE_VAR_FOR_LAND_MASK = "VHM0"     # any wave var works — all share the same NaN land mask
 
 
def get_land_mask(ds_t: xr.Dataset) -> np.ndarray:
    """1.0 over land, 0.0 over sea — derived from wave-label NaNs, not a
    separate coastline dataset. Used only for drawing the outline."""
    return ds_t[WAVE_VAR_FOR_LAND_MASK].isnull().values.astype(float)
 
 
def add_coastline(ax, lon, lat, land_mask):
    ax.contour(lon, lat, land_mask, levels=[0.5], colors="black", linewidths=1.2)
 
def plot_pressure(ax, ds_t, land_mask, lon, lat):
    msl = ds_t["msl"].values
 
    mesh = ax.pcolormesh(lon, lat, msl, shading="auto", cmap="viridis")
    plt.colorbar(mesh, ax=ax, label="Wind speed (m/s)", fraction=0.046, pad=0.04)

    add_coastline(ax, lon, lat, land_mask)
    ax.set_title("Input: mean sea level pressure")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
 
def plot_wind(ax, ds_t, land_mask, lon, lat, quiver_stride: int = None):
    u10 = ds_t["u10"].values
    v10 = ds_t["v10"].values
    speed = np.sqrt(u10 ** 2 + v10 ** 2)
 
    mesh = ax.pcolormesh(lon, lat, speed, shading="auto", cmap="viridis")
    plt.colorbar(mesh, ax=ax, label="Wind speed (m/s)", fraction=0.046, pad=0.04)
 
    # Sparse quiver arrows so the plot stays readable regardless of grid size
    if quiver_stride is None:
        quiver_stride = max(len(lon) // 25, 1)
    s = quiver_stride
    ax.quiver(
        lon[::s], lat[::s],
        u10[::s, ::s], v10[::s, ::s],
        color="white", alpha=0.7, scale=300, width=0.0025,
    )
 
    add_coastline(ax, lon, lat, land_mask)
    ax.set_title("Input: 10m wind (ERA5)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
 
def plot_wave_size_direction(ax, ds_t, land_mask, lon, lat, quiver_stride: int = None):
    u10 = ds_t["VMDR_cos"].values
    v10 = ds_t["VMDR_sin"].values
    wave_size = ds_t["VHM0"].values
    #direction_in_degrees = np.rad2deg(np.arctan2(v10, u10)) % 360#1#np.sqrt(u10 ** 2 + v10 ** 2)
    
    wave_size = np.where(land_mask > 0.5, np.nan, wave_size)
    u10 = np.where(land_mask > 0.5, np.nan, u10)
    v10 = np.where(land_mask > 0.5, np.nan, v10)

    mesh = ax.pcolormesh(lon, lat, wave_size, shading="auto", cmap="viridis", vmin=0.0)
    plt.colorbar(mesh, ax=ax, label="Wave size (m)", fraction=0.046, pad=0.04)
 
    # Sparse quiver arrows so the plot stays readable regardless of grid size
    if quiver_stride is None:
        quiver_stride = max(len(lon) // 25, 1)
    s = quiver_stride
    ax.quiver(
        lon[::s], lat[::s],
        u10[::s, ::s] * wave_size [::s, ::s] * 20, v10[::s, ::s] * wave_size [::s, ::s] * 20,
        color="white", alpha=0.7, scale=600, width=0.0025,
    )
 
    add_coastline(ax, lon, lat, land_mask)
    ax.set_title("Wave size and direction")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

 
def plot_wave(ax, ds_t, land_mask, lon, lat):
    wave = ds_t[WAVE_VAR_FOR_PLOT].values
 
    mesh = ax.pcolormesh(lon, lat, wave, shading="auto", cmap="plasma")
    plt.colorbar(mesh, ax=ax, label=f"{WAVE_VAR_FOR_PLOT} (m)", fraction=0.046, pad=0.04)
 
    add_coastline(ax, lon, lat, land_mask)
    ax.set_title(f"Output/label: {WAVE_VAR_FOR_PLOT} (CMEMS WW3)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
 
 
def load_model(checkpoint_path: Path) -> PlaceholderSwellCNN:
    in_channels = cfg.HISTORY_LEN * len(cfg.WIND_VARS)
    out_channels = len(cfg.MODEL_OUTPUT_VARS)
    model = PlaceholderSwellCNN(in_channels, out_channels).to(cfg.DEVICE)
    model.load_state_dict(torch.load(str(checkpoint_path), map_location=cfg.DEVICE))
    model.eval()
    return model
 
 
def build_input_window(ds: xr.Dataset, time_index: int) -> np.ndarray:
    """
    Build the (HISTORY_LEN * n_wind_vars, lat, lon) input the model
    expects, using the HISTORY_LEN timesteps ending at time_index. If
    fewer than HISTORY_LEN steps are available before time_index (e.g.
    plotting one of the first few timesteps in the dataset), the earliest
    available frame is repeated backward to pad the window — this keeps
    the plot usable even at the start of the dataset, but note the
    padded frames are not real data, so predictions near the very start
    of the time series are less meaningful than elsewhere.
    """
    start = time_index - cfg.HISTORY_LEN + 1
    if start < 0:
        pad = -start
        indices = [0] * pad + list(range(0, time_index + 1))
    else:
        indices = list(range(start, time_index + 1))
 
    wind = ds[cfg.WIND_VARS].isel(time=indices).to_array().transpose("time", "variable", "lat", "lon").values
    window = wind.reshape(-1, wind.shape[-2], wind.shape[-1]).astype("float32")
    #window = np.nan_to_num(window, nan=0.0)
    return window
 
 
def predict_timestep(ds: xr.Dataset, time_index: int, checkpoint_path: Path = CHECKPOINT_PATH) -> np.ndarray:
    model = load_model(checkpoint_path)
    X = build_input_window(ds, time_index)
    X_tensor = torch.from_numpy(X).unsqueeze(0).to(cfg.DEVICE)  # add batch dim
 
    with torch.no_grad():
        pred = model(X_tensor)[0].cpu().numpy()  # (n_wave_vars, lat, lon)
 
    #var_index = cfg.MODEL_OUTPUT_VARS.index(WAVE_VAR_FOR_PLOT)
    #return pred[var_index]
    return pred
 
def plot_prediction(ax, pred_field, land_mask, lon, lat):
    # Mask land cells for display consistency — the model was trained with
    # a masked loss that ignores land, so predictions there are arbitrary
    # and would be misleading to show at face value.
    pred_display = np.where(land_mask > 0.5, np.nan, pred_field)
 
    mesh = ax.pcolormesh(lon, lat, pred_display, shading="auto", cmap="plasma")
    plt.colorbar(mesh, ax=ax, label=f"Predicted {WAVE_VAR_FOR_PLOT} (m)", fraction=0.046, pad=0.04)
 
    add_coastline(ax, lon, lat, land_mask)
    ax.set_title("Model prediction")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
 
 
def plot_timestep(zarr_path: Path = cfg.OUTPUT_ZARR, time_index: int = TIME_INDEX, checkpoint_path: Path = CHECKPOINT_PATH):
    ds = xr.open_zarr(str(zarr_path), consolidated=False)
    ds_t = ds.isel(time=time_index)
 
    lon = ds_t["lon"].values
    lat = ds_t["lat"].values
    land_mask = get_land_mask(ds_t)
 
    fig, (ax_wind, ax_pressure, ax_wave, ax_pred) = plt.subplots(1, 4, figsize=(20, 6))
    plot_wind(ax_wind, ds_t, land_mask, lon, lat)
    plot_pressure(ax_pressure, ds_t, land_mask, lon, lat)
    plot_wave_size_direction(ax_wave, ds_t, land_mask, lon, lat)
    
    pred = predict_timestep(ds, time_index, checkpoint_path)

    pred_ds = prediction_to_dataset(pred, lat=lat, lon=lon, time_value=ds_t.time.values)
    plot_wave_size_direction(ax_pred, pred_ds, land_mask, lon, lat)
 
    fig.suptitle(f"Timestep {time_index} — {ds_t.time.values}")
    plt.tight_layout()
    plt.show()


def prediction_to_dataset(
    pred: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    time_value=None,
    wave_vars: list[str] = cfg.MODEL_OUTPUT_VARS,
) -> xr.Dataset:
    """
    Convert raw model output (n_wave_vars, lat, lon) into an xarray
    Dataset with the correct variable names as data_vars, matching the
    label data's naming convention.
 
    Wave direction stays encoded as VMDR_sin/VMDR_cos, matching how it's
    represented everywhere else in the pipeline (training labels, model
    output channels) -- it is not converted back into a VMDR degrees
    variable here.
 
    Args:
        pred: array of shape (len(wave_vars), lat, lon), e.g. straight
            from model(X)[0].cpu().numpy().
        lat, lon: 1D coordinate arrays matching the model's spatial grid.
        time_value: optional scalar (e.g. a numpy datetime64) to attach
            as a length-1 time coordinate -- convenient when comparing
            predictions against a specific label timestep.
        wave_vars: the channel ordering pred is assumed to follow.
            Defaults to this module's WAVE_VARS.
    """
    if pred.shape[0] != len(wave_vars):
        raise ValueError(
            f"pred has {pred.shape[0]} channels but wave_vars has {len(wave_vars)} "
            f"entries ({wave_vars}) -- channel count must match exactly."
        )
 
    data_vars = {name: (("lat", "lon"), pred[i]) for i, name in enumerate(wave_vars)}
 
    coords = {"lat": lat, "lon": lon}
    if time_value is not None:
        coords["time"] = time_value
 
    return xr.Dataset(data_vars=data_vars, coords=coords)   
 
 
if __name__ == "__main__":
    plot_timestep()