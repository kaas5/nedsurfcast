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
import xarray as xr

import config as cfg
 
#ZARR_PATH = Path("data/processed/northsea_training_data.zarr")
#TIME_INDEX = 0
 
WIND_VARS = ["u10", "v10"]
WAVE_VAR_FOR_PLOT = "VHM0"          # significant wave height, shown as the "output"
WAVE_VAR_FOR_LAND_MASK = "VHM0"     # any wave var works — all share the same NaN land mask
 
 
def get_land_mask(ds_t: xr.Dataset) -> np.ndarray:
    """1.0 over land, 0.0 over sea — derived from wave-label NaNs, not a
    separate coastline dataset. Used only for drawing the outline."""
    return ds_t[WAVE_VAR_FOR_LAND_MASK].isnull().values.astype(float)
 
 
def add_coastline(ax, lon, lat, land_mask):
    ax.contour(lon, lat, land_mask, levels=[0.5], colors="black", linewidths=1.2)
 
 
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
 
 
def plot_wave(ax, ds_t, land_mask, lon, lat):
    wave = ds_t[WAVE_VAR_FOR_PLOT].values
 
    mesh = ax.pcolormesh(lon, lat, wave, shading="auto", cmap="plasma")
    plt.colorbar(mesh, ax=ax, label=f"{WAVE_VAR_FOR_PLOT} (m)", fraction=0.046, pad=0.04)
 
    add_coastline(ax, lon, lat, land_mask)
    ax.set_title(f"Output/label: {WAVE_VAR_FOR_PLOT} (CMEMS WW3)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
 
 
def plot_timestep(time_index: int = 0, zarr_path: Path = cfg.OUTPUT_ZARR):
    ds = xr.open_zarr(str(zarr_path), consolidated=False)
    ds_t = ds.isel(time=time_index)
 
    lon = ds_t["lon"].values
    lat = ds_t["lat"].values
    land_mask = get_land_mask(ds_t)
 
    fig, (ax_wind, ax_wave) = plt.subplots(1, 2, figsize=(14, 6))
    plot_wind(ax_wind, ds_t, land_mask, lon, lat)
    plot_wave(ax_wave, ds_t, land_mask, lon, lat)
 
    fig.suptitle(f"Timestep {time_index} — {ds_t.time.values}")
    plt.tight_layout()
    plt.show()
 
 
if __name__ == "__main__":
    plot_timestep(time_index=1)
 