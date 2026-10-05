"""
QC visualization for the combined ERA5 + CMEMS WW3 + buoy Zarr dataset.
 
Produces three diagnostic views, each catching a different class of
problem discussed while building this pipeline:
 
1. Coverage heatmap   -> time-range mismatches / gaps per buoy (silent
                         reindex-to-NaN issues, timezone misalignment).
2. Location map        -> buoy position vs. its matched WW3 grid cell,
                         to catch bad nearest-sea-cell matches (e.g. a
                         buoy snapped to a cell far away, or on land).
3. Per-buoy time series -> buoy observation vs. WW3 at the matched cell,
                         for VHM0 and (circularly-aware) VMDR, so you can
                         eyeball whether the residual looks like a
                         plausible bias correction or like noise/garbage.
 
Prerequisites (pip install):
    matplotlib xarray zarr numpy
"""
 
from pathlib import Path
 
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
 
ZARR_PATH = Path("data/processed/northsea_training_data_nowcast_buoys.zarr")
HEIGHT_VAR = "VHM0"
DIR_VAR = "VMDR"
 
 
def load(zarr_path: Path = ZARR_PATH) -> xr.Dataset:
    return xr.open_zarr(str(zarr_path), consolidated=False)
 
 
# ---------------------------------------------------------------------------
# 1. Coverage heatmap — spot time-range mismatches / gaps per buoy
# ---------------------------------------------------------------------------
 
def plot_coverage(ds: xr.Dataset, var: str = HEIGHT_VAR):
    valid = ds[f"buoy_{var}"].notnull().values  # (buoy_id, time) bool array
 
    fig, ax = plt.subplots(figsize=(14, max(2, 0.4 * ds.sizes["buoy_id"])))
    ax.imshow(valid, aspect="auto", cmap="Greens", interpolation="none")
    ax.set_yticks(range(ds.sizes["buoy_id"]))
    ax.set_yticklabels(ds.buoy_id.values)
    ax.set_xlabel("Time index")
    ax.set_title(f"Data coverage: buoy_{var} valid (green) vs. missing (white)")
 
    coverage_pct = valid.mean(axis=1) * 100
    for i, pct in enumerate(coverage_pct):
        ax.text(ds.sizes["time"] * 1.01, i, f"{pct:.0f}%", va="center", fontsize=8)
 
    plt.tight_layout()
    plt.show()
 
 
# ---------------------------------------------------------------------------
# 2. Location map — buoy position vs. matched WW3 grid cell
# ---------------------------------------------------------------------------
 
def plot_locations(ds: xr.Dataset):
    land_mask = ds[HEIGHT_VAR].isel(time=0).isnull().values.astype(float)
    lon, lat = ds.lon.values, ds.lat.values
 
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.contour(lon, lat, land_mask, levels=[0.5], colors="black", linewidths=1.0)
 
    buoy_lat = ds["buoy_lat"].values
    buoy_lon = ds["buoy_lon"].values
    matched_lat = ds["matched_grid_lat"].values
    matched_lon = ds["matched_grid_lon"].values
 
    ax.scatter(buoy_lon, buoy_lat, c="tab:blue", label="Buoy (true position)", zorder=3)
    ax.scatter(matched_lon, matched_lat, c="tab:red", marker="x", label="Matched grid cell", zorder=3)
 
    for i in range(len(buoy_lat)):
        ax.plot([buoy_lon[i], matched_lon[i]], [buoy_lat[i], matched_lat[i]], "k--", linewidth=0.7, alpha=0.6)
        dist_km = np.sqrt(
            ((matched_lat[i] - buoy_lat[i]) * 111) ** 2 +
            ((matched_lon[i] - buoy_lon[i]) * 111 * np.cos(np.radians(buoy_lat[i]))) ** 2
        )
        ax.annotate(f"{str(ds.buoy_id.values[i])}\n{dist_km:.1f} km", (buoy_lon[i], buoy_lat[i]), fontsize=7)
 
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Buoy positions vs. matched WW3 grid cells")
    ax.legend()
    plt.tight_layout()
    plt.show()
 
 
# ---------------------------------------------------------------------------
# 3. Per-buoy time series — buoy vs. WW3-at-matched-cell, height + direction
# ---------------------------------------------------------------------------
 
def plot_buoy_timeseries(ds: xr.Dataset, buoy_id: str):
    b = ds.sel(buoy_id=buoy_id)
    grid_at_cell = ds.sel(
        lat=b["matched_grid_lat"].compute().item(),
        lon=b["matched_grid_lon"].compute().item(),
        method="nearest",
    )
 
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
 
    # Height: buoy vs. grid, plus residual
    axes[0].plot(ds.time, b[f"buoy_{HEIGHT_VAR}"], label="Buoy", linewidth=0.8)
    axes[0].plot(ds.time, grid_at_cell[HEIGHT_VAR], label="WW3 (matched cell)", linewidth=0.8, alpha=0.8)
    axes[0].set_ylabel(f"{HEIGHT_VAR} (m)")
    axes[0].legend()
    axes[0].set_title(f"{buoy_id} — significant wave height")
 
    axes[1].plot(ds.time, b[f"residual_{HEIGHT_VAR}"], color="tab:red", linewidth=0.8)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_ylabel(f"Residual (m)")
    axes[1].set_title("Residual (buoy − WW3) — large spikes may indicate a bad match or buoy error")
 
    # Direction: plot sin component only, since raw degrees would show
    # fake 360->0 discontinuities that look like data problems but aren't
    axes[2].plot(ds.time, np.sin(np.deg2rad(b[f"buoy_{DIR_VAR}"])), label="Buoy sin(direction)", linewidth=0.8)
    axes[2].plot(ds.time, np.sin(np.deg2rad(grid_at_cell['VMDR_sin'])), label="WW3 sin(direction)", linewidth=0.8, alpha=0.8)
    axes[2].set_ylabel("sin(direction)")
    axes[2].set_xlabel("Time")
    axes[2].legend()
    axes[2].set_title(f"{DIR_VAR} — shown as sin() to avoid a fake 0/360 wraparound artifact")
 
    plt.tight_layout()
    plt.show()
 
 
if __name__ == "__main__":
    ds = load()
    plot_coverage(ds)
    plot_locations(ds)
    plot_buoy_timeseries(ds, 'vlaktevanderaan.2-2639')
    #for buoy_id in ds.buoy_id.values:
        #plot_buoy_timeseries(ds, buoy_id)#vlaktevanderaan.2-2639
 