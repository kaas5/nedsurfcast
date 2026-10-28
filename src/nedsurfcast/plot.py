"""
Plots for the WW3 residual model
==================================

Shared by residual_regression.py (predicted vs. actual residual at the
buoys) and predict_residual.py (gridded correction along the Dutch coast).
Both use the same colour scales and labels, so a residual looks the same
in every figure.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from load_data import TARGET

RESIDUAL_LABEL = "buoy - WW3 (m)"


def height_style(vmax: float) -> dict:
    return dict(cmap="viridis", vmin=0, vmax=vmax)


def residual_style(rmax: float) -> dict:
    # Symmetric around 0 so under- and overestimation get equal weight
    return dict(cmap="RdBu_r", vmin=-rmax, vmax=rmax)


def symmetric_limit(*values) -> float:
    return float(np.nanmax([np.nanmax(np.abs(np.asarray(v, dtype=float))) for v in values if np.size(v)]))


# ---------------------------------------------------------------------------
# 1. PREDICTED VS. ACTUAL RESIDUAL AT THE BUOYS
# ---------------------------------------------------------------------------

def plot_pred_vs_actual(df: pd.DataFrame, pred: pd.Series, title: str):
    ok = pred.notna()
    d = df.loc[ok]
    lim = symmetric_limit(d["target"], pred[ok]) * 1.05

    fig, ax = plt.subplots(figsize=(7, 7))
    for buoy, g in d.groupby(level="buoy_id"):
        ax.scatter(g["target"], pred.loc[g.index], s=5, alpha=0.4, label=str(buoy))
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8, label="perfect prediction")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(f"Actual residual: {RESIDUAL_LABEL}")
    ax.set_ylabel(f"Predicted residual: {RESIDUAL_LABEL}")
    ax.legend(fontsize=7, markerscale=3)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 2. GRIDDED CORRECTION AT ONE TIMESTAMP
# ---------------------------------------------------------------------------

def plot_prediction(pred: xr.Dataset, buoys: pd.DataFrame, target: str = TARGET):
    lon, lat = pred.lon.values, pred.lat.values
    land_mask = pred[target].isnull().values.astype(float)
    ww3, residual, corrected = pred[target], pred[f"residual_{target}"], pred[f"{target}_corrected"]

    vmax = float(np.nanmax([ww3.max(), corrected.max(), buoys["obs"].max() if len(buoys) else 0]))
    rmax = symmetric_limit(residual.values, buoys["residual_obs"])

    panels = [
        (ww3, "WW3 wave height (m)", height_style(vmax), "obs"),
        (residual, f"Predicted residual: {RESIDUAL_LABEL}", residual_style(rmax), "residual_obs"),
        (corrected, "Corrected wave height (m)", height_style(vmax), "obs"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    for ax, (field, title, style, buoy_col) in zip(axes, panels):
        mesh = ax.pcolormesh(lon, lat, field.values, shading="nearest", **style)
        ax.contour(lon, lat, land_mask, levels=[0.5], colors="black", linewidths=0.8)
        # Buoys on the same colour scale: observed value, or observed residual in the middle panel
        ax.scatter(buoys["lon"], buoys["lat"], c=buoys[buoy_col], edgecolors="black", s=45, zorder=3, **style)
        fig.colorbar(mesh, ax=ax, shrink=0.8)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_aspect(1 / np.cos(np.deg2rad(lat.mean())))
    axes[0].set_ylabel("Latitude")

    fig.suptitle(f"WW3 residual correction at {pd.Timestamp(pred.time.values):%Y-%m-%d %H:%M} (dots: buoy observations)")
    plt.tight_layout()
    plt.show()
