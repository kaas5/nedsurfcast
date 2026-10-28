"""
Load the combined grid + buoy Zarr and turn it into model features
====================================================================

Shared by residual_regression.py (cross-validation on the buoys) and
predict_residual.py (gridded prediction). Both build their features with
build_features(), so a grid cell is treated exactly like a buoy's matched cell.

The Zarr itself is written by preprocess_dataset.build_combined_dataset().
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.cluster.hierarchy import fcluster, linkage

import config as cfg

TARGET = "VHM0"                                  # start with wave height; direction needs circular residuals
WW3_FEATURES = cfg.WAVE_VARS                     # WW3 local sea state
WIND_FEATURES = cfg.WIND_VARS                    # ERA5 at the matched cell
LAG_STEPS = [1, 2, 4]                            # in dataset time steps (e.g. 3 h each)


def load(zarr_path: Path = cfg.GRID_BUOYS_ZARR) -> xr.Dataset:
    return xr.open_zarr(str(zarr_path), consolidated=False)


def grid_variables(ds: xr.Dataset) -> list[str]:
    candidates = WW3_FEATURES + ["VMDR", "VMDR_sin", "VMDR_cos"] + WIND_FEATURES
    return [v for v in dict.fromkeys(candidates) if v in ds.data_vars]


def build_table(ds: xr.Dataset, target: str = TARGET) -> pd.DataFrame:
    buoy_ids = ds["buoy_id"].values
    cell_lat = xr.DataArray(ds["matched_grid_lat"].values, dims="buoy_id", coords={"buoy_id": buoy_ids})
    cell_lon = xr.DataArray(ds["matched_grid_lon"].values, dims="buoy_id", coords={"buoy_id": buoy_ids})

    # Pointwise (vectorized) selection of the matched cell for every buoy -> dims (time, buoy_id).
    # reset_coords drops the lat/lon coordinates so they can never leak in as features.
    local = ds[grid_variables(ds)].sel(lat=cell_lat, lon=cell_lon, method="nearest").compute().reset_coords(drop=True)
    features = build_features(local)

    obs = ds[f"buoy_{target}"].transpose("time", "buoy_id").compute()
    ww3_target = local[target]
    residual = obs - ww3_target  # same definition as residual_<var> in the Zarr

    table = xr.Dataset({**features, "ww3_target": ww3_target, "obs": obs, "target": residual})
    df = table.to_dataframe()
    df["buoy_group"] = df.index.get_level_values("buoy_id").map(buoy_groups(ds, target))
    return df.dropna(subset=["target"])


def build_features(local: xr.Dataset) -> xr.Dataset:
    """Features from WW3 + ERA5 at each point. Only per-point arithmetic and shifts along time,
    so `local` can be (time, buoy_id) for training or (time, lat, lon) for a gridded prediction."""
    f = {}
    for v in WW3_FEATURES:
        if v in local:
            f[f"ww3_{v}"] = local[v]

    if "VMDR" in local:
        rad = np.deg2rad(local["VMDR"])
        f["ww3_dir_sin"], f["ww3_dir_cos"] = np.sin(rad), np.cos(rad)
    elif "VMDR_sin" in local:
        f["ww3_dir_sin"], f["ww3_dir_cos"] = local["VMDR_sin"], local["VMDR_cos"]

    f["wind_u"], f["wind_v"] = local["u10"], local["v10"]
    f["wind_speed"] = np.hypot(local["u10"], local["v10"])
    f["msl_hpa"] = local["msl"] * 1000.0  # msl is stored in Pa / 100,000 (see preprocess_dataset.normalize_pressure)

    # Recent history: swell reflects wind over the previous hours/days, not just the current wind
    for k in LAG_STEPS:
        f[f"wind_speed_lag{k}"] = f["wind_speed"].shift(time=k)
        f[f"wind_u_lag{k}"] = f["wind_u"].shift(time=k)
        f[f"wind_v_lag{k}"] = f["wind_v"].shift(time=k)
        f[f"msl_tend{k}"] = f["msl_hpa"] - f["msl_hpa"].shift(time=k)  # pressure tendency: storm approaching or leaving

    return xr.Dataset(f)


def buoy_groups(ds: xr.Dataset, target: str = TARGET, max_km: float = cfg.BUOY_GROUP_DISTANCE_KM) -> pd.Series:
    """Single-linkage clusters of matched grid cells: a buoy joins a group when its cell is within
    max_km of any member. Buoys sharing a cell always end up together (distance 0)."""
    # Buoys without observations would otherwise chain neighbouring groups together
    ds = ds.sel(buoy_id=ds[f"buoy_{target}"].notnull().any("time").compute())
    lat, lon = ds["matched_grid_lat"].values, ds["matched_grid_lon"].values
    # Equirectangular km coordinates, accurate enough over the North Sea
    xy = np.column_stack([lon * 111.32 * np.cos(np.deg2rad(lat.mean())), lat * 110.57])
    labels = fcluster(linkage(xy, method="single"), t=max_km, criterion="distance")
    return pd.Series(labels, index=ds["buoy_id"].values, name="buoy_group")


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("ww3_target", "obs", "target", "buoy_group")]
