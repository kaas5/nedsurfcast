"""
Group nearby buoys, so leave-one-group validation never splits them between train and test
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy.cluster.hierarchy import fcluster, linkage

from nedsurfcast import config as cfg


def group_points(lat, lon, max_km: float = cfg.BUOY_GROUP_DISTANCE_KM) -> np.ndarray:
    """Single-linkage clusters: a point joins a group when it is within max_km of any member.
    Points at the same position always end up together (distance 0). -> group label per point."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    if len(lat) < 2:
        return np.ones(len(lat), dtype=int)
    # Equirectangular km coordinates, accurate enough over the North Sea
    xy = np.column_stack([lon * 111.32 * np.cos(np.deg2rad(lat.mean())), lat * 110.57])
    return fcluster(linkage(xy, method="single"), t=max_km, criterion="distance")


def buoy_groups(ds: xr.Dataset, target: str = "VHM0", max_km: float = cfg.BUOY_GROUP_DISTANCE_KM) -> pd.Series:
    """Clusters of matched grid cells (group_points), so buoys sharing or neighbouring a cell stay together."""
    # Buoys without observations would otherwise chain neighbouring groups together
    ds = ds.sel(buoy_id=ds[f"buoy_{target}"].notnull().any("time").compute())
    labels = group_points(ds["matched_grid_lat"].values, ds["matched_grid_lon"].values, max_km)
    return pd.Series(labels, index=ds["buoy_id"].values, name="buoy_group")
