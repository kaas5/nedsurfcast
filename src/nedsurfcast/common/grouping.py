"""
Group nearby buoys, so leave-one-group validation never splits them between train and test
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy.cluster.hierarchy import fcluster, linkage

from nedsurfcast import config as cfg


def buoy_groups(ds: xr.Dataset, target: str = "VHM0", max_km: float = cfg.BUOY_GROUP_DISTANCE_KM) -> pd.Series:
    """Single-linkage clusters of matched grid cells: a buoy joins a group when its cell is within
    max_km of any member. Buoys sharing a cell always end up together (distance 0)."""
    # Buoys without observations would otherwise chain neighbouring groups together
    ds = ds.sel(buoy_id=ds[f"buoy_{target}"].notnull().any("time").compute())
    lat, lon = ds["matched_grid_lat"].values, ds["matched_grid_lon"].values
    # Equirectangular km coordinates, accurate enough over the North Sea
    xy = np.column_stack([lon * 111.32 * np.cos(np.deg2rad(lat.mean())), lat * 110.57])
    labels = fcluster(linkage(xy, method="single"), t=max_km, criterion="distance")
    return pd.Series(labels, index=ds["buoy_id"].values, name="buoy_group")
