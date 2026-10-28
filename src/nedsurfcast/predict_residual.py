"""
Gridded WW3 residual prediction along the Dutch coast for one timestamp
=========================================================================

Trains the pooled residual model (models.py) on all buoys,
then applies it to every WW3 sea cell inside cfg.DUTCH_COAST at a single
chosen time. The features are built with the same build_features() as in
training, so a grid cell is treated exactly like a buoy's matched cell.

Output: a map of WW3 wave height, the predicted residual (buoy - WW3) and
the corrected wave height, with the buoy observations at that time overlaid.

Prerequisites (pip install):
    scikit-learn scipy pandas xarray zarr numpy matplotlib
"""

import numpy as np
import pandas as pd
import xarray as xr

import config as cfg
from load_data import LAG_STEPS, TARGET, build_features, build_table, feature_columns, grid_variables, load
from models import MODELS
from plot import plot_prediction

TIMESTAMP = "2024-12-06 09:00"   # storm Darragh; snapped to the nearest grid time
MODEL_NAME = "hist-gbr"


# ---------------------------------------------------------------------------
# 1. TRAIN ON ALL BUOYS
# ---------------------------------------------------------------------------

def train_final_model(table: pd.DataFrame, name: str = MODEL_NAME):
    cols = feature_columns(table)
    model = MODELS[name]()
    model.fit(table[cols], table["target"])
    return model, cols


# ---------------------------------------------------------------------------
# 2. PREDICT ON THE GRID
# ---------------------------------------------------------------------------

def select_box(ds: xr.Dataset, box: dict = cfg.DUTCH_COAST) -> xr.Dataset:
    # Boolean masks instead of slices, so it works for ascending and descending lat
    (lat_min, lat_max), (lon_min, lon_max) = box["lat"], box["lon"]
    return ds.sel(lat=(ds.lat >= lat_min) & (ds.lat <= lat_max), lon=(ds.lon >= lon_min) & (ds.lon <= lon_max))


def predict_at(ds: xr.Dataset, model, cols: list[str], timestamp, target: str = TARGET) -> xr.Dataset:
    i = int(ds.indexes["time"].get_indexer([pd.Timestamp(timestamp)], method="nearest")[0])
    print(f"Requested {timestamp}, using grid time {pd.Timestamp(ds.time.values[i])}")

    # The lag features look back max(LAG_STEPS) steps, so include those steps in the window
    window = ds[grid_variables(ds)].isel(time=slice(max(0, i - max(LAG_STEPS)), i + 1))
    features = build_features(select_box(window).compute()).isel(time=-1)

    ww3 = features[f"ww3_{target}"]
    X = np.column_stack([features[c].values.ravel() for c in cols])
    sea = ~np.isnan(ww3.values.ravel())
    residual = np.full(ww3.size, np.nan, dtype=float)
    residual[sea] = model.predict(X[sea])
    residual = xr.DataArray(residual.reshape(ww3.shape), coords=ww3.coords, dims=ww3.dims)

    return xr.Dataset({
        target: ww3,
        f"residual_{target}": residual,
        f"{target}_corrected": (ww3 + residual).clip(min=0),  # a correction can push calm seas below zero
    })


def buoys_at(ds: xr.Dataset, table: pd.DataFrame, model, cols: list[str], pred: xr.Dataset,
             target: str = TARGET) -> pd.DataFrame:
    """Buoys with an observation at the prediction time and their matched cell inside the box."""
    t = pred.time.values
    rows = table[table.index.get_level_values("time") == t].droplevel("time")
    lat_min, lat_max = cfg.DUTCH_COAST["lat"]
    lon_min, lon_max = cfg.DUTCH_COAST["lon"]

    out = []
    for buoy_id, row in rows.iterrows():
        b = ds.sel(buoy_id=buoy_id)
        cell_lat, cell_lon = float(b["matched_grid_lat"]), float(b["matched_grid_lon"])
        if not (lat_min <= cell_lat <= lat_max and lon_min <= cell_lon <= lon_max):
            continue
        cell = pred.sel(lat=cell_lat, lon=cell_lon, method="nearest")
        out.append({
            "buoy_id": buoy_id,
            "lat": float(b["buoy_lat"]),
            "lon": float(b["buoy_lon"]),
            "obs": row["obs"],
            "ww3": row["ww3_target"],
            "residual_obs": row["target"],
            "residual_pred": float(cell[f"residual_{target}"]),
            "corrected": float(cell[f"{target}_corrected"]),
            # Same model on the training row: must equal residual_pred, otherwise grid and training features differ
            "in_sample_pred": float(model.predict(row[cols].to_frame().T.astype(float))[0]),
        })
    return pd.DataFrame(out).set_index("buoy_id")


# ---------------------------------------------------------------------------
# 3. RUN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ds = load()
    table = build_table(ds)
    model, cols = train_final_model(table)

    pred = predict_at(ds, model, cols, TIMESTAMP)
    buoys = buoys_at(ds, table, model, cols, pred)

    print(f"\nBuoys inside the box at {pd.Timestamp(pred.time.values)}:")
    print(buoys.drop(columns=["lat", "lon"]).round(3).to_string())
    if len(buoys):
        print(f"\nmax |grid - in-sample| prediction difference: {(buoys['residual_pred'] - buoys['in_sample_pred']).abs().max():.2e}")

    plot_prediction(pred, buoys)
