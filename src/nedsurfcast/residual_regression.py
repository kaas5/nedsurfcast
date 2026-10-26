"""
Pooled point-wise regression: can local conditions predict WW3's error?
=========================================================================
 
A go/no-go test before investing in a grid-to-grid network. Every row is
one (buoy, time) pair:
 
    features = WW3 output + ERA5 wind/pressure at the buoy's matched grid cell
    target   = buoy observation - WW3 value at that cell   (the residual)
 
All buoys are pooled into ONE table and ONE model. No coordinates or buoy
ids are used as features, so whatever the model learns has to be a
relationship between local conditions and WW3 error, which is exactly
what would let it generalize to grid cells that have no buoy.
 
Two validation schemes answer two different questions:
  - leave-one-group-out: does the correction transfer to an unseen LOCATION?
                         (nearby buoys are held out together, see buoy_groups)
  - blocked-time       : does it transfer to unseen WEATHER (other time periods)?
 
Prerequisites (pip install):
    scikit-learn pandas xarray zarr numpy matplotlib
 
IMPORTANT: build the gridded Zarr with lead_time_hours=0 for this test.
With a lead-time offset, the grid time axis is shifted relative to real
time, and the buoy series (resampled onto that axis) would be compared
against WW3 values from a different real time, corrupting every residual.
"""
 
from pathlib import Path
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as cfg
  
TARGET = "VHM0"                                  # start with wave height; direction needs circular residuals
WW3_FEATURES = cfg.WAVE_VARS       # WW3 local sea state
WIND_FEATURES = cfg.WIND_VARS            # ERA5 at the matched cell
LAG_STEPS = [1, 2, 4]                            # in dataset time steps (e.g. 3 h each)
 
N_TIME_BLOCKS = 5
TIME_GAP_HOURS = 24                              # buffer around each test block to limit autocorrelation leakage
 

# ---------------------------------------------------------------------------
# 1. BUILD THE POOLED TABLE
# ---------------------------------------------------------------------------
 
def build_table(ds: xr.Dataset, target: str = TARGET) -> pd.DataFrame:
    buoy_ids = ds["buoy_id"].values
    cell_lat = xr.DataArray(ds["matched_grid_lat"].values, dims="buoy_id", coords={"buoy_id": buoy_ids})
    cell_lon = xr.DataArray(ds["matched_grid_lon"].values, dims="buoy_id", coords={"buoy_id": buoy_ids})
 
    candidates = WW3_FEATURES + ["VMDR", "VMDR_sin", "VMDR_cos"] + WIND_FEATURES
    grid_vars = [v for v in dict.fromkeys(candidates) if v in ds.data_vars]
 
    # Pointwise (vectorized) selection of the matched cell for every buoy -> dims (time, buoy_id).
    # reset_coords drops the lat/lon coordinates so they can never leak in as features.
    local = ds[grid_vars].sel(lat=cell_lat, lon=cell_lon, method="nearest").compute().reset_coords(drop=True)
 
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
    f["msl_hpa"] = local["msl"] * 1000.0  # msl is stored in Pa / 100,000 (see load_data.normalize_pressure)
 
    # Recent history: swell reflects wind over the previous hours/days, not just the current wind
    for k in LAG_STEPS:
        f[f"wind_speed_lag{k}"] = f["wind_speed"].shift(time=k)
        f[f"wind_u_lag{k}"] = f["wind_u"].shift(time=k)
        f[f"wind_v_lag{k}"] = f["wind_v"].shift(time=k)
        f[f"msl_tend{k}"] = f["msl_hpa"] - f["msl_hpa"].shift(time=k)  # pressure tendency: storm approaching or leaving
 
    obs = ds[f"buoy_{target}"].transpose("time", "buoy_id").compute()
    ww3_target = local[target]
    residual = obs - ww3_target  # same definition as residual_<var> in the Zarr
 
    table = xr.Dataset({**f, "ww3_target": ww3_target, "obs": obs, "target": residual})
    df = table.to_dataframe()
    df["buoy_group"] = df.index.get_level_values("buoy_id").map(buoy_groups(ds, target))
    return df.dropna(subset=["target"])


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
 
 
# ---------------------------------------------------------------------------
# 2. MODELS (all start from the same two baselines)
# ---------------------------------------------------------------------------
 
MODELS = {
    # Constant offset: how much of the error is plain average bias? Any real model must beat this.
    "mean-bias": lambda: DummyRegressor(strategy="mean"),
    "ridge": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)),
    # Small, heavily regularized trees; handles NaN (lags at the start, gaps) natively
    "hist-gbr": lambda: HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=50, l2_regularization=1.0
    ),
}
 
 
# ---------------------------------------------------------------------------
# 3. VALIDATION SCHEMES
# ---------------------------------------------------------------------------
 
# Nearby buoys see (almost) the same conditions, so a whole group is held out at once
def evaluate_leave_one_group_out(df, cols, make_model) -> pd.Series:
    groups = df["buoy_group"]
    pred = pd.Series(np.nan, index=df.index)
    for train_idx, test_idx in LeaveOneGroupOut().split(df, groups=groups):
        model = make_model()
        model.fit(df.iloc[train_idx][cols], df.iloc[train_idx]["target"])
        pred.iloc[test_idx] = model.predict(df.iloc[test_idx][cols])
    return pred
 
 
def evaluate_blocked_time(df, cols, make_model, n_blocks=N_TIME_BLOCKS, gap_hours=TIME_GAP_HOURS) -> pd.Series:
    times = df.index.get_level_values("time")
    blocks = np.array_split(np.sort(times.unique()), n_blocks)
    gap = pd.Timedelta(hours=gap_hours)
    pred = pd.Series(np.nan, index=df.index)
 
    for block in blocks:
        t0, t1 = pd.Timestamp(block[0]), pd.Timestamp(block[-1])
        is_test = (times >= t0) & (times <= t1)
        is_train = (times < t0 - gap) | (times > t1 + gap)  # gap rows are used by neither side
        if is_test.sum() == 0 or is_train.sum() == 0:
            continue
        model = make_model()
        model.fit(df.loc[is_train, cols], df.loc[is_train, "target"])
        pred.loc[is_test] = model.predict(df.loc[is_test, cols])
    return pred
 
 
SCHEMES = {
    "leave-one-group": evaluate_leave_one_group_out,
    "blocked-time": evaluate_blocked_time,
}
 
 
# ---------------------------------------------------------------------------
# 4. REPORTING
# ---------------------------------------------------------------------------
 
def rmse(x) -> float:
    return float(np.sqrt(np.mean(np.square(x))))
 
 
def summarize(name: str, scheme: str, df: pd.DataFrame, pred: pd.Series):
    ok = pred.notna()
    raw_err = df.loc[ok, "target"]        # error of uncorrected WW3
    new_err = raw_err - pred[ok]          # error of WW3 + predicted correction
    r_raw, r_new = rmse(raw_err), rmse(new_err)
    print(
        f"{scheme:<15} {name:<10} n={int(ok.sum()):>6}  "
        f"RMSE raw={r_raw:.3f} corrected={r_new:.3f}  skill={1 - r_new / r_raw:+.3f}  "
        f"bias raw={raw_err.mean():+.3f} corrected={new_err.mean():+.3f}"
    )
 
 
def _rmse_table(d: pd.DataFrame, grouper) -> pd.DataFrame:
    g = d.assign(sq_raw=d["err_raw"] ** 2, sq_new=d["err_new"] ** 2).groupby(grouper, observed=True)
    out = pd.DataFrame({"n": g.size(), "rmse_raw": np.sqrt(g["sq_raw"].mean()), "rmse_new": np.sqrt(g["sq_new"].mean())})
    out["skill"] = 1 - out["rmse_new"] / out["rmse_raw"]
    return out
 
 
def breakdown(df: pd.DataFrame, pred: pd.Series, label: str):
    ok = pred.notna()
    d = df.loc[ok].copy()
    d["err_raw"] = d["target"]
    d["err_new"] = d["target"] - pred[ok]
 
    print(f"\n--- {label}: per buoy ---")
    print(_rmse_table(d, d.index.get_level_values("buoy_id")).round(3))
 
    # Bin by the WW3 value (known at inference time), not by the observation
    d["ww3_sea_state"] = pd.qcut(d["ww3_target"], [0, 0.5, 0.8, 0.95, 1.0], duplicates="drop")
    print(f"\n--- {label}: by WW3 sea state (quantile bins, last bin = most extreme 5%) ---")
    print(_rmse_table(d, d["ww3_sea_state"]).round(3))
 
 
def plot_pred_vs_actual(df: pd.DataFrame, pred: pd.Series, title: str):
    ok = pred.notna()
    d = df.loc[ok]
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for buoy, g in d.groupby(level="buoy_id"):
        ax.scatter(g["target"], pred.loc[g.index], s=5, alpha=0.4, label=str(buoy))
    lim = float(np.nanmax(np.abs(d["target"]))) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8, label="perfect prediction")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Actual residual: buoy - WW3 (m)")
    ax.set_ylabel("Predicted residual (m)")
    ax.set_title(title)
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.show()
 
 
# ---------------------------------------------------------------------------
# 5. RUN
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    ds = xr.open_zarr(str(cfg.GRID_BUOYS_ZARR), consolidated=False)
    df = build_table(ds)
    cols = feature_columns(df)
 
    print("Rows per buoy (after dropping missing targets):")
    print(df.groupby(level="buoy_id").size().to_string())
    print("\nBuoy groups (held out together in leave-one-group):")
    print(df.reset_index().groupby("buoy_group")["buoy_id"].unique().apply(list).to_string())
    print(f"\nFeatures ({len(cols)}): {cols}\n")
 
    results = {}
    for scheme, evaluate in SCHEMES.items():
        for name, make_model in MODELS.items():
            pred = evaluate(df, cols, make_model)
            results[(scheme, name)] = pred
            summarize(name, scheme, df, pred)
        print()
 
    chosen = ("leave-one-group", "hist-gbr")
    breakdown(df, results[chosen], "leave-one-group / hist-gbr")
    plot_pred_vs_actual(df, results[chosen], "Leave-one-group-out: predicted vs. actual WW3 residual")
 