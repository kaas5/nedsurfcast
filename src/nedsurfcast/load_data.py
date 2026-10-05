"""
North Sea swell prediction — data pipeline skeleton
=====================================================
 
Goes from raw ERA5 wind NetCDF files + CMEMS WAVEWATCH III NetCDF files
to a single aligned, regridded Zarr store ready for ML training.
 
Prerequisites (pip install):
    xarray netCDF4 xesmf dask zarr cdsapi copernicusmarine numpy
 
This script assumes you've already downloaded the raw files locally
(see the download stubs below for how to do that with cdsapi /
copernicusmarine). It focuses on the processing pipeline itself.
"""
 
import xarray as xr
#import xesmf as xe
import numpy as np
from scipy.spatial import cKDTree
import datetime as dt
import config as cfg
 
# ---------------------------------------------------------------------------
# 2. LOAD
# ---------------------------------------------------------------------------
 
def load_wind():
    ds = xr.open_mfdataset(
        str(cfg.RAW_DIR / "era5_wind*.nc"),
        combine="by_coords",
    )
    # Newer CDS API downloads use "valid_time" instead of "time", and
    # "longitude"/"latitude" instead of "lon"/"lat". Handle both.
    rename_map = {}
    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        rename_map["valid_time"] = "time"
    if "longitude" in ds.coords:
        rename_map["longitude"] = "lon"
    if "latitude" in ds.coords:
        rename_map["latitude"] = "lat"
    ds = ds.rename(rename_map)
 
    # Some CDS downloads also add "number" (ensemble member) and "expver"
    # (experiment version) dims/coords even for single-realization
    # reanalysis requests. Drop them if present so they don't interfere
    # with merging/regridding later.
    drop_vars = [v for v in ("number", "expver") if v in ds.coords]
    if drop_vars:
        ds = ds.drop_vars(drop_vars)
 
    ds = ds.chunk({"time": 24})
    return ds
 
 
def load_waves():
    ds = xr.open_mfdataset(
        str(cfg.RAW_DIR / "ww3_northsea*.nc"),
        combine="by_coords",
        chunks={"time": 24},
    )
    # Rename here if CMEMS uses different dim names in your specific download
    if "longitude" in ds.coords:
        ds = ds.rename({"longitude": "lon", "latitude": "lat"})
    return ds
 
 
# ---------------------------------------------------------------------------
# 3. SUBSET TO DOMAIN + DATE RANGE (do this before heavy computation)
# ---------------------------------------------------------------------------
 
def subset(ds):
    sel_kwargs = {"time": cfg.DATE_RANGE}
    if ds.lat.values[0] < ds.lat.values[-1]:
        sel_kwargs["lat"] = slice(cfg.DOMAIN["lat"].stop, cfg.DOMAIN["lat"].start)
    else:
        sel_kwargs["lat"] = cfg.DOMAIN["lat"]
    sel_kwargs["lon"] = cfg.DOMAIN["lon"]
    return ds.sel(**sel_kwargs)
 
 
# ---------------------------------------------------------------------------
# 4. REGRID WIND ONTO THE WAVE GRID (reuse weights after first run)
# ---------------------------------------------------------------------------
 
def regrid_wind_to_wave_grid(ds_wind, ds_wave):
    # reuse = cfg.REGRID_WEIGHTS_FILE.exists()
    # regridder = xe.Regridder(
    #     ds_wind, ds_wave,
    #     method="bilinear",
    #     filename=str(cfg.REGRID_WEIGHTS_FILE),
    #     reuse_weights=reuse,
    # )
    ds_wind_regridded = ds_wind.interp_like(ds_wave, method="linear")
    return ds_wind_regridded
    #return regridder(ds_wind[cfg.WIND_VARS])
 
 
# ---------------------------------------------------------------------------
# 4. COARSEN THE WAVE GRID, THEN REGRID WIND ONTO IT
# ---------------------------------------------------------------------------
#
# The native CMEMS grid (~1.5 km) is about 147x finer than ERA5 (~28 km) —
# too large a mismatch to work with under tight memory constraints, and
# upsampling ERA5 onto the full native wave grid wastes memory without
# adding real information (ERA5 has no data at that resolution to begin
# with). Instead: coarsen the wave grid by a modest factor first (block-mean
# average), then interpolate ERA5 wind onto that coarser wave grid. This
# keeps meaningfully more coastal/shelf detail than ERA5's own resolution,
# while cutting the wave grid's memory footprint by roughly factor^2.

def coarsen_waves(ds_wave, factor: int = cfg.WAVE_COARSEN_FACTOR):
    return ds_wave.coarsen(lat=factor, lon=factor, boundary="trim").mean()

def regrid_wind_to_wave_grid(ds_wind, ds_wave_coarse):
    return ds_wind[cfg.WIND_VARS].interp_like(ds_wave_coarse, method="linear")

# ---------------------------------------------------------------------------
# 5. TEMPORAL ALIGNMENT — build forecast lead-time offset here if needed
# ---------------------------------------------------------------------------
 
def align_time(ds_wind, ds_wave, lead_time_hours: int = 0):
    """If lead_time_hours > 0, shift wave labels forward relative to wind
    features so the ML task becomes a genuine forecast, not a nowcast:
    wind at time t -> wave state at time t + lead_time_hours."""
    if lead_time_hours:
        ds_wave = ds_wave.assign_coords(
            time=ds_wave.time - np.timedelta64(lead_time_hours, "h")
        )
    common_times = np.intersect1d(ds_wind.time.values, ds_wave.time.values)
    return ds_wind.sel(time=common_times), ds_wave.sel(time=common_times)

# Change VMDR to a 2-component continuous representation
def wave_direction_to_sin_cos(ds_wave):
    theta_rad = np.deg2rad(ds_wave["VMDR"])
    ds_wave["VMDR_sin"] = np.sin(theta_rad)
    ds_wave["VMDR_cos"] = np.cos(theta_rad)
    ds_wave = ds_wave.drop_vars(["VMDR"])
    return ds_wave

# The pressure is given in an order of magnitude 100,000. Divide by 100,000 to stabilize training
def normalize_pressure(ds_wind):
    ds_wind["msl"] = ds_wind["msl"] / 100000
    return ds_wind
 
# ---------------------------------------------------------------------------
# 6. MERGE FEATURES + LABELS, SANITY CHECK, WRITE TO ZARR
# ---------------------------------------------------------------------------
 
def build_dataset(lead_time_hours: int = 0):
    ds_wind = load_wind()
    ds_wave = load_waves()
    
    print('data dimensions')
    print(ds_wind.dims)
    print(ds_wave.dims)

    ds_wind = subset(ds_wind)
    ds_wave = subset(ds_wave)

    ds_wind = normalize_pressure(ds_wind)

    ds_wave_coarse = coarsen_waves(ds_wave).astype("float32")
    ds_wind_regridded = regrid_wind_to_wave_grid(ds_wind, ds_wave_coarse).astype("float32")

    ds_wave_coarse = wave_direction_to_sin_cos(ds_wave_coarse)

    print('sanity check')
    print(f"features: {list(ds_wind_regridded.keys())}, labels: {list(ds_wave_coarse.keys())}")
    print("Wind time sample:", ds_wind_regridded.time.values[:3])
    print("Wave time sample:", ds_wave_coarse.time.values[:3])
 
    ds_wind_aligned, ds_wave_aligned = align_time(ds_wind_regridded, ds_wave_coarse, lead_time_hours)
 
    ds_combined = xr.merge([ds_wind_aligned, ds_wave_aligned[cfg.MODEL_OUTPUT_VARS]])
    ds_combined = ds_combined.chunk({"time": 24, "lat": -1, "lon": -1})
 
    ds_combined.to_zarr(str(cfg.OUTPUT_ZARR), mode="w")
    print(f"Saved aligned training data to {cfg.OUTPUT_ZARR}")
    return ds_combined
 
 
# ---------------------------------------------------------------------------
# 7. CHRONOLOGICAL TRAIN / VAL / TEST SPLIT (never shuffle randomly)
# ---------------------------------------------------------------------------

def chronological_split(ds, train_end, val_end):
    # train_end="2021-12-31", val_end="2022-12-31"
    train = ds.sel(time=slice(None, train_end))
    val = ds.sel(time=slice(train_end, val_end))
    test = ds.sel(time=slice(val_end, None))
    return train, val, test 


GRID_PATH = str(cfg.OUTPUT_ZARR)
BUOY_DIR = cfg.RAW_DIR / "buoys"  # one .nc per buoy, e.g. lauwers_oost.nc
OUTPUT_ZARR = str(cfg.PROCESSED_DIR / "northsea_training_data_nowcast_buoys.zarr")

BUOY_VARS = cfg.WAVE_VARS

def rename_buoy_variables(buoy_ds: xr.Dataset):
    buoy_ds.attrs["lat"] = buoy_ds.attrs.pop("Lat")
    buoy_ds.attrs["lon"] = buoy_ds.attrs.pop("Lon")
    
    if buoy_ds.attrs['Grootheid.Code'] == 'Hm0':
        buoy_ds.attrs['Grootheid.Code'] = 'VHM0'
    elif buoy_ds.attrs['Grootheid.Code'] == 'Tm-10':
        buoy_ds.attrs['Grootheid.Code'] = 'VTM10'
    elif buoy_ds.attrs['Grootheid.Code'] == 'Th0':
        buoy_ds.attrs['Grootheid.Code'] = 'VMDR'
    else:
        print('Fout in Grootheid.Code, code is:', buoy_ds.attrs['Grootheid.Code'])

    rename_map = {}
    if "Meetwaarde.Waarde_Numeriek" in buoy_ds.data_vars:
        rename_map["Meetwaarde.Waarde_Numeriek"] = buoy_ds.attrs['Grootheid.Code']
    buoy_ds = buoy_ds.rename(rename_map)

    return buoy_ds    
        
def resample_buoy_to_grid_time(buoy_ds: xr.Dataset, target_time: np.ndarray) -> xr.Dataset:
    """Resample a buoy's (likely finer-sampled) time series onto the
    gridded data's timestamps. Direction uses a circular mean via
    sin/cos decomposition; everything else uses a plain mean."""
    resampled = {}
    print('hallo', buoy_ds.attrs['Grootheid.Code'])
    print('buoy_data_vars', buoy_ds.data_vars)
    for var in cfg.WAVE_VARS:
        if var not in buoy_ds:
            resampled[var] = np.full(len(target_time), np.nan)
            continue

        if var == "VMDR":
            rad = np.deg2rad(buoy_ds[var])
            sin_mean = np.sin(rad).resample(time="3h").mean()
            cos_mean = np.cos(rad).resample(time="3h").mean()
            circular_mean = (np.rad2deg(np.arctan2(sin_mean, cos_mean)) % 360)
            resampled[var] = circular_mean.reindex(time=target_time).values
        else:
            grouped = buoy_ds[var].resample(time="3h").mean()
            resampled[var] = grouped.reindex(time=target_time).values

    return xr.Dataset(
        {var: (("time",), vals) for var, vals in resampled.items()},
        coords={"time": target_time},
    )

def match_to_sea_cell(ds_grid: xr.Dataset, buoy_lat: float, buoy_lon: float):
    """Nearest valid (non-land) grid cell — reuses the sea-mask-aware
    matching approach from earlier, since the naive nearest-cell lookup
    can land on a masked land cell near the coast."""
    sea_mask = ds_grid["VHM0"].isel(time=0).notnull().values
    lat2d, lon2d = np.meshgrid(ds_grid.lat.values, ds_grid.lon.values, indexing="ij")
    sea_points = np.column_stack([lat2d[sea_mask], lon2d[sea_mask]])

    tree = cKDTree(sea_points)
    _, idx = tree.query([buoy_lat, buoy_lon])
    matched_lat, matched_lon = sea_points[idx]
    return float(matched_lat), float(matched_lon)

def build_combined_dataset():
    ds_grid = xr.open_zarr(str(GRID_PATH), consolidated=False)

    buoy_ids, buoy_lats, buoy_lons = [], [], []
    matched_lats, matched_lons = [], []
    buoy_obs_by_var = {var: [] for var in BUOY_VARS}
    residual_by_var = {var: [] for var in BUOY_VARS}

    for nc_path in sorted(BUOY_DIR.glob("*.nc")):
        buoy_id = nc_path.stem
        buoy_ds = xr.open_dataset(nc_path)
        buoy_ds = rename_buoy_variables(buoy_ds)

        buoy_lat = float(buoy_ds.attrs["lat"])   # adjust to however lat/lon is actually stored in your files
        buoy_lon = float(buoy_ds.attrs["lon"])
        matched_lat, matched_lon = match_to_sea_cell(ds_grid, buoy_lat, buoy_lon)

        buoy_start, buoy_end = buoy_ds.time.values.min(), buoy_ds.time.values.max()
        grid_start, grid_end = ds_grid.time.values.min(), ds_grid.time.values.max()
        overlap_start = max(buoy_start, grid_start)
        overlap_end = min(buoy_end, grid_end)
        print(f'buoy {buoy_id}')
        print(f"{buoy_id}: buoy [{buoy_start}, {buoy_end}], grid [{grid_start}, {grid_end}], overlap: {(overlap_end - overlap_start).item() / (1000000000 * 3600 * 24)}")
        

        buoy_resampled = resample_buoy_to_grid_time(buoy_ds, ds_grid.time.values)
        grid_at_cell = ds_grid.sel(lat=matched_lat, lon=matched_lon, method="nearest")

        buoy_ids.append(buoy_id)
        buoy_lats.append(buoy_lat)
        buoy_lons.append(buoy_lon)
        matched_lats.append(matched_lat)
        matched_lons.append(matched_lon)

        for var in BUOY_VARS:
            buoy_obs_by_var[var].append(buoy_resampled[var].values)
            if var in grid_at_cell:
                residual_by_var[var].append(buoy_resampled[var].values - grid_at_cell[var].values)
            else:
                residual_by_var[var].append(np.full(len(ds_grid.time), np.nan))

    buoy_ds_combined = xr.Dataset(
        data_vars={
            **{f"buoy_{var}": (("buoy_id", "time"), np.array(buoy_obs_by_var[var])) for var in BUOY_VARS},
            **{f"residual_{var}": (("buoy_id", "time"), np.array(residual_by_var[var])) for var in BUOY_VARS},
            "buoy_lat": (("buoy_id",), np.array(buoy_lats)),
            "buoy_lon": (("buoy_id",), np.array(buoy_lons)),
            "matched_grid_lat": (("buoy_id",), np.array(matched_lats)),
            "matched_grid_lon": (("buoy_id",), np.array(matched_lons)),
        },
        coords={"buoy_id": buoy_ids, "time": ds_grid.time.values},
    )

    ds_combined = xr.merge([ds_grid, buoy_ds_combined])
    ds_combined.to_zarr(str(OUTPUT_ZARR), mode="w")
    print(f"Saved combined dataset to {OUTPUT_ZARR}")
    return ds_combined

 
# ---------------------------------------------------------------------------
# 8. CONVERT TO ML-READY ARRAYS (last step, right before feeding a model)
# ---------------------------------------------------------------------------
 
def to_arrays(ds):
    X = ds[cfg.WIND_VARS].to_array().transpose("time", "variable", "lat", "lon").values
    y = ds[cfg.MODEL_OUTPUT_VARS].to_array().transpose("time", "variable", "lat", "lon").values
    return X, y
 
 
if __name__ == "__main__":
    build_combined_dataset()

    # Example: build a 24-hour-ahead forecasting dataset
    """ds_combined = build_dataset(lead_time_hours=cfg.LEAD_TIME_HOURS)
    train_ds, val_ds, test_ds = chronological_split(ds_combined, cfg.TRAIN_END, cfg.VAL_END)
    X_train, y_train = to_arrays(train_ds)
    print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)"""
 