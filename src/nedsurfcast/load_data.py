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
 
# ---------------------------------------------------------------------------
# 8. CONVERT TO ML-READY ARRAYS (last step, right before feeding a model)
# ---------------------------------------------------------------------------
 
def to_arrays(ds):
    X = ds[cfg.WIND_VARS].to_array().transpose("time", "variable", "lat", "lon").values
    y = ds[cfg.MODEL_OUTPUT_VARS].to_array().transpose("time", "variable", "lat", "lon").values
    return X, y
 
 
if __name__ == "__main__":
    # Example: build a 24-hour-ahead forecasting dataset
    ds_combined = build_dataset(lead_time_hours=cfg.LEAD_TIME_HOURS)
    train_ds, val_ds, test_ds = chronological_split(ds_combined, cfg.TRAIN_END, cfg.VAL_END)
    X_train, y_train = to_arrays(train_ds)
    print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
 