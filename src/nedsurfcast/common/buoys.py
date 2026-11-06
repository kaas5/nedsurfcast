"""
RWS buoy loading, quality control and matching to a model grid
================================================================

Shared by the nowcast and forecast pipelines. Buoy variables are renamed to
the CMEMS names (cfg.WAVE_VARS), so every pipeline compares like with like.
"""

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

from nedsurfcast import config as cfg


# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------

def list_stations() -> dict[str, list[str]]:
    """Station id -> its downloaded grootheid files, skipping cfg.BUOY_IGNORE_LIST."""
    stations = {}
    for nc_path in sorted(cfg.BUOY_DIR.glob("*.nc")):
        buoy_id = nc_path.stem.split('-.-')[0]
        buoy_name = buoy_id.split('-')[0]
        if buoy_name in cfg.BUOY_IGNORE_LIST:
            print(f'ignoring {buoy_name}')
            continue
        #print(f'adding {buoy_name}')

        buoy_grootheid = nc_path.stem.split('-.-')[1]
        if buoy_id in list(stations.keys()):
            stations[buoy_id].append(buoy_grootheid)
        else:
            stations[buoy_id] = [buoy_grootheid]
    return stations


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

def combine_station_grootheid(station_name: str, station_grootheid_list: list):
    parts = []
    for grootheid in station_grootheid_list:
        ds = xr.open_dataset(str(cfg.BUOY_DIR / f'{station_name}-.-{grootheid}.nc'))
        code = str(ds["Grootheid.Code"].values.flat[0])  # preserved via always_preserve; check how it's stored in your files
        parts.append(ds["Meetwaarde.Waarde_Numeriek"].rename(cfg.RWS_TO_WAVE_VARS[code]).to_dataset())
    return xr.merge(parts)  # outer join on time, NaN where a code has no value at that time

def _attrs_equal(a, b) -> bool:
    try:
        return bool(np.array_equal(a, b))
    except Exception:
        return False

QUALITY_CODE = "WaarnemingMetadata.Kwaliteitswaardecode"

# RWS marks missing/unreliable measurements with quality code 99 (value 99999).
# ddlpy stores the code as an attribute instead of a variable when it is constant in a file.
def mask_bad_quality(da: xr.DataArray, ds: xr.Dataset) -> xr.DataArray:
    if QUALITY_CODE in ds.data_vars:
        return da.where(ds[QUALITY_CODE].isin(cfg.RWS_VALID_QUALITY_CODES))
    if str(ds.attrs.get(QUALITY_CODE)) in cfg.RWS_VALID_QUALITY_CODES:
        return da
    return da.where(False)

def load_station(station_id: str, station_grootheid_list: list) -> xr.Dataset:
    files = []
    for grootheid in station_grootheid_list:
        files.append(str(cfg.BUOY_DIR / f'{station_id}-.-{grootheid}.nc'))

    datasets = [xr.open_dataset(f) for f in sorted(files)]

    # Station-level attrs = keys present with identical values in every file
    first = datasets[0].attrs
    station_attrs = {
        k: v for k, v in first.items()
        if all(k in d.attrs and _attrs_equal(d.attrs[k], v) for d in datasets[1:])
    }

    parts = []
    for ds in datasets:
        code = str(ds["Grootheid.Code"].values.flat[0])  # check how this is stored in your files
        if code not in cfg.RWS_TO_WAVE_VARS:
            continue
        da = ds["Meetwaarde.Waarde_Numeriek"].rename(cfg.RWS_TO_WAVE_VARS[code])
        da = mask_bad_quality(da, ds)
        da = da.drop_duplicates("time")
        # Parameter-level metadata (units, description, ...) stays on the variable
        da.attrs = {k: v for k, v in ds.attrs.items() if k not in station_attrs}
        parts.append(da.to_dataset())

    merged = xr.merge(parts, combine_attrs="drop")  # outer join on time; NaN where a code has no value
    merged.attrs = station_attrs                    # station metadata lives on the Dataset
    merged.attrs["station_id"] = station_id
    return merged


# ---------------------------------------------------------------------------
# 2. CLEAN
# ---------------------------------------------------------------------------

def waveheight_cm_to_m(ds):
    if 'VHM0' in ds.data_vars:
        ds['VHM0'] = ds['VHM0'] / 100
        print('VHM0 in data_vars')
        return ds
    print('VHM0 NIET in data_vars')
    return ds

# Fill values and spikes that passed the quality code check, e.g. 19.4 m at borsselealpha or 9999.9 s periods
def mask_out_of_range(ds):
    for var, (lo, hi) in cfg.VALID_RANGE.items():
        if var in ds.data_vars:
            ds[var] = ds[var].where((ds[var] >= lo) & (ds[var] <= hi))
    return ds


# ---------------------------------------------------------------------------
# 3. ALIGN WITH A MODEL GRID
# ---------------------------------------------------------------------------

# Grid values are instantaneous, so average the buoy over [t-90min, t+90min) and label it t.
# offset alone only moves the bin edges; the labels stay at the bin start, hence the shift.
def resample_centered(da: xr.DataArray) -> xr.DataArray:
    out = da.resample(time="3h", offset="-90min").mean()
    return out.assign_coords(time=out.time + np.timedelta64(90, "m"))

def resample_buoy_to_grid_time(buoy_ds: xr.Dataset, target_time: np.ndarray) -> xr.Dataset:
    """Resample a buoy's (likely finer-sampled) time series onto the
    gridded data's timestamps. Direction uses a circular mean via
    sin/cos decomposition; everything else uses a plain mean."""
    resampled = {}
    #print('hallo', buoy_ds.attrs['Grootheid.Code'])
    print('buoy_data_vars', buoy_ds.data_vars)
    for var in cfg.WAVE_VARS:
        if var not in buoy_ds:
            resampled[var] = np.full(len(target_time), np.nan)
            continue

        if var == "VMDR":
            rad = np.deg2rad(buoy_ds[var])
            sin_mean = resample_centered(np.sin(rad))
            cos_mean = resample_centered(np.cos(rad))
            circular_mean = (np.rad2deg(np.arctan2(sin_mean, cos_mean)) % 360)
            resampled[var] = circular_mean.reindex(time=target_time).values
        else:
            grouped = resample_centered(buoy_ds[var])
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
