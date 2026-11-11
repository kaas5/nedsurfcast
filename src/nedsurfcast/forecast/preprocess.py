"""
Forecast residual dataset: archived GFS-Wave forecasts + RWS buoys
====================================================================

Following Pinto et al. (South Atlantic), the target is the error of the
archived wave forecast at the buoys:

    residual = buoy Hm0 at valid time - GFS HTSGW at the buoy's matched grid cell

One sample is one forecast run (issue time) at one lead time. Every sample
keeps the full GFS grid of the fields in fcfg.GFSWAVE_FIELDS, so the same
dataset serves a point-wise model (fields at the matched cells) and a U-Net
(whole grid in, residual at the buoy cells as sparse target).

Zarr layout:
    HTSGW, PERPW, ...         (time, lead_hours, lat, lon)   GFS forecast fields
    valid_time                (time, lead_hours)             issue time + lead
    buoy_VHM0                 (time, lead_hours, buoy_id)    observed Hm0 at valid_time (m)
    gfs_HTSGW_at_buoy         (time, lead_hours, buoy_id)    GFS at the matched cell
    residual_HTSGW            (time, lead_hours, buoy_id)    buoy - GFS
    buoy_lat/lon, matched_grid_lat/lon, match_distance_km   (buoy_id)
    sea_mask                  (lat, lon)

`time` is the ISSUE time of the run, not the valid time: the same valid time
appears in several runs at different lead times.

The grid stays at native GFS resolution (0.25 deg); any regridding for a model
belongs in the model's data loading, so this dataset stays a faithful copy.

Run from anywhere with: python -m nedsurfcast.forecast.preprocess
"""

import numpy as np
import pandas as pd
import xarray as xr

from nedsurfcast import config as cfg
from nedsurfcast.common.buoys import (list_stations, load_station, mask_out_of_range, match_to_sea_cell,
                                      resample_centered, waveheight_cm_to_m)
from nedsurfcast.forecast import config as fcfg

BUOY_TO_RWS = {v: k for k, v in cfg.RWS_TO_WAVE_VARS.items()}   # "VHM0" -> "Hm0"


# ---------------------------------------------------------------------------
# 1. GFS FORECAST RUNS
# ---------------------------------------------------------------------------

def gfs_field_names() -> list[str]:
    # "HTSGW:surface" -> "HTSGW"; "SWELL:1/2/3 in sequence" -> one "SWELL" with a swell_partition dimension
    return list(dict.fromkeys(field.split(":")[0] for field in fcfg.GFSWAVE_FIELDS))


def load_gfs_runs() -> xr.Dataset:
    """All downloaded runs (gfswave_YYYYMMDD_HHz.nc) stacked along issue time -> (time, lead_hours, lat, lon)."""
    files = sorted(fcfg.GFSWAVE_DIR.glob("gfswave_[0-9]*_[0-9][0-9]z.nc"))
    if not files:
        raise FileNotFoundError(f"No GFS-Wave runs in {fcfg.GFSWAVE_DIR}; run python -m nedsurfcast.download.gfs")

    runs = [xr.open_dataset(f, chunks={}) for f in files]
    ds = xr.concat(runs, dim="time", join="outer")   # outer: a run missing some lead times gets NaN there
    ds = ds.drop_vars([c for c in ("orderedSequenceData",) if c in ds.coords])
    ds = ds.rename({k: v for k, v in fcfg.GFSWAVE_VAR_NAMES.items() if k in ds.data_vars})
    ds = ds.sortby("time").sel(time=cfg.DATE_RANGE)

    missing = [v for v in gfs_field_names() if v not in ds.data_vars]
    if missing:
        raise ValueError(f"{missing} are in GFSWAVE_FIELDS but not in the downloaded runs; download again with overwrite=True")
    return ds[gfs_field_names()]


# ---------------------------------------------------------------------------
# 2. BUOY OBSERVATIONS ON THE VALID TIMES
# ---------------------------------------------------------------------------

def load_buoy_observations(valid_times: np.ndarray, var: str = fcfg.TARGET_BUOY_VAR) -> xr.Dataset:
    """Cleaned buoy series averaged over [t-90min, t+90min) at every valid time -> (valid_time, buoy_id)."""
    rws_code = BUOY_TO_RWS[var]
    obs, ids, lats, lons = [], [], [], []
    for station, grootheden in list_stations().items():
        if rws_code not in grootheden:
            continue
        buoy_ds = load_station(station, [rws_code])
        if var == "VHM0":
            buoy_ds = waveheight_cm_to_m(buoy_ds)
        buoy_ds = mask_out_of_range(buoy_ds)

        series = resample_centered(buoy_ds[var]).reindex(time=valid_times)
        if series.notnull().sum() == 0:
            print(f"{station}: no {var} observations during the forecast period, skipping")
            continue
        obs.append(series.values)
        ids.append(station)
        lats.append(float(buoy_ds.attrs["Lat"]))
        lons.append(float(buoy_ds.attrs["Lon"]))

    return xr.Dataset(
        {
            f"buoy_{var}": (("valid", "buoy_id"), np.column_stack(obs)),
            "buoy_lat": ("buoy_id", np.array(lats)),
            "buoy_lon": ("buoy_id", np.array(lons)),
        },
        coords={"valid": valid_times, "buoy_id": ids},
    )


def _distance_km(lat1, lon1, lat2, lon2):
    # Equirectangular, accurate enough over the North Sea
    dx = (np.asarray(lon2) - lon1) * 111.32 * np.cos(np.deg2rad(lat1))
    dy = (np.asarray(lat2) - lat1) * 110.57
    return np.hypot(dx, dy)


# ---------------------------------------------------------------------------
# 3. RESIDUAL DATASET
# ---------------------------------------------------------------------------

def build_residual_dataset() -> xr.Dataset:
    gfs = load_gfs_runs()
    print(f"GFS: {gfs.sizes['time']} runs x {gfs.sizes['lead_hours']} lead times on a "
          f"{gfs.sizes['lat']}x{gfs.sizes['lon']} grid, fields {list(gfs.data_vars)}")

    valid_times = np.unique(gfs["valid_time"].values)
    buoys = load_buoy_observations(valid_times)

    # Nearest GFS sea cell: at 0.25 deg most coastal buoys sit in (or next to) a land cell
    target = gfs[fcfg.TARGET_GFS_VAR]
    sea_mask = target.isel(time=0, lead_hours=0).notnull().compute()
    matched = [match_to_sea_cell(gfs, lat, lon, var=fcfg.TARGET_GFS_VAR)
               for lat, lon in zip(buoys["buoy_lat"].values, buoys["buoy_lon"].values)]
    cell_lat = xr.DataArray([m[0] for m in matched], dims="buoy_id", coords={"buoy_id": buoys.buoy_id})
    cell_lon = xr.DataArray([m[1] for m in matched], dims="buoy_id", coords={"buoy_id": buoys.buoy_id})

    # Observation at each sample's valid time: (valid, buoy_id) indexed with (time, lead_hours)
    obs = buoys[f"buoy_{fcfg.TARGET_BUOY_VAR}"].sel(valid=gfs["valid_time"]).drop_vars("valid")
    gfs_at_buoy = target.sel(lat=cell_lat, lon=cell_lon, method="nearest").drop_vars(["lat", "lon"])

    residuals = xr.Dataset(
        {
            f"buoy_{fcfg.TARGET_BUOY_VAR}": obs,
            f"gfs_{fcfg.TARGET_GFS_VAR}_at_buoy": gfs_at_buoy,
            f"residual_{fcfg.TARGET_GFS_VAR}": obs - gfs_at_buoy,
            "buoy_lat": buoys["buoy_lat"],
            "buoy_lon": buoys["buoy_lon"],
            "matched_grid_lat": cell_lat,
            "matched_grid_lon": cell_lon,
            "match_distance_km": ("buoy_id", _distance_km(buoys["buoy_lat"].values, buoys["buoy_lon"].values,
                                                          cell_lat.values, cell_lon.values)),
            "sea_mask": sea_mask.drop_vars([c for c in sea_mask.coords if c not in ("lat", "lon")]),
        }
    )
    ds = xr.merge([gfs, residuals])
    ds.attrs = {
        "description": "Archived NOAA GFS-Wave forecasts with RWS buoy residuals (buoy - GFS) at the matched cells",
        "residual": f"buoy_{fcfg.TARGET_BUOY_VAR} - gfs_{fcfg.TARGET_GFS_VAR}_at_buoy",
        "time": "issue time of the forecast run; valid_time = time + lead_hours",
    }

    # netCDF encodings (chunksizes, zlib, ...) are not valid for Zarr
    for v in ds.variables:
        ds[v].encoding = {}
    ds = ds.chunk({"time": 8, **{d: -1 for d in ds.dims if d != "time"}})
    ds.to_zarr(str(fcfg.RESIDUAL_ZARR), mode="w")
    print(f"Saved residual dataset to {fcfg.RESIDUAL_ZARR}")
    return ds


def summarize(ds: xr.Dataset):
    res = ds[f"residual_{fcfg.TARGET_GFS_VAR}"]
    n = res.notnull().sum(["time", "lead_hours"]).compute()
    per_buoy = pd.DataFrame({
        "n_residuals": n.values,
        "match_km": ds["match_distance_km"].values.round(1),
        "bias_m": res.mean(["time", "lead_hours"]).compute().values.round(3),
        "rmse_m": np.sqrt((res ** 2).mean(["time", "lead_hours"])).compute().values.round(3),
    }, index=ds.buoy_id.values)
    print(per_buoy.sort_values("match_km").to_string())

    by_lead = np.sqrt((res ** 2).mean(["time", "buoy_id"])).compute()
    print("\nRMSE of GFS against the buoys per lead time (m):")
    print(by_lead.to_series().round(3).to_string())


if __name__ == "__main__":
    ds = build_residual_dataset()
    summarize(ds)
