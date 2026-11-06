import ddlpy
import datetime as dt
import os
from concurrent.futures import ProcessPoolExecutor
import json
import numpy as np
import xarray as xr

# enabling debug logging so we can see what happens in the background
import logging
logging.basicConfig()
logging.getLogger("ddlpy").setLevel(logging.DEBUG)

import config as cfg

# ---------------------------------------------------------------------------
# 1. DOWNLOAD STUBS (run once, separately, not part of the main pipeline call)
# ---------------------------------------------------------------------------

def cds_time_selections(start: dt.datetime = cfg.START_DATE, end: dt.datetime = cfg.END_DATE,
                        step_hours: int = 3) -> list[dict]:
    """Year/month/day/time selections for the CDS API covering [start, end], one per calendar month.

    CDS returns the cross product of the year, month, day and time lists, so a single request for e.g.
    2024-11-15 .. 2025-01-10 would also fetch all of Jan-Feb 2024 and Nov-Dec 2025. Splitting per month
    keeps every request exact at day level (ECMWF also recommends monthly ERA5 requests). The first and
    last day are fetched for all hours; subset() trims to DATE_RANGE afterwards.
    """
    times = [f"{h:02d}:00" for h in range(0, 24, step_hours)]
    selections = []
    month_start = dt.datetime(start.year, start.month, 1)
    while month_start <= end:
        next_month = (month_start + dt.timedelta(days=32)).replace(day=1)
        first_day = max(start, month_start).day
        last_day = min(end, next_month - dt.timedelta(days=1)).day
        selections.append({
            "year": [str(month_start.year)],
            "month": [f"{month_start.month:02d}"],
            "day": [f"{d:02d}" for d in range(first_day, last_day + 1)],
            "time": times,
        })
        month_start = next_month
    return selections

def download_era5_wind_example(overwrite: bool = False):
    """Example CDS API call for ERA5 10m wind + pressure, one file per month
    (era5_wind_YYYY_MM.nc). Requires a configured ~/.cdsapirc with your API key."""
    import cdsapi

    c = cdsapi.Client(progress=True)
    for selection in cds_time_selections():
        filename = cfg.RAW_DIR / f"era5_wind_{selection['year'][0]}_{selection['month'][0]}.nc"
        if filename.exists() and not overwrite:
            print(f"{filename.name}: already exists and overwrite=False, skipping")
            continue

        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure"],
                **selection,
                "area": [62, -4, 51, 9],  # N, W, S, E
                "format": "netcdf",
                "download_format": "unarchived",
            },
            str(filename),
        )
 
def _gfswave_url(issue: dt.datetime, lead: int) -> str:
    return (f"{cfg.GFSWAVE_URL}/gfs.{issue:%Y%m%d}/{issue:%H}/wave/gridded/"
            f"gfswave.t{issue:%H}z.{cfg.GFSWAVE_GRID}.f{lead:03d}.grib2")


def _gfswave_byte_ranges(idx_text: str, fields: list[str]) -> list[tuple[int, int | None]]:
    """(start, end) byte ranges of the wanted GRIB messages, read from the .idx file.

    Lines look like "5:2894821:d=2024010100:HTSGW:surface:24 hour fcst:"; a message ends where the
    next one starts, the last one runs to the end of the file (end=None).
    """
    lines = [line.split(":") for line in idx_text.strip().splitlines()]
    offsets = [int(parts[1]) for parts in lines]
    ranges = []
    for i, parts in enumerate(lines):
        if f"{parts[3]}:{parts[4]}" in fields:
            end = offsets[i + 1] - 1 if i + 1 < len(offsets) else None
            ranges.append((offsets[i], end))
    return ranges


def _download_gfswave_lead(session, issue: dt.datetime, lead: int, tmp_dir) -> xr.Dataset | None:
    """Fetch only the wanted fields of one forecast step and crop them to cfg.DOMAIN."""
    url = _gfswave_url(issue, lead)
    idx = session.get(url + ".idx", timeout=60)
    if idx.status_code == 404:
        return None
    idx.raise_for_status()

    chunks = []
    for start, end in _gfswave_byte_ranges(idx.text, cfg.GFSWAVE_FIELDS):
        r = session.get(url, headers={"Range": f"bytes={start}-{'' if end is None else end}"}, timeout=120)
        r.raise_for_status()
        chunks.append(r.content)

    grib_path = tmp_dir / f"{issue:%Y%m%d%H}_f{lead:03d}.grib2"
    grib_path.write_bytes(b"".join(chunks))
    try:
        ds = xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs={"indexpath": ""}, decode_timedelta=True)
        # GRIB longitudes run 0..360; the North Sea domain crosses 0
        ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180).sortby("longitude")
        ds = ds.rename({"latitude": "lat", "longitude": "lon"}).sel(**cfg.DOMAIN).load()
    finally:
        grib_path.unlink()
    return ds.drop_vars([v for v in ("surface",) if v in ds.coords])


def download_gfswave_cycle(issue: dt.datetime, overwrite: bool = False):
    """One forecast run (all cfg.GFSWAVE_LEAD_HOURS) -> gfswave_YYYYMMDD_HHz.nc with dims (step, lat, lon)."""
    import requests

    filename = cfg.GFSWAVE_DIR / f"gfswave_{issue:%Y%m%d}_{issue:%H}z.nc"
    if filename.exists() and not overwrite:
        print(f"{filename.name}: already exists and overwrite=False, skipping")
        return
    tmp_dir = cfg.GFSWAVE_DIR / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    steps = []
    with requests.Session() as session:
        for lead in cfg.GFSWAVE_LEAD_HOURS:
            ds = _download_gfswave_lead(session, issue, lead, tmp_dir)
            if ds is None:
                print(f"{issue:%Y-%m-%d %H}z f{lead:03d}: not in archive")
                continue
            steps.append(ds)
    if not steps:
        print(f"{issue:%Y-%m-%d %H}z: no data, skipping")
        return

    # time = issue time, lead_hours = lead time, valid_time = time + lead
    ds = xr.concat(steps, dim="step")
    lead_hours = (ds.step.values / np.timedelta64(1, "h")).astype(int)
    ds = ds.drop_vars("step").assign_coords(step=lead_hours).rename(step="lead_hours")  # cfgrib's step attrs break reading back
    ds.to_netcdf(filename)
    print(f"{filename.name}: {len(steps)} lead times written")


def download_gfswave(overwrite: bool = False, max_workers: int = 4):
    """All forecast runs from cfg.START_DATE to cfg.END_DATE at cfg.GFSWAVE_CYCLES, a few runs in parallel."""
    from concurrent.futures import ThreadPoolExecutor

    cfg.GFSWAVE_DIR.mkdir(parents=True, exist_ok=True)
    issues = []
    day = cfg.START_DATE
    while day < cfg.END_DATE:
        issues += [day.replace(hour=h) for h in cfg.GFSWAVE_CYCLES]
        day += dt.timedelta(days=1)

    # downloading is I/O bound, so threads are enough; list() re-raises errors from the workers
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(lambda issue: download_gfswave_cycle(issue, overwrite), issues))


def download_cmems_wave_example():
    """Example Copernicus Marine Toolbox call for the North West Shelf
    WW3 reanalysis product. Requires `copernicusmarine login` once."""
    import copernicusmarine
    copernicusmarine.login(username='', password='', force_overwrite=True)
 
    copernicusmarine.subset(
        dataset_id="MetO-NWS-WAV-RAN",  # example North West Shelf WW3 hindcast dataset id — verify current id on CMEMS catalog
        variables=cfg.WAVE_VARS,
        minimum_longitude=-4, maximum_longitude=9,
        minimum_latitude=51, maximum_latitude=62,
        start_datetime=cfg.START_DATE.strftime('%Y-%m-%dT%H:%M:%S'), #"2024-01-01T00:00:00",
        end_datetime=cfg.END_DATE.strftime('%Y-%m-%dT%H:%M:%S'), #"2024-12-31T23:00:00",
        output_filename="ww3_northsea_raw.nc",
        output_directory=str(cfg.RAW_DIR),
    )

def get_buoy_data(location, start_date, end_date, dir_output, overwrite=True):
    station_id = location.name
    station_messageid = location["Locatie_MessageID"]
    grootheid = location["Grootheid.Code"].replace("/", "_")  # codes like "H1/3" would break the path
    filename = os.path.join(dir_output, f"{station_id}-{station_messageid}-.-{grootheid}.nc")

    if os.path.isfile(filename) and overwrite is False:
        print(f"{station_id}: netcdf file already exists and overwrite=False, skipping")
        return

    measurements = ddlpy.measurements(location, start_date=start_date, end_date=end_date)

    if measurements.empty:
        print(f"{station_id}: no measurements found")
        return

    print(f"{station_id}: writing retrieved data to netcdf file")

    # convert to xarray: constant columns are converted to attributes to save disk space
    # except the columns in always_preserve
    always_preserve = [
        "Grootheid.Code",
        "Meetwaarde.Waarde_Numeriek",
    ]
    ds = ddlpy.dataframe_to_xarray(measurements, always_preserve=always_preserve)

    ds.to_netcdf(filename)

def download_buoys():
    dir_output = str(cfg.RAW_DIR / 'buoys')
    os.makedirs(dir_output, exist_ok=True)

    # get locations
    locations = ddlpy.locations()
    bool_grootheid = locations["Grootheid.Code"].isin(cfg.RWS_WAVE_VARS)
    bool_groepering = locations['Groepering.Code'].isin(['']) # timeseries ("") versus extremes (GETETM2/GETETMSL2/GETETBRKD2/GETETBRKDMSL2)
    selected = locations.loc[
        #bool_stations &
        bool_grootheid &
        bool_groepering
    ]

    # normal code
    # for station_code, location in selected.iterrows():
    #     get_data(location, start_date, end_date, dir_output)

    # parallel code
    with ProcessPoolExecutor(max_workers=3) as executor:
        for station_code, location in selected.iterrows():
            executor.submit(get_buoy_data, location, cfg.START_DATE, cfg.END_DATE, dir_output)

def debug_station_grootheid():
    locations = ddlpy.locations()
    hoi = {}
    for grootheid in cfg.RWS_WAVE_VARS:
        bool_grootheid = locations["Grootheid.Code"].isin([grootheid])
        bool_groepering = locations['Groepering.Code'].isin(['']) # timeseries ("") versus extremes (GETETM2/GETETMSL2/GETETBRKD2/GETETBRKDMSL2)
        selected = locations.loc[
            #bool_stations &
            bool_grootheid &
            bool_groepering
        ]

        print(f'GROOTHEID: {grootheid}')
        for station_code, location in selected.iterrows():
            station_id = location.name
            station_messageid = location["Locatie_MessageID"]
            station = f'{station_id}-{station_messageid}'
            if station in list(hoi.keys()):
                hoi[station].append(grootheid)
            else:
                hoi[station] = [grootheid]

    obj = json.dumps(hoi, indent=4)
    print(obj)

    return

if __name__=='__main__':
    download_era5_wind_example()
    download_cmems_wave_example()
    download_buoys()
    #download_gfswave()