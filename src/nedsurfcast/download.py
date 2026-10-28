import ddlpy
import datetime as dt
import os
from concurrent.futures import ProcessPoolExecutor
import json

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
    #download_cmems_wave_example()
    #download_buoys()