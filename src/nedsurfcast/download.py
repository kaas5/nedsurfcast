import ddlpy
import datetime as dt
import os
from concurrent.futures import ProcessPoolExecutor

# enabling debug logging so we can see what happens in the background
import logging
logging.basicConfig()
logging.getLogger("ddlpy").setLevel(logging.DEBUG)

import config as cfg

# ---------------------------------------------------------------------------
# 1. DOWNLOAD STUBS (run once, separately, not part of the main pipeline call)
# ---------------------------------------------------------------------------

def download_era5_wind_example():
    """Example CDS API call for ERA5 10m wind + pressure. Requires a
    configured ~/.cdsapirc with your API key."""
    import cdsapi

    c = cdsapi.Client(progress=True)
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure"],
            "year": [str(y) for y in range(cfg.START_DATE.year, cfg.END_DATE.year + 1 if cfg.START_DATE.year == cfg.END_DATE.year else cfg.END_DATE.year)],
            "month": [f"{m:02d}" for m in range(cfg.START_DATE.month, cfg.END_DATE.month)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(0, 24, 3)],
            "area": [62, -4, 51, 9],  # N, W, S, E
            "format": "netcdf",
        },
        str(cfg.RAW_DIR / "era5_wind_raw.nc"),
    )
 
def download_cmems_wave_example():
    """Example Copernicus Marine Toolbox call for the North West Shelf
    WW3 reanalysis product. Requires `copernicusmarine login` once."""
    import copernicusmarine
    copernicusmarine.login(username='', password='')
 
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
    filename = os.path.join(dir_output, f"{station_id}-{station_messageid}.nc")

    if os.path.isfile(filename) and overwrite is False:
        print("{station_id}: netcdf file already exists and overwrite=False, skipping")
        return

    measurements = ddlpy.measurements(location, start_date=start_date, end_date=end_date)

    if measurements.empty:
        print(f"{station_id}: no measurements found")
        return

    print(f"{station_id}: writing retrieved data to netcdf file")

    # convert to xarray: constant columns are converted to attributes to save disk space
    # except the columns in always_preserve
    always_preserve = [
        #"Grootheid.Code",
        "Meetwaarde.Waarde_Numeriek",
    ]
    ds = ddlpy.dataframe_to_xarray(measurements, always_preserve=always_preserve)

    ds.to_netcdf(filename)

def download_buoys():
    dir_output = str(cfg.RAW_DIR / 'buoys')
    os.makedirs(dir_output, exist_ok=True)

    # get locations
    locations = ddlpy.locations()
    #bool_stations = locations.index.isin(['ijmuiden.munitiestort.3', 'ijgeul.2', 'europlatform'])
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

if __name__=='__main__':
    #download_era5_wind_example()
    #download_cmems_wave_example()
    download_buoys()