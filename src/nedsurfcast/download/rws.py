"""
Download RWS buoy measurements with ddlpy (shared by nowcast and forecast)
===========================================================================

Run from anywhere with: python -m nedsurfcast.download.rws
"""

import ddlpy
import os
from concurrent.futures import ProcessPoolExecutor
import json

# enabling debug logging so we can see what happens in the background
import logging
logging.basicConfig()
logging.getLogger("ddlpy").setLevel(logging.DEBUG)

from nedsurfcast import config as cfg


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
    dir_output = str(cfg.BUOY_DIR)
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
    download_buoys()
