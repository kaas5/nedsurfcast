import xarray as xr
#import xesmf as xe
import numpy as np
import matplotlib.pyplot as plt
import glob 
import os

import ddlpy

import config as cfg

def load_buoy():
    ds = xr.open_mfdataset(
        str(cfg.RAW_DIR / "dutch buoys/ijgeul.2-7368.nc"),
        #combine="by_coords",
        chunks={"time": 24},
    )
    # Rename here if CMEMS uses different dim names in your specific download
    #if "longitude" in ds.coords:
    #    ds = ds.rename({"longitude": "lon", "latitude": "lat"})
    return ds

def preprocess_steps():
    ds = load_buoy()
    #print(ds)

    #print('hallo 1', ds.time)

    # eerst direction naar sin/cos


    #ds = ds.resample(time='1h').mean()
    #print('hallo 2', ds.time)

    #print(type(ds))
    print(list(ds.keys()))
    #print(ds.coords)
    print(np.unique(ds["Grootheid.Code"].values))
    print(ds['Meetwaarde.Waarde_Numeriek'].values)
        
def data_bekijken():
    locations = ddlpy.locations()
    bool_stations = locations.index.isin(['ijmuiden.munitiestort.3', 'ijgeul.2', 'europlatform'])
    bool_grootheid = locations["Grootheid.Code"].isin(["Hm0", "Tm01", "Th0"]) # golfhoogte, golfperiode, golfrichting
    bool_groepering = locations['Groepering.Code'].isin(['']) # timeseries ("") versus extremes (GETETM2/GETETMSL2/GETETBRKD2/GETETBRKDMSL2)
    #bool_hoedanigheid = locations['Hoedanigheid.Code'].isin(['NAP']) # vertical reference (NAP/MSL)
    selected = locations.loc[
        bool_stations &
        bool_grootheid &
        bool_groepering
    ]

    print(selected['Grootheid.Code'].unique())

    #print(locations['Grootheid.Code'].unique())

    #print(locations.columns.tolist())  # confirm the exact description column name in your version

    #wave_dir_candidates = locations[locations["Grootheid.Code"].isin(['T', 'T1/3', 'TE' ,'TE3' ,'TELFAFA' ,'TELFBTA', 'TELFH3' ,'Th0', 'T_H1/3', 'Th3', 'T_Hmax', 'Tm01' ,'Tm02', 'Tm-10' ,'Tmax', 'Hmax'])]
    #print(wave_dir_candidates[["Grootheid.Code", "Grootheid.Omschrijving"]].drop_duplicates())

    #selected.to_json(r'locations.json', orient='records')

    ## Create a spatial plot
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.plot(selected.Lon, selected.Lat, 'k.')
    plt.show()


if __name__ == '__main__':
    #preprocess_steps()
    #data_bekijken()

    file_list = glob.glob(os.path.join(str(cfg.RAW_DIR / "dutch buoys/*.nc")))
    print(file_list)
    for file_nc in file_list:
        ds = xr.open_dataset(file_nc)
        station_code = ds.attrs["Code"]
        station_naam = ds.attrs["Naam"]
        print(f'{station_naam}-{station_code} Grootheid.Code: {np.unique(ds["Grootheid.Code"].values)}')
        #ds["Meetwaarde.Waarde_Numeriek"].plot(ax=ax, label=f"{station_code} ({station_naam})")

