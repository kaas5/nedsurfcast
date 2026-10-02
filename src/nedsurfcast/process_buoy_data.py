import xarray as xr
#import xesmf as xe
import numpy as np
import matplotlib.pyplot as plt

import ddlpy

import config as cfg

def load_buoy():
    ds = xr.open_mfdataset(
        str(cfg.RAW_DIR / "dutch buoys/ameland.nes-6200.nc"),
        #combine="by_coords",
        chunks={"time": 24},
    )
    # Rename here if CMEMS uses different dim names in your specific download
    #if "longitude" in ds.coords:
    #    ds = ds.rename({"longitude": "lon", "latitude": "lat"})
    return ds

if __name__ == '__main__':
    """ds = load_buoy()
    print(type(ds))
    print(list(ds.keys()))
    print(ds['Meetwaarde.Waarde_Numeriek'].values)""" 


    locations = ddlpy.locations()
    #locations.to_json(r'locations.json', orient='records')
    #selected = locations.loc['ameland.nes']

    bool_grootheid = locations['Grootheid.Code'].isin(['Hm0']) # waterlevel (WATHTE)
    bool_groepering = locations['Groepering.Code'].isin(['']) # timeseries ("") versus extremes (GETETM2/GETETMSL2/GETETBRKD2/GETETBRKDMSL2)
    #bool_hoedanigheid = locations['Hoedanigheid.Code'].isin(['NAP']) # vertical reference (NAP/MSL)
    selected = locations.loc[bool_grootheid & bool_groepering]
    #selected

    print(selected.shape)

    ## Create a spatial plot
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.plot(selected.Lon, selected.Lat, 'k.')
    plt.show()

    print(type(locations))
    print(locations['Grootheid.Code'].nunique())
    #print(selected)
