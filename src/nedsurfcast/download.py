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
            "year": [str(y) for y in range(2024, 2025)],
            "month": [f"{m:02d}" for m in range(1, 13)],
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
    copernicusmarine.login(username='kaas5', password='Deens8356!')
 
    copernicusmarine.subset(
        dataset_id="MetO-NWS-WAV-RAN",  # example North West Shelf WW3 hindcast dataset id — verify current id on CMEMS catalog
        variables=cfg.WAVE_VARS,
        minimum_longitude=-4, maximum_longitude=9,
        minimum_latitude=51, maximum_latitude=62,
        start_datetime="2024-01-01T00:00:00",
        end_datetime="2024-12-31T23:00:00",
        output_filename="ww3_northsea_raw.nc",
        output_directory=str(cfg.RAW_DIR),
    )

if __name__=='__main__':
    #download_era5_wind_example()
    download_cmems_wave_example()