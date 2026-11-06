"""
Download the CMEMS North West Shelf WW3 hindcast (nowcast pipeline)
=====================================================================

Run from anywhere with: python -m nedsurfcast.download.cmems
"""

from nedsurfcast import config as cfg
from nedsurfcast.nowcast import config as ncfg


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
        output_directory=str(ncfg.CMEMS_DIR),
    )


if __name__ == '__main__':
    download_cmems_wave_example()
