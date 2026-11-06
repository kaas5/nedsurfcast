"""
Forecast configuration: archived NOAA GFS-Wave forecasts + RWS buoys
======================================================================
"""

from nedsurfcast import config as cfg

FORECAST_DIR = cfg.PROCESSED_DIR / "forecast"
FORECAST_DIR.mkdir(parents=True, exist_ok=True)

""" Archived NOAA GFS-Wave (WAVEWATCH III) forecasts, s3://noaa-gfs-bdp-pds (available from 2021-03) """
GFSWAVE_URL = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
GFSWAVE_GRID = "global.0p25"     # global.0p16 stops at 52.5N, atlocn.0p16 is the western Atlantic only
GFSWAVE_CYCLES = [0]             # issue hours per day (0, 6, 12, 18 available)
GFSWAVE_LEAD_HOURS = list(range(0, 73, 3))
# GRIB messages to keep, as "<VAR>:<level>" in the .idx files; each message is the full global field
GFSWAVE_FIELDS = [
    "HTSGW:surface",             # significant wave height (m)
    "PERPW:surface",             # peak period (s)
    "DIRPW:surface",             # peak direction (deg)
    "UGRD:surface",              # 10 m wind forcing (m/s)
    "VGRD:surface",
    "SWELL:1 in sequence",       # primary swell partition: height, period, direction
    "SWPER:1 in sequence",
    "SWDIR:1 in sequence",
]
GFSWAVE_DIR = cfg.RAW_DIR / "gfswave"
