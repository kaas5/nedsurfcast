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
    "UGRD:surface",              # 10 m wind forcing (m/s); WIND/WDIR in the file are the same wind as speed/direction
    "VGRD:surface",
    "WVHGT:surface",             # wind sea partition: height, period, direction
    "WVPER:surface",
    "WVDIR:surface",
    # swell partitions, "1 in sequence" is the most energetic; the order swaps between time steps
    *[f"{var}:{n} in sequence" for n in (1, 2, 3) for var in ("SWELL", "SWPER", "SWDIR")],
]
GFSWAVE_DIR = cfg.RAW_DIR / "gfswave"

# cfgrib variable name -> NCEP name used in GFSWAVE_FIELDS; the preprocessed dataset uses the NCEP names
GFSWAVE_VAR_NAMES = {
    "swh": "HTSGW", "perpw": "PERPW", "dirpw": "DIRPW", "u": "UGRD", "v": "VGRD",
    "shts": "SWELL", "mpts": "SWPER", "swdir": "SWDIR",
    "shww": "WVHGT", "mpww": "WVPER", "wvdir": "WVDIR",
}

""" Measured RWS wave spectra from the RWsOS viewer API (https://rwsos.rws.nl/viewer/) """
# The API only keeps about one year of spectra: older events exist but carry null values
# (checked 2026-11: values start 2025-10-31 for every station). Download regularly to build an archive.
RWS_SPECTRA_URL = "https://rwsos.rws.nl/wb-api/dd/2.0"
RWS_SPECTRA_SOURCE = "S_1"                    # "meting" (measurement)
RWS_SPECTRA_SERIES = {                        # API observationTypeId -> variable name, 51 frequency bins each
    "GD.28-78": "energy_density",             # cm2 s (converted to m2/Hz on download)
    "GR.1-51": "direction",                   # mean direction per frequency (deg, coming from)
    "GR.52-102": "spread",                    # directional spread per frequency (deg)
}
RWS_SPECTRA_NBINS = 51                        # bin i covers [10 i, 10 i + 10) mHz
RWS_SPECTRA_DF = 0.01                         # Hz
RWS_SPECTRA_STATIONS = [                      # open sea and open coast; estuaries/sheltered buoys left out
    "K133",                                   # K13 Alpha (53.22N 3.22E), also a GFS-Wave station
    # EPL/EPL3, A12/A122 and MUNS/MUN1 publish identical spectra where both exist (copies, not separate
    # sensors): use one code per pair. SCHB/SCB2 are two different buoys at the same position.
    "EPL3",                                   # Euro platform (51.99N 3.28E), also a GFS-Wave station
    "A122",                                   # A12 platform (55.42N 3.82E); hardly any data after 2026-02
    "E131",                                   # Maasgeul E13 buoy
    "MUN1",                                   # Noordzee munitiestortplaats
    "SCHB", "SCB2",                           # Schouwenbank directional waveriders
    "HKZWN", "HKZWZ",                         # Hollandse Kust Zuid wind farm
    "HKNWN", "HKNWZ",                         # Hollandse Kust Noord wind farm
    "HKWWN", "HKWWZ",                         # Hollandse Kust West wind farm
    "IJG1",                                   # IJgeul IJ11 buoy
    "MMND",                                   # Maasmond
    "SGAT",                                   # Schulpengat
]
RWS_SPECTRA_DIR = cfg.RAW_DIR / "rws_spectra"

# Buoy spectrum partitioning (common/partition.py), tuned with forecast/check_partitions.py
BUOY_SPECTRA_WINDOW = "60min"                 # spectra averaged over [t - window, t + window] before partitioning
BUOY_PARTITION_KWARGS = dict(                 # passed to partition_spectrum
    f_range=(0.03, 0.5),                      # Hz; below 0.03 the buoy data is unreliable
    trough_ratio=0.6,
    min_df=0.02,
    max_dir_diff=60.0,
    min_hs=0.1,
)

""" Partition dataset: GFS wave systems matched to buoy wave systems (forecast/preprocess.py) """
MERGE_GFS = True                              # merge GFS partitions a 1D buoy cannot resolve (common.partition.merge_unresolvable)
MATCH_KWARGS = dict(                          # common.partition.match_partitions; use the same for training and evaluation
    max_dist=2.0,                             # normalised distance gate
    sigma_f=0.02,                             # Hz per unit distance
    sigma_dir=30.0,                           # degrees per unit distance
)
PARTITION_ZARR = FORECAST_DIR / "gfswave_partitions.zarr"
