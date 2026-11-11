"""
Shared configuration for the nowcast and forecast pipelines
=============================================================

Pipeline-specific settings live in nowcast/config.py and forecast/config.py.
"""

from pathlib import Path
import datetime as dt

# ---------------------------------------------------------------------------
# 0. PATHS — absolute, so scripts work from any working directory
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
BUOY_DIR = RAW_DIR / "buoys"

# North Sea bounding box (lon_min, lon_max, lat_min, lat_max)
DOMAIN = dict(lon=slice(-4, 9), lat=slice(62, 51))  # note: lat often descending in ERA5

# Period of the measured RWS wave spectra (the RWsOS API keeps about one year; values start 2025-10-31).
# The nowcast data (ERA5, CMEMS, buoys in data/raw) was downloaded for 2024-01-01 .. 2025-01-01.
START_DATE = dt.datetime(2025, 11, 1)
END_DATE = dt.datetime(2026, 11, 11)
DATE_RANGE = slice(START_DATE, END_DATE)

DUTCH_COAST = dict(lat=(51.2, 53.9), lon=(2.5, 7.3))  # (min, max) box for gridded residual predictions

# ---------------------------------------------------------------------------
# 1. RWS BUOYS
# ---------------------------------------------------------------------------

WAVE_VARS = ["VHM0", "VTM10", "VMDR"]    # sig. wave height, period, mean direction (CMEMS names, also used for the buoys)
RWS_WAVE_VARS = ["Hm0", "Tm-10", "Th0"]   # golfhoogte (cm), golfperiode (s, spectraal moment, m-10 nadruk op swell), golfrichting (graad)
RWS_TO_WAVE_VARS = {"Hm0": "VHM0", "Tm-10": "VTM10", "Th0": "VMDR"}

BUOY_IGNORE_LIST = ['honte.kaloot',
                    'pasvanterneuzen',
                    'vaarwaterlangshoofdplaat',
                    'vaarwaterlangshoofdplaat.4',
                    'valkenisse.overloop.nabijboei65a',
                    'marollegat',
                    'keeten',
                    'oosterschelde.4',
                    'hansweert.2',
                    'hansweert.2overloop',
                    'hansweert',
                    'bath.2',
                    'rilland.bath',
                    'markermeer.midden',
                    'markermeer.trintelzand',
                    'rotterdamsehoek.meetpaalfl2',
                    'ameland.nes',
                    'denhoorntexel.waddenzee.gatvandestier',
                    'denhelder.waddenzee.mondingnieuwehaven',
                    'oudewestereems.zuid',
                    'uithuizerwad',
                    'eemshaven.waddenzee',
                    # sheltered behind structures / in estuaries, not representable on the WW3 grid
                    'hoekvanholland.maeslantkering.beneden.noord',
                    'haringvlietsluizen.buiten',
                    'oudewestereems.noord.boei',
                    'stortemelk.oost',
                    'stortemelk.west']

RWS_VALID_QUALITY_CODES = ['00']  # 00 = normal value, 99 = missing/unreliable
# Physically plausible (min, max) per buoy variable; values outside are measurement errors or fill values
VALID_RANGE = {
    "VHM0": (0.0, 20.0),    # m
    "VTM10": (1.0, 25.0),   # s
    "VMDR": (0.0, 360.0),   # degrees
}

BUOY_GROUP_DISTANCE_KM = 15.0     # buoys whose matched cells are closer than this are held out together
