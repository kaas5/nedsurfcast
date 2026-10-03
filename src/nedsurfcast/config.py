from pathlib import Path
import torch
import datetime as dt
 
# ---------------------------------------------------------------------------
# 0. CONFIG — adjust to your setup
# ---------------------------------------------------------------------------

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# North Sea bounding box (lon_min, lon_max, lat_min, lat_max)
DOMAIN = dict(lon=slice(-4, 9), lat=slice(62, 51))  # note: lat often descending in ERA5

START_DATE = dt.datetime(2024, 1, 1)
END_DATE = dt.datetime(2024, 2, 1)
DATE_RANGE = slice(START_DATE, END_DATE)

TRAIN_END="2024-11-01"
VAL_END="2024-12-01"

WIND_VARS = ["u10", "v10", "msl"]       # ERA5: 10m wind components + mean sea level pressure
WAVE_VARS = ["VHM0", "VTM10", "VMDR"]    # CMEMS WW3: sig. wave height, peak period, mean direction
RWS_WAVE_VARS = ["Hm0", "Tm-10", "Th0"]   # golfhoogte (cm), golfperiode (s, spectraal moment, nadruk op swell), golfrichting (graad)

MODEL_OUTPUT_VARS = ["VHM0", "VTM10", "VMDR_sin", "VMDR_cos"]
LEAD_TIME_HOURS = 0

OUTPUT_ZARR = PROCESSED_DIR / "northsea_training_data_nowcast.zarr"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE = "cpu"

WAVE_COARSEN_FACTOR = 3  # e.g. 3 -> ~4.5 km cells, ~9x less memory than native 1.5 km

""" Model training parameters """
HISTORY_LEN = 4          # number of past wind timesteps stacked as input channels
BEST_MODEL_NAME = 'best_model_nowcast.pt'
