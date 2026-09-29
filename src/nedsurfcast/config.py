from pathlib import Path
import torch
 
# ---------------------------------------------------------------------------
# 0. CONFIG — adjust to your setup
# ---------------------------------------------------------------------------

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# North Sea bounding box (lon_min, lon_max, lat_min, lat_max)
DOMAIN = dict(lon=slice(-4, 9), lat=slice(62, 51))  # note: lat often descending in ERA5
 
DATE_RANGE = slice("2024-01-01", "2024-12-31")

TRAIN_END="2024-11-01"
VAL_END="2024-12-01"

WIND_VARS = ["u10", "v10", "msl"]       # ERA5: 10m wind components + mean sea level pressure
WAVE_VARS = ["VHM0", "VTPK", "VMDR"]    # CMEMS WW3: sig. wave height, peak period, mean direction

MODEL_OUTPUT_VARS = ["VHM0", "VTPK", "VMDR_sin", "VMDR_cos"]
LEAD_TIME_HOURS = 0
 
REGRID_WEIGHTS_FILE = PROCESSED_DIR / "wind_to_wave_weights.nc"
OUTPUT_ZARR = PROCESSED_DIR / "northsea_training_data_nowcast.zarr"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE = "cpu"

WAVE_COARSEN_FACTOR = 3  # e.g. 3 -> ~4.5 km cells, ~9x less memory than native 1.5 km

""" Model training parameters """
HISTORY_LEN = 4          # number of past wind timesteps stacked as input channels
BEST_MODEL_NAME = 'best_model_nowcast.pt'
