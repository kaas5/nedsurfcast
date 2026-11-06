"""
Nowcast configuration: ERA5 + CMEMS WW3 hindcast + RWS buoys
==============================================================
"""

import torch

from nedsurfcast import config as cfg

ERA5_DIR = cfg.RAW_DIR / "era5"
CMEMS_DIR = cfg.RAW_DIR / "cmems"
NOWCAST_DIR = cfg.PROCESSED_DIR / "nowcast"
NOWCAST_DIR.mkdir(parents=True, exist_ok=True)

GRID_ZARR = NOWCAST_DIR / "northsea_training_data_nowcast.zarr"
GRID_BUOYS_ZARR = NOWCAST_DIR / "northsea_training_data_buoys_nowcast.zarr"

TRAIN_END="2024-11-01"
VAL_END="2024-12-01"
LEAD_TIME_HOURS = 0

WIND_VARS = ["u10", "v10", "msl"]       # ERA5: 10m wind components + mean sea level pressure
MODEL_OUTPUT_VARS = ["VHM0", "VTM10", "VMDR_sin", "VMDR_cos"]  # CNN targets; sin/cos are derived from VMDR (degrees) on load

WAVE_COARSEN_FACTOR = 3  # e.g. 3 -> ~4.5 km cells, ~9x less memory than native 1.5 km

""" Model training parameters """
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE = "cpu"
HISTORY_LEN = 4          # number of past wind timesteps stacked as input channels
BEST_MODEL_NAME = 'best_model_nowcast.pt'
