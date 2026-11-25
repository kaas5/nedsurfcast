"""
Final PyTorch models on all data, saved for later use
=======================================================

Trains every PyTorch model of models.MODELS (mlp, mlp-hlf, cnn, cnn-hlf; not ridge) once on all spectra: every
station (including fcfg.TRAIN_ONLY_STATIONS) and every month. Early stopping validates on time blocks of these
same rows (as in the evaluation). How good these models are on unseen stations and months is what
train_vs_buoys.py / train_vs_gfs_point.py measure; nothing is scored here.

Saved by forecast/checkpoints.py in fcfg.CHECKPOINT_DIR as <name>_<timestamp>.pt (ensemble members
<name>_<timestamp>_<n>.pt); all models of one run share the timestamp. Load with checkpoints.load_model(name).

Run from anywhere with: python -m nedsurfcast.forecast.train_final [--models NAME ...]
"""

from datetime import datetime

import numpy as np

from nedsurfcast.forecast.checkpoints import TIMESTAMP_FORMAT, is_torch_model, save_model
from nedsurfcast.forecast.load_data import features, load_data, targets
from nedsurfcast.forecast.models import MODELS, fit_model

TORCH_MODELS = {k: v for k, v in MODELS.items() if is_torch_model(v[0]())}


def run(models: dict = TORCH_MODELS):
    data = load_data()
    m = data["meta"]
    Y = targets(data)
    times = m["time"].to_numpy()
    trained_on = {"rows": int(len(m)), "stations": sorted(map(str, m["station"].unique())),
                  "time_min": str(times.min()), "time_max": str(times.max())}
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    print(f"{len(m)} spectra at {len(trained_on['stations'])} stations, {trained_on['time_min'][:10]} to "
          f"{trained_on['time_max'][:10]}; models: {list(models)}; timestamp {timestamp}")
    for name, (make, fs) in models.items():
        model = make()
        if not is_torch_model(model):
            print(f"  {name}: not a PyTorch model, skipped")
            continue
        fit_model(model, features(data, fs), Y, times, data["freq"])
        paths = save_model(model, name, fs, timestamp, Y, data["freq"], trained_on)
        print(f"  {name}: saved {', '.join(p.name for p in paths)}", flush=True)
    return timestamp


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Train the PyTorch models on all data and save them")
    ap.add_argument("--models", nargs="+", choices=list(TORCH_MODELS), help="subset of the models (default all)")
    args = ap.parse_args()
    run({k: v for k, v in TORCH_MODELS.items() if args.models is None or k in args.models})
