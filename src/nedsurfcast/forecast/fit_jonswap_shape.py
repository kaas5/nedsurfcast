"""
Fit the JONSWAP shape used to rebuild a spectrum from GFS partitions
=====================================================================

The GFS spectrum used as model input is rebuilt from the GFS partitions (hs, Tp, dir) with a JONSWAP shape
per partition (common.partition.synthesize_spectrum). Classic JONSWAP (gamma 3.3, f^-5 tail) is a young
wind-sea shape; GFS's spectra are broader, especially at long periods.

Here gamma and the tail exponent n depend on the partition's peak period (common.partition.shape_parameters):
the "lo" values below T_LO seconds, the "hi" values above T_HI, linear in between. The four values are
fitted so that the rebuilt spectrum is as close as possible to GFS's OWN point spectrum at K13 and EURO
(fcfg.GFSWAVE_SPEC_STATIONS), with the GFS partitions of the nearest grid cell as input. Only GFS data is
used, no buoy data, so the fit does not leak buoy information into the evaluation at K133/EPL3.

Loss: RMSE of sqrt(E) per bin (the energy part of the spectrum model's loss). Cross-station check: fit on one
station, score on the other. The final fit on both stations is what goes into fcfg.GFS_SPECTRUM_SHAPE.

Run from anywhere with: python -m nedsurfcast.forecast.fit_spectrum_shape
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from nedsurfcast.common.partition import jonswap, shape_parameters
from nedsurfcast.forecast.preprocess import load

T_LO, T_HI = 8.0, 14.0                       # s; gamma and tail interpolate between these peak periods
BOUNDS = [(1.0, 20.0), (1.0, 20.0), (3.0, 15.0), (3.0, 15.0)]   # gamma_lo, gamma_hi, n_lo, n_hi
START = [3.3, 3.3, 5.0, 5.0]                 # classic JONSWAP
BANDS = {"< 6 s": (1 / 6, np.inf), "6-10 s": (0.1, 1 / 6), "> 10 s": (0, 0.1)}


def load_cases() -> dict:
    """GFS partitions at the nearest cell and the GFS point spectrum, for every run, lead and point station."""
    point = load("gfs_station_spectra").load()
    stations = load("stations").load()
    grid = load("grid")
    parts, energy, station = [], [], []
    for k, st in enumerate(point.station.values):
        cell = grid.sel(lat=float(stations["nearest_lat"].sel(station=st)),
                        lon=float(stations["nearest_lon"].sel(station=st))).load()
        p = np.stack([np.concatenate([cell[w].values[..., None], np.moveaxis(cell[s].values, 2, -1)], -1)
                      for w, s in (("WVHGT", "SWELL"), ("WVPER", "SWPER"), ("WVDIR", "SWDIR"))], -1)
        p = p.reshape(-1, 4, 3)                                          # (time * lead, partition, hs/tp/dir)
        e = point["gfs_spec_energy"].values[:, :, k].reshape(-1, point.sizes["freq"])
        ok = np.isfinite(e).all(1) & (np.nan_to_num(p[:, :, 0]) > 0).any(1)
        parts.append(p[ok]), energy.append(e[ok]), station.append(np.full(ok.sum(), st))
    return {"parts": np.concatenate(parts), "energy": np.concatenate(energy).astype(float),
            "station": np.concatenate(station), "freq": point["freq"].values}


def make_shape(x) -> dict:
    return dict(t_lo=T_LO, t_hi=T_HI, gamma=(float(x[0]), float(x[1])), tail=(float(x[2]), float(x[3])))


def rebuild(cases: dict, rows: np.ndarray, shape: dict | None) -> np.ndarray:
    """All present partitions of the selected cases at once -> rebuilt 1D spectra (n_rows, n_freq)."""
    p = cases["parts"][rows]
    i, j = np.nonzero(np.nan_to_num(p[:, :, 0]) > 0)
    hs, fp = p[i, j, 0], 1 / p[i, j, 1]
    gamma, n = shape_parameters(fp, shape)
    e_k = jonswap(cases["freq"], hs, fp, gamma, n)
    out = np.zeros((len(p), len(cases["freq"])))
    np.add.at(out, i, e_k)
    return out


def rmse_sqrt(e_model, e_ref) -> float:
    return float(np.sqrt(np.mean((np.sqrt(e_model) - np.sqrt(e_ref)) ** 2)))


def band_scores(e_model, e_ref, freq) -> dict:
    df = np.gradient(freq)
    out = {}
    for name, (lo, hi) in BANDS.items():
        sel = (freq >= lo) & (freq < hi)
        h_m, h_r = 4 * np.sqrt(e_model[:, sel] @ df[sel]), 4 * np.sqrt(e_ref[:, sel] @ df[sel])
        out[f"Hm0 {name} rmse"] = float(np.sqrt(np.mean((h_m - h_r) ** 2)))
        out[f"Hm0 {name} bias"] = float(np.mean(h_m - h_r))
    return out


def fit(cases: dict, rows: np.ndarray, max_rows: int = 6000, seed: int = 0) -> dict:
    """Minimise the sqrt(E) RMSE on (a random subset of) the rows; Nelder-Mead in the bounded box."""
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(rows)
    if len(idx) > max_rows:
        idx = rng.choice(idx, max_rows, replace=False)
    sub = np.zeros(len(rows), bool)
    sub[idx] = True
    ref = cases["energy"][sub]
    lo, hi = np.array(BOUNDS).T

    def loss(x):
        return rmse_sqrt(rebuild(cases, sub, make_shape(np.clip(x, lo, hi))), ref)

    res = minimize(loss, START, method="Nelder-Mead", options=dict(maxiter=400, xatol=0.02, fatol=1e-6))
    return make_shape(np.clip(res.x, lo, hi))


def score(cases: dict, rows: np.ndarray, shape: dict | None) -> dict:
    e = rebuild(cases, rows, shape)
    ref = cases["energy"][rows]
    return {"rmse sqrt(E)": rmse_sqrt(e, ref)} | band_scores(e, ref, cases["freq"])


def describe(shape: dict) -> str:
    return (f"gamma {shape['gamma'][0]:.2f} -> {shape['gamma'][1]:.2f}, "
            f"tail f^-{shape['tail'][0]:.2f} -> f^-{shape['tail'][1]:.2f} (Tp {T_LO:g} -> {T_HI:g} s)")


def run():
    cases = load_cases()
    stations = sorted(set(cases["station"]))
    print(f"{len(cases['energy'])} GFS point spectra with GFS partitions at the nearest cell: "
          f"{ {s: int((cases['station'] == s).sum()) for s in stations} }\n")

    rows_out = []
    for test in stations:                                   # cross-station: fit on the other station
        train = cases["station"] != test
        shape = fit(cases, train)
        print(f"fitted on {stations[1 - stations.index(test)]}: {describe(shape)}")
        is_test = cases["station"] == test
        rows_out.append({"test station": test, "shape": "classic JONSWAP"} | score(cases, is_test, None))
        rows_out.append({"test station": test, "shape": "fitted on other station"} | score(cases, is_test, shape))

    final = fit(cases, np.ones(len(cases["energy"]), bool))
    print(f"fitted on both:  {describe(final)}\n")
    table = pd.DataFrame(rows_out).set_index(["test station", "shape"])
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print("Rebuilt spectrum vs GFS point spectrum (lower is better; bias = rebuilt - point):")
    print(table.round(3).to_string())
    print(f"\nFor forecast/config.py:\nGFS_SPECTRUM_SHAPE = dict(t_lo={T_LO:g}, t_hi={T_HI:g}, "
          f"gamma=({final['gamma'][0]:.2f}, {final['gamma'][1]:.2f}), "
          f"tail=({final['tail'][0]:.2f}, {final['tail'][1]:.2f}))")
    return final, table


if __name__ == "__main__":
    run()
