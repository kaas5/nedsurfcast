"""
Show interesting out-of-sample spectra from the spectrum correction model
===========================================================================

Reads the out-of-sample predictions saved by forecast/train_spectrum.py (oos_spectra.zarr) and
plots the buoy, GFS and corrected spectrum for a set of interesting cases:

    largest swell        highest measured swell energy (Tp > 6 s)
    best correction      model swell Hm0 much closer to the buoy than GFS
    worst correction     model swell Hm0 much further from the buoy than GFS
    GFS missed swell     buoy swell much higher than GFS forecast
    GFS false swell      GFS forecast much more swell than measured
    random               for an unbiased impression

Each panel: energy per frequency (left axis) and direction per frequency (right axis, dots, only where
there is energy), the swell Hm0 of the three spectra and their wave systems after partitioning.
Cases of one category are at least MIN_GAP_DAYS apart in valid time, so they show different sea states.

Run from anywhere with:
    python -m nedsurfcast.forecast.plot_spectra
    python -m nedsurfcast.forecast.plot_spectra --scheme blocked-time --model ridge --station K133 --save spectra.png
    python -m nedsurfcast.forecast.plot_spectra --station EPL3 --time 2026-01-19T00 --lead 24
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from nedsurfcast.common.partition import partition_spectrum
from nedsurfcast.forecast import config as fcfg

SWELL_F_MAX = 1 / 6             # Hz; Tp > 6 s counts as swell
MIN_SWELL_HM0 = 0.3             # m; only cases where buoy or GFS has at least this much swell
MIN_GAP_DAYS = 2                # cases of one category are at least this far apart in valid time
DIR_MIN_ENERGY = 0.02           # m2/Hz; directions are drawn only where the spectrum has this much energy
COLORS = {"buoy": "k", "GFS": "tab:grey", "corrected": "tab:blue"}


# ---------------------------------------------------------------------------
# 1. DATA
# ---------------------------------------------------------------------------

def load_predictions(scheme: str, model: str, station: str | None = None) -> dict:
    ds = xr.open_zarr(str(fcfg.SPECTRUM_MODEL_DIR / "oos_spectra.zarr"))
    if scheme not in ds.scheme.values or model not in ds.model.values:
        raise ValueError(f"choose scheme from {list(ds.scheme.values)} and model from {list(ds.model.values)}")
    keep = np.ones(ds.sizes["spectrum"], bool) if station is None else (ds["station"].values == station)
    if not keep.any():
        raise ValueError(f"no spectra for station {station}; stations: {sorted(set(ds['station'].values))}")
    sel = lambda v: ds[v].values[keep]
    return {
        "freq": ds["freq"].values,
        "buoy": (sel("buoy_energy"), sel("buoy_direction")),
        "GFS": (sel("gfs_energy"), sel("gfs_direction")),
        "corrected": (ds["corrected_energy"].sel(scheme=scheme, model=model).values[keep],
                      ds["corrected_direction"].sel(scheme=scheme, model=model).values[keep]),
        "meta": pd.DataFrame({v: sel(v) for v in ("time", "station", "lead_hours", "valid_time")}),
        "label": f"{model}, {scheme}",
    }


def swell_hm0(energy: np.ndarray, f: np.ndarray) -> np.ndarray:
    df = float(np.median(np.diff(f)))
    return 4 * np.sqrt(energy[..., f < SWELL_F_MAX].sum(-1) * df)


# ---------------------------------------------------------------------------
# 2. PICK CASES
# ---------------------------------------------------------------------------

def _spaced(order: np.ndarray, times: np.ndarray, k: int) -> list[int]:
    """First k indices from order whose valid times are at least MIN_GAP_DAYS apart."""
    chosen, gap = [], np.timedelta64(MIN_GAP_DAYS, "D")
    for i in order:
        if all(abs(times[i] - times[j]) >= gap for j in chosen):
            chosen.append(int(i))
        if len(chosen) == k:
            break
    return chosen


def pick_cases(p: dict, per_category: int = 2, seed: int = 0) -> dict[str, int]:
    f = p["freq"]
    hb, hg, hc = (swell_hm0(p[k][0], f) for k in ("buoy", "GFS", "corrected"))
    times = p["meta"]["valid_time"].values
    relevant = (np.maximum(hb, hg) >= MIN_SWELL_HM0)
    gain = np.abs(hg - hb) - np.abs(hc - hb)                  # > 0: correction moved swell Hm0 towards the buoy

    def top(score):
        score = np.where(relevant, score, -np.inf)
        return _spaced(np.argsort(-score), times, per_category)

    cases = {}
    for name, score in (("largest swell", hb), ("best correction", gain), ("worst correction", -gain),
                        ("GFS missed swell", hb - hg), ("GFS false swell", hg - hb)):
        for n, i in enumerate(top(score)):
            cases[f"{name} #{n + 1}"] = i
    rng = np.random.default_rng(seed)
    for n, i in enumerate(rng.choice(np.flatnonzero(relevant), per_category, replace=False)):
        cases[f"random #{n + 1}"] = int(i)
    return cases


def find_case(p: dict, time: str, lead: int) -> dict[str, int]:
    m = p["meta"]
    hit = np.flatnonzero((m["time"] == pd.Timestamp(time)).to_numpy() & (m["lead_hours"] == lead).to_numpy())
    if len(hit) == 0:
        raise ValueError(f"no spectrum for run {time} at lead {lead} h (with the chosen station)")
    return {f"run {pd.Timestamp(time):%Y-%m-%d %H}z +{lead} h, {m['station'].iloc[i]}": int(i) for i in hit}


# ---------------------------------------------------------------------------
# 3. PLOT
# ---------------------------------------------------------------------------

def _systems_text(f, energy, direction) -> str:
    parts = partition_spectrum(f, energy, direction, **fcfg.BUOY_PARTITION_KWARGS)
    return "  ".join(f"{q['hs']:.2f}m {q['tp']:.1f}s {q['dir']:.0f}°" for q in parts[:3]) or "-"


def plot_cases(p: dict, cases: dict[str, int], ncols: int = 3):
    f = p["freq"]
    n = len(cases)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 5.8 * nrows), squeeze=False)
    for ax, (title, i) in zip(axes.flat, cases.items()):
        lines = []
        ax_dir = ax.twinx()
        for key, style in (("buoy", dict(drawstyle="steps-mid", lw=1.4)), ("GFS", dict(ls="--", lw=1.2)),
                           ("corrected", dict(lw=1.6))):
            e, d = p[key][0][i], p[key][1][i]
            ax.plot(f, e, color=COLORS[key], label=key, **style)
            show = e >= DIR_MIN_ENERGY
            ax_dir.scatter(f[show], d[show], s=7, color=COLORS[key], alpha=0.6, marker="o" if key == "buoy" else "x")
            lines.append(f"{key:<9} swell {swell_hm0(e, f):.2f} m | {_systems_text(f, e, d)}")
        ax.axvline(SWELL_F_MAX, color="tab:orange", lw=0.8, ls=":")
        ax.set_xlim(0.03, 0.4)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("E (m²/Hz)")
        ax_dir.set_ylim(0, 360)
        ax_dir.set_yticks([0, 90, 180, 270, 360])
        ax_dir.tick_params(labelsize=7)
        ax_dir.set_ylabel("direction (deg)", fontsize=8, color="tab:grey")
        m = p["meta"].iloc[i]
        ax.set_title(f"{title}\n{m['station']}  run {pd.Timestamp(m['time']):%Y-%m-%d}  +{m['lead_hours']} h  "
                     f"(valid {pd.Timestamp(m['valid_time']):%m-%d %H}h)", fontsize=9)
        # anchored just below the x-axis label (x in axes fraction, y relative to the label), so it never
        # overlaps the axis or the spectra, whatever the figure size or screen resolution
        ax.annotate("\n".join(lines), xy=(0.0, 0.0), xycoords=(ax.transAxes, ax.xaxis.label),
                    xytext=(0, -6), textcoords="offset points", ha="left", va="top", fontsize=7,
                    family="monospace")
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    axes.flat[0].legend(loc="center right", fontsize=8)
    fig.suptitle(f"Out-of-sample spectra ({p['label']}): lines = energy, dots/crosses = direction (right axis), "
                 f"dotted line = 6 s; systems: Hs Tp direction", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96), pad=1.5, h_pad=4.0, w_pad=3.5)
    return fig


def summary(p: dict, cases: dict[str, int]) -> pd.DataFrame:
    f = p["freq"]
    rows = []
    for title, i in cases.items():
        m = p["meta"].iloc[i]
        hb, hg, hc = (float(swell_hm0(p[k][0][i], f)) for k in ("buoy", "GFS", "corrected"))
        rows.append(dict(case=title, station=m["station"], run=pd.Timestamp(m["time"]), lead=m["lead_hours"],
                         swell_buoy=hb, swell_gfs=hg, swell_corrected=hc))
    return pd.DataFrame(rows).set_index("case").round(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scheme", default="leave-one-group", help="leave-one-group or blocked-time")
    ap.add_argument("--model", default="mlp", help="mean-bias, ridge or mlp")
    ap.add_argument("--station", help="only this station, e.g. K133")
    ap.add_argument("--time", help="issue time of one run, e.g. 2026-01-19T00 (with --lead)")
    ap.add_argument("--lead", type=int, default=0, help="lead time in hours for --time")
    ap.add_argument("--per-category", type=int, default=2, help="cases per category")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", metavar="FILE", help="save the figure instead of showing it")
    a = ap.parse_args()

    p = load_predictions(a.scheme, a.model, a.station)
    cases = find_case(p, a.time, a.lead) if a.time else pick_cases(p, a.per_category, a.seed)
    pd.set_option("display.width", 200)
    print(summary(p, cases).to_string())
    fig = plot_cases(p, cases)
    if a.save:
        fig.savefig(a.save, dpi=110)
        print(f"Saved {a.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
