"""
Spectra dominated by long-period swell: GFS point spectrum vs corrected vs measured
====================================================================================

Long-period swell (Tp > 10 s) is where the corrected spectrum still loses to GFS's own forecast spectrum
(forecast/compare_gfs_spectra.py). This script shows such cases at the stations with a GFS point spectrum
(K133, EPL3):

    measured         the buoy spectrum (the truth)
    GFS raw          GFS's own point spectrum: the raw forecast to beat
    corrected        ridge on GFS input only (no issue-time measurement), out-of-sample: trained with the
                     station's group held out, as in compare_gfs_spectra.py
    GFS rebuilt      thin dotted: the spectrum rebuilt from the GFS partitions, which is what ridge corrects

A spectrum is "dominated by long-period swell" when at least LONG_FRACTION of its energy (m0) lies at
Tp > 10 s and that band holds at least MIN_LONG_HM0, in the measured spectrum or in GFS's point spectrum
(so GFS swell that did not arrive is included). Only lead times >= EVAL_MIN_LEAD. Cases:

    largest long swell   highest measured Hm0 at Tp > 10 s
    worst correction     corrected Hm0 (Tp > 10 s) much further from the buoy than GFS's point spectrum
    best correction      corrected Hm0 (Tp > 10 s) much closer to the buoy than GFS's point spectrum
    random               for an unbiased impression

It also prints the bias and RMSE per band over all dominated spectra, to see whether the loss is systematic.

Run from anywhere with:
    python -m nedsurfcast.forecast.plot_long_swell
    python -m nedsurfcast.forecast.plot_long_swell --station K133 --per-category 3 --save long_swell.png
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nedsurfcast.forecast.compare_gfs_spectra import MODEL_VARIANTS, attach_point_spectra, predict_held_out
from nedsurfcast.forecast.plot_spectra import DIR_MIN_ENERGY, _spaced, _systems_text
from nedsurfcast.forecast.train_spectrum import EVAL_MIN_LEAD, band_hm0, corrected_spectrum, load_data

MODEL = "ridge (GFS)"
LONG_F_MAX = 0.1                # Hz; Tp > 10 s
LONG_FRACTION = 0.5             # share of m0 at Tp > 10 s for a spectrum to count as dominated
MIN_LONG_HM0 = 0.3              # m; Hm0 at Tp > 10 s
BANDS = {"Tp > 10 s": (0.0, LONG_F_MAX), "6-10 s": (LONG_F_MAX, 1 / 6), "Tp < 6 s": (1 / 6, 1.0)}
SPECTRA = {                     # key -> label, line style
    "buoy": ("measured", dict(color="k", drawstyle="steps-mid", lw=1.4)),
    "point": ("GFS raw (point spectrum)", dict(color="tab:red", ls="--", lw=1.3)),
    "corrected": (f"corrected ({MODEL})", dict(color="tab:blue", lw=1.6)),
    "rebuilt": ("GFS rebuilt (model input)", dict(color="tab:grey", ls=":", lw=1.0)),
}


# ---------------------------------------------------------------------------
# 1. DATA AND PREDICTIONS
# ---------------------------------------------------------------------------

def load_cases(station: str | None = None) -> dict:
    """Rows of K133/EPL3 with a GFS point spectrum: the four spectra as (energy, direction) + meta."""
    data = load_data()
    has_point = attach_point_spectra(data)
    pred = predict_held_out(data, {MODEL: MODEL_VARIANTS[MODEL]})[MODEL]
    m = data["meta"]
    rows = has_point & np.isfinite(pred).all(1) & (m["lead_hours"].to_numpy() >= EVAL_MIN_LEAD)
    if station:
        rows &= (m["station"] == station).to_numpy()
    corr_e, corr_d = corrected_spectrum(data, np.nan_to_num(pred))
    meta = m[rows].reset_index(drop=True)
    meta["valid_time"] = meta["time"] + pd.to_timedelta(meta["lead_hours"], unit="h")
    return {
        "freq": data["freq"], "meta": meta,
        "buoy": (data["buoy_energy"][rows], data["buoy_direction"][rows]),
        "point": (data["point_energy"][rows], data["point_direction"][rows]),
        "corrected": (corr_e[rows], corr_d[rows]),
        "rebuilt": (data["gfs_energy"][rows], data["gfs_direction"][rows]),
    }


def long_dominated(p: dict) -> np.ndarray:
    """Spectra where the measured or the GFS point spectrum is dominated by Tp > 10 s."""
    f = p["freq"]
    out = np.zeros(len(p["meta"]), bool)
    for key in ("buoy", "point"):
        long = band_hm0(p[key][0], f, *BANDS["Tp > 10 s"])
        total = band_hm0(p[key][0], f, 0, 1)
        out |= (long >= MIN_LONG_HM0) & (long ** 2 >= LONG_FRACTION * total ** 2)
    return out


# ---------------------------------------------------------------------------
# 2. SUMMARY AND CASES
# ---------------------------------------------------------------------------

def band_errors(p: dict, rows: np.ndarray) -> pd.DataFrame:
    """Bias and RMSE of the band Hm0 against the measured spectrum, per spectrum, over the given rows."""
    f, eb = p["freq"], p["buoy"][0][rows]
    records = []
    for key in ("point", "rebuilt", "corrected"):
        e = p[key][0][rows]
        rec = {"spectrum": SPECTRA[key][0]}
        for band, (lo, hi) in BANDS.items():
            err = band_hm0(e, f, lo, hi) - band_hm0(eb, f, lo, hi)
            rec[f"{band} bias"] = float(err.mean())
            rec[f"{band} rmse"] = float(np.sqrt((err ** 2).mean()))
        records.append(rec)
    return pd.DataFrame(records).set_index("spectrum")


def pick_cases(p: dict, rows: np.ndarray, per_category: int = 2, seed: int = 0) -> dict[str, int]:
    f = p["freq"]
    hb, hp, hc = (band_hm0(p[k][0], f, *BANDS["Tp > 10 s"]) for k in ("buoy", "point", "corrected"))
    gain = np.abs(hp - hb) - np.abs(hc - hb)                  # > 0: correction closer to the buoy than GFS raw
    times = p["meta"]["valid_time"].to_numpy()

    def top(score):
        return _spaced(np.argsort(-np.where(rows, score, -np.inf)), times, per_category)

    cases = {}
    for name, score in (("largest long swell", hb), ("worst correction", -gain), ("best correction", gain)):
        for n, i in enumerate(top(score)):
            cases[f"{name} #{n + 1}"] = i
    rng = np.random.default_rng(seed)
    for n, i in enumerate(rng.choice(np.flatnonzero(rows), min(per_category, int(rows.sum())), replace=False)):
        cases[f"random #{n + 1}"] = int(i)
    return cases


def case_table(p: dict, cases: dict[str, int]) -> pd.DataFrame:
    f = p["freq"]
    records = []
    for title, i in cases.items():
        m = p["meta"].iloc[i]
        rec = dict(case=title, station=m["station"], run=f"{pd.Timestamp(m['time']):%Y-%m-%d}", lead=int(m["lead_hours"]))
        for key in ("buoy", "point", "corrected"):
            rec[f"Hm0>10s {key}"] = float(band_hm0(p[key][0][i:i + 1], f, *BANDS["Tp > 10 s"])[0])
        records.append(rec)
    return pd.DataFrame(records).set_index("case").round(2)


# ---------------------------------------------------------------------------
# 3. PLOT
# ---------------------------------------------------------------------------

def plot_cases(p: dict, cases: dict[str, int], ncols: int = 4):
    f = p["freq"]
    n = len(cases)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 6.0 * nrows), squeeze=False)
    for ax, (title, i) in zip(axes.flat, cases.items()):
        ax_dir = ax.twinx()
        lines = []
        for key, (label, style) in SPECTRA.items():
            e, d = p[key][0][i], p[key][1][i]
            ax.plot(f, e, label=label, **style)
            if key != "rebuilt":
                show = e >= DIR_MIN_ENERGY
                ax_dir.scatter(f[show], d[show], s=7, color=style["color"], alpha=0.6,
                               marker="o" if key == "buoy" else "x")
                hm0 = {b: float(band_hm0(e[None], f, lo, hi)[0]) for b, (lo, hi) in BANDS.items()}
                lines.append(f"{key:<9} >10s {hm0['Tp > 10 s']:.2f}  6-10s {hm0['6-10 s']:.2f}  <6s "
                             f"{hm0['Tp < 6 s']:.2f} m | {_systems_text(f, e, d)}")
        ax.axvline(LONG_F_MAX, color="tab:orange", lw=0.8, ls=":")
        ax.axvline(1 / 6, color="tab:orange", lw=0.6, ls=":", alpha=0.5)
        ax.set_xlim(0.03, 0.3)
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
        ax.annotate("\n".join(lines), xy=(0.0, 0.0), xycoords=(ax.transAxes, ax.xaxis.label),
                    xytext=(0, -6), textcoords="offset points", ha="left", va="top", fontsize=6.5,
                    family="monospace")
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    axes.flat[0].legend(loc="upper right", fontsize=7)
    fig.suptitle("Spectra dominated by long-period swell: lines = energy, dots/crosses = direction (right axis), "
                 "dotted lines = 10 s and 6 s; systems: Hs Tp direction", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96), pad=1.5, h_pad=4.5, w_pad=3.5)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--station", help="only K133 or EPL3")
    ap.add_argument("--per-category", type=int, default=2, help="cases per category")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", metavar="FILE", help="save the figure instead of showing it")
    a = ap.parse_args()

    p = load_cases(a.station)
    rows = long_dominated(p)
    n_meas = int((rows & (band_hm0(p["buoy"][0], p["freq"], *BANDS["Tp > 10 s"]) >= MIN_LONG_HM0)).sum())
    print(f"\n{int(rows.sum())} of {len(rows)} spectra dominated by long-period swell "
          f"(measured or GFS raw; {n_meas} with >= {MIN_LONG_HM0} m measured at Tp > 10 s)")
    if not rows.any():
        return
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    print("\nBand Hm0 error against the measured spectrum over these spectra (m; bias = model - measured):")
    print(band_errors(p, rows).round(3).to_string())
    for st in sorted(p["meta"]["station"].unique()):
        r = rows & (p["meta"]["station"] == st).to_numpy()
        if r.any():
            print(f"\n{st} ({int(r.sum())} spectra):")
            print(band_errors(p, r).round(3).to_string())

    cases = pick_cases(p, rows, a.per_category, a.seed)
    print("\nCases:")
    print(case_table(p, cases).to_string())
    fig = plot_cases(p, cases)
    if a.save:
        fig.savefig(a.save, dpi=110)
        print(f"Saved {a.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
