"""
Visual check of the buoy spectrum partitioning
================================================

Tune the parameters of common/partition.py by eye before using the partitions as targets:
    figure 1  example spectra (storms, mixed seas, long-period swell, calm, random) with the
              partition boundaries and the mean direction per frequency
    figure 2  two weeks around the largest storm: spectrogram with the partition peak
              frequencies on top, and partition Hs / Tp coloured by direction. Real wave systems
              show up as continuous tracks; random splits show up as scattered points.

What to look for:
    - every clear peak in E(f) is its own partition, small noise wiggles are not
    - a partition boundary where the direction jumps is a real system boundary
    - tracks in figure 2 are continuous: swell drifts slowly to higher frequencies
      (dispersion), wind sea follows the wind

Run from anywhere with: python -m nedsurfcast.forecast.check_partitions
"""

import matplotlib.pyplot as plt
import numpy as np

from nedsurfcast.common.partition import average_spectra, load_spectra, partition_dataset
from nedsurfcast.forecast import config as fcfg

STATION = "K133"
AVERAGE_WINDOW = fcfg.BUOY_SPECTRA_WINDOW
PARTITION_KWARGS = fcfg.BUOY_PARTITION_KWARGS     # change these in forecast/config.py while checking
F_RANGE = PARTITION_KWARGS["f_range"]
SEED = 0
COLORS = plt.get_cmap("tab10").colors


def pick_examples(parts, n_random=2, seed=SEED) -> dict[str, np.datetime64]:
    """A few informative moments: largest seas, most systems, long-period swell, calm, random."""
    hs_total = parts["hs_total"].to_series().dropna()
    n = parts["n_partitions"].to_series()
    swell = ((parts["tp"] > 9) & (parts["hs"] > 0.5)).any("partition").to_series()

    def spaced(index, k, min_gap="2D"):
        """First k times from an ordered index that are at least min_gap apart."""
        chosen = []
        for t in index:
            if all(abs(t - c) >= np.timedelta64(int(min_gap[:-1]), "D") for c in chosen):
                chosen.append(t)
            if len(chosen) == k:
                break
        return chosen

    examples = {}
    for i, t in enumerate(spaced(hs_total.sort_values(ascending=False).index, 2)):
        examples[f"largest Hs #{i + 1}"] = t
    busy = hs_total[(n.reindex(hs_total.index) >= 3) & (hs_total > hs_total.median())]
    for i, t in enumerate(spaced(busy.sort_values(ascending=False).index, 2)):
        examples[f">= 3 systems #{i + 1}"] = t
    long_swell = hs_total[swell.reindex(hs_total.index).fillna(False).astype(bool)]
    for i, t in enumerate(spaced(long_swell.sample(frac=1, random_state=seed).index, 2)):
        examples[f"swell Tp > 9 s #{i + 1}"] = t
    examples["calm"] = hs_total[hs_total < hs_total.quantile(0.05)].sample(1, random_state=seed).index[0]
    for i, t in enumerate(hs_total.sample(n_random, random_state=seed + 1).index):
        examples[f"random #{i + 1}"] = t
    return examples


def plot_examples(spectra, parts, examples):
    n = len(examples)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False)
    sel_f = spectra["freq"].sel(freq=slice(*F_RANGE))
    for ax, (label, t) in zip(axes.flat, examples.items()):
        spec = spectra.sel(time=t, freq=sel_f)
        p = parts.sel(time=t)
        ax.step(sel_f, spec["energy_density"], where="mid", color="k", lw=1)
        lines = []
        for k in range(p.sizes["partition"]):
            if not np.isfinite(p["hs"][k]):
                continue
            c = COLORS[k % len(COLORS)]
            ax.axvspan(float(p["f_lo"][k]), float(p["f_hi"][k]), color=c, alpha=0.2, lw=0)
            ax.axvline(float(p["fp"][k]), color=c, lw=1, ls="--")
            lines.append(f"Hs {float(p['hs'][k]):.2f} m  Tp {float(p['tp'][k]):4.1f} s  "
                         f"{float(p['dir'][k]):3.0f}°  ±{float(p['spread'][k]):2.0f}°")
        ax.text(0.98, 0.97, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=7,
                family="monospace", bbox=dict(facecolor="white", alpha=0.8, lw=0))
        ax.set_xlim(*F_RANGE)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("E (m²/Hz)")
        ax.set_title(f"{label}: {np.datetime_as_string(np.datetime64(t), 'm')}  Hs {float(p['hs_total']):.2f} m",
                     fontsize=9)

        ax_dir = ax.twinx()
        ax_dir.scatter(sel_f, spec["direction"], s=6, color="tab:gray", alpha=0.7)
        ax_dir.set_ylim(0, 360)
        ax_dir.set_yticks([0, 90, 180, 270, 360])
        ax_dir.set_ylabel("direction (°)", color="tab:gray", fontsize=8)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.suptitle(f"{STATION}: partitioned spectra (shaded = partition, dashed = peak, grey dots = direction)")
    fig.tight_layout()
    return fig


def plot_period(spectra, parts, days=7):
    """Two weeks around the largest Hs: spectrogram + partition tracks."""
    t0 = parts["hs_total"].idxmax("time").values
    window = slice(t0 - np.timedelta64(days, "D"), t0 + np.timedelta64(days, "D"))
    s = spectra.sel(time=window, freq=slice(*F_RANGE))
    p = parts.sel(time=window)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    mesh = axes[0].pcolormesh(s.time, s.freq, np.log10(s["energy_density"].T.clip(min=1e-4)), cmap="viridis",
                              shading="nearest")
    fig.colorbar(mesh, ax=axes[0], label="log10 E (m²/Hz)", pad=0.01)
    times = np.repeat(p.time.values[:, None], p.sizes["partition"], axis=1)
    big = (p["hs"] > 0.2).values
    axes[0].scatter(times[big], p["fp"].values[big], s=3, color="w")
    axes[0].set_ylabel("frequency (Hz)")
    axes[0].set_title(f"{STATION}: spectrogram with partition peak frequencies (Hs > 0.2 m, white)")

    axes[1].plot(p.time, p["hs_total"], color="k", lw=1, label="total Hs")
    sc = axes[1].scatter(times, p["hs"].values, c=p["dir"].values, cmap="twilight", vmin=0, vmax=360, s=5)
    axes[1].set_ylabel("Hs (m)")
    axes[1].legend(loc="upper left")
    fig.colorbar(sc, ax=axes[1], label="direction (°)", pad=0.01)

    sc = axes[2].scatter(times, p["tp"].values, c=p["dir"].values, cmap="twilight", vmin=0, vmax=360,
                         s=np.nan_to_num(p["hs"].values) * 15 + 1)
    axes[2].set_ylabel("Tp (s), size ~ Hs")
    axes[2].set_ylim(0, 20)
    fig.colorbar(sc, ax=axes[2], label="direction (°)", pad=0.01)
    fig.tight_layout()
    return fig


def summarize(spectra, parts):
    valid = spectra["energy_density"].notnull().any("freq")
    n = parts["n_partitions"].where(valid, drop=True).values.astype(int)
    print(f"{STATION}: {int(valid.sum())} averaged spectra, number of partitions per spectrum:",
          dict(zip(*np.unique(n, return_counts=True))))
    for k in range(parts.sizes["partition"]):
        q = parts.isel(partition=k)
        print(f"  partition {k}: present {float(q['hs'].notnull().where(valid, drop=True).mean()):.0%}, "
              f"Hs p50 {float(q['hs'].median()):.2f} m, Tp p50 {float(q['tp'].median()):.1f} s")


if __name__ == "__main__":
    spectra = average_spectra(load_spectra(fcfg.RWS_SPECTRA_DIR, STATION), window=AVERAGE_WINDOW)
    parts = partition_dataset(spectra, **PARTITION_KWARGS)
    summarize(spectra, parts)
    plot_examples(spectra, parts, pick_examples(parts))
    plot_period(spectra, parts)
    plt.show()
