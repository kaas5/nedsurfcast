"""
Wave spectrum partitioning and partition matching
===================================================

Turns measured buoy spectra (energy density E(f) plus mean direction and spread per
frequency, as published by RWS) into wave systems ("partitions") with Hs, Tp, direction,
comparable to the GFS-Wave wind sea / swell partitions, and matches partitions of two
sources (buoy vs GFS, buoy vs model) without relying on their order.

1D partitioning, since the buoys give no 2D spectrum:
    1. split E(f) at the minimum between every two peaks
    2. also split where the mean direction jumps (crossing seas at overlapping frequencies)
    3. merge neighbours with a shallow trough or nearly the same peak frequency,
       unless their directions differ (then they are separate systems)
    4. merge partitions below min_hs into their neighbour
    5. per partition: Hs = 4 sqrt(m0), peak frequency (parabolic fit), energy-weighted
       mean direction and spread

Average spectra over a time window BEFORE partitioning (average_spectra): averaging
partitions afterwards brings back the order problem.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks

PARTITION_VARS = ["hs", "tp", "fp", "dir", "spread", "f_lo", "f_hi"]


# ---------------------------------------------------------------------------
# 1. LOAD AND AVERAGE SPECTRA
# ---------------------------------------------------------------------------

def load_spectra(spectra_dir: Path, station: str, start=None, end=None) -> xr.Dataset:
    """All monthly files of a station in spectra_dir/<station>/ (download/rws_spectra.py) -> (time, freq)."""
    files = sorted((Path(spectra_dir) / station).glob(f"{station}_*.nc"))
    if not files:
        raise FileNotFoundError(f"No spectra for {station}; run python -m nedsurfcast.download.rws_spectra")
    ds = xr.open_mfdataset(files, combine="nested", concat_dim="time").load()
    ds = ds.sortby("time").drop_duplicates("time")
    return ds.sel(time=slice(start, end))


def _angle_diff(a, b):
    """Smallest absolute difference between two directions (deg)."""
    return np.abs((np.asarray(a) - np.asarray(b) + 180) % 360 - 180)


def average_spectra(ds: xr.Dataset, window: str = "60min", step: str = "30min", min_fraction: float = 0.6) -> xr.Dataset:
    """Centred moving average over [t - window, t + window] on a regular time grid.

    Energy is averaged directly, direction as an energy-weighted vector and spread energy-weighted.
    Times with less than min_fraction of the window measured become NaN.
    """
    grid = np.arange(ds.time.values[0], ds.time.values[-1] + np.timedelta64(1, "s"), np.timedelta64(_minutes(step), "m"))
    ds = ds.reindex(time=grid)
    n = 2 * _minutes(window) // _minutes(step) + 1

    e = ds["energy_density"]
    rad = np.deg2rad(ds["direction"])
    roll = dict(time=n, center=True, min_periods=max(1, int(np.ceil(min_fraction * n))))
    e_mean = e.rolling(**roll).mean()
    e_sin = (e * np.sin(rad)).rolling(**roll).mean()
    e_cos = (e * np.cos(rad)).rolling(**roll).mean()
    e_spread = (e * ds["spread"]).rolling(**roll).mean()

    out = xr.Dataset({
        "energy_density": e_mean,
        "direction": (np.rad2deg(np.arctan2(e_sin, e_cos)) % 360).where(ds["direction"].notnull().any("time")),
        "spread": e_spread / e_mean.where(e_mean > 0),
    }, attrs=ds.attrs)
    for v in out.data_vars:
        out[v].attrs = ds[v].attrs
    return out


def _minutes(text: str) -> int:
    """'60min' -> 60, '3h' -> 180."""
    return int(pd.Timedelta(text) / pd.Timedelta(minutes=1))


# ---------------------------------------------------------------------------
# 2. PARTITION ONE SPECTRUM
# ---------------------------------------------------------------------------

def _weighted_direction(e, theta):
    rad = np.deg2rad(theta)
    ok = np.isfinite(theta) & np.isfinite(e)
    if not ok.any() or e[ok].sum() <= 0:
        return np.nan
    return float(np.rad2deg(np.arctan2(np.sum(e[ok] * np.sin(rad[ok])), np.sum(e[ok] * np.cos(rad[ok])))) % 360)


def _segments_from_peaks(e):
    """Split at the minimum between consecutive peaks -> list of [lo, hi) index ranges."""
    padded = np.r_[0.0, e, 0.0]                     # lets a peak sit at the first or last bin
    peaks, _ = find_peaks(padded)
    peaks = peaks - 1
    if len(peaks) == 0:
        return [[0, len(e)]]
    cuts = [p1 + int(np.argmin(e[p1:p2 + 1])) for p1, p2 in zip(peaks[:-1], peaks[1:])]
    edges = [0] + [c + 1 for c in cuts] + [len(e)]   # the trough bin stays with the lower-frequency side
    return [[lo, hi] for lo, hi in zip(edges[:-1], edges[1:]) if hi > lo]


def _split_on_direction(segments, e, theta, max_jump, min_energy):
    """Cut a segment where the mean direction of the two bins on either side differs > max_jump."""
    out = []
    for lo, hi in segments:
        start = lo
        for i in range(lo + 2, hi - 1):            # boundary between bin i-1 and bin i
            left, right = slice(max(start, i - 2), i), slice(i, min(hi, i + 2))
            if e[left].min() < min_energy or e[right].min() < min_energy:
                continue                            # direction is noise where there is hardly any energy
            if _angle_diff(_weighted_direction(e[left], theta[left]),
                           _weighted_direction(e[right], theta[right])) > max_jump:
                out.append([start, i])
                start = i
        out.append([start, hi])
    return out


def _merge_pair(segments, k):
    segments[k] = [segments[k][0], segments[k + 1][1]]
    del segments[k + 1]


def partition_spectrum(f, e, theta, spread=None, *, f_range=(0.03, 0.5), trough_ratio=0.6, min_df=0.02,
                       max_dir_diff=60.0, min_hs=0.1, rel_energy_dir=0.05):
    """Partition one 1D spectrum.

    f (Hz, bin centres, equally spaced), e (m2/Hz), theta (deg, coming from), spread (deg) per bin.
    trough_ratio  merge neighbours whose trough is above this fraction of the lower peak
    min_df        merge neighbours whose peaks are closer than this (Hz)
    max_dir_diff  ... but never merge systems whose mean directions differ more than this (deg),
                  and split a peak where the direction jumps more than this
    min_hs        partitions below this height (m) are merged into a neighbour
    -> list of dicts (keys PARTITION_VARS), sorted by hs, largest first
    """
    f, e, theta = np.asarray(f, float), np.asarray(e, float), np.asarray(theta, float)
    spread = np.full_like(e, np.nan) if spread is None else np.asarray(spread, float)
    keep = (f >= f_range[0]) & (f <= f_range[1])
    f, e, theta, spread = f[keep], np.nan_to_num(e[keep]), theta[keep], spread[keep]
    if e.sum() <= 0:
        return []
    df = float(np.median(np.diff(f)))

    segments = _segments_from_peaks(e)
    segments = _split_on_direction(segments, e, theta, max_dir_diff, rel_energy_dir * e.max())

    def peak(seg):
        return seg[0] + int(np.argmax(e[seg[0]:seg[1]]))

    def direction(seg):
        return _weighted_direction(e[seg[0]:seg[1]], theta[seg[0]:seg[1]])

    def hs(seg):
        return 4 * np.sqrt(e[seg[0]:seg[1]].sum() * df)   # bins hold mean densities, so sum * df = m0

    # 3. merge neighbours that are the same system
    merged = True
    while merged and len(segments) > 1:
        merged = False
        for k in range(len(segments) - 1):
            a, b = segments[k], segments[k + 1]
            trough = min(e[a[1] - 1], e[b[0]])
            shallow = trough > trough_ratio * min(e[peak(a)], e[peak(b)])
            close = f[peak(b)] - f[peak(a)] < min_df
            same_direction = not (_angle_diff(direction(a), direction(b)) > max_dir_diff)   # NaN counts as same
            if (shallow or close) and same_direction:
                _merge_pair(segments, k)
                merged = True
                break

    # 4. too small to be a system: merge into the neighbour across the higher boundary
    while len(segments) > 1:
        heights = [hs(s) for s in segments]
        k = int(np.argmin(heights))
        if heights[k] >= min_hs:
            break
        left = e[segments[k][0]] if k > 0 else -np.inf
        right = e[segments[k][1] - 1] if k < len(segments) - 1 else -np.inf
        _merge_pair(segments, k - 1 if left >= right else k)

    parts = []
    for seg in segments:
        lo, hi = seg
        h = hs(seg)   # only below min_hs when it is the last system left (calm sea): keep it
        i = peak(seg)
        # parabolic fit through the peak bin and its neighbours for a sub-bin peak frequency
        offset = 0.0
        if lo < i < hi - 1:
            denom = e[i - 1] - 2 * e[i] + e[i + 1]
            if denom < 0:
                offset = float(np.clip(0.5 * (e[i - 1] - e[i + 1]) / denom, -0.5, 0.5))
        fp = f[i] + offset * df
        w = e[lo:hi]
        ok = np.isfinite(spread[lo:hi])
        parts.append(dict(
            hs=float(h), fp=float(fp), tp=float(1 / fp), dir=direction(seg),
            spread=float(np.sum(w[ok] * spread[lo:hi][ok]) / w[ok].sum()) if ok.any() and w[ok].sum() > 0 else np.nan,
            f_lo=float(f[lo] - df / 2), f_hi=float(f[hi - 1] + df / 2),
        ))
    return sorted(parts, key=lambda p: -p["hs"])


# ---------------------------------------------------------------------------
# 3. PARTITION A TIME SERIES
# ---------------------------------------------------------------------------

def partition_dataset(ds: xr.Dataset, max_partitions: int = 4, **kwargs) -> xr.Dataset:
    """(time, freq) spectra -> (time, partition) with PARTITION_VARS, NaN for absent partitions.

    Partition 0 is the largest. The slot number is only a storage order: compare
    partitions between sources with match_partitions, never slot by slot.
    """
    f = ds["freq"].values
    e_all, th_all = ds["energy_density"].values, ds["direction"].values
    sp_all = ds["spread"].values if "spread" in ds else np.full_like(e_all, np.nan)

    out = {v: np.full((ds.sizes["time"], max_partitions), np.nan) for v in PARTITION_VARS}
    n_found = np.zeros(ds.sizes["time"], dtype=int)
    for t in range(ds.sizes["time"]):
        if not np.isfinite(e_all[t]).any():
            continue
        parts = partition_spectrum(f, e_all[t], th_all[t], sp_all[t], **kwargs)
        n_found[t] = len(parts)
        for k, p in enumerate(parts[:max_partitions]):
            for v in PARTITION_VARS:
                out[v][t, k] = p[v]

    res = xr.Dataset({v: (("time", "partition"), out[v]) for v in PARTITION_VARS},
                     coords={"time": ds.time, "partition": np.arange(max_partitions)}, attrs=ds.attrs)
    res["n_partitions"] = ("time", n_found)
    df = float(np.median(np.diff(f)))
    res["hs_total"] = 4 * np.sqrt(ds["energy_density"].sel(freq=slice(*kwargs.get("f_range", (0.03, 0.5)))).sum("freq", min_count=1) * df)
    units = dict(hs="m", tp="s", fp="Hz", dir="degree (coming from)", spread="degree", f_lo="Hz", f_hi="Hz", hs_total="m")
    for v, u in units.items():
        res[v].attrs["units"] = u
    return res


# ---------------------------------------------------------------------------
# 4. MATCH PARTITIONS OF TWO SOURCES
# ---------------------------------------------------------------------------

def partition_distance(fp_a, dir_a, fp_b, dir_b, sigma_f=0.02, sigma_dir=30.0):
    """Normalised distance in (peak frequency, direction) space; height is deliberately left out."""
    return np.hypot((np.asarray(fp_a) - np.asarray(fp_b)) / sigma_f, _angle_diff(dir_a, dir_b) / sigma_dir)


def match_partitions(a, b, max_dist=2.0, sigma_f=0.02, sigma_dir=30.0):
    """Optimal one-to-one matching of two partition sets.

    a, b: arrays (n, >=3) with columns (hs, fp, dir, ...); rows with NaN hs are ignored.
    -> (pairs [(i, j)], unmatched rows of a, unmatched rows of b), indices into the original arrays.
    Use the same settings for every source you compare (GFS vs buoy, model vs buoy).
    """
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    ia = np.flatnonzero(np.isfinite(a[:, 0]) & (a[:, 0] > 0)) if a.size else np.array([], int)
    ib = np.flatnonzero(np.isfinite(b[:, 0]) & (b[:, 0] > 0)) if b.size else np.array([], int)
    if len(ia) == 0 or len(ib) == 0:
        return [], list(ia), list(ib)
    cost = partition_distance(a[ia, None, 1], a[ia, None, 2], b[None, ib, 1], b[None, ib, 2], sigma_f, sigma_dir)
    cost = np.where(np.isfinite(cost), cost, 1e6)   # missing frequency or direction: never a match
    rows, cols = linear_sum_assignment(cost)
    pairs = [(int(ia[r]), int(ib[c])) for r, c in zip(rows, cols) if cost[r, c] <= max_dist]
    matched_a, matched_b = {i for i, _ in pairs}, {j for _, j in pairs}
    return pairs, [int(i) for i in ia if i not in matched_a], [int(j) for j in ib if j not in matched_b]
