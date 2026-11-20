"""
Folds and scores of the spectrum correction model
===================================================

Folds (combined unseen station and unseen months): the issue times are split in TEST_BLOCKS equal ~monthly
blocks. A fold tests a held-out station group inside its test blocks and trains on the other groups outside
those blocks and TEST_GAP_HOURS around them (forecasts up to 72 h of neighbouring runs share valid times, plus
a day against autocorrelation). Alternating blocks keep every season in training.
    point_station_folds   the stations with a GFS point spectrum: station k tests blocks k, k + n, ...
    station_group_folds   every station group twice: on the even and on the odd blocks

Metrics, per spectrum compared with the measured (averaged buoy) spectrum, columns METRICS:
    sqrt(E)       amplitude per bin, sqrt(E) - sqrt(E_buoy): the energy part of the training target
    direction     energy-weighted direction vectors per bin: the direction part of the target
    Hm0           total wave height over the buoy's frequency range
    Hm0 T<6 s / Hm0 6-10 s / Hm0 T>10 s    wave height of the energy in the bins of that period band
                  (T <= 6 s | 6 < T <= 10 s | T > 10 s)
Scores, pooled over all scored rows (sums of squares over rows, so larger groups weigh more):
    rmse          sqrt(sum err^2 / n)
    nrmse         sqrt(sum err^2 / sum measured^2): the error relative to the size of what was measured
                  (for direction: the measured energy-weighted vectors)
    skill         1 - sqrt(sum err^2 / sum err_baseline^2): improvement over a baseline spectrum
Bootstrap intervals resample whole forecast runs (consecutive lead times of one run are correlated).
"""

import numpy as np
import pandas as pd

from nedsurfcast.forecast import config as fcfg
from nedsurfcast.forecast.load_data import PERIOD_BANDS, weighted_vector

EVAL_MIN_LEAD = 3               # h; lead 0 is the GFS analysis, not a forecast
SCORE_LEADS = [3, 6, 12, 24, 48, 72]
TEST_BLOCKS = 12
TEST_GAP_HOURS = max(fcfg.GFSWAVE_LEAD_HOURS) + 24
BOOTSTRAP = 1000
SEED = 0
METRICS = ["sqrt(E)", "direction", "Hm0", "Hm0 T<6 s", "Hm0 6-10 s", "Hm0 T>10 s"]
KINDS = ("rmse", "nrmse", "skill")


# ---------------------------------------------------------------------------
# 1. FOLDS
# ---------------------------------------------------------------------------

def test_block(meta: pd.DataFrame, k: int, n_folds: int = 2) -> tuple[np.ndarray, np.ndarray, list]:
    """Test blocks of fold k: every n_folds-th of the TEST_BLOCKS blocks of issue times, starting at k.
    Returns the rows in its blocks, the rows training must leave out (the blocks plus TEST_GAP_HOURS on both
    sides of each) and the blocks' (start, end)."""
    times = meta["time"].to_numpy()
    gap = np.timedelta64(TEST_GAP_HOURS, "h")
    in_block, near_block = np.zeros(len(times), bool), np.zeros(len(times), bool)
    edges = []
    for block in np.array_split(np.unique(times), TEST_BLOCKS)[k::n_folds]:
        in_block |= (times >= block[0]) & (times <= block[-1])
        near_block |= (times >= block[0] - gap) & (times <= block[-1] + gap)
        edges.append((pd.Timestamp(block[0]), pd.Timestamp(block[-1])))
    return in_block, near_block, edges


def point_station_folds(data: dict) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """One fold per station with a GFS point spectrum (alternating blocks): its group and test blocks held
    out, tested on that station in its blocks."""
    m = data["meta"]
    station, groups = m["station"].to_numpy(), m["buoy_group"].to_numpy()
    stations = sorted(np.unique(station[data["has_point"]]))
    folds = []
    for k, st in enumerate(stations):
        in_block, near_block, _ = test_block(m, k, len(stations))
        g = groups[station == st][0]
        folds.append((f"test {st} (blocks {k}::{len(stations)})", (groups != g) & ~near_block,
                      (station == st) & in_block))
    return folds


def station_group_folds(data: dict) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Two folds per station group (even and odd blocks), each holding out the group and its test blocks."""
    m = data["meta"]
    station, groups = m["station"].to_numpy(), m["buoy_group"].to_numpy()
    blocks = [test_block(m, k, 2)[:2] for k in range(2)]
    folds = []
    for g in np.unique(groups):
        names = "/".join(sorted(np.unique(station[groups == g])))
        for k, (in_block, near_block) in enumerate(blocks):
            folds.append((f"{names} {('even', 'odd')[k]} months", (groups != g) & ~near_block, (groups == g) & in_block))
    return folds


# ---------------------------------------------------------------------------
# 2. SCORES
# ---------------------------------------------------------------------------

def band_hm0(energy: np.ndarray, f: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    df = float(np.median(np.diff(f)))
    sel = (f >= lo) & (f < hi)
    return 4 * np.sqrt(energy[:, sel].sum(1) * df)


def _terms(sets: dict, data: dict, rows: np.ndarray) -> tuple[dict, dict]:
    """Per scored row: squared error per metric of every set, and the squared measured value per metric."""
    f = data["freq"]
    eb, db = data["buoy_energy"][rows], data["buoy_direction"][rows]
    bs, bc = weighted_vector(eb, db)
    hb = {"Hm0": band_hm0(eb, f)} | {f"Hm0 {b}": band_hm0(eb, f, lo, hi) for b, (lo, hi) in PERIOD_BANDS.items()}
    ref = {"sqrt(E)": eb.mean(1), "direction": ((bs ** 2 + bc ** 2) / 2).mean(1)} | {k: v ** 2 for k, v in hb.items()}
    err = {}
    for name, (e, d) in sets.items():
        e, d = e[rows], d[rows]
        vs, vc = weighted_vector(e, d)
        t = {"sqrt(E)": ((np.sqrt(e) - np.sqrt(eb)) ** 2).mean(1),
             "direction": (((vs - bs) ** 2 + (vc - bc) ** 2) / 2).mean(1),
             "Hm0": (band_hm0(e, f) - hb["Hm0"]) ** 2}
        t |= {f"Hm0 {b}": (band_hm0(e, f, lo, hi) - hb[f"Hm0 {b}"]) ** 2 for b, (lo, hi) in PERIOD_BANDS.items()}
        err[name] = t
    return err, ref


def _score(err: float, ref: float, base: float, n: int, kind: str) -> float:
    if kind == "rmse":
        return np.sqrt(err / n)
    if kind == "nrmse":
        return np.sqrt(err / ref)
    return 1 - np.sqrt(err / base)


def score_table(sets: dict, data: dict, rows: np.ndarray, kind: str, baseline: str | None = None,
                by: np.ndarray | None = None) -> pd.DataFrame:
    """Score per set (and per value of `by` over the rows, e.g. station), columns n + METRICS.
    kind: "rmse", "nrmse" (vs the measurements) or "skill" (vs the set named `baseline`)."""
    assert kind in KINDS and (kind != "skill" or baseline in sets)
    err, ref = _terms(sets, data, rows)
    keys = np.full(int(rows.sum()), "all", dtype=object) if by is None else by[rows]
    records = []
    for g in pd.unique(keys):
        r = keys == g
        for name in sets:
            rec = {"group": g, "spectrum": name, "n": int(r.sum())}
            for k in METRICS:
                base = err[baseline][k][r].sum() if kind == "skill" else np.nan
                rec[k] = float(_score(err[name][k][r].sum(), ref[k][r].sum(), base, r.sum(), kind))
            records.append(rec)
    return pd.DataFrame(records).set_index(["group", "spectrum"])


def bootstrap(sets: dict, data: dict, rows: np.ndarray, kind: str, baseline: str | None = None) -> pd.DataFrame:
    """Score with a 95% interval ("x [lo, hi]") over BOOTSTRAP resamples of whole forecast runs."""
    err, ref = _terms(sets, data, rows)
    run = pd.factorize(data["meta"]["time"].to_numpy()[rows])[0]
    n_runs = run.max() + 1
    rng = np.random.default_rng(SEED)
    w = np.stack([np.bincount(rng.integers(0, n_runs, n_runs), minlength=n_runs) for _ in range(BOOTSTRAP)])
    per_run = lambda v: np.bincount(run, v, n_runs)
    counts = per_run(np.ones(len(run)))
    records = []
    for name in sets:
        if kind == "skill" and name == baseline:
            continue
        rec = {"spectrum": name}
        for k in METRICS:
            e, r = per_run(err[name][k]), per_run(ref[k])
            b = per_run(err[baseline][k]) if kind == "skill" else np.full(n_runs, np.nan)
            point = _score(e.sum(), r.sum(), b.sum(), counts.sum(), kind)
            samples = _score(w @ e, w @ r, w @ b, w @ counts, kind)
            lo, hi = np.percentile(samples, [2.5, 97.5])
            fmt = "{:.3f}" if kind == "rmse" else "{:+.1%}" if kind == "skill" else "{:.1%}"
            rec[k] = f"{fmt.format(point)} [{fmt.format(lo)}, {fmt.format(hi)}]"
        records.append(rec)
    return pd.DataFrame(records).set_index("spectrum")


def by_lead(sets: dict, data: dict, rows: np.ndarray, kind: str, baseline: str | None = None,
            metrics=("sqrt(E)", "Hm0")) -> pd.DataFrame:
    """Score per lead time in SCORE_LEADS: rows = sets, columns = (metric, lead time)."""
    lead = data["meta"]["lead_hours"].to_numpy()
    t = score_table(sets, data, rows & np.isin(lead, SCORE_LEADS), kind, baseline, by=lead)
    t = t[list(metrics)].reset_index().rename(columns={"group": "lead (h)"})
    return t.pivot(index="spectrum", columns="lead (h)", values=list(metrics)).reindex(list(sets))


# ---------------------------------------------------------------------------
# 3. PRINTING
# ---------------------------------------------------------------------------

def formatted(table: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Percent for skill and nrmse, 3 decimals for rmse; n stays an integer."""
    fmt = {"rmse": "{:.3f}", "nrmse": "{:.1%}", "skill": "{:+.1%}"}[kind]
    out = table.copy().astype(object)
    for c in table.columns:
        if c != "n":
            out[c] = [fmt.format(v) if np.isfinite(v) else "" for v in table[c]]
    return out


def setup_pandas():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", 500)
