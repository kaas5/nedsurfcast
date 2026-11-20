"""
Does the corrected spectrum beat GFS's own forecast spectrum?
===============================================================

The spectrum model (forecast/train_spectrum.py) is trained against a GFS spectrum that is RECONSTRUCTED
from the GFS partitions (JONSWAP shape per partition). Part of its skill can therefore be repairing that
reconstruction instead of GFS. GFS also publishes its real forecast spectra at a few stations; two of them
sit at an RWS spectrum buoy (fcfg.GFSWAVE_SPEC_STATIONS: K13 -> K133, EURO -> EPL3). There the fair test is:

    1. measured spectrum vs GFS point spectrum (GFS's own forecast, computed at the buoy position)
    2. measured spectrum vs corrected spectrum (ridge, out-of-sample)

If 2 beats 1, the model beats the GFS forecast. The reconstructed GFS spectrum (the model input) is shown as well.

Ridge and the MLP (same settings as train_spectrum.py) are trained on two folds that combine leave-one-group
and blocked time: each fold leaves out the station group of K133 or EPL3 AND its test blocks of time (the
year in TEST_BLOCKS ~monthly blocks, alternating between the two folds, with TEST_GAP_HOURS on both sides of
each block unused). The test set is the held-out station inside its test blocks: a station and periods the
model has never seen, together covering the whole year, while training still spans all seasons.
Variants: GFS input only, and GFS + the
change of the GFS spectrum over the previous hours. No buoy measurement as input: it does not exist at grid
cells without a buoy, so such a model would not generalise. Choose the models with MODEL_VARIANTS or
--models, e.g. --models "ridge (GFS)" "ridge (GFS + history per band)".

Scores over lead times >= EVAL_MIN_LEAD and spectra where the GFS point spectrum exists:
    rmse sqrt(E)    RMSE of the amplitude per bin: the energy part of the training loss
    rmse dir vector RMSE of the energy-weighted direction vectors per bin: the direction part of the loss
    Hm0 per band, energy-weighted swell direction error
Bootstrap over forecast runs (BOOTSTRAP samples) gives 95% intervals for the skill against the GFS point
spectrum: consecutive lead times of one run are strongly correlated, so runs are resampled as a whole.

Only tables are written (fcfg.SPECTRUM_MODEL_DIR/gfs_point_*.csv), no spectra.

Run from anywhere with: python -m nedsurfcast.forecast.compare_gfs_spectra
(needs python -m nedsurfcast.download.gfs --spectra and python -m nedsurfcast.forecast.preprocess --station-spectra)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from nedsurfcast.common.angles import wrap_angle
from nedsurfcast.common.validation import TIME_GAP_HOURS, rmse
from nedsurfcast.forecast import config as fcfg
from nedsurfcast.forecast.preprocess import load
from nedsurfcast.forecast.train_spectrum import (EVAL_MIN_LEAD, MODELS, PERIOD_BANDS, SCORE_LEADS, SWELL_F_MAX,
                                                 _mlp, _weighted_vector, band_hm0, corrected_spectrum, features,
                                                 load_data, targets)

BOOTSTRAP = 1000
SEED = 0
# Test blocks: the issue times split in TEST_BLOCKS equal parts (~monthly), alternating between the folds: the
# K133 fold tests on blocks 0, 2, 4, ..., the EPL3 fold on 1, 3, 5, ... Together the test set covers the whole
# year, and each fold still trains on every season. Training leaves out each test block and TEST_GAP_HOURS on
# both sides: forecasts (up to 72 h) of runs next to a block share its valid times, plus TIME_GAP_HOURS
# against autocorrelation.
TEST_BLOCKS = 12
TEST_GAP_HOURS = max(fcfg.GFSWAVE_LEAD_HOURS) + TIME_GAP_HOURS
MODEL_VARIANTS = {                      # name -> (model factory, feature set)
    "ridge (GFS)": (MODELS["ridge"][0], "gfs"),
    "ridge (GFS + history)": (MODELS["ridge"][0], "gfs+history"),
    "ridge (GFS + history per band)": (MODELS["ridge"][0], "gfs+history-bands"),
    "mlp (GFS)": (_mlp, "gfs"),
    "mlp (GFS + history per band)": (_mlp, "gfs+history-bands"),
    "ridge (GFS + location)": (MODELS["ridge"][0], "gfs+location"),
    "mlp (GFS + location)": (_mlp, "gfs+location"),
    "ridge (GFS + history per band + location)": (MODELS["ridge"][0], "gfs+history-bands+location"),
    "mlp (GFS + history per band + location)": (_mlp, "gfs+history-bands+location"),
    "ridge (GFS + location + fetch)": (MODELS["ridge"][0], "gfs+location+fetch"),
    "ridge (GFS + history per band + location + fetch)": (MODELS["ridge"][0], "gfs+history-bands+location+fetch"),
    "mlp (GFS + location + fetch)": (_mlp, "gfs+location+fetch"),
    "mlp (GFS + history per band + location + fetch)": (_mlp, "gfs+history-bands+location+fetch"),
}
BASELINE = "GFS point spectrum"

# Evaluation "station-bias" (--eval station-bias): can the models learn a bias correction PER STATION? All
# stations train, K133 and EPL3 included; only the last autumn (issue times from AUTUMN_START, with
# TEST_GAP_HOURS before it) is left out and is the test set at K133/EPL3. A station identifier (one-hot) is
# offered as a feature; "station mean bias" is the mean training residual per station and bin. Diagnostic
# only: neither the station identity nor training data at the station exist at grid cells without a buoy.
AUTUMN_START = np.datetime64("2026-09-01")
STATION_BIAS_VARIANTS = {
    "station mean bias": (lambda: Ridge(alpha=1e-6), "station"),
    "ridge (GFS)": MODEL_VARIANTS["ridge (GFS)"],
    "ridge (GFS + station)": (MODELS["ridge"][0], "gfs+station"),
    "ridge (GFS + location)": MODEL_VARIANTS["ridge (GFS + location)"],
    "mlp (GFS)": MODEL_VARIANTS["mlp (GFS)"],
    "mlp (GFS + station)": (_mlp, "gfs+station"),
    "mlp (GFS + location)": MODEL_VARIANTS["mlp (GFS + location)"],
}
# "alternating-epl3": the alternating evaluation tested on EPL3 only (two folds: even and odd blocks), because
# K133 lies far offshore, outside the training distribution of mostly coastal stations; K133 still trains.
EVALUATIONS = {"alternating": MODEL_VARIANTS, "alternating-epl3": MODEL_VARIANTS,
               "station-bias": STATION_BIAS_VARIANTS}


# ---------------------------------------------------------------------------
# 1. DATA
# ---------------------------------------------------------------------------

def attach_point_spectra(data: dict) -> np.ndarray:
    """Add the GFS point spectra to data (aligned with its rows); returns the rows that have one."""
    ps = load("gfs_station_spectra").load()
    m = data["meta"]
    times = ps["time"].values
    it = np.searchsorted(times, m["time"].to_numpy())
    il = np.searchsorted(ps["lead_hours"].values, m["lead_hours"].to_numpy())
    stations = list(ps["station"].values)
    ist = np.array([stations.index(s) if s in stations else -1 for s in m["station"]])
    ok = (ist >= 0) & (it < len(times))
    ok[ok] &= times[it[ok]] == m["time"].to_numpy()[ok]
    nf = len(data["freq"])
    energy = np.full((len(m), nf), np.nan)
    direction = np.full((len(m), nf), np.nan)
    energy[ok] = ps["gfs_spec_energy"].values[it[ok], il[ok], ist[ok]]
    direction[ok] = ps["gfs_spec_direction"].values[it[ok], il[ok], ist[ok]]
    data["point_energy"], data["point_direction"] = energy, direction
    return ok & np.isfinite(energy).all(1)


# ---------------------------------------------------------------------------
# 2. MODELS ON THE FOLDS THAT HOLD OUT THE POINT-SPECTRUM STATIONS
# ---------------------------------------------------------------------------

def predict_held_out(data: dict, variants: dict = MODEL_VARIANTS, test_stations=None) -> dict[str, np.ndarray]:
    """Out-of-sample residuals of each model variant (name -> (factory, feature set)) for the test stations
    (default K133 and EPL3), each fold trained with the station's group and its test blocks held out.

    Two stations: station k is tested on the alternating blocks k, k + 2, ... One station: two folds, on the
    even and on the odd blocks, so that station is tested over the whole year."""
    m = data["meta"]
    Y = targets(data)
    X = {fs: features(data, fs) for _, fs in variants.values()}
    pred = {name: np.full(Y.shape, np.nan) for name in variants}
    stations = list(test_stations or fcfg.GFSWAVE_SPEC_STATIONS.values())
    folds = [(stations[0], 0), (stations[0], 1)] if len(stations) == 1 else list(zip(stations, range(len(stations))))
    n_folds = len(folds)
    for station, k in folds:
        in_block, near_block, edges = test_block(m, k, n_folds)
        print(f"  test blocks of {station}: {', '.join(f'{a:%Y-%m-%d}..{b:%m-%d}' for a, b in edges)}; "
              f"training also leaves out {TEST_GAP_HOURS} h on both sides of each")
        group = m.loc[m["station"] == station, "buoy_group"].iloc[0]
        is_train = (m["buoy_group"] != group).to_numpy() & ~near_block
        is_test = (m["station"] == station).to_numpy() & in_block
        held_out = sorted(m.loc[m["buoy_group"] == group, "station"].unique())
        for name, (make, fs) in variants.items():
            model = make().fit(X[fs][is_train], Y[is_train])
            pred[name][is_test] = np.clip(model.predict(X[fs][is_test]), Y[is_train].min(0), Y[is_train].max(0))
        print(f"  fold holding out {held_out} and the test block: trained on {int(is_train.sum())} spectra, "
              f"predicted {int(is_test.sum())} spectra of {station}")
    return pred


def predict_station_bias(data: dict, variants: dict = STATION_BIAS_VARIANTS) -> dict[str, np.ndarray]:
    """Evaluation "station-bias": train on every station before the last autumn, predict K133/EPL3 in it."""
    m = data["meta"]
    Y = targets(data)
    X = {fs: features(data, fs) for _, fs in variants.values()}
    times = m["time"].to_numpy()
    is_train = times < AUTUMN_START - np.timedelta64(TEST_GAP_HOURS, "h")
    is_test = (times >= AUTUMN_START) & m["station"].isin(fcfg.GFSWAVE_SPEC_STATIONS.values()).to_numpy()
    pred = {name: np.full(Y.shape, np.nan) for name in variants}
    for name, (make, fs) in variants.items():
        model = make().fit(X[fs][is_train], Y[is_train])
        pred[name][is_test] = np.clip(model.predict(X[fs][is_test]), Y[is_train].min(0), Y[is_train].max(0))
    print(f"  trained on {int(is_train.sum())} spectra of all stations (issue times before "
          f"{AUTUMN_START - np.timedelta64(TEST_GAP_HOURS, 'h')}), predicted {int(is_test.sum())} spectra "
          f"of K133/EPL3 from {AUTUMN_START}")
    return pred


def test_block(meta: pd.DataFrame, k: int, n_folds: int = 2) -> tuple[np.ndarray, np.ndarray, list]:
    """Test blocks of fold k: the issue times split in TEST_BLOCKS equal parts, fold k takes every n_folds-th
    block starting at k (alternating). Returns the rows in its blocks, the rows training must leave out (the
    blocks plus TEST_GAP_HOURS on both sides of each) and the blocks' (start, end)."""
    times = meta["time"].to_numpy()
    gap = np.timedelta64(TEST_GAP_HOURS, "h")
    in_block, near_block = np.zeros(len(times), bool), np.zeros(len(times), bool)
    edges = []
    for block in np.array_split(np.unique(times), TEST_BLOCKS)[k::n_folds]:
        in_block |= (times >= block[0]) & (times <= block[-1])
        near_block |= (times >= block[0] - gap) & (times <= block[-1] + gap)
        edges.append((pd.Timestamp(block[0]), pd.Timestamp(block[-1])))
    return in_block, near_block, edges


# ---------------------------------------------------------------------------
# 3. SCORES
# ---------------------------------------------------------------------------

def squared_errors(f, e, d, eb, db) -> dict[str, np.ndarray]:
    """Per spectrum: squared errors that the scores are built from (so they can be bootstrapped by run)."""
    vs, vc = _weighted_vector(e, d)
    bs, bc = _weighted_vector(eb, db)
    out = {
        "rmse sqrt(E)": ((np.sqrt(e) - np.sqrt(eb)) ** 2).mean(1),
        "rmse dir vector": (((vs - bs) ** 2 + (vc - bc) ** 2) / 2).mean(1),
        "rmse Hm0": (band_hm0(e, f, 0, 1) - band_hm0(eb, f, 0, 1)) ** 2,
    }
    for band, (lo, hi) in PERIOD_BANDS.items():
        out[f"rmse Hm0 {band}"] = (band_hm0(e, f, lo, hi) - band_hm0(eb, f, lo, hi)) ** 2
    return out


def swell_dir_error(f, d, eb, db) -> float:
    swell = f < SWELL_F_MAX
    diff = np.abs(wrap_angle(d[:, swell] - db[:, swell]))
    w = np.where(np.isfinite(diff), eb[:, swell], 0)
    return float(np.sum(np.nan_to_num(diff) * w) / np.sum(w))


def score_table(sets: dict, data: dict, rows: np.ndarray, by=None) -> pd.DataFrame:
    """RMSE per set (rows of the table), optionally per group of `by` (e.g. station)."""
    f = data["freq"]
    eb, db = data["buoy_energy"], data["buoy_direction"]
    groups = {"all": rows} if by is None else {g: rows & (by == g) for g in np.unique(by[rows])}
    records = []
    for g, r in groups.items():
        for name, (e, d) in sets.items():
            sq = squared_errors(f, e[r], d[r], eb[r], db[r])
            rec = {"group": g, "spectrum": name, "n": int(r.sum())}
            rec |= {k: float(np.sqrt(v.mean())) for k, v in sq.items()}
            rec["swell dir error (deg)"] = swell_dir_error(f, d[r], eb[r], db[r])
            records.append(rec)
    return pd.DataFrame(records).set_index(["group", "spectrum"])


def skill(table: pd.DataFrame, baseline: str = BASELINE) -> pd.DataFrame:
    cols = [c for c in table.columns if c.startswith("rmse")]
    base = table.xs(baseline, level="spectrum")[cols]
    return 1 - table[cols].div(base.reindex(table.index, level="group"))


def by_lead(sets: dict, data: dict, rows: np.ndarray) -> pd.DataFrame:
    f, lead = data["freq"], data["meta"]["lead_hours"].to_numpy()
    eb, db = data["buoy_energy"], data["buoy_direction"]
    records = []
    for name, (e, d) in sets.items():
        sq = squared_errors(f, e, d, eb, db)
        for h in SCORE_LEADS:
            r = rows & (lead == h)
            records.append({"spectrum": name, "lead (h)": h,
                            "rmse sqrt(E)": float(np.sqrt(sq["rmse sqrt(E)"][r].mean())),
                            "rmse Hm0 swell (Tp > 6 s)": float(np.sqrt(
                                ((band_hm0(e[r], f, 0, SWELL_F_MAX) - band_hm0(eb[r], f, 0, SWELL_F_MAX)) ** 2).mean()))})
    t = pd.DataFrame(records).pivot(index="spectrum", columns="lead (h)")
    return t.reindex(list(sets))


def bootstrap_skill(sets: dict, data: dict, rows: np.ndarray, metrics=("rmse sqrt(E)", "rmse Hm0 Tp > 10 s",
                                                                        "rmse Hm0 6-10 s", "rmse Hm0"),
                    baseline: str = BASELINE) -> pd.DataFrame:
    """95% interval of 1 - RMSE_set / RMSE_baseline, resampling whole forecast runs (issue times)."""
    f = data["freq"]
    eb, db = data["buoy_energy"][rows], data["buoy_direction"][rows]
    run = pd.factorize(data["meta"]["time"].to_numpy()[rows])[0]
    n_runs = run.max() + 1
    sums = {name: {k: np.bincount(run, v, n_runs) for k, v in squared_errors(f, e[rows], d[rows], eb, db).items()}
            for name, (e, d) in sets.items()}
    counts = np.bincount(run, minlength=n_runs)
    rng = np.random.default_rng(SEED)
    weights = np.stack([np.bincount(rng.integers(0, n_runs, n_runs), minlength=n_runs) for _ in range(BOOTSTRAP)])
    records = []
    for name in sets:
        if name == baseline:
            continue
        rec = {"spectrum": name}
        for k in metrics:
            mse = (weights @ sums[name][k]) / (weights @ counts)
            mse_base = (weights @ sums[baseline][k]) / (weights @ counts)
            s = 1 - np.sqrt(mse / mse_base)
            point = 1 - np.sqrt(sums[name][k].sum() / sums[baseline][k].sum())
            lo, hi = np.percentile(s, [2.5, 97.5])
            rec[k] = f"{point:+.1%} [{lo:+.1%}, {hi:+.1%}]"
        records.append(rec)
    return pd.DataFrame(records).set_index("spectrum")


# ---------------------------------------------------------------------------
# 4. RUN
# ---------------------------------------------------------------------------

def run(models: list[str] | None = None, evaluation: str = "alternating"):
    """models: names from the evaluation's variants to train (default all); evaluation: see EVALUATIONS."""
    variants = {k: v for k, v in EVALUATIONS[evaluation].items() if models is None or k in models}
    data = load_data()
    has_point = attach_point_spectra(data)
    m = data["meta"]
    print(f"{int(has_point.sum())} spectra of {sorted(fcfg.GFSWAVE_SPEC_STATIONS.values())} have a GFS point spectrum; "
          f"models: {list(variants)}")

    if evaluation == "station-bias":
        pred = predict_station_bias(data, variants)
    else:
        pred = predict_held_out(data, variants, ["EPL3"] if evaluation == "alternating-epl3" else None)
    rows = has_point & (m["lead_hours"].to_numpy() >= EVAL_MIN_LEAD)
    for p in pred.values():
        rows &= np.isfinite(p).all(1) | ~rows        # only rows that were predicted (K133/EPL3 in the test block)
    print(f"Scoring {int(rows.sum())} spectra (lead times >= {EVAL_MIN_LEAD} h, "
          f"{m.loc[rows, 'time'].nunique()} forecast runs)\n")

    sets = {BASELINE: (data["point_energy"], data["point_direction"]),
            "GFS reconstruction": (data["gfs_energy"], data["gfs_direction"])}
    for name, y in pred.items():
        sets[name] = corrected_spectrum(data, np.nan_to_num(y))

    station = m["station"].to_numpy()
    overall = score_table(sets, data, rows)
    per_station = score_table(sets, data, rows, by=station)
    leads = by_lead(sets, data, rows)
    boot = bootstrap_skill(sets, data, rows)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print("Errors against the measured spectra, both stations (lower is better):")
    print(overall.droplevel("group").round(3).to_string())
    print(f"\nSkill vs the {BASELINE} (1 - RMSE / RMSE_{BASELINE.replace(' ', '_')}):")
    print(skill(overall).droplevel("group").map(lambda v: f"{v:+.1%}").to_string())
    print(f"\nSkill with 95% bootstrap interval over forecast runs ({BOOTSTRAP} samples):")
    print(boot.to_string())
    print("\nPer station:")
    print(per_station.round(3).to_string())
    print(f"\nSkill vs the {BASELINE} per station:")
    print(skill(per_station).map(lambda v: f"{v:+.1%}").to_string())
    print("\nPer lead time:")
    print(leads.round(3).to_string())

    out = fcfg.SPECTRUM_MODEL_DIR
    out.mkdir(parents=True, exist_ok=True)
    sfx = "" if evaluation == "alternating" else "_" + evaluation.replace("-", "_")
    overall.droplevel("group").to_csv(out / f"gfs_point_comparison{sfx}.csv")
    per_station.to_csv(out / f"gfs_point_comparison_per_station{sfx}.csv")
    leads.to_csv(out / f"gfs_point_comparison_by_lead{sfx}.csv")
    boot.to_csv(out / f"gfs_point_comparison_bootstrap{sfx}.csv")
    print(f"\nTables saved to {out} (gfs_point_comparison*{sfx}.csv)")
    return overall, per_station, leads, boot


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Compare the corrected spectra with GFS's point spectra")
    ap.add_argument("--eval", choices=list(EVALUATIONS), default="alternating",
                    help="alternating: unseen station and months (default); alternating-epl3: the same, tested "
                         "on EPL3 only; station-bias: all stations train, the last autumn is the test set")
    ap.add_argument("--models", nargs="+", metavar="NAME", help="subset of the evaluation's model variants")
    args = ap.parse_args()
    unknown = set(args.models or []) - set(EVALUATIONS[args.eval])
    if unknown:
        ap.error(f"unknown models {sorted(unknown)}; choose from {list(EVALUATIONS[args.eval])}")
    run(args.models, args.eval)
