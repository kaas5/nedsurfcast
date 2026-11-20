"""
Correct the GFS wave spectrum towards the buoy spectrum, then partition it
============================================================================

Instead of correcting GFS wave systems one by one (which needs matching to know which buoy system
belongs to which GFS system), the model corrects the whole 1D spectrum at a station:

    input   GFS spectrum rebuilt from its partitions (forecast/preprocess.py, group "spectra"), scaled to
            GFS's HTSGW of the same time step (SCALE_TO_HTSGW, htsgw_factor):
            log energy and energy-weighted direction per 10 mHz bin, plus bulk GFS context
            + (feature set "+history") the change of the rebuilt GFS spectrum over the previous 6/12/24 h
              (HISTORY_LAGS) per bin, or ("+history-bands") per period band: Hm0, mean frequency and
              mean direction; from the same run, or before the issue time from the GFS nowcast
              (f000/f003 of the 06/12/18z runs), so only data known at the issue time
            + (models marked "gfs+obs") the GFS error at ISSUE time: the buoy spectrum measured around
              the issue time minus the GFS spectrum at lead 0, in the same form as the target. A GFS run
              is published ~3.5 h after its issue time, so this measurement is known when the forecast
              is used. Missing measurements (gaps) are set to 0 with an indicator feature.
    target  per bin: sqrt(E_buoy) - sqrt(E_GFS)   (amplitude, additive)
            and the difference of the energy-weighted direction vectors (buoy - GFS)
            -> an output of 0 is the uncorrected GFS spectrum

            Why amplitude and not log energy: a log ratio is dominated by nearly empty bins (the
            buoy's noise floor against GFS's steep JONSWAP flanks) and turns into a multiplicative
            factor 10^y that explodes when a real peak lands in such a bin. sqrt(E) is additive,
            weighs energetic bins roughly like wave height, and makes the noise floor negligible.
            Predictions are clipped to the range of the training targets against extrapolation.
    output  corrected spectrum, partitioned with the SAME partition_spectrum as the buoy, so the
            wave systems have exactly the buoy's definition and any swell definition can be
            derived afterwards

The output has a fixed size (one value per bin), so training needs no matching at all.
Matching is only used to score wave systems, identically for GFS and the model.

Validation (common/validation.py): leave-one-group (unseen location) and blocked-time (unseen
weather; the gap covers the longest lead time).

Models: mean-bias, persistence (GFS + the issue-time error at every lead time, no fitting), ridge and
mlp with the issue-time measurement, and mlp-gfs-only without it (shows what the measurement adds).

Scores against the buoy, GFS vs corrected, over lead times >= EVAL_MIN_LEAD (at lead 0 the issue-time
measurement IS the target):
    spectrum  RMSE of log10 energy per bin, Hm0 per period band, swell-band direction
    systems   hit rate, false alarms and Hs/Tp/direction errors of matched systems >= 0.3 m
    by lead   RMSE of swell Hm0 (Tp > 6 s) per lead time

Saved in fcfg.SPECTRUM_MODEL_DIR:
    oos_spectra.zarr      out-of-sample corrected spectra of every model and scheme, with the buoy and
                          GFS spectra, along (spectrum, freq); time/station/lead_hours identify a spectrum
    spectrum_scores.csv   spectrum scores per scheme and model (GFS included)
    spectrum_skill.csv    skill vs GFS of the RMSE columns
    system_scores.csv     wave-system scores after partitioning
    swell_hm0_by_lead.csv RMSE of swell Hm0 per lead time

Run from anywhere with: python -m nedsurfcast.forecast.train_spectrum
"""

import matplotlib.pyplot as plt
import json
import shutil

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nedsurfcast.common.angles import wrap_angle
from nedsurfcast.common.partition import match_partitions, partition_spectrum
from nedsurfcast.common.validation import FOLDS, N_TIME_BLOCKS, rmse
from nedsurfcast.forecast import config as fcfg
from nedsurfcast.forecast.location import fetch_limited_log_ratio, load_station_location, lookup, tanh_kh
from nedsurfcast.forecast.preprocess import load

EPS = 1e-3                      # m2/Hz added before logs (features, log score) and in the direction weights
HS_MIN = 0.3                    # m; systems below this are not scored
TIME_GAP_HOURS = max(fcfg.GFSWAVE_LEAD_HOURS) + 24
PERIOD_BANDS = {"Tp > 10 s": (0.0, 0.1), "6-10 s": (0.1, 1 / 6), "Tp < 6 s": (1 / 6, 1.0)}   # Hz
SWELL_F_MAX = 1 / 6             # Hz; bins below this count as the swell band for the direction score

EVAL_MIN_LEAD = 3               # h; scores leave out lead 0, where the issue-time measurement is the target
SCORE_LEADS = [3, 6, 12, 24, 48, 72]
HISTORY_LAGS = (6, 12, 24)      # h; feature set "history": change of the GFS spectrum over these lags
SCALE_TO_HTSGW = True           # scale every rebuilt GFS spectrum to Hm0 = GFS's HTSGW (see htsgw_factor)


class PersistIssueError:
    """Baseline: the GFS error measured at issue time, applied unchanged at every lead time.

    Reads it from the issue-error block of the "gfs+obs" features (the last 3 x n_freq columns
    before the indicator), which has exactly the layout of the target."""

    def fit(self, X, Y):
        self.n_out = Y.shape[1]
        return self

    def predict(self, X):
        return X[:, -(self.n_out + 1):-1]


MLP_BACKEND = "torch"           # "torch": common.torch_mlp on the GPU if available; "sklearn": MLPRegressor (CPU)
MLP_KWARGS = dict(hidden_layer_sizes=(128, 64), alpha=1e-3, learning_rate_init=1e-3, batch_size=256,
                  max_iter=40, early_stopping=True, n_iter_no_change=5, random_state=0)


def _mlp():
    if MLP_BACKEND == "torch":
        from nedsurfcast.common.torch_mlp import TorchMLPRegressor
        net = TorchMLPRegressor(**MLP_KWARGS)
    else:
        net = MLPRegressor(**MLP_KWARGS)
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), net)


# name -> (factory, feature set): "gfs" = GFS spectrum + context, "gfs+obs" = also the issue-time GFS error
MODELS = {
    "mean-bias": (lambda: DummyRegressor(strategy="mean"), "gfs"),
    "persistence": (PersistIssueError, "gfs+obs"),
    "ridge": (lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)), "gfs+obs"),
    "mlp-gfs-only": (_mlp, "gfs"),
    "mlp": (_mlp, "gfs+obs"),
}
SYSTEM_MODELS = ["persistence", "mlp-gfs-only", "mlp"]   # partitioning is slow; score systems for these only


# ---------------------------------------------------------------------------
# 1. DATA
# ---------------------------------------------------------------------------

def _weighted_vector(energy, direction):
    """Direction as a vector whose length says how much energy carries it (0 in empty bins)."""
    w = energy / (energy + EPS)
    rad = np.deg2rad(np.nan_to_num(direction))
    return w * np.sin(rad), w * np.cos(rad)


def htsgw_factor(energy: np.ndarray, htsgw: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Energy factor (HTSGW / Hm0)^2 per spectrum, Hm0 of the rebuilt spectrum over its frequency grid (the
    buoy's range): the scaled spectrum has GFS's total wave height of the predicted time step and the shape
    of the partitions. 1 where it cannot be computed (empty spectrum, missing HTSGW) or SCALE_TO_HTSGW is off."""
    if not SCALE_TO_HTSGW:
        return np.ones(energy.shape[:-1])
    df = float(np.median(np.diff(f)))
    hm0 = 4 * np.sqrt(energy.sum(-1) * df)
    with np.errstate(invalid="ignore", divide="ignore"):
        fac = (htsgw / hm0) ** 2
    return np.where(np.isfinite(fac) & (hm0 > 0), fac, 1.0)


def load_data() -> dict:
    """Every (run, lead, station) with a buoy spectrum -> arrays (n, n_freq) + a meta table for the folds."""
    spec = load("spectra").load()
    st = load("station_times").load()
    stations = load("stations").load()

    has = st["has_buoy"].values                                   # (time, lead, station)
    it, il, ist = np.nonzero(has)
    take = lambda a: a[it, il, ist]

    variables = ("buoy_energy", "buoy_direction", "gfs_energy", "gfs_direction")
    data = {v: take(spec[v].values).astype(float) for v in variables}
    # the same run and station at lead 0: the measurement around the issue time and GFS's analysis
    for v in variables:
        data[f"issue_{v}"] = spec[v].values[it, 0, ist].astype(float)
    data["has_issue_obs"] = has[it, 0, ist]
    # the reconstruction scaled to GFS's HTSGW of the predicted time step (and of lead 0 for the issue time)
    reb = load("gfs_rebuilt")
    for c in ("time", "lead_hours", "station"):
        assert (reb[c].values == spec[c].values).all(), f"gfs_rebuilt and spectra differ in {c}"
    # HTSGW bilinear at the station (as scored against the buoy); nearest cell where that is missing
    hts = st["gfs_htsgw"].values
    hts = np.where(np.isfinite(hts), hts, reb["htsgw"].values)
    f = spec["freq"].values
    data["gfs_energy_unscaled"] = data["gfs_energy"].copy()
    data["htsgw_factor"] = htsgw_factor(data["gfs_energy"], hts[it, il, ist], f)
    data["gfs_energy"] *= data["htsgw_factor"][:, None]
    data["issue_gfs_energy"] *= htsgw_factor(data["issue_gfs_energy"], hts[it, 0, ist], f)[:, None]
    data["index"] = (it, il, ist)                                 # into (time, lead_hours, station)
    data["freq"] = spec["freq"].values
    meta = pd.DataFrame({
        "time": spec["time"].values[it], "station": spec["station"].values[ist],
        "lead_hours": spec["lead_hours"].values[il], "buoy_group": stations["station_group"].values[ist],
    })
    for v in ("gfs_htsgw", "gfs_u10", "gfs_v10", "gfs_perpw", "gfs_dirpw"):
        meta[v] = take(st[v].values)
    data["meta"] = meta.set_index(["time", "station", "lead_hours"], drop=False)
    return data


def issue_error(data: dict) -> np.ndarray:
    """GFS error at issue time (buoy - GFS at lead 0), same layout as the target; 0 where nothing was measured."""
    err = residual(data["issue_buoy_energy"], data["issue_buoy_direction"],
                   data["issue_gfs_energy"], data["issue_gfs_direction"])
    err[~data["has_issue_obs"]] = 0.0
    return np.nan_to_num(err)


def _history_pairs(data: dict, lags=HISTORY_LAGS) -> list[tuple]:
    """Per lag: (energy, direction) of the rebuilt GFS spectrum at the valid time and `lag` hours earlier
    (group "gfs_rebuilt"); the earlier one is NaN where it does not exist.

    The earlier spectrum comes from the same run if lead - lag >= 0. Before the issue time it is the GFS
    nowcast at that valid time (group "gfs_nowcast": f000/f003 of the latest 00/06/12/18z run), and only
    where that is missing the run issued 24 h earlier (lead - lag + 24). Everything exists at the issue
    time, no future information.
    """
    if "history_pairs" in data:
        return data["history_pairs"]
    reb = load("gfs_rebuilt")
    e_all, d_all = reb["energy"].values.astype(float), reb["direction"].values.astype(float)
    e_all *= htsgw_factor(e_all, reb["htsgw"].values, reb["freq"].values)[..., None]   # nearest cell, as e_nc
    times, leads = reb["time"].values, list(reb["lead_hours"].values)
    nc = load("gfs_nowcast")
    assert list(nc["station"].values) == list(reb["station"].values)
    e_nc, d_nc, nc_times = nc["energy"].values.astype(float), nc["direction"].values.astype(float), nc["valid_time"].values
    e_nc *= htsgw_factor(e_nc, nc["htsgw"].values, nc["freq"].values)[..., None]
    it, il, ist = data["index"]
    lead = np.asarray(leads)[il]
    now = (e_all[it, il, ist], d_all[it, il, ist])
    pairs = []
    for lag in lags:
        same_run = lead - lag >= 0
        prev_lead = np.where(same_run, lead - lag, lead - lag + 24)
        prev_time = np.where(same_run, times[it], times[it] - np.timedelta64(24, "h"))
        jt = np.searchsorted(times, prev_time)
        ok = (jt < len(times)) & (prev_lead <= max(leads))
        ok[ok] &= times[jt[ok]] == prev_time[ok]
        jl = np.searchsorted(leads, np.clip(prev_lead, 0, max(leads)))
        e_prev, d_prev = np.full_like(now[0], np.nan), np.full_like(now[1], np.nan)
        e_prev[ok], d_prev[ok] = e_all[jt[ok], jl[ok], ist[ok]], d_all[jt[ok], jl[ok], ist[ok]]

        # before the issue time: nowcast at the earlier valid time, where it exists
        valid_prev = times[it] + (lead - lag).astype("timedelta64[h]")
        jn = np.clip(np.searchsorted(nc_times, valid_prev), 0, len(nc_times) - 1)
        use = ~same_run & (nc_times[jn] == valid_prev)
        use[use] &= np.isfinite(e_nc[jn[use], ist[use]]).all(1)
        e_prev[use], d_prev[use] = e_nc[jn[use], ist[use]], d_nc[jn[use], ist[use]]
        print(f"  history lag {lag} h: {(~same_run).mean():.1%} before the issue time, "
              f"of which {use.sum() / max((~same_run).sum(), 1):.1%} from the nowcast")
        pairs.append((now, (e_prev, d_prev)))
    data["history_pairs"] = pairs
    return pairs


def history_features(data: dict) -> np.ndarray:
    """Per bin, per lag: change of sqrt(E) and of the energy-weighted direction vectors (valid time minus
    `lag` hours earlier). Missing earlier spectra (no previous run) -> 0, i.e. "no change"."""
    blocks = []
    for (e_now, d_now), (e_prev, d_prev) in _history_pairs(data):
        s_now, c_now = _weighted_vector(e_now, d_now)
        s_prev, c_prev = _weighted_vector(e_prev, d_prev)
        blocks.append(np.nan_to_num(np.hstack([np.sqrt(e_now) - np.sqrt(e_prev), s_now - s_prev, c_now - c_prev])))
    return np.hstack(blocks)


def _band_summary(energy: np.ndarray, direction: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Per period band: Hm0, energy-weighted mean frequency and the unit vector of the energy-weighted mean
    direction -> (n, 4 * n_bands); mean frequency and direction are NaN in an empty band."""
    df = float(np.median(np.diff(f)))
    rad = np.deg2rad(np.nan_to_num(direction))
    out = []
    for lo, hi in PERIOD_BANDS.values():
        sel = (f >= lo) & (f < hi)
        e = np.nan_to_num(energy[:, sel])
        m0 = e.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            f_mean = (e * f[sel]).sum(1) / m0
            vs, vc = (e * np.sin(rad[:, sel])).sum(1), (e * np.cos(rad[:, sel])).sum(1)
            norm = np.hypot(vs, vc)
            out += [4 * np.sqrt(m0 * df), np.where(m0 > 0, f_mean, np.nan), vs / norm, vc / norm]
    hm0_missing = ~np.isfinite(energy).all(1)
    arr = np.column_stack(out)
    arr[hm0_missing] = np.nan
    return arr


def _station_location_rows(data: dict):
    """The station location table and, per data row, the index of its station in it."""
    loc = load_station_location()
    names = loc["station"].values
    st = data["meta"]["station"].to_numpy()
    order = np.argsort(names)
    k = order[np.clip(np.searchsorted(names, st, sorter=order), 0, len(names) - 1)]
    assert (names[k] == st).all(), "stations missing from the location table: rebuild it"
    return loc, k


def _wind_from(data: dict) -> tuple[np.ndarray, np.ndarray]:
    m = data["meta"]
    u, v = m["gfs_u10"].to_numpy(), m["gfs_v10"].to_numpy()
    return np.rad2deg(np.arctan2(-u, -v)) % 360, np.hypot(u, v)


def fetch_features(data: dict) -> np.ndarray:
    """Fetch mismatch between reality and GFS (forecast/location.py): in the wind direction log10(D_real /
    D_GFS), the expected log10 Hs ratio of fetch-limited growth and log10(g D_real / U^2); in the mean
    direction of the < 6 s and 6-10 s bands log10(D_real / D_GFS). 5 features, all 0 where GFS's coastline is
    right, so not a fixed station identifier."""
    loc, k = _station_location_rows(data)
    real, gfs = loc["distance_to_land"].values[k], loc["gfs_distance_to_land"].values[k]
    wind_from, speed = _wind_from(data)
    xr_w, xg_w = lookup(real, wind_from), lookup(gfs, wind_from)
    out = [np.log10(xr_w / xg_w), fetch_limited_log_ratio(xr_w, xg_w, speed),
           np.log10(9.81 * xr_w * 1e3 / np.maximum(speed, 1.0) ** 2)]
    summary = _band_summary(data["gfs_energy"], data["gfs_direction"], data["freq"])
    for band in ("Tp < 6 s", "6-10 s"):
        b = list(PERIOD_BANDS).index(band)
        theta = np.rad2deg(np.arctan2(summary[:, 4 * b + 2], summary[:, 4 * b + 3])) % 360
        out.append(np.log10(lookup(real, theta) / lookup(gfs, theta)))
    return np.column_stack(out)


def location_features(data: dict) -> np.ndarray:
    """Location as seen by the incoming waves (forecast/location.py): per period band of the GFS spectrum,
    log10 of the over-water distance to land in the band's mean direction, tanh(k h) at the station and
    tanh(k h) along the incoming ray at the band's mean frequency; plus log10 of the fetch in the wind
    direction. 10 features; NaN in an empty band (imputed). No fixed per-station values."""
    loc, k = _station_location_rows(data)
    dist, ray, h0 = loc["distance_to_land"].values[k], loc["ray_depth"].values[k], loc["depth"].values[k]
    summary = _band_summary(data["gfs_energy"], data["gfs_direction"], data["freq"])   # (n, 4 per band)
    out = []
    for b in range(len(PERIOD_BANDS)):
        f_mean, vs, vc = summary[:, 4 * b + 1], summary[:, 4 * b + 2], summary[:, 4 * b + 3]
        theta = np.rad2deg(np.arctan2(vs, vc)) % 360
        out += [np.log10(lookup(dist, theta)),
                tanh_kh(f_mean, h0),
                tanh_kh(f_mean, lookup(ray, theta))]
    m = data["meta"]
    wind_from = np.rad2deg(np.arctan2(-m["gfs_u10"].to_numpy(), -m["gfs_v10"].to_numpy())) % 360
    out.append(np.log10(lookup(dist, wind_from)))
    return np.column_stack(out)


def history_band_features(data: dict) -> np.ndarray:
    """Per period band and lag: change of Hm0, mean frequency (dispersion drift) and mean direction vector.
    36 features for 3 bands and 3 lags. Missing values (empty band, no previous run) -> 0 ("no change")."""
    f = data["freq"]
    blocks = [np.nan_to_num(_band_summary(*now, f) - _band_summary(*prev, f)) for now, prev in _history_pairs(data)]
    return np.hstack(blocks)


def features(data: dict, feature_set: str = "gfs") -> np.ndarray:
    """feature_set: "gfs", optionally with "+obs" (issue-time GFS error) and/or the change of the GFS
    spectrum over HISTORY_LAGS: "+history" (per bin) or "+history-bands" (per period band).
    "+station" adds a one-hot station identifier, "station" alone is only that (per-station mean bias). Only
    for diagnostics: a station identity does not exist at grid cells without a buoy.
    "+location" adds depth and exposure as seen by the incoming bands (location_features): they can be
    computed for any grid cell. "+fetch" adds the fetch mismatch between reality and GFS (fetch_features)."""
    parts = feature_set.split("+")
    blocks = [_gfs_features(data)] if parts[0] == "gfs" else []
    if "station" in parts:
        st = data["meta"]["station"].to_numpy()
        blocks.append((st[:, None] == np.unique(st)[None, :]).astype(float))
    if "location" in parts:
        blocks.append(location_features(data))
    if "fetch" in parts:
        blocks.append(fetch_features(data))
    if "history" in parts:
        blocks.append(history_features(data))
    if "history-bands" in parts:
        blocks.append(history_band_features(data))
    if "obs" in parts:                       # last: PersistIssueError reads the issue error from the end
        blocks += [issue_error(data), data["has_issue_obs"][:, None].astype(float)]
    return np.hstack(blocks)


def _gfs_features(data: dict) -> np.ndarray:
    eg, dg = data["gfs_energy"], data["gfs_direction"]
    m = data["meta"]
    gsin, gcos = _weighted_vector(eg, dg)
    wind_from = np.deg2rad(np.rad2deg(np.arctan2(-m["gfs_u10"], -m["gfs_v10"])) % 360)
    context = np.column_stack([
        m["gfs_htsgw"], np.hypot(m["gfs_u10"], m["gfs_v10"]), np.sin(wind_from), np.cos(wind_from),
        m["gfs_perpw"], np.sin(np.deg2rad(m["gfs_dirpw"])), np.cos(np.deg2rad(m["gfs_dirpw"])), m["lead_hours"],
    ])
    return np.hstack([np.log10(eg + EPS), np.sqrt(eg), gsin, gcos, context])


def targets(data: dict) -> np.ndarray:
    return residual(data["buoy_energy"], data["buoy_direction"], data["gfs_energy"], data["gfs_direction"])


def residual(eb, db, eg, dg) -> np.ndarray:
    """Per bin: sqrt(E_buoy) - sqrt(E_GFS), then the differences of the energy-weighted direction vectors."""
    bsin, bcos = _weighted_vector(eb, db)
    gsin, gcos = _weighted_vector(eg, dg)
    return np.hstack([np.sqrt(eb) - np.sqrt(eg), bsin - gsin, bcos - gcos])


def corrected_spectrum(data: dict, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Model output (residuals) -> corrected energy and direction per bin."""
    nf = len(data["freq"])
    eg, dg = data["gfs_energy"], data["gfs_direction"]
    energy = np.clip(np.sqrt(eg) + y[:, :nf], 0, None) ** 2
    gsin, gcos = _weighted_vector(eg, dg)
    direction = np.rad2deg(np.arctan2(gsin + y[:, nf:2 * nf], gcos + y[:, 2 * nf:])) % 360
    return energy, direction


# ---------------------------------------------------------------------------
# 2. CROSS-VALIDATION
# ---------------------------------------------------------------------------

def cross_validate(data: dict, X: dict[str, np.ndarray], Y: np.ndarray, scheme: str) -> dict[str, np.ndarray]:
    """X: feature matrix per feature set. Every model is trained on its own feature set, all on the same folds."""
    kwargs = dict(n_blocks=N_TIME_BLOCKS, gap_hours=TIME_GAP_HOURS) if scheme == "blocked-time" else {}
    pred = {name: np.full(Y.shape, np.nan) for name in MODELS}
    for k, (is_train, is_test) in enumerate(FOLDS[scheme](data["meta"], **kwargs)):
        for name, (make, feature_set) in MODELS.items():
            X_set = X[feature_set]
            model = make()
            model.fit(X_set[is_train], Y[is_train])
            # no corrections outside what was seen in training (the MLP extrapolates in unseen seasons)
            pred[name][is_test] = np.clip(model.predict(X_set[is_test]), Y[is_train].min(0), Y[is_train].max(0))
        print(f"  {scheme}: fold {k + 1} done ({int(is_test.sum())} test spectra)")
    return pred


# ---------------------------------------------------------------------------
# 3. SCORES
# ---------------------------------------------------------------------------

def band_hm0(energy: np.ndarray, f: np.ndarray, lo: float, hi: float) -> np.ndarray:
    df = float(np.median(np.diff(f)))
    sel = (f >= lo) & (f < hi)
    return 4 * np.sqrt(energy[:, sel].sum(1) * df)


def spectrum_scores(data: dict, energy: np.ndarray, direction: np.ndarray, rows=slice(None)) -> dict:
    f, eb, db = data["freq"], data["buoy_energy"][rows], data["buoy_direction"][rows]
    energy, direction = energy[rows], direction[rows]
    out = {"rmse_log10_energy": rmse(np.log10(energy + EPS) - np.log10(eb + EPS))}
    out["rmse_hm0"] = rmse(band_hm0(energy, f, 0, 1) - band_hm0(eb, f, 0, 1))
    for band, (lo, hi) in PERIOD_BANDS.items():
        out[f"rmse_hm0 {band}"] = rmse(band_hm0(energy, f, lo, hi) - band_hm0(eb, f, lo, hi))
    swell = f < SWELL_F_MAX
    w = eb[:, swell]
    diff = np.abs(wrap_angle(direction[:, swell] - db[:, swell]))
    ok = np.isfinite(diff)
    out["swell_dir_error"] = float(np.sum(np.where(ok, w * diff, 0)) / np.sum(np.where(ok, w, 0)))
    return out


def partition_all(f, energy, direction) -> list[np.ndarray]:
    kw = fcfg.BUOY_PARTITION_KWARGS
    out = []
    for e, d in zip(energy, direction):
        parts = partition_spectrum(f, e, d, **kw)
        out.append(np.array([[p["hs"], p["fp"], p["dir"]] for p in parts]).reshape(-1, 3))
    return out


def system_scores(pred_systems: list, buoy_systems: list) -> dict:
    """Match every prediction against the buoy with the dataset's matching settings; score systems >= HS_MIN."""
    hits = n_buoy = false = n_pred = 0
    d_hs, d_tp, d_dir, swell = [], [], [], []
    for a, b in zip(pred_systems, buoy_systems):
        pairs, ua, ub = match_partitions(a, b, **fcfg.MATCH_KWARGS)
        big_b = b[:, 0] >= HS_MIN if len(b) else np.zeros(0, bool)
        big_a = a[:, 0] >= HS_MIN if len(a) else np.zeros(0, bool)
        n_buoy += big_b.sum(); hits += sum(1 for _, j in pairs if big_b[j])
        n_pred += big_a.sum(); false += sum(1 for i in ua if big_a[i])
        for i, j in pairs:
            if max(a[i, 0], b[j, 0]) >= HS_MIN:
                d_hs.append(a[i, 0] - b[j, 0]); d_tp.append(1 / a[i, 1] - 1 / b[j, 1])
                d_dir.append(abs(wrap_angle(a[i, 2] - b[j, 2]))); swell.append(1 / b[j, 1] >= 6)
    d_hs, d_tp, d_dir, swell = map(np.asarray, (d_hs, d_tp, d_dir, swell))
    return {"hit_rate": hits / n_buoy, "false_alarm_rate": false / max(n_pred, 1), "n_pairs": len(d_hs),
            "rmse_hs": rmse(d_hs), "bias_hs": float(d_hs.mean()), "rmse_tp": rmse(d_tp), "mae_dir": float(d_dir.mean()),
            "rmse_hs_swell": rmse(d_hs[swell]), "rmse_tp_swell": rmse(d_tp[swell]), "mae_dir_swell": float(d_dir[swell].mean())}


def skill_per_bin(data: dict, energy: np.ndarray) -> np.ndarray:
    eb, eg = data["buoy_energy"], data["gfs_energy"]
    err_m = np.log10(energy + EPS) - np.log10(eb + EPS)
    err_g = np.log10(eg + EPS) - np.log10(eb + EPS)
    return 1 - np.sqrt(np.mean(err_m ** 2, 0)) / np.sqrt(np.mean(err_g ** 2, 0))


def plot(data: dict, results: dict, examples: int = 4, seed: int = 0):
    f = data["freq"]
    fig, axes = plt.subplots(2, max(examples, 2), figsize=(4.2 * max(examples, 2), 7.5))
    ax = axes[0, 0]
    for scheme, (energy, _) in results.items():
        ax.plot(f, skill_per_bin(data, energy), "o-", ms=3, label=scheme)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("skill log10 energy vs GFS"); ax.legend(); ax.grid(alpha=0.3)
    ax = axes[0, 1]
    m = data["meta"]
    for scheme, (energy, _) in results.items():
        e = pd.Series(np.abs(band_hm0(energy, f, 0, 1 / 6) - band_hm0(data["buoy_energy"], f, 0, 1 / 6)) ** 2)
        g = pd.Series(np.abs(band_hm0(data["gfs_energy"], f, 0, 1 / 6) - band_hm0(data["buoy_energy"], f, 0, 1 / 6)) ** 2)
        lead = m["lead_hours"].values
        ax.plot(sorted(set(lead)), np.sqrt(e.groupby(lead).mean()), "o-", ms=3, label=f"corrected ({scheme})")
    ax.plot(sorted(set(lead)), np.sqrt(g.groupby(lead).mean()), "o-", ms=3, color="k", label="GFS")
    ax.set_xlabel("lead time (h)"); ax.set_ylabel("RMSE Hm0 for Tp > 6 s (m)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    for a in axes[0, 2:]:
        a.set_visible(False)

    energy, _ = results["leave-one-group"]
    swell_energy = band_hm0(data["buoy_energy"], f, 0, 1 / 6)
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.flatnonzero(swell_energy > 0.5), size=examples, replace=False)
    for a, i in zip(axes[1], idx):
        a.step(f, data["buoy_energy"][i], where="mid", color="k", label="buoy")
        a.plot(f, data["gfs_energy"][i], color="grey", ls="--", label="GFS")
        a.plot(f, energy[i], color="tab:blue", label="corrected")
        t, st, lead = m.index[i]
        a.set_title(f"{st} {pd.Timestamp(t):%Y-%m-%d} +{lead} h", fontsize=9)
        a.set_xlim(0.03, 0.4); a.set_xlabel("frequency (Hz)")
    axes[1, 0].set_ylabel("E (m²/Hz)"); axes[1, 0].legend(fontsize=8)
    fig.suptitle("Spectrum correction (out-of-sample); examples from leave-one-group with swell Hm0 > 0.5 m")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. RUN
# ---------------------------------------------------------------------------

def swell_hm0_by_lead(data: dict, energy: np.ndarray) -> dict:
    """RMSE of swell Hm0 (Tp > 6 s) against the buoy at the SCORE_LEADS lead times."""
    f, lead = data["freq"], data["meta"]["lead_hours"].to_numpy()
    err = band_hm0(energy, f, 0, SWELL_F_MAX) - band_hm0(data["buoy_energy"], f, 0, SWELL_F_MAX)
    return {f"lead {h} h": rmse(err[lead == h]) for h in SCORE_LEADS}


def run():
    data = load_data()
    X = {feature_set: features(data, feature_set) for feature_set in ("gfs", "gfs+obs")}
    Y = targets(data)
    f = data["freq"]
    scored = data["meta"]["lead_hours"].to_numpy() >= EVAL_MIN_LEAD
    print(f"{len(Y)} spectra (run x lead x station with a buoy spectrum), {len(f)} frequency bins, "
          f"{X['gfs'].shape[1]} GFS features (+{X['gfs+obs'].shape[1] - X['gfs'].shape[1]} issue-time measurement), "
          f"{Y.shape[1]} targets; issue-time measurement available for {data['has_issue_obs'].mean():.0%}; "
          f"scores over lead times >= {EVAL_MIN_LEAD} h ({int(scored.sum())} spectra)")

    print("Partitioning buoy and GFS spectra for the system scores ...")
    buoy_systems = [s for s, k in zip(partition_all(f, data["buoy_energy"], data["buoy_direction"]), scored) if k]
    gfs_systems = [s for s, k in zip(partition_all(f, data["gfs_energy"], data["gfs_direction"]), scored) if k]
    gfs_spec = spectrum_scores(data, data["gfs_energy"], data["gfs_direction"], rows=scored)
    gfs_sys = system_scores(gfs_systems, buoy_systems)
    gfs_lead = swell_hm0_by_lead(data, data["gfs_energy"])

    corrected, spec_rows, sys_rows, lead_rows = {}, [], [], []
    for scheme in FOLDS:
        pred = cross_validate(data, X, Y, scheme)
        spec_rows.append(dict(scheme=scheme, model="GFS", **gfs_spec))
        sys_rows.append(dict(scheme=scheme, model="GFS", **gfs_sys))
        lead_rows.append(dict(scheme=scheme, model="GFS", **gfs_lead))
        for name, y in pred.items():
            energy, direction = corrected_spectrum(data, y)
            corrected[(scheme, name)] = (energy.astype(np.float32), direction.astype(np.float32))
            spec_rows.append(dict(scheme=scheme, model=name, **spectrum_scores(data, energy, direction, rows=scored)))
            lead_rows.append(dict(scheme=scheme, model=name, **swell_hm0_by_lead(data, energy)))
            if name in SYSTEM_MODELS:
                print(f"  partitioning {name} spectra ({scheme}) ...")
                systems_pred = partition_all(f, energy[scored], direction[scored])
                sys_rows.append(dict(scheme=scheme, model=name, **system_scores(systems_pred, buoy_systems)))

    spec = pd.DataFrame(spec_rows).set_index(["scheme", "model"]).round(3)
    systems = pd.DataFrame(sys_rows).set_index(["scheme", "model"]).round(3)
    by_lead = pd.DataFrame(lead_rows).set_index(["scheme", "model"]).round(3)
    print(f"\nSpectrum scores against the buoy, leads >= {EVAL_MIN_LEAD} h "
          "(lower is better; Hm0 in m, direction in degrees, energy-weighted):")
    print(spec.to_string())
    hm0_cols = [c for c in spec.columns if c.startswith("rmse")]
    skill = 1 - spec[hm0_cols].div(spec.xs("GFS", level="model")[hm0_cols].reindex(spec.index, level="scheme"))
    print("\nSkill vs GFS (1 - RMSE_model / RMSE_GFS):")
    print(skill.round(3).to_string())
    print(f"\nWave systems >= {HS_MIN} m after partitioning, matched against the buoy systems (swell: buoy Tp >= 6 s):")
    print(systems.to_string())
    print("\nRMSE of swell Hm0 (Tp > 6 s) per lead time (m):")
    print(by_lead.to_string())

    save_results(data, corrected, spec, skill, systems, by_lead)
    results = {scheme: corrected[(scheme, SYSTEM_MODELS[-1])] for scheme in FOLDS}   # for the figure
    return data, results, spec, systems


def save_results(data: dict, corrected: dict, spec: pd.DataFrame, skill: pd.DataFrame, systems: pd.DataFrame,
                 by_lead: pd.DataFrame):
    """Out-of-sample corrected spectra and score tables -> fcfg.SPECTRUM_MODEL_DIR."""
    out_dir = fcfg.SPECTRUM_MODEL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    schemes = list(FOLDS)
    models = list(MODELS)
    meta = data["meta"]
    shape = (len(schemes), len(models), len(meta), len(data["freq"]))
    energy = np.full(shape, np.nan, dtype=np.float32)
    direction = np.full(shape, np.nan, dtype=np.float32)
    for (scheme, name), (e, d) in corrected.items():
        energy[schemes.index(scheme), models.index(name)] = e
        direction[schemes.index(scheme), models.index(name)] = d

    dims = ("spectrum", "freq")
    ds = xr.Dataset(
        {
            "corrected_energy": (("scheme", "model") + dims, energy, {"units": "m2/Hz"}),
            "corrected_direction": (("scheme", "model") + dims, direction, {"units": "degree (coming from)"}),
            "buoy_energy": (dims, data["buoy_energy"].astype(np.float32), {"units": "m2/Hz"}),
            "buoy_direction": (dims, data["buoy_direction"].astype(np.float32), {"units": "degree (coming from)"}),
            "gfs_energy": (dims, data["gfs_energy"].astype(np.float32), {"units": "m2/Hz"}),
            "gfs_direction": (dims, data["gfs_direction"].astype(np.float32), {"units": "degree (coming from)"}),
            "time": ("spectrum", meta["time"].to_numpy(), {"long_name": "issue time of the GFS run"}),
            "station": ("spectrum", meta["station"].to_numpy().astype(str)),
            "lead_hours": ("spectrum", meta["lead_hours"].to_numpy()),
            "valid_time": ("spectrum", (meta["time"] + pd.to_timedelta(meta["lead_hours"], unit="h")).to_numpy()),
            "buoy_group": ("spectrum", meta["buoy_group"].to_numpy()),
        },
        coords={"scheme": schemes, "model": models, "freq": data["freq"]},
        attrs={
            "description": "Out-of-sample corrected spectra: every spectrum predicted by a model that did not see "
                           "its station group (leave-one-group) or its time block (blocked-time)",
            "target": "sqrt(E_buoy) - sqrt(E_GFS) per bin + difference of energy-weighted direction vectors",
            "models": json.dumps({name: feature_set for name, (_, feature_set) in MODELS.items()}),
            "issue_obs": "gfs+obs models also get the GFS error at issue time (buoy - GFS at lead 0); "
                         "lead 0 is left out of the scores because there that input is the target",
            "eval_min_lead_hours": EVAL_MIN_LEAD,
            "eps": EPS, "time_gap_hours": TIME_GAP_HOURS, "n_time_blocks": N_TIME_BLOCKS,
            "partition_zarr": str(fcfg.PARTITION_ZARR),
            "buoy_partition_kwargs": json.dumps(fcfg.BUOY_PARTITION_KWARGS),
        },
    )
    path = out_dir / "oos_spectra.zarr"
    if path.exists():
        shutil.rmtree(path)
    ds.chunk({"scheme": 1, "model": 1, "spectrum": 20000, "freq": -1}).to_zarr(str(path), mode="w")

    spec.to_csv(out_dir / "spectrum_scores.csv")
    skill.round(4).to_csv(out_dir / "spectrum_skill.csv")
    systems.to_csv(out_dir / "system_scores.csv")
    by_lead.to_csv(out_dir / "swell_hm0_by_lead.csv")
    print(f"\nSaved out-of-sample spectra and score tables to {out_dir}")


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    data, results, spec, systems = run()
    plot(data, results)
    plt.show()
