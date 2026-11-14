"""
Two-stage correction of GFS wave systems: does the system exist, and how high is it?
======================================================================================

Every row is one GFS wave system at a station (forecast/preprocess.py, group "samples").
About 30% of the GFS systems have no buoy counterpart (false alarms). A single regression on
"buoy Hs - GFS Hs" with buoy Hs = 0 for false alarms learns to shrink every uncertain system,
which damages the real ones. So the problem is split:

    1. classifier  P(system exists)          target: matched with a buoy system, trained on all systems
    2. regressor   buoy Hs - GFS Hs          trained on matched systems only
    3. combined    Hs = GFS Hs + residual    if P >= threshold, else the system is removed (Hs = 0)
                   (also reported: the expected height P * corrected Hs)

Features are the GFS system itself, the other GFS systems at that moment (order-invariant summary),
bulk GFS context at the station and the lead time; no station ids or coordinates, so the model has
to learn relationships that transfer to locations without a buoy.

Validation (common/validation.py), both stages on the same folds:
    leave-one-group : unseen LOCATION (station groups held out)
    blocked-time    : unseen WEATHER; the gap around each block covers the longest lead time

Scores:
    classifier  Brier skill vs the base rate, ROC AUC, real systems kept / false alarms removed
    regressor   RMSE of matched-system Hs vs the buoy, GFS vs corrected
    combined    RMSE of system Hs vs the buoy over all systems (false alarm: buoy Hs = 0)

Run from anywhere with: python -m nedsurfcast.forecast.train_residual
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from nedsurfcast.common.angles import wrap_angle
from nedsurfcast.common.models import CLASSIFIERS, MODELS
from nedsurfcast.common.validation import FOLDS, N_TIME_BLOCKS, rmse
from nedsurfcast.forecast import config as fcfg
from nedsurfcast.forecast.preprocess import load

GRAVITY = 9.81
TIME_GAP_HOURS = max(fcfg.GFSWAVE_LEAD_HOURS) + 24   # blocked-time gap: longest lead + a day of autocorrelation
THRESHOLD = 0.5                                       # remove a system when P(exists) is below this
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
CLASSIFIER_NAMES = ["prior", "logistic", "hist-gbc"]
REGRESSOR_NAMES = ["mean-bias", "ridge", "hist-gbr"]
DETAIL = ("hist-gbc", "hist-gbr")                     # combination used for the breakdowns and the figure


# ---------------------------------------------------------------------------
# 1. TABLE
# ---------------------------------------------------------------------------

def build_table() -> pd.DataFrame:
    samples = load("samples").load().to_dataframe()
    context = load("station_times").load()
    stations = load("stations").load()

    # bulk context at the station for every sample, via the (run, lead, station) indices
    st_index = {s: i for i, s in enumerate(stations.station.values)}
    it, il = samples["time_idx"].values, samples["lead_idx"].values
    ist = samples["station"].map(st_index).values
    for v in ("gfs_htsgw", "gfs_u10", "gfs_v10", "gfs_perpw", "gfs_dirpw"):
        samples[v] = context[v].values[it, il, ist]
    samples["buoy_group"] = stations["station_group"].values[ist]

    samples["buoy_hs_target"] = samples["buoy_hs"].where(samples["matched"], 0.0)   # false alarm: no system
    samples["target"] = samples["buoy_hs"] - samples["gfs_hs"]                        # NaN for false alarms

    df = add_features(samples)
    return df.set_index(["time", "station", df.index.rename("sample")]).sort_index()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features known at forecast time; directions as sin/cos, wind as 'coming from' like the waves."""
    df = df.copy()
    df["gfs_tp"] = 1 / df["gfs_fp"]
    for name, deg in (("gfs_dir", df["gfs_dir"]), ("other_dir", df["other_dir"]), ("gfs_dirpw", df["gfs_dirpw"])):
        df[f"{name}_sin"], df[f"{name}_cos"] = np.sin(np.deg2rad(deg)), np.cos(np.deg2rad(deg))

    df["wind_speed"] = np.hypot(df["gfs_u10"], df["gfs_v10"])
    wind_from = np.rad2deg(np.arctan2(-df["gfs_u10"], -df["gfs_v10"])) % 360
    df["wind_from_sin"], df["wind_from_cos"] = np.sin(np.deg2rad(wind_from)), np.cos(np.deg2rad(wind_from))
    # angle between wind and wave system, and inverse wave age U10 cos(angle) / c_p (> ~0.8: wind sea)
    angle = wrap_angle(df["gfs_dir"] - wind_from)
    df["wind_wave_angle_cos"] = np.cos(np.deg2rad(angle))
    phase_speed = GRAVITY / (2 * np.pi * df["gfs_fp"])
    df["inverse_wave_age"] = df["wind_speed"] * df["wind_wave_angle_cos"] / phase_speed
    df["share_of_energy"] = df["gfs_hs"] ** 2 / (df["gfs_hs"] ** 2 + df["other_hs"] ** 2)
    return df


FEATURES = [
    "gfs_hs", "gfs_tp", "gfs_dir_sin", "gfs_dir_cos", "gfs_rank", "n_gfs_systems", "share_of_energy",
    "other_hs", "other_dir_sin", "other_dir_cos",
    "gfs_htsgw", "gfs_perpw", "gfs_dirpw_sin", "gfs_dirpw_cos",
    "wind_speed", "wind_from_sin", "wind_from_cos", "wind_wave_angle_cos", "inverse_wave_age",
    "lead_hours",
]


# ---------------------------------------------------------------------------
# 2. CROSS-VALIDATION OF BOTH STAGES
# ---------------------------------------------------------------------------

def folds(df: pd.DataFrame, scheme: str):
    kwargs = dict(n_blocks=N_TIME_BLOCKS, gap_hours=TIME_GAP_HOURS) if scheme == "blocked-time" else {}
    return list(FOLDS[scheme](df, **kwargs))


def cross_validate(df: pd.DataFrame, fold_masks, classifiers=CLASSIFIER_NAMES, regressors=REGRESSOR_NAMES):
    """Out-of-sample P(exists) per classifier and residual per regressor, on the same folds.

    The regressor is fitted on the matched training systems only, but predicts for every test system
    (the combination needs a corrected height for every system the classifier keeps).
    """
    X = df[FEATURES]
    exists = df["matched"].to_numpy()
    target = df["target"].to_numpy()
    prob = {name: np.full(len(df), np.nan) for name in classifiers}
    resid = {name: np.full(len(df), np.nan) for name in regressors}
    for is_train, is_test in fold_masks:
        for name in classifiers:
            clf = CLASSIFIERS[name]()
            clf.fit(X[is_train], exists[is_train])
            prob[name][is_test] = clf.predict_proba(X[is_test])[:, list(clf.classes_).index(True)]
        train_matched = is_train & exists
        for name in regressors:
            reg = MODELS[name]()
            reg.fit(X[train_matched], target[train_matched])
            resid[name][is_test] = reg.predict(X[is_test])
    return prob, resid


# ---------------------------------------------------------------------------
# 3. SCORES
# ---------------------------------------------------------------------------

def classifier_scores(df: pd.DataFrame, prob: dict) -> pd.DataFrame:
    exists = df["matched"].to_numpy()
    base = exists.mean()
    brier_ref = brier_score_loss(exists, np.full(len(exists), base))
    rows = {}
    for name, p in prob.items():
        keep = p >= THRESHOLD
        rows[name] = dict(
            brier=brier_score_loss(exists, p), brier_skill=1 - brier_score_loss(exists, p) / brier_ref,
            auc=roc_auc_score(exists, p) if np.unique(p).size > 1 else 0.5,
            real_kept=(keep & exists).sum() / exists.sum(),
            false_alarms_removed=(~keep & ~exists).sum() / (~exists).sum(),
            false_alarm_rate_after=(keep & ~exists).sum() / max(keep.sum(), 1),
        )
    out = pd.DataFrame(rows).T
    out.loc["GFS (keep all)"] = dict(brier=np.nan, brier_skill=np.nan, auc=np.nan, real_kept=1.0,
                                     false_alarms_removed=0.0, false_alarm_rate_after=(~exists).mean())
    return out.round(3)


def regressor_scores(df: pd.DataFrame, resid: dict) -> pd.DataFrame:
    m = df["matched"].to_numpy()
    gfs, buoy = df["gfs_hs"].to_numpy()[m], df["buoy_hs"].to_numpy()[m]
    swell = df["gfs_tp"].to_numpy()[m] >= 6
    err_gfs = gfs - buoy
    rows = {}
    for name, r in resid.items():
        err = np.clip(gfs + r[m], 0, None) - buoy
        rows[name] = dict(rmse_gfs=rmse(err_gfs), rmse_corrected=rmse(err), skill=1 - rmse(err) / rmse(err_gfs),
                          bias_corrected=err.mean(),
                          skill_swell=1 - rmse(err[swell]) / rmse(err_gfs[swell]),
                          skill_wind_sea=1 - rmse(err[~swell]) / rmse(err_gfs[~swell]))
    return pd.DataFrame(rows).T.round(3)


def combined_heights(df: pd.DataFrame, p: np.ndarray, r: np.ndarray, threshold: float | None) -> np.ndarray:
    """Corrected system height; threshold None = expected height P * corrected."""
    corrected = np.clip(df["gfs_hs"].to_numpy() + r, 0, None)
    return p * corrected if threshold is None else np.where(p >= threshold, corrected, 0.0)


def combined_errors(df: pd.DataFrame, p: np.ndarray, r: np.ndarray, threshold: float | None = THRESHOLD) -> pd.DataFrame:
    d = df.copy()
    d["err_gfs"] = d["gfs_hs"] - d["buoy_hs_target"]
    d["err_corrected"] = combined_heights(df, p, r, threshold) - d["buoy_hs_target"]
    return d


def combined_scores(df: pd.DataFrame, p: np.ndarray, r: np.ndarray) -> pd.DataFrame:
    rows = {}
    for thr in THRESHOLDS + [None]:
        d = combined_errors(df, p, r, thr)
        m = d["matched"]
        rows["expected P*Hs" if thr is None else f"threshold {thr}"] = dict(
            rmse_gfs=rmse(d["err_gfs"]), rmse_corrected=rmse(d["err_corrected"]),
            skill=1 - rmse(d["err_corrected"]) / rmse(d["err_gfs"]),
            skill_matched=1 - rmse(d.loc[m, "err_corrected"]) / rmse(d.loc[m, "err_gfs"]),
            skill_false_alarms=1 - rmse(d.loc[~m, "err_corrected"]) / rmse(d.loc[~m, "err_gfs"]),
        )
    return pd.DataFrame(rows).T.round(3)


def score_table(d: pd.DataFrame, grouper) -> pd.DataFrame:
    g = d.groupby(grouper, observed=True)
    out = pd.DataFrame({
        "n": g.size(),
        "rmse_gfs": g["err_gfs"].apply(rmse),
        "rmse_corrected": g["err_corrected"].apply(rmse),
        "bias_gfs": g["err_gfs"].mean(),
        "bias_corrected": g["err_corrected"].mean(),
    })
    out["skill"] = 1 - out["rmse_corrected"] / out["rmse_gfs"]
    return out.round(3)


def breakdown(d: pd.DataFrame, label: str):
    print(f"\n--- combined {DETAIL[0]} + {DETAIL[1]}, threshold {THRESHOLD}, {label} ---")
    d = d.assign(
        system=np.where(d["matched"], "matched", "false alarm"),
        kind=np.where(d["gfs_tp"] >= 6, "Tp >= 6 s", "Tp < 6 s"),
        gfs_size=pd.cut(d["gfs_hs"], [0, 0.3, 0.5, 1, 2, 10], right=False),
    )
    print("by period and system:"); print(score_table(d, ["kind", "system"]).to_string())
    print("\nby GFS system height (m):"); print(score_table(d, "gfs_size").to_string())
    print("\nby station:"); print(score_table(d, d.index.get_level_values("station")).to_string())
    by_lead = score_table(d, "lead_hours")
    print("\nby lead time:"); print(by_lead.loc[[l for l in (0, 24, 48, 72) if l in by_lead.index]].to_string())


def plot_by_lead(results: dict):
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4), squeeze=False, sharey=True)
    for ax, (scheme, d) in zip(axes[0], results.items()):
        t = score_table(d, "lead_hours")
        tm = score_table(d[d["matched"]], "lead_hours")
        ax.plot(t.index, t["rmse_gfs"], "o-", color="k", label="GFS, all systems")
        ax.plot(t.index, t["rmse_corrected"], "o-", color="tab:blue", label="two-stage, all systems")
        ax.plot(tm.index, tm["rmse_gfs"], "s--", color="grey", label="GFS, matched systems")
        ax.plot(tm.index, tm["rmse_corrected"], "s--", color="tab:orange", label="two-stage, matched systems")
        ax.set_title(scheme)
        ax.set_xlabel("lead time (h)")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("RMSE system Hs vs buoy (m)")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"Wave-system height error per lead time (out-of-sample, {DETAIL[0]} + {DETAIL[1]}, threshold {THRESHOLD})")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. RUN
# ---------------------------------------------------------------------------

def run():
    df = build_table()
    print(f"{len(df)} GFS systems ({int(df['matched'].sum())} matched, {int((~df['matched']).sum())} false alarms), "
          f"{df.index.get_level_values('station').nunique()} stations in {df['buoy_group'].nunique()} groups, "
          f"{len(FEATURES)} features")

    details = {}
    for scheme in FOLDS:
        prob, resid = cross_validate(df, folds(df, scheme))
        print(f"\n===== {scheme} =====")
        print(f"\nStage 1, classifier P(system exists), threshold {THRESHOLD}:")
        print(classifier_scores(df, prob).to_string())
        print("\nStage 2, regressor on matched systems (buoy Hs - GFS Hs):")
        print(regressor_scores(df, resid).to_string())
        print(f"\nCombined {DETAIL[0]} + {DETAIL[1]}, all systems (false alarm: buoy Hs = 0):")
        print(combined_scores(df, prob[DETAIL[0]], resid[DETAIL[1]]).to_string())
        details[scheme] = combined_errors(df, prob[DETAIL[0]], resid[DETAIL[1]])

    for scheme, d in details.items():
        breakdown(d, scheme)
    return details


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    details = run()
    plot_by_lead(details)
    plt.show()
