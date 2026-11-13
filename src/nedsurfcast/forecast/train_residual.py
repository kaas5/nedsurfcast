"""
Can GFS wave-system errors be predicted? Residual models on the partition dataset
===================================================================================

Every row is one GFS wave system at a station (forecast/preprocess.py, group "samples"):

    target   = buoy Hs - GFS Hs of that system
               (false alarm: the buoy has no matching system, so buoy Hs = 0 and target = -GFS Hs)
    features = the GFS system itself, the other GFS systems at that moment (order-invariant summary),
               bulk GFS context at the station (HTSGW, wind, peak period/direction) and the lead time

No station ids or coordinates are features, so whatever the model learns is a relationship between
forecast conditions and GFS error, which is what lets it correct locations without a buoy.

Two validation schemes (common/validation.py):
    leave-one-group : does the correction transfer to an unseen LOCATION? (station groups held out)
    blocked-time    : does it transfer to unseen WEATHER? Blocks of issue times; the gap around each
                      block covers the longest lead time, so no valid time is in both train and test.

Scores: RMSE of the system height against the buoy, uncorrected (GFS) vs corrected
(GFS + predicted residual, clipped at 0 m). Skill = 1 - RMSE_corrected / RMSE_GFS.

Run from anywhere with: python -m nedsurfcast.forecast.train_residual
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nedsurfcast.common.angles import wrap_angle
from nedsurfcast.common.models import MODELS
from nedsurfcast.common.validation import N_TIME_BLOCKS, SCHEMES, rmse
from nedsurfcast.forecast import config as fcfg
from nedsurfcast.forecast.preprocess import load

INCLUDE_FALSE_ALARMS = True     # False: only matched systems (the target is then "how wrong is the height")
GRAVITY = 9.81
TIME_GAP_HOURS = max(fcfg.GFSWAVE_LEAD_HOURS) + 24   # blocked-time gap: longest lead + a day of autocorrelation


# ---------------------------------------------------------------------------
# 1. TABLE
# ---------------------------------------------------------------------------

def build_table(include_false_alarms: bool = INCLUDE_FALSE_ALARMS) -> pd.DataFrame:
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

    if not include_false_alarms:
        samples = samples[samples["matched"]]
    samples["buoy_hs_target"] = samples["buoy_hs"].where(samples["matched"], 0.0)
    samples["target"] = samples["buoy_hs_target"] - samples["gfs_hs"]

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
# 2. VALIDATION
# ---------------------------------------------------------------------------

def predict(df: pd.DataFrame, scheme: str, model_name: str) -> pd.Series:
    evaluate = SCHEMES[scheme]
    kwargs = dict(n_blocks=N_TIME_BLOCKS, gap_hours=TIME_GAP_HOURS) if scheme == "blocked-time" else {}
    return evaluate(df, FEATURES, MODELS[model_name], **kwargs)


def errors(df: pd.DataFrame, pred: pd.Series) -> pd.DataFrame:
    """Height errors against the buoy before and after correction; corrected heights cannot go below 0 m."""
    ok = pred.notna()
    d = df.loc[ok].copy()
    d["err_gfs"] = d["gfs_hs"] - d["buoy_hs_target"]
    d["err_corrected"] = (d["gfs_hs"] + pred[ok]).clip(lower=0) - d["buoy_hs_target"]
    return d


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
    print(f"\n--- {label} ---")
    d = d.assign(
        system=np.where(d["matched"], "matched", "false alarm"),
        kind=np.where(d["gfs_tp"] >= 6, "Tp >= 6 s", "Tp < 6 s"),
        gfs_size=pd.cut(d["gfs_hs"], [0, 0.3, 0.5, 1, 2, 10], right=False),
    )
    print("by station:")
    print(score_table(d, d.index.get_level_values("station")).to_string())
    print("\nmatched systems vs false alarms:"); print(score_table(d, "system").to_string())
    print("\nby period (swell-like vs wind-sea-like):"); print(score_table(d, "kind").to_string())
    print("\nby GFS system height (m):"); print(score_table(d, "gfs_size").to_string())
    by_lead = score_table(d, "lead_hours")
    print("\nby lead time:"); print(by_lead.loc[[l for l in (0, 24, 48, 72) if l in by_lead.index]].to_string())


def plot_by_lead(results: dict):
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4), squeeze=False, sharey=True)
    for ax, (scheme, d) in zip(axes[0], results.items()):
        t = score_table(d, "lead_hours")
        ax.plot(t.index, t["rmse_gfs"], "o-", color="k", label="GFS")
        ax.plot(t.index, t["rmse_corrected"], "o-", color="tab:blue", label="GFS + hist-gbr correction")
        ax.set_title(scheme)
        ax.set_xlabel("lead time (h)")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("RMSE system Hs vs buoy (m)")
    axes[0][0].legend()
    fig.suptitle("Wave-system height error per lead time (out-of-sample)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. RUN
# ---------------------------------------------------------------------------

def run(model_names=("mean-bias", "ridge", "hist-gbr"), detail_model: str = "hist-gbr"):
    df = build_table()
    print(f"{len(df)} samples ({int(df['matched'].sum())} matched, {int((~df['matched']).sum())} false alarms), "
          f"{df.index.get_level_values('station').nunique()} stations in {df['buoy_group'].nunique()} groups, "
          f"{len(FEATURES)} features")
    print(f"Target buoy Hs - GFS Hs: mean {df['target'].mean():+.3f} m, std {df['target'].std():.3f} m\n")

    rows, details = [], {}
    for scheme in SCHEMES:
        for name in model_names:
            d = errors(df, predict(df, scheme, name))
            r_gfs, r_cor = rmse(d["err_gfs"]), rmse(d["err_corrected"])
            m = d[d["matched"]]
            rows.append(dict(scheme=scheme, model=name, n=len(d), rmse_gfs=r_gfs, rmse_corrected=r_cor,
                             skill=1 - r_cor / r_gfs, bias_corrected=d["err_corrected"].mean(),
                             skill_matched=1 - rmse(m["err_corrected"]) / rmse(m["err_gfs"])))
            print(f"{scheme:<16} {name:<10} RMSE GFS {r_gfs:.3f} -> {r_cor:.3f} m  skill {1 - r_cor / r_gfs:+.3f}")
            if name == detail_model:
                details[scheme] = d
    summary = pd.DataFrame(rows).set_index(["scheme", "model"]).round(3)
    print("\nSummary (skill_matched = skill on matched systems only):")
    print(summary.to_string())
    for scheme, d in details.items():
        breakdown(d, f"{detail_model}, {scheme}")
    return summary, details


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    summary, details = run()
    plot_by_lead(details)
    plt.show()
