"""
Model inputs and targets of the spectrum correction model, from the processed dataset
=======================================================================================

Reads the groups written by forecast/preprocess.py (fcfg.PARTITION_ZARR) and prepares everything the models
see. Nothing here is written to disk.

    load_data()     every (run, lead time, station) with a buoy spectrum: buoy spectrum, GFS spectrum rebuilt
                    from its partitions, GFS bulk context, GFS's own point spectrum where it exists, and a meta
                    table (time, station, lead_hours, buoy_group) for the folds and the score tables
    features()      model input for a feature set, e.g. "gfs" or "gfs+history-bands+location+fetch"
    targets()       per bin: sqrt(E_buoy) - sqrt(E_GFS) and the difference of the energy-weighted direction
                    vectors (buoy - GFS); 0 = the uncorrected GFS spectrum
    corrected_spectrum()  the inverse: GFS spectrum + model output -> corrected energy and direction

Feature blocks (only GFS-derived input: the correction must also work at grid cells without a buoy):
    gfs             log10 energy, sqrt(E) and energy-weighted direction vectors per bin of the rebuilt GFS
                    spectrum, plus GFS HTSGW, wind, peak period/direction and the lead time
    history-bands   per period band and lag (HISTORY_LAGS): change of Hm0, mean frequency and mean direction
                    of the rebuilt GFS spectrum since `lag` hours earlier. From the same run, or before the
                    issue time from the GFS nowcast (f000/f003 of the 06/12/18z runs): known at the issue time
    location        per period band, as seen from the band's mean direction and frequency: log10 distance to
                    land, tanh(k h) at the buoy and along the incoming ray; log10 fetch in the wind direction.
                    Functions of the incoming waves, so not a fixed station identifier
    fetch           fetch mismatch between reality and GFS's 0.25 deg coastline: log10(D_real / D_GFS) in the
                    wind direction and the < 6 s and 6-10 s band directions, the expected log10 Hs ratio of
                    fetch-limited growth and log10(g D_real / U^2)
The station name is only used to look up the station's row in the location table, never as a feature.
"""

import numpy as np
import pandas as pd

from nedsurfcast.forecast import config as fcfg
from nedsurfcast.forecast.preprocess import load

EPS = 1e-3                      # m2/Hz added before logs and in the direction weights
PERIOD_BANDS = {"T>10 s": (0.0, 0.1), "6-10 s": (0.1, 1 / 6), "T<6 s": (1 / 6, 1.0)}   # Hz, [lo, hi)
HISTORY_LAGS = (6, 12, 24)      # h
G = 9.81
FULLY_DEVELOPED = 2.3e4         # dimensionless fetch g X / U^2 beyond which wind sea stops growing


# ---------------------------------------------------------------------------
# 1. DATA
# ---------------------------------------------------------------------------

def load_data() -> dict:
    """Every (run, lead, station) with a buoy spectrum -> arrays (n, n_freq) and a meta table (n rows).
    Only the stations in fcfg.RWS_SPECTRA_STATIONS (the zarr may hold stations that were dropped later)."""
    spec = load("spectra").load()
    st = load("station_times").load()
    stations = load("stations").load()

    use = np.isin(spec["station"].values, fcfg.RWS_SPECTRA_STATIONS)
    it, il, ist = np.nonzero(st["has_buoy"].values & use)         # (time, lead, station)
    take = lambda a: a[it, il, ist]
    data = {v: take(spec[v].values).astype(float)
            for v in ("buoy_energy", "buoy_direction", "gfs_energy", "gfs_direction")}
    data["index"] = (it, il, ist)                                 # into (time, lead_hours, station)
    data["freq"] = spec["freq"].values
    meta = pd.DataFrame({
        "time": spec["time"].values[it], "station": spec["station"].values[ist],
        "lead_hours": spec["lead_hours"].values[il], "buoy_group": stations["station_group"].values[ist],
    })
    for v in ("gfs_htsgw", "gfs_u10", "gfs_v10", "gfs_perpw", "gfs_dirpw"):
        meta[v] = take(st[v].values)
    data["meta"] = meta
    _attach_point_spectra(data)
    return data


def _attach_point_spectra(data: dict):
    """GFS's own forecast spectrum (group gfs_station_spectra) aligned with the rows: point_energy and
    point_direction (NaN where there is none) and the boolean has_point."""
    ps = load("gfs_station_spectra").load()
    m = data["meta"]
    times = ps["time"].values
    it = np.searchsorted(times, m["time"].to_numpy())
    il = np.searchsorted(ps["lead_hours"].values, m["lead_hours"].to_numpy())
    names = list(ps["station"].values)
    ist = np.array([names.index(s) if s in names else -1 for s in m["station"]])
    ok = (ist >= 0) & (it < len(times))
    ok[ok] &= times[it[ok]] == m["time"].to_numpy()[ok]
    nf = len(data["freq"])
    energy, direction = np.full((len(m), nf), np.nan), np.full((len(m), nf), np.nan)
    energy[ok] = ps["gfs_spec_energy"].values[it[ok], il[ok], ist[ok]]
    direction[ok] = ps["gfs_spec_direction"].values[it[ok], il[ok], ist[ok]]
    data["point_energy"], data["point_direction"] = energy, direction
    data["has_point"] = ok & np.isfinite(energy).all(1)


def weighted_vector(energy, direction):
    """Direction as a vector whose length says how much energy carries it (0 in empty bins)."""
    w = energy / (energy + EPS)
    rad = np.deg2rad(np.nan_to_num(direction))
    return w * np.sin(rad), w * np.cos(rad)


# ---------------------------------------------------------------------------
# 2. TARGETS
# ---------------------------------------------------------------------------

def residual(eb, db, eg, dg) -> np.ndarray:
    """Per bin: sqrt(E_buoy) - sqrt(E_GFS), then the differences of the energy-weighted direction vectors."""
    bsin, bcos = weighted_vector(eb, db)
    gsin, gcos = weighted_vector(eg, dg)
    return np.hstack([np.sqrt(eb) - np.sqrt(eg), bsin - gsin, bcos - gcos])


def targets(data: dict) -> np.ndarray:
    return residual(data["buoy_energy"], data["buoy_direction"], data["gfs_energy"], data["gfs_direction"])


def corrected_spectrum(data: dict, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Model output (residuals, the layout of targets()) -> corrected energy and direction per bin."""
    nf = len(data["freq"])
    eg, dg = data["gfs_energy"], data["gfs_direction"]
    energy = np.clip(np.sqrt(eg) + y[:, :nf], 0, None) ** 2
    gsin, gcos = weighted_vector(eg, dg)
    direction = np.rad2deg(np.arctan2(gsin + y[:, nf:2 * nf], gcos + y[:, 2 * nf:])) % 360
    return energy, direction


# ---------------------------------------------------------------------------
# 3. FEATURES
# ---------------------------------------------------------------------------

def _wind_from(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """GFS wind direction (deg, coming from) and speed (m/s) at the station."""
    m = data["meta"]
    u, v = m["gfs_u10"].to_numpy(), m["gfs_v10"].to_numpy()
    return np.rad2deg(np.arctan2(-u, -v)) % 360, np.hypot(u, v)


def gfs_features(data: dict) -> np.ndarray:
    eg, dg = data["gfs_energy"], data["gfs_direction"]
    m = data["meta"]
    gsin, gcos = weighted_vector(eg, dg)
    wind_from, speed = _wind_from(data)
    context = np.column_stack([
        m["gfs_htsgw"], speed, np.sin(np.deg2rad(wind_from)), np.cos(np.deg2rad(wind_from)),
        m["gfs_perpw"], np.sin(np.deg2rad(m["gfs_dirpw"])), np.cos(np.deg2rad(m["gfs_dirpw"])), m["lead_hours"],
    ])
    return np.hstack([np.log10(eg + EPS), np.sqrt(eg), gsin, gcos, context])


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
    arr = np.column_stack(out)
    arr[~np.isfinite(energy).all(1)] = np.nan
    return arr


def _band_directions(data: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per period band of the GFS spectrum: (mean frequency, mean direction in deg)."""
    s = _band_summary(data["gfs_energy"], data["gfs_direction"], data["freq"])
    return {band: (s[:, 4 * b + 1], np.rad2deg(np.arctan2(s[:, 4 * b + 2], s[:, 4 * b + 3])) % 360)
            for b, band in enumerate(PERIOD_BANDS)}


def _history_pairs(data: dict, lags=HISTORY_LAGS) -> list[tuple]:
    """Per lag: (energy, direction) of the rebuilt GFS spectrum at the valid time and `lag` hours earlier
    (group "gfs_rebuilt"); the earlier one is NaN where it does not exist.

    The earlier spectrum comes from the same run if lead - lag >= 0. Before the issue time it is the GFS
    nowcast at that valid time (group "gfs_nowcast"), and only where that is missing the run issued 24 h
    earlier (lead - lag + 24). Everything exists at the issue time, no future information."""
    if "history_pairs" in data:
        return data["history_pairs"]
    reb = load("gfs_rebuilt")
    nc = load("gfs_nowcast")
    assert list(nc["station"].values) == list(reb["station"].values)
    data["history_pairs"] = history_pairs_from(
        reb["energy"].values.astype(float), reb["direction"].values.astype(float), reb["time"].values,
        reb["lead_hours"].values, nc["energy"].values.astype(float), nc["direction"].values.astype(float),
        nc["valid_time"].values, data["index"], lags)
    return data["history_pairs"]


def history_pairs_from(e_all, d_all, times, leads, e_nc, d_nc, nc_times, index, lags=HISTORY_LAGS) -> list[tuple]:
    """The pairs of _history_pairs from arrays (also used for live forecasts, forecast/predict.py):
        e_all, d_all    rebuilt spectra (time, lead_hours, station, freq) of the runs at issue `times` (sorted)
        e_nc, d_nc      rebuilt nowcast spectra (valid_time, station, freq) at `nc_times` (sorted)
        index           (it, il, ist) per data row into (time, lead_hours, station)"""
    leads = list(leads)
    it, il, ist = index
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
        pairs.append((now, (e_prev, d_prev)))
    return pairs


def history_band_features(data: dict) -> np.ndarray:
    """Per period band and lag: change of Hm0, mean frequency (dispersion drift) and mean direction vector.
    36 features for 3 bands and 3 lags. Missing values (empty band, no earlier spectrum) -> 0 ("no change")."""
    f = data["freq"]
    return np.hstack([np.nan_to_num(_band_summary(*now, f) - _band_summary(*prev, f))
                      for now, prev in _history_pairs(data)])


def _station_location(data: dict):
    """The station_location group and, per data row, the index of its station in it."""
    loc = load("station_location").load()
    names = loc["station"].values
    st = data["meta"]["station"].to_numpy()
    order = np.argsort(names)
    k = order[np.clip(np.searchsorted(names, st, sorter=order), 0, len(names) - 1)]
    assert (names[k] == st).all(), "stations missing from station_location: python -m nedsurfcast.forecast.preprocess --location"
    return loc, k


def lookup(table: np.ndarray, direction_deg: np.ndarray) -> np.ndarray:
    """table (n, n_directions, evenly spaced from 0 deg) at per-row directions (n,) -> (n,); NaN direction -> NaN."""
    n_dir = table.shape[1]
    ok = np.isfinite(direction_deg)
    k = np.zeros(len(direction_deg), int)
    k[ok] = np.round(direction_deg[ok] / (360 / n_dir)).astype(int) % n_dir
    out = table[np.arange(len(k)), k].astype(float)
    out[~ok] = np.nan
    return out


def tanh_kh(f: np.ndarray, h: np.ndarray) -> np.ndarray:
    """tanh(k h) from the linear dispersion relation (2 pi f)^2 = g k tanh(k h); 1 in deep water."""
    omega2_h_g = (2 * np.pi * f) ** 2 * h / G                     # = k h tanh(k h)
    kh = np.sqrt(omega2_h_g) * (1 + 0.2 * omega2_h_g)             # shallow-water start, then Newton
    for _ in range(20):
        t = np.tanh(kh)
        kh = kh - (kh * t - omega2_h_g) / (t + kh * (1 - t ** 2))
    return np.tanh(kh)


def fetch_limited_log_ratio(x_real_km, x_gfs_km, wind_speed) -> np.ndarray:
    """Expected log10(Hs_real / Hs_GFS) of the wind sea from the fetch difference alone: fetch-limited growth
    Hs ~ (g X / U^2)^0.5 (JONSWAP), with g X / U^2 capped at full development, where fetch no longer matters."""
    u2 = np.maximum(wind_speed, 1.0) ** 2
    x_real = np.minimum(G * x_real_km * 1e3 / u2, FULLY_DEVELOPED)
    x_gfs = np.minimum(G * x_gfs_km * 1e3 / u2, FULLY_DEVELOPED)
    return 0.5 * np.log10(x_real / x_gfs)


def location_features(data: dict) -> np.ndarray:
    """Per period band: log10 distance to land in the band's mean direction, tanh(k h) at the buoy and along the
    incoming ray at the band's mean frequency; plus log10 fetch in the wind direction. 10 features; NaN in an
    empty band (imputed by the model pipeline)."""
    loc, k = _station_location(data)
    dist, ray, h0 = loc["distance_to_land"].values[k], loc["ray_depth"].values[k], loc["depth"].values[k]
    out = []
    for f_mean, theta in _band_directions(data).values():
        out += [np.log10(lookup(dist, theta)), tanh_kh(f_mean, h0), tanh_kh(f_mean, lookup(ray, theta))]
    wind_from, _ = _wind_from(data)
    out.append(np.log10(lookup(dist, wind_from)))
    return np.column_stack(out)


def fetch_features(data: dict) -> np.ndarray:
    """Fetch mismatch between reality and GFS: in the wind direction log10(D_real / D_GFS), the expected
    log10 Hs ratio of fetch-limited growth and log10(g D_real / U^2); in the mean direction of the < 6 s and
    6-10 s bands log10(D_real / D_GFS). 5 features, all 0 where GFS's coastline is right."""
    loc, k = _station_location(data)
    real, gfs = loc["distance_to_land"].values[k], loc["gfs_distance_to_land"].values[k]
    wind_from, speed = _wind_from(data)
    xr_w, xg_w = lookup(real, wind_from), lookup(gfs, wind_from)
    out = [np.log10(xr_w / xg_w), fetch_limited_log_ratio(xr_w, xg_w, speed),
           np.log10(G * xr_w * 1e3 / np.maximum(speed, 1.0) ** 2)]
    bands = _band_directions(data)
    for band in ("T<6 s", "6-10 s"):
        theta = bands[band][1]
        out.append(np.log10(lookup(real, theta) / lookup(gfs, theta)))
    return np.column_stack(out)


FEATURE_BLOCKS = {                      # token -> builder; appended after the gfs block in this order
    "location": location_features,
    "fetch": fetch_features,
    "history-bands": history_band_features,
}


def features(data: dict, feature_set: str = "gfs") -> np.ndarray:
    """feature_set: "gfs" followed by any of "+location", "+fetch", "+history-bands" (see the module docstring)."""
    parts = feature_set.split("+")
    unknown = set(parts[1:]) - set(FEATURE_BLOCKS)
    if parts[0] != "gfs" or unknown:
        raise ValueError(f"bad feature set {feature_set!r}: 'gfs' + any of {list(FEATURE_BLOCKS)}")
    return np.hstack([gfs_features(data)] + [build(data) for token, build in FEATURE_BLOCKS.items() if token in parts])
