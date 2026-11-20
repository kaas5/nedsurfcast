"""
Location properties of the stations, as a function of the incoming wave direction and period
================================================================================================

GFS-Wave runs on a 0.25 deg grid: it cannot resolve the local depth and how sheltered a position is by the
coast. Those effects depend on the incoming waves: a swell from the north-west at K13 crosses ~500 km of open
water, the same swell at a buoy behind the Wadden islands crosses a few km; a 14 s swell feels 20 m of water,
a 5 s wind sea does not. So the location is not used as a fixed property (that would only be a station
identifier), but as a function of the direction and period of each incoming band:

    per station, from the EMODnet Bathymetry DTM (download_bathymetry):
        depth at the station                        h0
        per direction (1 deg, coming-from):
            over-water distance to land             D(theta), capped at RAY_MAX_KM ("open sea")
            mean depth along the first RAY_DEPTH_KM of that ray     h_ray(theta)

    per spectrum and period band (forecast/train_spectrum.py, feature token "location"), with the band's mean
    frequency f and mean direction theta from the GFS spectrum:
        log10 D(theta)              exposure / fetch in the direction the band comes from
        tanh(k h0)                  how much the band feels the bottom at the station (k from the dispersion
                                    relation at frequency f)
        tanh(k h_ray(theta))        how much it felt the bottom on its way in (shoaling, refraction, sandbanks)
    plus log10 D(wind direction): the fetch of the local wind sea.

    fetch mismatch (feature token "fetch"): GFS grows its wind sea over the fetch of its own 0.25 deg coastline
    from its own grid cell (gfs_distance_to_land), the buoy experiences the real fetch. In the wind direction:
        log10(D_real / D_GFS), the expected log10 Hs ratio from fetch-limited growth (fetch_limited_log_ratio,
        0 when fully developed) and log10(g D_real / U^2); and log10(D_real / D_GFS) in the mean direction of
        the < 6 s and 6-10 s bands.

The same quantities can be computed for any grid cell, so the correction can generalise to cells without buoy.

Run from anywhere with: python -m nedsurfcast.forecast.location   (downloads the DTM and builds the table)
"""

import io

import numpy as np
import xarray as xr

from nedsurfcast import config as cfg
from nedsurfcast.forecast import config as fcfg

WCS_URL = "https://ows.emodnet-bathymetry.eu/wcs"
COVERAGE = "emodnet__mean"           # EMODnet DTM: negative = below sea level, positive = land elevation
BOX = dict(lat=(49.0, 61.0), lon=(-5.0, 12.0))   # deg; covers the North Sea and its coasts
RESOLUTION = 0.01                    # deg (~1 km); the DTM itself is ~115 m
TILE_DEG = 1.0                       # deg; BOX must be a whole number of tiles
BATHYMETRY_FILE = cfg.RAW_DIR / "emodnet" / "bathymetry_north_sea.nc"
LOCATION_FILE = fcfg.FORECAST_DIR / "station_location.nc"

N_DIRECTIONS = 360                   # 1 deg
RAY_STEP_KM = 0.5                    # finer than the grid, so narrow dunes/islands are not stepped over
RAY_MAX_KM = 500.0                   # longer rays (or leaving BOX) count as open sea
RAY_DEPTH_KM = 50.0                  # mean depth over the first part of the ray
MIN_DEPTH = 1.0                      # m; floor for depths (buoys near the surf zone, DTM noise)
G = 9.81
KM_PER_DEG = 111.195


# ---------------------------------------------------------------------------
# 1. BATHYMETRY
# ---------------------------------------------------------------------------

def download_bathymetry(overwrite: bool = False) -> xr.DataArray:
    """EMODnet DTM resampled to RESOLUTION over BOX -> elevation (lat, lon) in m, negative below sea level."""
    if BATHYMETRY_FILE.exists() and not overwrite:
        return xr.open_dataarray(BATHYMETRY_FILE)
    import requests
    from PIL import Image

    def tile(la0, la1, lo0, lo1):           # the WCS refuses large areas: fetch TILE_DEG tiles
        n_la, n_lo = round((la1 - la0) / RESOLUTION), round((lo1 - lo0) / RESOLUTION)
        params = [("service", "WCS"), ("version", "2.0.1"), ("request", "GetCoverage"), ("coverageId", COVERAGE),
                  ("subset", f"Lat({la0},{la1})"), ("subset", f"Long({lo0},{lo1})"), ("format", "image/tiff"),
                  ("scalesize", f"i({n_lo}),j({n_la})")]          # i = longitude, j = latitude
        r = requests.get(WCS_URL, params=params, timeout=300)
        r.raise_for_status()
        if "tiff" not in r.headers.get("content-type", ""):
            raise RuntimeError(f"EMODnet WCS returned {r.headers.get('content-type')}: {r.text[:500]}")
        a = np.array(Image.open(io.BytesIO(r.content)), dtype=np.float32)     # row 0 = north
        assert a.shape == (n_la, n_lo), a.shape
        return a

    (lat0, lat1), (lon0, lon1) = BOX["lat"], BOX["lon"]
    n_lat, n_lon = round((lat1 - lat0) / RESOLUTION), round((lon1 - lon0) / RESOLUTION)
    lat_edges, lon_edges = np.arange(lat0, lat1 + 1e-9, TILE_DEG), np.arange(lon0, lon1 + 1e-9, TILE_DEG)
    rows = []
    for la1, la0 in zip(lat_edges[::-1][:-1], lat_edges[::-1][1:]):           # north to south
        rows.append(np.hstack([tile(la0, la1, lo0, lo1) for lo0, lo1 in zip(lon_edges[:-1], lon_edges[1:])]))
    elev = np.vstack(rows)
    assert elev.shape == (n_lat, n_lon), elev.shape
    lat = lat1 - (np.arange(n_lat) + 0.5) * (lat1 - lat0) / n_lat
    lon = lon0 + (np.arange(n_lon) + 0.5) * (lon1 - lon0) / n_lon
    da = xr.DataArray(elev[::-1], coords={"lat": lat[::-1], "lon": lon}, dims=("lat", "lon"), name="elevation",
                      attrs={"units": "m", "source": f"{WCS_URL} {COVERAGE}", "positive": "up"})
    BATHYMETRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    da.to_netcdf(BATHYMETRY_FILE)
    print(f"bathymetry: {da.shape} written to {BATHYMETRY_FILE}")
    return da


# ---------------------------------------------------------------------------
# 2. PER-STATION TABLES
# ---------------------------------------------------------------------------

def _sample(elev: xr.DataArray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Nearest-cell elevation; NaN outside BOX."""
    la, lo = elev["lat"].values, elev["lon"].values
    i = np.round((lat - la[0]) / (la[1] - la[0])).astype(int)
    j = np.round((lon - lo[0]) / (lo[1] - lo[0])).astype(int)
    inside = (i >= 0) & (i < len(la)) & (j >= 0) & (j < len(lo))
    out = np.full(lat.shape, np.nan)
    out[inside] = elev.values[i[inside], j[inside]]
    return out


_S = np.arange(1, int(RAY_MAX_KM / RAY_STEP_KM) + 1) * RAY_STEP_KM          # km along a ray


def _rays(elev: xr.DataArray, lat: float, lon: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Elevation along a ray in every direction (direction, step), the index of the first land step (len(_S)
    if none) and the over-water distance to land (km, RAY_MAX_KM if none). Land = elevation >= 0; NaN
    (outside the grid) is open sea."""
    theta = np.deg2rad(np.arange(N_DIRECTIONS) * 360 / N_DIRECTIONS)
    # flat-earth steps are accurate enough over a few hundred km at these latitudes
    ray_lat = lat + np.outer(np.cos(theta), _S) / KM_PER_DEG
    ray_lon = lon + np.outer(np.sin(theta), _S) / (KM_PER_DEG * np.cos(np.deg2rad(lat)))
    e = _sample(elev, ray_lat, ray_lon)
    land = e >= 0
    first = np.where(land.any(1), land.argmax(1), len(_S))
    dist = np.where(first < len(_S), _S[np.minimum(first, len(_S) - 1)], RAY_MAX_KM)
    return e, first, dist


def gfs_distance_to_land(lats, lons) -> np.ndarray:
    """Over-water distance to land per direction (n, N_DIRECTIONS) as GFS sees it: rays from the given GFS
    cells over the 0.25 deg GFS sea mask (an approximation of the coastline of GFS-Wave's model grids)."""
    from nedsurfcast.forecast.preprocess import load

    sea = load("grid")["sea_mask"].load()
    elev = xr.where(sea, -1.0, 1.0)                     # same convention as the DTM: land >= 0
    return np.array([_rays(elev, lat, lon)[2] for lat, lon in zip(lats, lons)])


def location_table(lats, lons, elev: xr.DataArray) -> dict[str, np.ndarray]:
    """Depth h0 (n,), distance to land D (n, N_DIRECTIONS) in km and mean ray depth h_ray (n, N_DIRECTIONS)."""
    s = _S
    h0, dist, h_ray = [], [], []
    for lat, lon in zip(lats, lons):
        e, first, d = _rays(elev, lat, lon)                                    # (direction, step)
        dist.append(d)
        near = (s[None, :] <= RAY_DEPTH_KM) & (np.arange(len(s))[None, :] < first[:, None])
        depth = np.where(near & np.isfinite(e), np.maximum(-e, MIN_DEPTH), np.nan)
        with np.errstate(invalid="ignore"):
            h_ray.append(np.where(near.any(1), np.nanmean(depth, 1), MIN_DEPTH))
        h0.append(max(-float(_sample(elev, np.array([lat]), np.array([lon]))[0]), MIN_DEPTH))
    return {"depth": np.array(h0), "distance_to_land": np.array(dist), "ray_depth": np.array(h_ray)}


def build_station_location() -> xr.Dataset:
    """Location table at the buoy positions of the forecast dataset's stations -> LOCATION_FILE."""
    from nedsurfcast.forecast.preprocess import load

    stations = load("stations").load()
    t = location_table(stations["lat"].values, stations["lon"].values, download_bathymetry())
    gfs_dist = gfs_distance_to_land(stations["nearest_lat"].values, stations["nearest_lon"].values)
    ds = xr.Dataset(
        {"depth": ("station", t["depth"], {"units": "m"}),
         "distance_to_land": (("station", "direction"), t["distance_to_land"],
                              {"units": "km", "long_name": "over-water distance to land towards direction",
                               "capped_at": RAY_MAX_KM}),
         "gfs_distance_to_land": (("station", "direction"), gfs_dist,
                                  {"units": "km", "long_name": "distance to land on the 0.25 deg GFS sea mask, "
                                   "from the station's GFS cell", "capped_at": RAY_MAX_KM}),
         "ray_depth": (("station", "direction"), t["ray_depth"],
                       {"units": "m", "long_name": f"mean depth over the first {RAY_DEPTH_KM:g} km of the ray"})},
        coords={"station": stations["station"].values, "direction": np.arange(N_DIRECTIONS) * 360 / N_DIRECTIONS},
        attrs={"description": "station depth and per-direction exposure from the EMODnet DTM (forecast/location.py)",
               "direction": "deg, coming-from (the ray points towards where the waves come from)"})
    ds.to_netcdf(LOCATION_FILE)
    return ds


def load_station_location() -> xr.Dataset:
    if LOCATION_FILE.exists():
        ds = xr.open_dataset(LOCATION_FILE).load()
        if "gfs_distance_to_land" in ds:
            return ds
    return build_station_location()


# ---------------------------------------------------------------------------
# 3. WAVE-DEPENDENT FEATURES
# ---------------------------------------------------------------------------

def tanh_kh(f: np.ndarray, h: np.ndarray) -> np.ndarray:
    """tanh(k h) from the linear dispersion relation (2 pi f)^2 = g k tanh(k h); 1 in deep water."""
    omega2_h_g = (2 * np.pi * f) ** 2 * h / G                     # = k h tanh(k h)
    kh = np.sqrt(omega2_h_g) * (1 + 0.2 * omega2_h_g)             # shallow-water start, then Newton
    for _ in range(20):
        t = np.tanh(kh)
        kh = kh - (kh * t - omega2_h_g) / (t + kh * (1 - t ** 2))
    return np.tanh(kh)


FULLY_DEVELOPED = 2.3e4             # dimensionless fetch g X / U^2 beyond which wind sea stops growing


def fetch_limited_log_ratio(x_real_km, x_gfs_km, wind_speed) -> np.ndarray:
    """Expected log10(Hs_real / Hs_GFS) of the wind sea from the fetch difference alone: fetch-limited growth
    Hs ~ (g X / U^2)^0.5 (JONSWAP), with g X / U^2 capped at full development, where fetch no longer matters."""
    u2 = np.maximum(wind_speed, 1.0) ** 2
    xr_ = np.minimum(G * x_real_km * 1e3 / u2, FULLY_DEVELOPED)
    xg = np.minimum(G * x_gfs_km * 1e3 / u2, FULLY_DEVELOPED)
    return 0.5 * np.log10(xr_ / xg)


def lookup(table: np.ndarray, direction_deg: np.ndarray) -> np.ndarray:
    """table (n, N_DIRECTIONS) at per-row directions (n,) -> (n,); NaN direction -> NaN."""
    ok = np.isfinite(direction_deg)
    k = np.zeros(len(direction_deg), int)
    k[ok] = np.round(direction_deg[ok] / (360 / N_DIRECTIONS)).astype(int) % N_DIRECTIONS
    out = table[np.arange(len(k)), k].astype(float)
    out[~ok] = np.nan
    return out


if __name__ == "__main__":
    ds = build_station_location()
    import pandas as pd

    d = ds["distance_to_land"]
    print(f"station location table written to {LOCATION_FILE}")
    print(pd.DataFrame({"depth (m)": ds["depth"].values, **{f"D from {n} (km)": d.sel(direction=a).values
                        for n, a in (("N", 0), ("E", 90), ("SW", 225), ("W", 270), ("NW", 315))},
                        "ray depth NW (m)": ds["ray_depth"].sel(direction=315).values},
                       index=ds["station"].values).round(1).to_string())
