"""
Download measured RWS wave spectra from the RWsOS API (forecast pipeline)
===========================================================================

Per station and month one file rws_spectra/<station>/<station>_YYYY_MM.nc with dims (time, freq):
    energy_density   (m2/Hz)        variance density per 10 mHz bin
    direction        (deg)          mean direction per frequency, nautical "coming from"
    spread           (deg)          directional spread per frequency

Spectra arrive every 30 minutes. The API only holds about one year of values (older events
are null), so run this regularly: finished months are skipped, the running month is refreshed.

Run from anywhere with: python -m nedsurfcast.download.rws_spectra
"""

import datetime as dt

import numpy as np
import xarray as xr

from nedsurfcast import config as cfg
from nedsurfcast.forecast import config as fcfg

MISSING = 99999
HEADERS = {"User-Agent": "nedsurfcast"}


def _get_json(session, path: str, **params) -> dict:
    r = session.get(f"{fcfg.RWS_SPECTRA_URL}/{path}", params=params, headers=HEADERS, timeout=300)
    r.raise_for_status()
    return r.json()


def list_spectrum_stations(session) -> dict[str, tuple[str, float, float]]:
    """Station code -> (name, lat, lon) for every station that publishes spectra."""
    d = _get_json(session, "locations/geojson", sourceName=fcfg.RWS_SPECTRA_SOURCE,
                  observationTypeId=next(iter(fcfg.RWS_SPECTRA_SERIES)), boundingBox="[2,48.56,7.5,57]")
    out = {}
    for feature in d["features"]:
        props = feature["properties"]
        lon, lat = feature["geometry"]["coordinates"][:2]
        out.setdefault(props["id"], (props.get("locationName", ""), lat, lon))
    return out


def _fetch_series(session, station: str, obs_id: str, start: dt.datetime, end: dt.datetime):
    """One series over [start, end) -> (times, array (n_times, nbins)) with NaN for missing values."""
    # endTime is inclusive in the API, stop one second early so months do not overlap
    d = _get_json(session, "timeseries", observationTypeId=obs_id, sourceName=fcfg.RWS_SPECTRA_SOURCE,
                  locationCode=station, startTime=f"{start:%Y-%m-%dT%H:%M:%SZ}",
                  endTime=f"{end - dt.timedelta(seconds=1):%Y-%m-%dT%H:%M:%SZ}")
    rows, meta = {}, {}
    for result in d.get("results", []):
        meta = result
        for event in result["events"]:
            values = event.get("value")
            if isinstance(values, list) and len(values) == fcfg.RWS_SPECTRA_NBINS:
                t = np.datetime64(event["timeStamp"].replace("Z", ""), "ns")
                rows[t] = [np.nan if v is None or v == MISSING else v for v in values]
    times = np.array(sorted(rows), dtype="datetime64[ns]")
    values = np.array([rows[t] for t in times], dtype=float).reshape(-1, fcfg.RWS_SPECTRA_NBINS)
    return times, values, meta


def download_station_month(session, station: str, month: dt.datetime, overwrite: bool = False):
    month_end = (month.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    out_dir = fcfg.RWS_SPECTRA_DIR / station
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"{station}_{month:%Y_%m}.nc"
    month_finished = month_end <= dt.datetime.utcnow()
    if filename.exists() and month_finished and not overwrite:
        print(f"{filename.name}: already exists and overwrite=False, skipping")
        return

    series, meta = {}, {}
    for obs_id, name in fcfg.RWS_SPECTRA_SERIES.items():
        times, values, m = _fetch_series(session, station, obs_id, month, month_end)
        series[name] = xr.DataArray(values, dims=("time", "freq"), coords={"time": times})
        meta = meta or m

    ds = xr.Dataset(series)   # outer join on time
    ds["energy_density"] = ds["energy_density"] / 1e4               # cm2 s = cm2/Hz -> m2/Hz
    ds = ds.dropna("time", how="all", subset=["energy_density"])    # the API keeps empty events for old dates
    if ds.sizes["time"] == 0:
        print(f"{station} {month:%Y-%m}: no spectra with values, nothing written")
        return

    df = fcfg.RWS_SPECTRA_DF
    ds = ds.assign_coords(freq=np.arange(fcfg.RWS_SPECTRA_NBINS) * df + df / 2)   # bin centres (Hz)
    ds["freq"].attrs = {"units": "Hz", "long_name": "centre of the 10 mHz frequency bin"}
    ds["energy_density"].attrs = {"units": "m2/Hz", "long_name": "variance density"}
    ds["direction"].attrs = {"units": "degree", "long_name": "mean wave direction per frequency (coming from)"}
    ds["spread"].attrs = {"units": "degree", "long_name": "directional spread per frequency"}

    lon, lat = meta.get("location", {}).get("geometry", {}).get("coordinates", [np.nan, np.nan])[:2]
    ds.attrs = {"station": station, "station_name": meta.get("location", {}).get("properties", {}).get("locationName", ""),
                "lat": lat, "lon": lon, "source": f"{fcfg.RWS_SPECTRA_URL} ({fcfg.RWS_SPECTRA_SOURCE})"}
    ds.to_netcdf(filename)
    print(f"{filename.name}: {ds.sizes['time']} spectra written")


def download_rws_spectra(stations=None, start: dt.datetime = cfg.START_DATE, end: dt.datetime = cfg.END_DATE,
                         overwrite: bool = False):
    """All months touching [start, end) for every station (default fcfg.RWS_SPECTRA_STATIONS)."""
    import requests

    stations = stations or fcfg.RWS_SPECTRA_STATIONS
    with requests.Session() as session:
        for station in stations:
            month = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            while month < end:
                download_station_month(session, station, month, overwrite)
                month = (month.replace(day=28) + dt.timedelta(days=4)).replace(day=1)


if __name__ == "__main__":
    download_rws_spectra()
