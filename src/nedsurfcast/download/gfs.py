"""
Download archived NOAA GFS-Wave forecasts (forecast pipeline)
===============================================================

Run from anywhere with: python -m nedsurfcast.download.gfs
"""

import datetime as dt
import threading
import numpy as np
import xarray as xr

from nedsurfcast import config as cfg
from nedsurfcast.forecast import config as fcfg

# eccodes (used by cfgrib) is not thread-safe: parallel decoding crashes it. Downloads still run in parallel.
_GRIB_LOCK = threading.Lock()


def _gfswave_url(issue: dt.datetime, lead: int) -> str:
    return (f"{fcfg.GFSWAVE_URL}/gfs.{issue:%Y%m%d}/{issue:%H}/wave/gridded/"
            f"gfswave.t{issue:%H}z.{fcfg.GFSWAVE_GRID}.f{lead:03d}.grib2")


def _gfswave_byte_ranges(idx_text: str, fields: list[str]) -> list[tuple[int, int | None]]:
    """(start, end) byte ranges of the wanted GRIB messages, read from the .idx file.

    Lines look like "5:2894821:d=2024010100:HTSGW:surface:24 hour fcst:"; a message ends where the
    next one starts, the last one runs to the end of the file (end=None).
    """
    lines = [line.split(":") for line in idx_text.strip().splitlines()]
    offsets = [int(parts[1]) for parts in lines]
    ranges = []
    for i, parts in enumerate(lines):
        if f"{parts[3]}:{parts[4]}" in fields:
            end = offsets[i + 1] - 1 if i + 1 < len(offsets) else None
            if ranges and ranges[-1][1] is not None and ranges[-1][1] + 1 == offsets[i]:
                ranges[-1] = (ranges[-1][0], end)     # adjacent message: extend the previous request
            else:
                ranges.append((offsets[i], end))
    return ranges


def _download_gfswave_lead(session, issue: dt.datetime, lead: int, tmp_dir) -> xr.Dataset | None:
    """Fetch only the wanted fields of one forecast step and crop them to cfg.DOMAIN."""
    url = _gfswave_url(issue, lead)
    idx = session.get(url + ".idx", timeout=60)
    if idx.status_code == 404:
        return None
    idx.raise_for_status()

    chunks = []
    for start, end in _gfswave_byte_ranges(idx.text, fcfg.GFSWAVE_FIELDS):
        r = session.get(url, headers={"Range": f"bytes={start}-{'' if end is None else end}"}, timeout=120)
        r.raise_for_status()
        chunks.append(r.content)

    grib_path = tmp_dir / f"{issue:%Y%m%d%H}_f{lead:03d}.grib2"
    grib_path.write_bytes(b"".join(chunks))
    try:
        with _GRIB_LOCK:
            ds = xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs={"indexpath": ""}, decode_timedelta=True)
            # GRIB longitudes run 0..360; the North Sea domain crosses 0. Rounding removes float noise
            # (359.750016) that would otherwise drop the domain edge at 9.0 from the selection.
            ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180).round(4),
                                  latitude=ds.latitude.round(4)).sortby("longitude")
            ds = ds.rename({"latitude": "lat", "longitude": "lon"}).sel(**cfg.DOMAIN).load()
    finally:
        grib_path.unlink()
    # Swell partitions "n in sequence" arrive as a dimension orderedSequenceData (1., 2., 3.); with only one
    # sequence requested it is a scalar coordinate instead, which carries no information
    if "orderedSequenceData" in ds.dims:
        ds = (ds.assign_coords(orderedSequenceData=ds.orderedSequenceData.values.astype(int))
                .rename(orderedSequenceData="swell_partition"))
    return ds.drop_vars([v for v in ("surface", "orderedSequenceData") if v in ds.coords])


def _session():
    """requests session that retries dropped connections and server errors (S3 closes connections now and then)."""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    retry = Retry(total=6, connect=6, read=6, status=6, backoff_factor=2,
                  status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def download_gfswave_cycle(issue: dt.datetime, overwrite: bool = False):
    """One forecast run (all fcfg.GFSWAVE_LEAD_HOURS) -> gfswave_YYYYMMDD_HHz.nc with dims (lead_hours, lat, lon).

    The file is only written when every lead time is in, so an interrupted run leaves no partial file.
    """
    filename = fcfg.GFSWAVE_DIR / f"gfswave_{issue:%Y%m%d}_{issue:%H}z.nc"
    if filename.exists() and not overwrite:
        print(f"{filename.name}: already exists and overwrite=False, skipping")
        return
    tmp_dir = fcfg.GFSWAVE_DIR / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    steps = []
    with _session() as session:
        for lead in fcfg.GFSWAVE_LEAD_HOURS:
            ds = _download_gfswave_lead(session, issue, lead, tmp_dir)
            if ds is None:
                print(f"{issue:%Y-%m-%d %H}z f{lead:03d}: not in archive")
                continue
            steps.append(ds)
    if not steps:
        print(f"{issue:%Y-%m-%d %H}z: no data, skipping")
        return

    # time = issue time, lead_hours = lead time, valid_time = time + lead
    ds = xr.concat(steps, dim="step")
    lead_hours = (ds.step.values / np.timedelta64(1, "h")).astype(int)
    ds = ds.drop_vars("step").assign_coords(step=lead_hours).rename(step="lead_hours")  # cfgrib's step attrs break reading back
    ds.to_netcdf(filename)
    print(f"{filename.name}: {len(steps)} lead times written")


def download_gfswave(overwrite: bool = False, max_workers: int = 4):
    """All forecast runs from cfg.START_DATE to cfg.END_DATE at fcfg.GFSWAVE_CYCLES, a few runs in parallel."""
    from concurrent.futures import ThreadPoolExecutor

    fcfg.GFSWAVE_DIR.mkdir(parents=True, exist_ok=True)
    issues = []
    day = cfg.START_DATE
    while day < cfg.END_DATE:
        issues += [day.replace(hour=h) for h in fcfg.GFSWAVE_CYCLES]
        day += dt.timedelta(days=1)

    def safe_download(issue):
        try:
            download_gfswave_cycle(issue, overwrite)
        except Exception as err:   # one failing run should not stop the others; rerun to fill the gaps
            print(f"{issue:%Y-%m-%d %H}z: FAILED ({type(err).__name__}: {err})")
            return issue

    # downloading is I/O bound, so threads are enough
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        failed = [issue for issue in executor.map(safe_download, issues) if issue is not None]
    if failed:
        print(f"{len(failed)} runs failed, run again to retry them: {[f'{i:%Y-%m-%d %H}z' for i in failed]}")


if __name__ == '__main__':
    download_gfswave()
