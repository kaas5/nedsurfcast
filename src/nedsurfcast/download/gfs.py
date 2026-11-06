"""
Download archived NOAA GFS-Wave forecasts (forecast pipeline)
===============================================================

Run from anywhere with: python -m nedsurfcast.download.gfs
"""

import datetime as dt
import numpy as np
import xarray as xr

from nedsurfcast import config as cfg
from nedsurfcast.forecast import config as fcfg


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
        ds = xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs={"indexpath": ""}, decode_timedelta=True)
        # GRIB longitudes run 0..360; the North Sea domain crosses 0
        ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180).sortby("longitude")
        ds = ds.rename({"latitude": "lat", "longitude": "lon"}).sel(**cfg.DOMAIN).load()
    finally:
        grib_path.unlink()
    return ds.drop_vars([v for v in ("surface",) if v in ds.coords])


def download_gfswave_cycle(issue: dt.datetime, overwrite: bool = False):
    """One forecast run (all fcfg.GFSWAVE_LEAD_HOURS) -> gfswave_YYYYMMDD_HHz.nc with dims (lead_hours, lat, lon)."""
    import requests

    filename = fcfg.GFSWAVE_DIR / f"gfswave_{issue:%Y%m%d}_{issue:%H}z.nc"
    if filename.exists() and not overwrite:
        print(f"{filename.name}: already exists and overwrite=False, skipping")
        return
    tmp_dir = fcfg.GFSWAVE_DIR / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    steps = []
    with requests.Session() as session:
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

    # downloading is I/O bound, so threads are enough; list() re-raises errors from the workers
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(lambda issue: download_gfswave_cycle(issue, overwrite), issues))


if __name__ == '__main__':
    download_gfswave()
