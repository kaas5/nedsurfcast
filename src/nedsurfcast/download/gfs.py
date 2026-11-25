"""
Download archived NOAA GFS-Wave forecasts (forecast pipeline)
===============================================================

Run from anywhere with:
    python -m nedsurfcast.download.gfs              gridded fields (fcfg.GFSWAVE_FIELDS)
    python -m nedsurfcast.download.gfs --spectra    point spectra at fcfg.GFSWAVE_SPEC_STATIONS
    python -m nedsurfcast.download.gfs --nowcast    analyses (f000/f003) of the 06/12/18z runs
    python -m nedsurfcast.download.gfs --live [--issue YYYYMMDDHH]   inputs of a live forecast (forecast/predict.py)
"""

import datetime as dt
import threading
import numpy as np
import xarray as xr

from nedsurfcast import config as cfg
from nedsurfcast.forecast import config as fcfg

# eccodes (used by cfgrib) is not thread-safe: parallel decoding crashes it. Downloads still run in parallel.
_GRIB_LOCK = threading.Lock()
# netCDF/HDF5 is not thread-safe either: concurrent writes fail or crash the process
_NETCDF_LOCK = threading.Lock()


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


def download_gfswave_cycle(issue: dt.datetime, overwrite: bool = False, leads=None, out_dir=None):
    """One forecast run (fcfg.GFSWAVE_LEAD_HOURS, or `leads`) -> <out_dir>/gfswave_YYYYMMDD_HHz.nc with dims
    (lead_hours, lat, lon); out_dir defaults to fcfg.GFSWAVE_DIR.

    The file is only written when every lead time is in, so an interrupted run leaves no partial file.
    """
    leads = fcfg.GFSWAVE_LEAD_HOURS if leads is None else leads
    out_dir = fcfg.GFSWAVE_DIR if out_dir is None else out_dir
    filename = out_dir / f"gfswave_{issue:%Y%m%d}_{issue:%H}z.nc"
    if filename.exists() and not overwrite:
        print(f"{filename.name}: already exists and overwrite=False, skipping")
        return
    tmp_dir = out_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    steps = []
    with _session() as session:
        for lead in leads:
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
    with _NETCDF_LOCK:
        ds.to_netcdf(filename)
    print(f"{filename.name}: {len(steps)} lead times written")


def _issue_times(cycles=None) -> list[dt.datetime]:
    """Every forecast run from cfg.START_DATE to cfg.END_DATE at fcfg.GFSWAVE_CYCLES (or `cycles`)."""
    cycles = fcfg.GFSWAVE_CYCLES if cycles is None else cycles
    issues = []
    day = cfg.START_DATE
    while day < cfg.END_DATE:
        issues += [day.replace(hour=h) for h in cycles]
        day += dt.timedelta(days=1)
    return issues


def _run_parallel(download_one, issues, max_workers: int):
    """download_one(issue) for every run, a few in parallel; a failing run is reported, not fatal."""
    from concurrent.futures import ThreadPoolExecutor

    def safe_download(issue):
        try:
            download_one(issue)
        except Exception as err:   # one failing run should not stop the others; rerun to fill the gaps
            print(f"{issue:%Y-%m-%d %H}z: FAILED ({type(err).__name__}: {err})")
            return issue

    # downloading is I/O bound, so threads are enough
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        failed = [issue for issue in executor.map(safe_download, issues) if issue is not None]
    if failed:
        print(f"{len(failed)} runs failed, run again to retry them: {[f'{i:%Y-%m-%d %H}z' for i in failed]}")


def download_gfswave(overwrite: bool = False, max_workers: int = 4):
    """All gridded forecast runs from cfg.START_DATE to cfg.END_DATE, a few runs in parallel."""
    fcfg.GFSWAVE_DIR.mkdir(parents=True, exist_ok=True)
    _run_parallel(lambda issue: download_gfswave_cycle(issue, overwrite), _issue_times(), max_workers)


# ---------------------------------------------------------------------------
# POINT SPECTRA (station output)
# ---------------------------------------------------------------------------
# gfswave.<ID>.spec is an ASCII WAVEWATCH III spectral file with a fixed layout:
#   header   14 lines, 985 bytes: title line, 50 frequencies (Hz, log-spaced 0.035-0.964), 36 directions (rad)
#   per hour 260 lines, 20137 bytes: "YYYYMMDD HHMMSS", station line (name, lat, lon, depth, wind speed and
#            direction, current speed and direction), 1800 values E(f, theta), frequency fastest
# Hours 0..384 follow each other, so lead L starts at byte 985 + L * 20137 and a range request
# fetches only the lead times needed.
# Units m2/Hz/rad; directions nautical "going to" (checked against the RWS buoys: + 180 deg = "coming from").
_SPEC_HEADER_BYTES = 985
_SPEC_STEP_BYTES = 20137
_SPEC_STEP_LINES = 260
_SPEC_NF, _SPEC_ND = 50, 36


def _gfswave_spec_url(issue: dt.datetime, station: str) -> str:
    return (f"{fcfg.GFSWAVE_URL}/gfs.{issue:%Y%m%d}/{issue:%H}/wave/station/bulls.t{issue:%H}z/"
            f"gfswave.{station}.spec")


def parse_gfswave_spec(text: str):
    """ASCII spectrum file (or its first hours) -> frequencies (Hz), directions (deg, going to),
    valid times, energy (time, direction, frequency) in m2/Hz/rad, station lines."""
    lines = text.splitlines()
    header = " ".join(lines[1:14]).split()
    freq = np.array(header[:_SPEC_NF], float)
    direction = np.rad2deg(np.array(header[_SPEC_NF:_SPEC_NF + _SPEC_ND], float)) % 360
    times, energy, station_lines = [], [], []
    for i in range(14, len(lines) - _SPEC_STEP_LINES + 1, _SPEC_STEP_LINES):
        times.append(dt.datetime.strptime(lines[i].strip(), "%Y%m%d %H%M%S"))
        station_lines.append(lines[i + 1])
        values = np.array(" ".join(lines[i + 2:i + _SPEC_STEP_LINES]).split(), float)
        energy.append(values.reshape(_SPEC_ND, _SPEC_NF))
    return freq, direction, times, np.array(energy), station_lines


def download_gfswave_spectra_cycle(issue: dt.datetime, overwrite: bool = False):
    """Point spectra of one run at fcfg.GFSWAVE_SPEC_STATIONS, lead times fcfg.GFSWAVE_LEAD_HOURS
    -> gfswave_spec_YYYYMMDD_HHz.nc with energy (station, lead_hours, direction, frequency)."""
    filename = fcfg.GFSWAVE_SPEC_DIR / f"gfswave_spec_{issue:%Y%m%d}_{issue:%H}z.nc"
    if filename.exists() and not overwrite:
        print(f"{filename.name}: already exists and overwrite=False, skipping")
        return
    max_lead = max(fcfg.GFSWAVE_LEAD_HOURS)
    last_byte = _SPEC_HEADER_BYTES + (max_lead + 1) * _SPEC_STEP_BYTES - 1

    stations, spectra, lats, lons, depths = [], [], [], [], []
    freq = direction = None
    with _session() as session:
        for station in fcfg.GFSWAVE_SPEC_STATIONS:
            r = session.get(_gfswave_spec_url(issue, station), headers={"Range": f"bytes=0-{last_byte}"}, timeout=120)
            if r.status_code == 404:
                print(f"{issue:%Y-%m-%d %H}z {station}: not in archive")
                continue
            r.raise_for_status()
            freq, direction, times, energy, station_lines = parse_gfswave_spec(r.text)
            if not times or times[0] != issue:
                raise ValueError(f"{station}: first time {times[:1]} is not the issue time {issue}")
            leads = [int((t - issue).total_seconds() // 3600) for t in times]
            out = np.full((len(fcfg.GFSWAVE_LEAD_HOURS), _SPEC_ND, _SPEC_NF), np.nan, dtype=np.float32)
            for k, h in enumerate(fcfg.GFSWAVE_LEAD_HOURS):
                if h in leads:
                    out[k] = energy[leads.index(h)]
            spectra.append(out)
            name_end = station_lines[0].index("'", 1) + 1        # values after the quoted name are fixed width
            rest = station_lines[0][name_end:]
            lats.append(float(rest[0:7])); lons.append(float(rest[7:14])); depths.append(float(rest[14:24]))
            stations.append(station)
    if not stations:
        print(f"{issue:%Y-%m-%d %H}z: no station spectra, skipping")
        return

    ds = xr.Dataset(
        {"energy": (("station", "lead_hours", "direction", "frequency"), np.stack(spectra),
                    {"units": "m2/Hz/rad", "long_name": "WAVEWATCH III variance density E(f, theta)"}),
         "lat": ("station", np.array(lats)), "lon": ("station", np.array(lons)),
         "depth": ("station", np.array(depths), {"units": "m"})},
        coords={"station": stations, "lead_hours": np.array(fcfg.GFSWAVE_LEAD_HOURS),
                "direction": ("direction", direction, {"units": "degree", "long_name": "nautical, going to"}),
                "frequency": ("frequency", freq, {"units": "Hz"}),
                "time": np.datetime64(issue, "ns")},
        attrs={"source": _gfswave_spec_url(issue, "<station>")},
    )
    with _NETCDF_LOCK:
        ds.to_netcdf(filename)
    print(f"{filename.name}: {len(stations)} stations written")


def download_gfswave_nowcast(overwrite: bool = False, max_workers: int = 4):
    """Nowcast fields: the shortest lead times (fcfg.GFSWAVE_NOWCAST_LEAD_HOURS) of the runs at
    fcfg.GFSWAVE_NOWCAST_CYCLES, into fcfg.GFSWAVE_NOWCAST_DIR. Together with the 00z runs this gives the
    latest available GFS state at every 3-hourly valid time (history features of the spectrum model)."""
    fcfg.GFSWAVE_NOWCAST_DIR.mkdir(parents=True, exist_ok=True)
    _run_parallel(lambda issue: download_gfswave_cycle(issue, overwrite, fcfg.GFSWAVE_NOWCAST_LEAD_HOURS,
                                                       fcfg.GFSWAVE_NOWCAST_DIR),
                  _issue_times(fcfg.GFSWAVE_NOWCAST_CYCLES), max_workers)


# ---------------------------------------------------------------------------
# LIVE DATA (forecast/predict.py)
# ---------------------------------------------------------------------------

LIVE_PREVIOUS_LEADS = list(range(0, 24, 3))   # previous day's run: nowcast at 00/03z, history fallback


def run_complete(issue: dt.datetime, lead: int = None) -> bool:
    """Whether the run is published up to `lead` (default the last of fcfg.GFSWAVE_LEAD_HOURS)."""
    lead = max(fcfg.GFSWAVE_LEAD_HOURS) if lead is None else lead
    with _session() as session:
        return session.head(_gfswave_url(issue, lead) + ".idx", timeout=60).status_code == 200


def latest_issue(cycle: int = None, days_back: int = 3) -> dt.datetime:
    """The newest complete run at `cycle` (default fcfg.GFSWAVE_CYCLES[0], i.e. 00z) of the last days."""
    cycle = fcfg.GFSWAVE_CYCLES[0] if cycle is None else cycle
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    today = now.replace(hour=cycle, minute=0, second=0, microsecond=0)
    for back in range(days_back + 1):
        issue = today - dt.timedelta(days=back)
        if issue <= now and run_complete(issue):
            return issue
    raise RuntimeError(f"no complete {cycle:02d}z run in the last {days_back} days")


def _has_leads(filename, leads) -> bool:
    if not filename.exists():
        return False
    with xr.open_dataset(filename) as ds:
        return set(leads) <= set(int(h) for h in ds["lead_hours"].values)


def download_live(issue: dt.datetime = None, overwrite: bool = False) -> dt.datetime:
    """Everything a live forecast from run `issue` (default: the newest complete 00z run) needs, into
    fcfg.GFSWAVE_LIVE_DIR: the run itself, the previous day's run (LIVE_PREVIOUS_LEADS) and the previous day's
    nowcasts (fcfg.GFSWAVE_NOWCAST_CYCLES, fcfg.GFSWAVE_NOWCAST_LEAD_HOURS) in GFSWAVE_LIVE_DIR/nowcast.
    Files that already hold the needed lead times are kept. Returns the issue time."""
    issue = latest_issue() if issue is None else issue
    previous = issue - dt.timedelta(days=1)
    live, nowcast = fcfg.GFSWAVE_LIVE_DIR, fcfg.GFSWAVE_LIVE_DIR / "nowcast"
    nowcast.mkdir(parents=True, exist_ok=True)
    jobs = [(issue, fcfg.GFSWAVE_LEAD_HOURS, live), (previous, LIVE_PREVIOUS_LEADS, live)]
    jobs += [(previous.replace(hour=h), fcfg.GFSWAVE_NOWCAST_LEAD_HOURS, nowcast) for h in fcfg.GFSWAVE_NOWCAST_CYCLES]
    for run, leads, out_dir in jobs:
        filename = out_dir / f"gfswave_{run:%Y%m%d}_{run:%H}z.nc"
        if not overwrite and _has_leads(filename, leads):
            print(f"{filename.name}: already there")
            continue
        download_gfswave_cycle(run, overwrite=True, leads=leads, out_dir=out_dir)
    return issue


def download_gfswave_spectra(overwrite: bool = False, max_workers: int = 4):
    """Point spectra of every run from cfg.START_DATE to cfg.END_DATE at fcfg.GFSWAVE_SPEC_STATIONS."""
    fcfg.GFSWAVE_SPEC_DIR.mkdir(parents=True, exist_ok=True)
    _run_parallel(lambda issue: download_gfswave_spectra_cycle(issue, overwrite), _issue_times(), max_workers)


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description="Download archived GFS-Wave forecasts")
    ap.add_argument("--spectra", action="store_true", help="download the point spectra instead of the gridded fields")
    ap.add_argument("--nowcast", action="store_true", help="download the nowcast fields (06/12/18z f000/f003)")
    ap.add_argument("--live", action="store_true", help="download what a live forecast needs (newest 00z run)")
    ap.add_argument("--issue", help="with --live: issue time YYYYMMDDHH instead of the newest run")
    args = ap.parse_args()
    if args.live:
        download_live(dt.datetime.strptime(args.issue, "%Y%m%d%H") if args.issue else None)
    elif args.spectra:
        download_gfswave_spectra()
    elif args.nowcast:
        download_gfswave_nowcast()
    else:
        download_gfswave()
