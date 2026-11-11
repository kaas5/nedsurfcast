#!/usr/bin/env python3
"""
3D plot of one day of measured wave spectra for a Rijkswaterstaat station.

  x-axis: frequency (mHz)
  y-axis: energy per frequency bin (cm^2) = energy density * bin width (10 mHz)
  z-axis: time of the measurement (UTC)

Data comes from the public JSON API behind the RWsOS viewer
(https://rwsos.rws.nl/viewer/):  https://rwsos.rws.nl/wb-api/dd/2.0/

Spectrum parameters (source "S_1" = measurements), 51 frequency bins each,
bin i covers [10*i, 10*i + 10) mHz:
    GD.28-78   energy density            (cm^2 s)
    GR.1-51    mean wave direction       (degrees, "coming from")
    GR.52-102  directional spread        (degrees)

Examples:
    python rws_wave_spectrum.py --list
    python rws_wave_spectrum.py DEUR 2026-11-08
    python rws_wave_spectrum.py EPL 2026-11-08 --save epl.png
"""
import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

BASE = "https://rwsos.rws.nl/wb-api/dd/2.0"
BBOX = "[2,48.56,7.5,57]"
SOURCE = "S_1"
ENERGY = "GD.28-78"
MISSING = 99999
NBINS = 51
BIN_MHZ = 10
F_MID_MHZ = np.arange(NBINS) * BIN_MHZ + BIN_MHZ / 2
F_RANGE_MHZ = (30, 500)  # below 30 mHz the buoy data is (near) zero / unreliable


def get_json(path, **params):
    url = f"{BASE}/{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "rws-wave-spectrum-script"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            print(f"  retry after error: {e}", file=sys.stderr)
            time.sleep(2 ** attempt)


def stations():
    d = get_json("locations/geojson", sourceName=SOURCE, observationTypeId=ENERGY, boundingBox=BBOX)
    out = {}
    for f in d["features"]:
        p = f["properties"]
        lon, lat = f["geometry"]["coordinates"][:2]
        out.setdefault(p["id"], (p.get("locationName", ""), lat, lon))
    return out


def fetch(station, obs_id, start, end):
    """Return (times, array[n_times, 51]) with NaN for missing values."""
    d = get_json("timeseries", observationTypeId=obs_id, sourceName=SOURCE, locationCode=station,
                 startTime=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                 endTime=end.strftime("%Y-%m-%dT%H:%M:%SZ"))
    rows = {}
    for res in d.get("results", []):
        for ev in res["events"]:
            v = ev.get("value")
            if isinstance(v, list) and len(v) == NBINS:
                t = datetime.fromisoformat(ev["timeStamp"].replace("Z", "+00:00"))
                rows[t] = [np.nan if x is None or x == MISSING else x for x in v]
    times = sorted(rows)
    return times, np.array([rows[t] for t in times], dtype=float).reshape(-1, NBINS)


def plot_day(station, day, save=None):
    end = day + timedelta(days=1) - timedelta(seconds=1)
    names = stations()
    label = f"{station} – {names[station][0]}" if station in names else station

    times, E = fetch(station, ENERGY, day, end)
    if not times:
        sys.exit(f"No spectra for {station} on {day:%Y-%m-%d}. Check the code with --list.")

    energy = np.nan_to_num(E) * (BIN_MHZ / 1000)  # cm^2 s * Hz -> cm^2 per bin
    hours = np.array([(t - day).total_seconds() / 3600 for t in times])
    keep = (F_MID_MHZ >= F_RANGE_MHZ[0]) & (F_MID_MHZ <= F_RANGE_MHZ[1])
    f, energy = F_MID_MHZ[keep], energy[:, keep]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(projection="3d")
    cmap = plt.get_cmap("viridis")
    colors = [cmap(h / 24) for h in hours]

    # Draw back to front so nearer spectra overlap the later ones visually
    order = np.argsort(hours)
    polys = [np.column_stack([np.r_[f[0], f, f[-1]], np.r_[0, energy[i], 0]]) for i in order]
    fill = PolyCollection(polys, facecolors=[colors[i] for i in order], alpha=0.25, edgecolors="none")
    ax.add_collection3d(fill, zs=hours[order], zdir="z")
    for i in order:
        ax.plot(f, energy[i], zs=hours[i], zdir="z", color=colors[i], lw=1.2)

    # Frequency on x, energy on y (vertical), time on z (depth)
    ax.view_init(elev=25, azim=30, vertical_axis="y")
    ax.set_xlim(*F_RANGE_MHZ)
    ax.set_ylim(0, energy.max() * 1.05)
    ax.set_zlim(0, 24)  # most recent measurement in front, 00:00 at the back
    ax.set_zticks(range(0, 25, 3))
    ax.set_zticklabels([f"{h:02d}:00" for h in range(0, 25, 3)])
    ax.set_xlabel("Frequency [mHz]", labelpad=10)
    ax.set_ylabel("Energy per bin [cm²]", labelpad=10)
    ax.set_zlabel("Time (UTC)", labelpad=12)
    ax.set_box_aspect((1.4, 0.8, 1.4))
    ax.set_title(f"Wave spectra {label}, {day:%Y-%m-%d} ({len(times)} measurements)", pad=20)

    fig.subplots_adjust(left=0, right=1, bottom=0.02, top=0.95)
    if save:
        fig.savefig(save, dpi=130)
        print(f"Saved {save}", file=sys.stderr)
    else:
        plt.show()  # drag with the mouse to rotate


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("station", nargs="?", default="DEUR", help="location code (default DEUR, see --list)")
    ap.add_argument("date", nargs="?", help="day in UTC, YYYY-MM-DD (default: yesterday)")
    ap.add_argument("--save", metavar="FILE", help="save the figure to FILE instead of showing it")
    ap.add_argument("--list", action="store_true", help="list stations that publish spectra")
    a = ap.parse_args()

    if a.list:
        for code, (name, lat, lon) in stations().items():
            print(f"{code:<10} {lat:8.4f} {lon:8.4f}  {name}")
        return

    if a.date:
        day = datetime.strptime(a.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        day = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
    plot_day(a.station, day, a.save)


if __name__ == "__main__":
    main()
