"""
Download ERA5 wind + pressure from the Copernicus CDS (nowcast pipeline)
==========================================================================

Run from anywhere with: python -m nedsurfcast.download.era5
"""

import datetime as dt

from nedsurfcast import config as cfg
from nedsurfcast.nowcast import config as ncfg


def cds_time_selections(start: dt.datetime = cfg.START_DATE, end: dt.datetime = cfg.END_DATE,
                        step_hours: int = 3) -> list[dict]:
    """Year/month/day/time selections for the CDS API covering [start, end], one per calendar month.

    CDS returns the cross product of the year, month, day and time lists, so a single request for e.g.
    2024-11-15 .. 2025-01-10 would also fetch all of Jan-Feb 2024 and Nov-Dec 2025. Splitting per month
    keeps every request exact at day level (ECMWF also recommends monthly ERA5 requests). The first and
    last day are fetched for all hours; subset() trims to DATE_RANGE afterwards.
    """
    times = [f"{h:02d}:00" for h in range(0, 24, step_hours)]
    selections = []
    month_start = dt.datetime(start.year, start.month, 1)
    while month_start <= end:
        next_month = (month_start + dt.timedelta(days=32)).replace(day=1)
        first_day = max(start, month_start).day
        last_day = min(end, next_month - dt.timedelta(days=1)).day
        selections.append({
            "year": [str(month_start.year)],
            "month": [f"{month_start.month:02d}"],
            "day": [f"{d:02d}" for d in range(first_day, last_day + 1)],
            "time": times,
        })
        month_start = next_month
    return selections

def download_era5_wind_example(overwrite: bool = False):
    """Example CDS API call for ERA5 10m wind + pressure, one file per month
    (era5_wind_YYYY_MM.nc). Requires a configured ~/.cdsapirc with your API key."""
    import cdsapi

    ncfg.ERA5_DIR.mkdir(parents=True, exist_ok=True)
    c = cdsapi.Client(progress=True)
    for selection in cds_time_selections():
        filename = ncfg.ERA5_DIR / f"era5_wind_{selection['year'][0]}_{selection['month'][0]}.nc"
        if filename.exists() and not overwrite:
            print(f"{filename.name}: already exists and overwrite=False, skipping")
            continue

        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure"],
                **selection,
                "area": [62, -4, 51, 9],  # N, W, S, E
                "format": "netcdf",
                "download_format": "unarchived",
            },
            str(filename),
        )


if __name__ == '__main__':
    download_era5_wind_example()
