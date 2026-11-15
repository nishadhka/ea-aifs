#!/usr/bin/env python3
"""ERA5 weekly tas / mslp / pr on the AI-WQ 1.5 deg grid, from ARCO-ERA5 (ERA5T).

Needed because the AI-WQ archive stops at the week of 2026-08-24
(ftp observations/ latest folder), so the Sep-Nov verification weeks of the past
cycles have no AI-WQ obs. ARCO full_37-1h-0p25deg-chunk-1.zarr-v3 carries ERA5T to
~5 days behind real time (attr valid_time_stop_era5t).

Weekly definitions, found by `validate` against AI-WQ's own files for weeks that
exist in both (20260727: tas rms 0.0002 K, mslp 0.02 Pa, pr 0.002 mm -- i.e. identical):
    tas, mslp   mean of the 6-hourly fields 00/06/12/18 UTC, Monday 00 .. Sunday 18 (28)
                (an hourly mean differs by 0.12 K / 15 Pa rms)
    pr          sum of hourly tp * 1000 (mm), Monday 01 .. next Monday 00 UTC (168)
    grid        every 6th 0.25 deg point (the 1.5 deg grid is a subset); an area mean
                differs by 0.57 K / 31 Pa / 6.7 mm rms

    era5_weekly.py validate 20260727 20260803 20260824
    era5_weekly.py build 20261005 20261012 ...      -> CACHE/<var>_<monday>.nc
"""

import datetime as dt
import os
import sys

import numpy as np
import xarray as xr

ARCO = ("https://storage.googleapis.com/gcp-public-data-arco-era5/ar/"
        "full_37-1h-0p25deg-chunk-1.zarr-v3")
CACHE = os.path.join(os.environ.get("BC_CACHE", "/tank/projects/bc-aifs"), "era5_weekly")
ARCO_VAR = {"tas": "2m_temperature", "mslp": "mean_sea_level_pressure",
            "pr": "total_precipitation"}
DEFAULT = dict(tas_mslp="6-hourly mean Mon00..Sun18", pr="hourly sum Mon01..Mon00",
               grid="0.25 deg every 6th point")

LAT15 = np.linspace(90, -90, 121)
LON15 = np.arange(240) * 1.5


def _open():
    # chunks=None: no dask. Recent dask materialises the store's 1.3 M-chunk graph at
    # compute and was OOM-killed on Colab (2026-11-12); lazy indexing reads only the chunks
    # selected, the same bytes.
    return xr.open_zarr(ARCO, chunks=None)


def weekly(monday):
    """dict var -> (121, 240) DataArray, AI-WQ definitions (module docstring)."""
    from concurrent.futures import ThreadPoolExecutor
    t0 = dt.datetime.strptime(monday, "%Y%m%d")
    ds = _open()
    hours = dict(tas=range(0, 168, 6), mslp=range(0, 168, 6), pr=range(1, 169))
    jobs = [(v, h) for v in hours for h in hours[v]]

    def read(job):
        v, h = job
        a = ds[ARCO_VAR[v]].sel(time=t0 + dt.timedelta(hours=h))[::6, ::6]
        return job, np.asarray(a.values, np.float64)

    with ThreadPoolExecutor(max_workers=16) as ex:
        got = dict(ex.map(read, jobs))
    res = {}
    for v in hours:
        x = np.stack([got[(v, h)] for h in hours[v]])
        if not np.isfinite(x).all(axis=(1, 2)).all():
            raise RuntimeError(f"{monday} {v}: missing hours in ARCO (ERA5T not out yet?)")
        res[v] = xr.DataArray(x.sum(0) * 1000 if v == "pr" else x.mean(0),
                              dims=("latitude", "longitude"),
                              coords=dict(latitude=LAT15, longitude=LON15))
    return res


def build(mondays):
    os.makedirs(CACHE, exist_ok=True)
    for m in mondays:
        if all(os.path.exists(f"{CACHE}/{v}_{m}.nc") for v in ARCO_VAR):
            print(f"{m}: cached")
            continue
        w = weekly(m)
        for v, a in w.items():
            a = a.rename(v)
            a.attrs.update(source="ARCO-ERA5 (ERA5T)", **DEFAULT)
            a.to_netcdf(f"{CACHE}/{v}_{m}.nc")
        print(f"{m}: built")


def load(monday, var):
    f = f"{CACHE}/{var}_{monday}.nc"
    return xr.open_dataarray(f) if os.path.exists(f) else None


def validate(mondays):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import past_cycle_bc as B
    w = np.cos(np.radians(LAT15))[:, None] * np.ones((1, 240))
    for m in mondays:
        mine = weekly(m)
        for v, a in mine.items():
            ref = B.obs(m, v)
            if ref is None:
                print(f"{m} {v}: no AI-WQ obs")
                continue
            d = a.values - np.asarray(ref.values, np.float64)
            print(f"{m} {v:5s} bias {np.sum(d*w)/np.sum(w):+.5f}  "
                  f"rms {np.sqrt(np.sum(d**2*w)/np.sum(w)):.5f}  max|d| {np.abs(d).max():.5f}")


if __name__ == "__main__":
    cmd, days = sys.argv[1], sys.argv[2:]
    {"validate": validate, "build": build}[cmd](days)
