#!/usr/bin/env python3
"""Compact hand-off of a Colab cycle for the evaluation / correction (RUNBOOK_20261112 §4).

Reads the per-member products on source.coop and writes, under OUT (a BC_CACHE-like tree):

    weekly/hc<YYYY><MMDD>.npz      tas, mslp, pr: (2 weeks, 11 members, 121, 240) float64,
                                   the past_cycle_bc.aggregate definition (weeks 3 and 4)
    weekly/<RTDATE>.npz            the same for the real-time members (year RTYEAR)
    hc15/<YYYY><MMDD>/m<NNN>/      ts_days.nc, mjo_bands.npz, mjo_probs.nc, indices.npz
    rt15/<RTDATE>/m<NNN>/          the same for the real-time members
    era5_weekly/, truth_idx/       copied from BC_CACHE if present (truth)

then zips OUT and puts it on source.coop at hc-products/<MMDD>/bundle_<MMDD>.zip.

    hc_bundle.py --mmdd 1112 --years 2006-2025 --rt-year 2026 --out /content/bundle1112
"""

import argparse
import os
import shutil
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import sc_io  # noqa: E402
import past_cycle_bc as B  # noqa: E402

SMALL = ("ts_days.nc", "mjo_bands.npz", "mjo_probs.nc", "indices.npz", "timing.json")


def members_of(mmdd, year):
    keys = sc_io.ls(sc_io.BASE + f"/hc-products/{mmdd}/{year}/")
    done = sorted({k.split("/")[-2] for k, _ in keys if k.endswith("/DONE")})
    return [int(m[1:]) for m in done]


def fetch_member(mmdd, year, m, work):
    key = sc_io.product_prefix(mmdd, year, m) + "aifs_ensemble_forecast_1p5deg_member001.nc"
    path = os.path.join(work, f"{year}_m{m:03d}.nc")
    sc_io.get(key, path)
    return path


def weekly_member(path):
    """netCDF/HDF5 is not thread-safe (segfault with 8 threads): open one file at a time."""
    out = {}
    with xr.open_dataset(path) as d:
        for v in B.VARS:
            x = d[B.RAW[v]].isel(member=0)
            arr = np.empty((2, 121, 240))
            for w, (chunks, nsteps) in enumerate(B.WEEKS):
                s = sum(x.isel(time=t).sum("step", skipna=True) for t in chunks).values
                arr[w] = s * 1000 * 7.0 / (nsteps * 6 / 24) if v == "pr" else s / nsteps
            out[v] = arr
    os.remove(path)
    return out


def one_set(mmdd, year, tag, small_dir, out, work):
    mem = members_of(mmdd, year)
    with ThreadPoolExecutor(8) as ex:                 # downloads in parallel
        paths = list(ex.map(lambda m: fetch_member(mmdd, year, m, work), mem))
    res = [weekly_member(p) for p in paths]           # reads one at a time
    arr = {v: np.stack([r[v] for r in res], axis=1) for v in B.VARS}       # (2, n, 121, 240)
    os.makedirs(os.path.join(out, "weekly"), exist_ok=True)
    np.savez(os.path.join(out, "weekly", f"{tag}.npz"), **arr,
             files=np.array([f"m{m:03d}" for m in mem]))
    for m in mem:
        d = os.path.join(out, small_dir, f"m{m:03d}")
        os.makedirs(d, exist_ok=True)
        for n in SMALL:
            k = sc_io.product_prefix(mmdd, year, m) + n
            if sc_io.size(k):
                sc_io.get(k, os.path.join(d, n))
    print(f"{tag}: {len(mem)} members", flush=True)
    return len(mem)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmdd", required=True)
    ap.add_argument("--years", default=None, help="hindcast years, e.g. 2006-2025 (omit: real-time only)")
    ap.add_argument("--rt-year", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-upload", action="store_true")
    a = ap.parse_args()
    work = a.out + "_work"
    os.makedirs(work, exist_ok=True)
    if a.years:
        y0, _, y1 = a.years.partition("-")
        for y in range(int(y0), int(y1 or y0) + 1):
            one_set(a.mmdd, y, f"hc{y}{a.mmdd}", f"hc15/{y}{a.mmdd}", a.out, work)
    if a.rt_year:
        rt = f"{a.rt_year}{a.mmdd}"
        one_set(a.mmdd, a.rt_year, rt, f"rt15/{rt}", a.out, work)
    cache = os.environ.get("BC_CACHE", "/content/bc")
    for sub in ("era5_weekly", "truth_idx"):
        if os.path.isdir(os.path.join(cache, sub)):
            shutil.copytree(os.path.join(cache, sub), os.path.join(a.out, sub), dirs_exist_ok=True)
    z = f"{a.out}.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(a.out):
            for f in files:
                p = os.path.join(root, f)
                zf.write(p, os.path.relpath(p, os.path.dirname(a.out)))
    print(f"zip {os.path.getsize(z)/1e6:.0f} MB -> {z}", flush=True)
    if not a.no_upload:
        key = f"{sc_io.BASE}/hc-products/{a.mmdd}/bundle_{a.mmdd}.zip"
        sc_io.put(z, key)
        print(f"uploaded -> {key}", flush=True)
