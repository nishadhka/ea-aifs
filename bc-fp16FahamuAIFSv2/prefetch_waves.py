#!/usr/bin/env python3
"""Prefetch the MARS wave files era5_state_pkl.py needs, so tape waits overlap.

era5_state_pkl.py fetches ARCO, then MARS, one date after another; a MARS request that
waits for tape (40+ min seen for ikg0 2023) stalls every later date. This fetches the
same files (same names fetch_j1r2.fetch_cycle looks for) into a staging folder and
renames them into the cache only when complete, so the builder never reads a partial
GRIB. Run it in the opposite year order to the builder.

    prefetch_waves.py 1105 2006 2022
"""
import datetime as dt
import os
import shutil
import sys

sys.path.insert(0, "/home/ezra/ea-aifs/fp16FahamuAIFSv2/run-pre50r1-dates")
import fetch_j1r2 as J  # noqa: E402

CACHE = "/tank/projects/bc-aifs/mars_waves"
STAGE = "/tank/projects/bc-aifs/mars_waves_stage"

md, y0, y1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
os.makedirs(STAGE, exist_ok=True)
ids = list(J.PARAMS)                      # all_wave=True: all 11 params
for y in range(y0, y1 + 1):
    t1 = dt.datetime(y, int(md[:2]), int(md[2:]), 0)
    ev = J.expver_for(t1.strftime("%Y%m%d"))
    for t in (t1 - dt.timedelta(hours=6), t1):
        d, hh = t.strftime("%Y%m%d"), t.strftime("%H")
        name = f"{ev}_all_{d}_{hh}.grib"
        if os.path.exists(os.path.join(CACHE, name)):
            print(f"{name}: cached", flush=True)
            continue
        tmp = os.path.join(STAGE, name)
        try:
            J.retrieve(d, hh, ids, tmp, grid="N320", expver=ev)
            shutil.move(tmp, os.path.join(CACHE, name))   # same filesystem: atomic rename
            print(f"{name}: fetched", flush=True)
        except Exception as e:
            print(f"{name}: FAILED {e}", flush=True)
