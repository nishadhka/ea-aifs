#!/usr/bin/env python3
"""Phase A on a Colab CPU runtime: build ERA5 starting-state pkls and put them on
source.coop (COLAB_PLAN.md section 3).

For each year, runs era5_state_pkl.py (ARCO-ERA5 read inside Google's network + MARS ecWAM
waves), uploads the pkl to hc-inputs-tmp/<MMDD>/<YYYY>/, reads its size back, and deletes
the local copy. A year whose pkl is already on source.coop is skipped, so a relaunch
resumes. JOBS years run at once (MARS tape waits then overlap).

    hc_phaseA.py --mmdd 1105 --years 2025-2006 [--member 1] [--jobs 4]

Needs: /root/.ecmwfapirc (MARS), /root/.sc.env (source.coop write), BC_CACHE with
ref_wave_land_n320.npy.
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sc_io  # noqa: E402

BUILDER = os.path.join(os.path.dirname(HERE), "era5_state_pkl.py")
WORK = os.environ.get("HC_STATES", "/content/states")


def years_of(spec):
    a, _, b = spec.partition("-")
    a, b = int(a), int(b or a)
    return list(range(a, b - 1, -1)) if a >= b else list(range(a, b + 1))


def one(mmdd, year, member):
    date = f"{year}{mmdd}_0000"
    key = sc_io.input_key(mmdd, year, member)
    if sc_io.size(key):
        print(f"[{date}] on source.coop already, skip", flush=True)
        return date, "skip", 0.0
    out = os.path.join(WORK, date)
    log = os.path.join(WORK, f"build_{date}.log")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    with open(log, "w") as fh:
        r = subprocess.run([sys.executable, "-u", BUILDER, "--date", date, "--out", out,
                            "--member", str(member)], stdout=fh, stderr=subprocess.STDOUT)
    tb = time.time() - t0
    pkl = os.path.join(out, f"input_state_member_{member:03d}.pkl")
    if r.returncode or not os.path.exists(pkl):
        print(f"[{date}] BUILD FAILED after {tb:.0f}s, see {log}", flush=True)
        return date, "fail", tb
    t1 = time.time()
    try:
        sc_io.put(pkl, key)
    except Exception as e:
        print(f"[{date}] UPLOAD FAILED: {e}; pkl kept at {pkl}", flush=True)
        return date, "fail", tb
    tu = time.time() - t1
    mb = os.path.getsize(pkl) / 1e6
    os.remove(pkl)
    print(f"[{date}] built {tb:.0f}s, uploaded {mb:.0f} MB in {tu:.0f}s -> {key}", flush=True)
    return date, "ok", tb


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmdd", required=True)
    ap.add_argument("--years", required=True, help="e.g. 2025-2006 or 2025")
    ap.add_argument("--member", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(lambda y: one(a.mmdd, y, a.member), years_of(a.years)))
    ok = [d for d, s, _ in res if s == "ok"]
    bad = [d for d, s, _ in res if s == "fail"]
    print(f"PHASE A done in {(time.time()-t0)/60:.1f} min: {len(ok)} built, "
          f"{sum(s == 'skip' for _, s, _ in res)} skipped, failed {bad or 'none'}")
    sys.exit(1 if bad else 0)
