#!/usr/bin/env python3
"""Phase A on a Colab CPU runtime: build ERA5 starting-state pkls and put them on
source.coop (COLAB_PLAN.md section 3).

For each year, runs era5_state_pkl.py (ARCO-ERA5 read inside Google's network + MARS ecWAM
waves), uploads the pkl to hc-inputs-tmp/<MMDD>/<YYYY>/, reads its size back, and deletes
the local copy. A year whose pkl is already on source.coop is skipped, so a relaunch
resumes. JOBS years run at once (MARS tape waits then overlap).

    hc_phaseA.py --mmdd 1105 --years 2025-2006 [--member 1] [--jobs 4]
    hc_phaseA.py --mmdd 1105 --years 2025-2006 --perturbed 2-11 [--jobs 2]
    hc_phaseA.py --realtime 20261112 [--rt-members 1-51] [--jobs 4]   -> 1112/2026/m001..051

--perturbed: members 2..11 from the control already on source.coop plus the ERA5 EDA
(era5_eda_perturb.py): per year, fetch the EDA (CDS, one request at a time across all
threads via api_polite), download the control, build the missing members, upload each,
delete local pkls. Years with every member on source.coop are skipped.

Needs: /root/.ecmwfapirc (MARS), /root/.cdsapirc (CDS, --perturbed), /root/.sc.env
(source.coop write), BC_CACHE with ref_wave_land_n320.npy (control).
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
PERTURB = os.path.join(os.path.dirname(HERE), "era5_eda_perturb.py")
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
    for attempt in range(3):      # earthkit's sqlite cache can be locked by parallel builds
        with open(log, "w") as fh:
            r = subprocess.run([sys.executable, "-u", BUILDER, "--date", date, "--out", out,
                                "--member", str(member)], stdout=fh, stderr=subprocess.STDOUT)
        if r.returncode == 0:
            break
        print(f"[{date}] build attempt {attempt+1} failed, retrying", flush=True)
        time.sleep(30 * (attempt + 1))
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


PROTO = os.path.join(os.path.dirname(os.path.dirname(HERE)), "s3_grib_pkl",
                     "s3_grib_pkl_input_aifsens_v2_proto.py")


def one_realtime(date, member, control_member=51):
    """Real-time IFS open-data member (pf 1..50; 51 = unperturbed control from HRES 0 h,
    as the real-time chain) -> hc-inputs-tmp/<MMDD>/<YYYY>/ like a hindcast member, so the
    GPU worker runs it unchanged. Open data is anonymous GCS: no MARS/CDS."""
    year, mmdd = int(date[:4]), date[4:8]
    key = sc_io.input_key(mmdd, year, member)
    if sc_io.size(key):
        return member, "skip"
    out = os.path.join(WORK, f"rt_{date}_m{member:03d}")
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "-u", PROTO, "--date", f"{date}_0000", "--member", str(member),
           "--source", "gcs", "--out", out]
    if member == control_member:
        cmd += ["--control-member", str(control_member)]
    pkl = os.path.join(out, f"proto_input_state_member_{member:03d}.pkl")
    for attempt in range(3):   # earthkit cache races between parallel builds (4/51 on 19 Nov)
        with open(os.path.join(WORK, f"rt_{date}_m{member:03d}.log"), "w") as fh:
            r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
        if r.returncode == 0 and os.path.exists(pkl):
            break
        time.sleep(20 * (attempt + 1))
    if r.returncode or not os.path.exists(pkl):
        print(f"[{date} m{member:03d}] REALTIME BUILD FAILED", flush=True)
        return member, "fail"
    try:
        sc_io.put(pkl, key)
    except Exception as e:
        print(f"[{date} m{member:03d}] UPLOAD FAILED: {e}", flush=True)
        return member, "fail"
    os.remove(pkl)
    print(f"[{date} m{member:03d}] real-time pkl -> {key}", flush=True)
    return member, "ok"


def eda_cache(mmdd, year, direction):
    """Restore the year's 4 EDA GRIBs from source.coop ('get') or save them ('put')."""
    sys.path.insert(0, os.path.dirname(HERE))
    import era5_eda_perturb as E
    for path in E.eda_paths(year, mmdd).values():
        key = sc_io.eda_key(mmdd, year, os.path.basename(path))
        if direction == "get" and not os.path.exists(path) and sc_io.size(key):
            sc_io.get(key, path)
        elif direction == "put" and os.path.exists(path) and sc_io.size(key) != os.path.getsize(path):
            sc_io.put(path, key)


def one_perturbed(mmdd, year, members, inflate=1.0):
    date = f"{year}{mmdd}_0000"
    todo = [k for k in members if not sc_io.size(sc_io.input_key(mmdd, year, k))]
    if not todo:
        print(f"[{date}] members {members[0]}-{members[-1]} on source.coop already, skip", flush=True)
        return date, "skip", 0.0
    if not sc_io.size(sc_io.input_key(mmdd, year, 1)):
        print(f"[{date}] no control pkl on source.coop yet: skipped, rerun later", flush=True)
        return date, "fail", 0.0
    out = os.path.join(WORK, date)
    log = os.path.join(WORK, f"perturb_{date}.log")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    eda_cache(mmdd, year, "get")             # never ask CDS for what is already stored
    with open(log, "w") as fh:
        r = subprocess.run([sys.executable, "-u", PERTURB, "fetch", "--mmdd", mmdd,
                            "--years", str(year)], stdout=fh, stderr=subprocess.STDOUT)
        if r.returncode:
            print(f"[{date}] EDA FETCH FAILED, see {log}", flush=True)
            return date, "fail", time.time() - t0
        eda_cache(mmdd, year, "put")
        tf = time.time() - t0
        ctl = os.path.join(out, "input_state_member_001.pkl")
        if not os.path.exists(ctl):
            sc_io.get(sc_io.input_key(mmdd, year, 1), ctl)
        spec = ",".join(map(str, todo))
        r = subprocess.run([sys.executable, "-u", PERTURB, "build", "--mmdd", mmdd,
                            "--year", str(year), "--control", ctl, "--out", out,
                            "--members", f"{todo[0]}-{todo[-1]}", "--inflate", str(inflate)],
                           stdout=fh, stderr=subprocess.STDOUT)
    tb = time.time() - t0 - tf
    if r.returncode:
        print(f"[{date}] PERTURB BUILD FAILED, see {log}", flush=True)
        return date, "fail", time.time() - t0
    t1 = time.time()
    for k in todo:
        pkl = os.path.join(out, f"input_state_member_{k:03d}.pkl")
        try:
            sc_io.put(pkl, sc_io.input_key(mmdd, year, k))
        except Exception as e:
            print(f"[{date}] m{k:03d} UPLOAD FAILED: {e}; pkl kept", flush=True)
            return date, "fail", time.time() - t0
        os.remove(pkl)
    os.remove(os.path.join(out, "input_state_member_001.pkl"))
    print(f"[{date}] members {spec}: EDA {tf:.0f}s, build {tb:.0f}s, "
          f"upload {time.time()-t1:.0f}s", flush=True)
    return date, "ok", time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmdd", default=None)
    ap.add_argument("--years", default=None, help="e.g. 2025-2006 or 2025")
    ap.add_argument("--realtime", default=None,
                    help="YYYYMMDD: build the 51 real-time IFS open-data members instead")
    ap.add_argument("--rt-members", default="1-51")
    ap.add_argument("--member", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--perturbed", default=None, help="e.g. 2-11: EDA-perturbed members")
    ap.add_argument("--inflate", type=float, default=1.0, help="EDA perturbation scale factor")
    a = ap.parse_args()
    t0 = time.time()
    if a.realtime:
        lo, _, hi = a.rt_members.partition("-")
        mem = list(range(int(lo), int(hi or lo) + 1))
        with ThreadPoolExecutor(a.jobs) as ex:
            res = list(ex.map(lambda m: one_realtime(a.realtime, m), mem))
        bad = [m for m, st in res if st == "fail"]
        print(f"REALTIME {a.realtime} done in {(time.time()-t0)/60:.1f} min: "
              f"{sum(st == 'ok' for _, st in res)} built, failed {bad or 'none'}")
        sys.exit(1 if bad else 0)
    if a.perturbed:
        lo, _, hi = a.perturbed.partition("-")
        mem = list(range(int(lo), int(hi or lo) + 1))
        job = lambda y: one_perturbed(a.mmdd, y, mem, a.inflate)
    else:
        job = lambda y: one(a.mmdd, y, a.member)
    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(job, years_of(a.years)))
    ok = [d for d, s, _ in res if s == "ok"]
    bad = [d for d, s, _ in res if s == "fail"]
    print(f"PHASE A done in {(time.time()-t0)/60:.1f} min: {len(ok)} built, "
          f"{sum(s == 'skip' for _, s, _ in res)} skipped, failed {bad or 'none'}")
    sys.exit(1 if bad else 0)
