#!/usr/bin/env python3
"""Phase B on a Colab GPU runtime: claim-and-skip worker over the pkls on source.coop
(COLAB_PLAN.md section 3, routine B3).

Loop: list hc-inputs-tmp/<MMDD>/, take the next (year, member) without a DONE marker in
hc-products/ and without a fresh CLAIM by another runtime, download its pkl (anonymous),
run hc_member.sh unchanged (792 h rollout + extraction, stores deleted at the end), upload
the products plus timing.json, write DONE last. Any number of runtimes can run this at
once; a killed runtime leaves a CLAIM that expires after --claim-ttl and is then retaken.

    hc_gpu_worker.py --mmdd 1105 [--years 2025-2006] [--members 1-11] [--max 1]

Needs: /content/venv (hc_gpu_setup.sh), /root/.sc.env (source.coop write).
"""

import argparse
import json
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sc_io  # noqa: E402

BC = os.path.dirname(HERE)
ROOT = os.path.dirname(BC)
HOST = f"{socket.gethostname()}-{os.getpid()}"
ENV = dict(os.environ,
           HC_PY="/content/venv/bin/python",
           HC_REPO=os.path.join(ROOT, "fp16FahamuAIFSv2"),
           HC_WORK="/content/hc_work",
           HC_VPM="/content/aux",
           HC_LSM_1P5="/content/aux/land_sea_mask_1pt5DEG.nc",
           HC_OPTIONAL="1",
           HF_HOME=os.environ.get("HF_HOME", "/content/hf_cache"))


ZLIB = """
import glob, os, sys, xarray as xr
for p in glob.glob(os.path.join(sys.argv[1], "*.nc")):
    ds = xr.load_dataset(p)
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(p + ".z", encoding=enc)
    os.replace(p + ".z", p)
"""


def rng(spec):
    if not spec:
        return None
    out = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        a, b = int(a), int(b or a)
        out |= set(range(min(a, b), max(a, b) + 1))
    return out


def todo(mmdd, years, members):
    pat = re.compile(rf"hc-inputs-tmp/{mmdd}/(\d{{4}})/input_state_member_(\d{{3}})\.pkl$")
    items = []
    for key, size in sc_io.ls(f"{sc_io.BASE}/hc-inputs-tmp/{mmdd}/"):
        m = pat.search(key)
        if m and size > 0:
            y, n = int(m.group(1)), int(m.group(2))
            if (years is None or y in years) and (members is None or n in members):
                items.append((y, n, key))
    return sorted(items, key=lambda t: (-t[0], t[1]))


def _read_claim(prefix):
    """CLAIM body via the S3 API (not the public URL, which may be cached), or None."""
    try:
        body = sc_io._anon().get_object(Bucket=sc_io.BUCKET, Key=prefix + "CLAIM")["Body"].read()
        return json.loads(body)
    except Exception:
        return None


def claim(prefix, ttl):
    """Write a CLAIM, wait, read it back: only the runtime whose claim survived proceeds.
    (Without the read-back two workers that looked at the same moment both ran the member,
    2026-11-13.)"""
    if sc_io.size(prefix + "DONE"):
        return False
    info = _read_claim(prefix)
    if info and info["host"] != HOST and time.time() - info["t"] < ttl:
        return False
    sc_io.put_bytes(json.dumps({"host": HOST, "t": time.time()}).encode(), prefix + "CLAIM")
    time.sleep(8 + random.random() * 4)
    info = _read_claim(prefix)
    return bool(info) and info["host"] == HOST


class GpuPeak(threading.Thread):
    """Peak GPU memory (MiB) while a member runs, from nvidia-smi every 2 s."""
    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.stop = 0, threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                      "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True).stdout
                self.peak = max(self.peak, int(out.split()[0]))
            except Exception:
                pass
            self.stop.wait(2)


def stage_times(log):
    """Seconds per hc_member.sh stage from its '### HH:MM:SS <date> <stage>' lines."""
    marks = re.findall(r"^### (\d\d):(\d\d):(\d\d) \S+ (.*)$", log, re.M)
    secs = [(int(h) * 3600 + int(m) * 60 + int(s), what) for h, m, s, what in marks]
    out = {}
    for (t0, what), (t1, _) in zip(secs, secs[1:]):
        out[what.split(" (")[0].split(":")[0]] = (t1 - t0) % 86400
    return out


def run_one(mmdd, year, member, key):
    date = f"{year}{mmdd}_0000"
    prefix = sc_io.product_prefix(mmdd, year, member)
    state = f"/content/states/{date}_m{member:03d}"
    out = f"/content/hc_out/{date}_m{member:03d}"
    shutil.rmtree(out, ignore_errors=True)
    t0 = time.time()
    sc_io.get(key, os.path.join(state, "input_state_member_001.pkl"))
    t_dl = time.time() - t0
    gpu = GpuPeak(); gpu.start()
    t1 = time.time()
    r = subprocess.run(["bash", os.path.join(BC, "hc_member.sh"), date, state, out],
                       env=ENV, capture_output=True, text=True)
    t_run = time.time() - t1
    gpu.stop.set()
    print(r.stdout, end="", flush=True)
    shutil.rmtree(state, ignore_errors=True)
    if r.returncode or not os.path.exists(os.path.join(out, "DONE")):
        print(f"[{date} m{member:03d}] FAILED rc={r.returncode}\n{r.stderr[-2000:]}", flush=True)
        sc_io.rm(prefix + "CLAIM")
        return None
    timing = dict(date=date, member=member, host=HOST, gpu=gpu_name(),
                  download_s=round(t_dl, 1), member_s=round(t_run, 1),
                  stages_s=stage_times(r.stdout), peak_gpu_mib=gpu.peak,
                  finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with open(os.path.join(out, "timing.json"), "w") as fh:
        json.dump(timing, fh, indent=1)
    # lossless zlib on the NetCDF products before upload: the 1.5 deg file is written as
    # uncompressed float64 (209 MB; "41 MB" on the GPU box is ZFS's on-disk size)
    subprocess.run([ENV["HC_PY"], "-c", ZLIB, out], check=False)
    t2 = time.time()
    names = sorted(n for n in os.listdir(out) if n != "DONE")
    for n in names:
        sc_io.put(os.path.join(out, n), prefix + n)
    sc_io.put(os.path.join(out, "DONE"), prefix + "DONE")
    sc_io.rm(prefix + "CLAIM")
    timing["upload_s"] = round(time.time() - t2, 1)
    shutil.rmtree(out, ignore_errors=True)
    print(f"[{date} m{member:03d}] DONE {json.dumps(timing)}", flush=True)
    return timing


def gpu_name():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "?"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmdd", required=True)
    ap.add_argument("--years", default=None, help="e.g. 2006-2025")
    ap.add_argument("--members", default=None, help="e.g. 1 or 1-11")
    ap.add_argument("--max", type=int, default=0, help="stop after this many members (0 = all)")
    ap.add_argument("--claim-ttl", type=float, default=3 * 3600)
    ap.add_argument("--wait", type=float, default=0,
                    help="when nothing is claimable, poll every 2 min for new pkls for up to "
                         "this many seconds (pkls still being built), then exit")
    a = ap.parse_args()
    years, members = rng(a.years), rng(a.members)
    time.sleep(random.random() * 30)          # stagger workers started together
    done, failed, t0 = [], set(), time.time()
    idle_since = None
    while True:
        picked = None
        for y, n, key in todo(a.mmdd, years, members):
            if (y, n) in failed:          # failed in this process: leave it for a rerun
                continue
            if claim(sc_io.product_prefix(a.mmdd, y, n), a.claim_ttl):
                picked = (y, n, key)
                break
        if not picked:
            idle_since = idle_since or time.time()
            if a.wait and time.time() - idle_since < a.wait:
                time.sleep(120)
                continue
            break
        idle_since = None
        t = run_one(a.mmdd, *picked)
        if t:
            done.append(t)
        else:
            failed.add(picked[:2])
            if len(failed) >= 3:
                print("WORKER: 3 members failed, stopping"); break
        if a.max and len(done) >= a.max:
            break
    if done:
        avg = sum(t["member_s"] for t in done) / len(done)
        print(f"WORKER done: {len(done)} members in {(time.time()-t0)/60:.1f} min, "
              f"mean {avg/60:.2f} min per member, peak GPU "
              f"{max(t['peak_gpu_mib'] for t in done)} MiB")
    else:
        print("WORKER: nothing left to run")
    if failed:
        print(f"WORKER failed: {sorted(failed)}")
