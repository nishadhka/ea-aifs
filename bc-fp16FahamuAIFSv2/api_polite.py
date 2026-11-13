"""One request at a time to ECMWF MARS (Web API, .ecmwfapirc) and to the CDS (.cdsapirc).

Both services rate-limit and can block an account that floods them, so every script in
this folder that talks to them goes through `polite(service)`:

    from api_polite import polite
    with polite("mars"):
        ECMWFDataServer().retrieve(req)

- a cross-process file lock per service: however many threads or processes build states
  on this machine, at most ONE request per service is in flight (queued, running or
  downloading). Threads and parallel builds still overlap their other work (ARCO reads,
  regridding, writing).
- a minimum gap between the end of one request and the start of the next
  (MIN_GAP_S, default 20 s for CDS, 5 s for MARS).
- callers check their cache before calling, so a rerun never repeats a request.

The lock is per machine (API_LOCK_DIR, default ~/.cache/api_locks). Run MARS/CDS fetching
on ONE machine or Colab runtime at a time; never fan it out across runtimes.
"""

import contextlib
import fcntl
import os
import time

LOCK_DIR = os.environ.get("API_LOCK_DIR", os.path.expanduser("~/.cache/api_locks"))
MIN_GAP_S = {"cds": float(os.environ.get("CDS_MIN_GAP_S", 20)),
             "mars": float(os.environ.get("MARS_MIN_GAP_S", 5))}


@contextlib.contextmanager
def polite(service):
    os.makedirs(LOCK_DIR, exist_ok=True)
    path = os.path.join(LOCK_DIR, f"{service}.lock")
    with open(path, "a+") as fh:
        t0 = time.time()
        fcntl.flock(fh, fcntl.LOCK_EX)            # wait for any other request to finish
        waited = time.time() - t0
        if waited > 1:
            print(f"  [{service}] waited {waited:.0f} s for the previous request", flush=True)
        fh.seek(0)
        try:
            last = float(fh.read().strip() or 0)
        except ValueError:
            last = 0.0
        gap = MIN_GAP_S.get(service, 10) - (time.time() - last)
        if gap > 0:
            time.sleep(gap)
        try:
            yield
        finally:
            fh.seek(0)
            fh.truncate()
            fh.write(f"{time.time():.0f}")
            fh.flush()
            fcntl.flock(fh, fcntl.LOCK_UN)
