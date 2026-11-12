#!/usr/bin/env python3
"""ERA5 truth for the Africa / Indian Ocean indices (README 0.2, variables 6-15), computed
with the SAME function as the model side (hc_extract.compute), so model and truth share
one definition.

Data: CDS reanalysis-era5-single-levels / -pressure-levels, clipped to AREA at 1 deg,
6-hourly for instantaneous fields (the model's step times) and hourly for tp (summed into
the model's 6 h accumulation windows). Dates D .. D+33 for a start date D, two requests
per month per dataset, cached as GRIB in CACHE/era5_idx/<YYYYMMDD>/.

Not used for TS days or MJO: their truth is global (IBTrACS, AI-WQ MJO files), see
hc_truth.py. This script stops if any index box or weekly-map domain lies outside AREA,
so an index can never be scored against clipped data by mistake.

    era5_truth_indices.py 1105 2006 2025          -> CACHE/truth_idx/<YYYYMMDD>.npz
"""

import datetime as dt
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/ezra/ea-aifs/fp16FahamuAIFSv2/run-pre50r1-dates")
import hc_extract as X  # noqa: E402

CACHE = "/tank/projects/bc-aifs"
AREA = (40, -25, -40, 120)                    # N, W, S, E (CDS order)
GRID = 1.0
NLAT, NLON = 81, 146
LAT = np.linspace(AREA[0], AREA[2], NLAT)
LON = np.linspace(AREA[1], AREA[3], NLON)
PL_VARS = {"u_component_of_wind": "u", "v_component_of_wind": "v", "specific_humidity": "q"}
PL_LEVELS = ["925", "850", "700", "600", "200", "150"]
SL_VARS = {"skin_temperature": "skt", "volumetric_soil_water_layer_1": "swvl1",
           "volumetric_soil_water_layer_2": "swvl2", "mean_sea_level_pressure": "msl",
           "total_column_water": "tcw", "significant_height_of_combined_wind_waves_and_swell": "swh",
           "mean_wave_period": "mwp", "2m_temperature": "2t", "2m_dewpoint_temperature": "2d"}


def check_domain():
    """Every box and the weekly-map domain must lie inside AREA."""
    n, w, s, e = AREA
    bad = [k for k, (_, (la0, la1, lo0, lo1), _) in X.BOXES.items()
           if la0 < s or la1 > n or lo0 < w or lo1 > e]
    la0, la1, lo0, lo1 = X.DOMAIN
    if la0 < s or la1 > n or lo0 < w or lo1 > e:
        bad.append("DOMAIN")
    for sec, (lo0, lo1) in X.ITCZ_SECTORS.items():
        if lo0 < w or lo1 > e or -20 < s or 20 > n:
            bad.append(f"itcz_{sec}")
    if bad:
        raise SystemExit(f"outside the downloaded area {AREA}: {bad}")


def cds_get(dataset, inputs, target):
    import fetch_era5_l10 as F
    if os.path.exists(target) and os.path.getsize(target) > 0:
        return target
    tok = F.token()
    body = {"inputs": dict(inputs, area=list(AREA), grid=[str(GRID), str(GRID)],
                           data_format="grib", download_format="unarchived")}
    for attempt in range(5):
        try:
            job = F._req(f"{F.CDS}/retrieve/v1/processes/{dataset}/execute",
                         json.dumps(body).encode(), tok)
            jid, t0 = job["jobID"], time.time()
            while True:
                st = F._req(f"{F.CDS}/retrieve/v1/jobs/{jid}", None, tok)["status"]
                if st == "successful":
                    break
                if st in ("failed", "dismissed"):
                    raise RuntimeError(f"CDS job {jid} {st}")
                if time.time() - t0 > 7200:
                    raise RuntimeError(f"CDS job {jid} still {st} after 2 h")
                time.sleep(10)
            url = F._req(f"{F.CDS}/retrieve/v1/jobs/{jid}/results", None, tok)["asset"]["value"]["href"]
            tmp = target + ".part"
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, target)
            return target
        except Exception as e:                       # network blips, CDS 5xx
            print(f"  {os.path.basename(target)} attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(60 * (attempt + 1))
    raise RuntimeError(f"gave up on {target}")


def month_chunks(d0, d1):
    """[(year, month, [days])] covering d0..d1 inclusive."""
    out, d = [], d0
    while d <= d1:
        y, m, days = d.year, d.month, []
        while d <= d1 and d.month == m:
            days.append(f"{d.day:02d}")
            d += dt.timedelta(days=1)
        out.append((str(y), f"{m:02d}", days))
    return out


def fetch(init):
    d0 = dt.datetime.strptime(init, "%Y%m%d")
    d1 = d0 + dt.timedelta(days=X.NDAYS)
    wd = os.path.join(CACHE, "era5_idx", init)
    os.makedirs(wd, exist_ok=True)
    six_h = ["00:00", "06:00", "12:00", "18:00"]
    jobs = []
    for y, m, days in month_chunks(d0, d1):
        base = dict(product_type=["reanalysis"], year=[y], month=[m], day=days)
        jobs.append(("reanalysis-era5-pressure-levels",
                     dict(base, variable=list(PL_VARS), pressure_level=PL_LEVELS, time=six_h),
                     os.path.join(wd, f"pl_{y}{m}.grib")))
        jobs.append(("reanalysis-era5-single-levels",
                     dict(base, variable=list(SL_VARS), time=six_h),
                     os.path.join(wd, f"sl_{y}{m}.grib")))
        jobs.append(("reanalysis-era5-single-levels",
                     dict(base, variable=["total_precipitation"],
                          time=[f"{h:02d}:00" for h in range(24)]),
                     os.path.join(wd, f"tp_{y}{m}.grib")))
    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(lambda j: cds_get(*j), jobs))
    return wd


def read_gribs(wd):
    """{(name, valid datetime): (NLAT, NLON) float64}, land/missing as NaN."""
    import eccodes
    out = {}
    for f in sorted(os.listdir(wd)):
        if not f.endswith(".grib"):
            continue
        with open(os.path.join(wd, f), "rb") as fh:
            while (h := eccodes.codes_grib_new_from_file(fh)) is not None:
                try:
                    eccodes.codes_set(h, "missingValue", 1e20)
                    sn = eccodes.codes_get(h, "shortName")
                    lev = eccodes.codes_get(h, "level") if eccodes.codes_get(h, "typeOfLevel") == "isobaricInhPa" else None
                    vd, vt = eccodes.codes_get(h, "validityDate"), eccodes.codes_get(h, "validityTime")
                    vals = np.asarray(eccodes.codes_get_values(h), np.float64).reshape(NLAT, NLON)
                    vals[vals >= 1e19] = np.nan
                    name = f"{sn}_{lev}" if lev is not None else sn
                    t = dt.datetime.strptime(f"{vd}{vt:04d}", "%Y%m%d%H%M")
                    out[(name, t)] = vals
                finally:
                    eccodes.codes_release(h)
    return out


def truth(init):
    check_domain()
    out_f = os.path.join(CACHE, "truth_idx", f"{init}.npz")
    if os.path.exists(out_f):
        return out_f
    wd = fetch(init)
    g = read_gribs(wd)
    d0 = dt.datetime.strptime(init, "%Y%m%d")
    nstep = 4 * X.NDAYS
    six = {}
    for v in X.FIELDS:
        arr = np.empty((nstep, NLAT * NLON), np.float32)
        for i in range(nstep):
            lead = 6 * (i + 1)
            if v == "tp":                              # hourly (t-1h, t] summed over the step
                hs = [d0 + dt.timedelta(hours=lead - k) for k in range(6)]
                arr[i] = sum(g[("tp", t)] for t in hs).ravel()
            else:
                arr[i] = g[(v, d0 + dt.timedelta(hours=lead))].ravel()
        six[v] = arr
    lat2, lon2 = np.meshgrid(LAT, LON, indexing="ij")
    lat, lon = lat2.ravel(), lon2.ravel()
    res = X.compute(six, lat, lon, X.land_mask(lat, lon))
    res["meta"] = X.meta(init, -1, f"ERA5 CDS 1deg area {AREA}")
    os.makedirs(os.path.dirname(out_f), exist_ok=True)
    np.savez_compressed(out_f, **res)
    return out_f


if __name__ == "__main__":
    md, y0, y1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    check_domain()
    inits = [f"{y}{md}" for y in range(y1, y0 - 1, -1)]
    def one(i):
        try:
            f = truth(i)
            print(f"{i}: {f}", flush=True)
        except Exception as e:
            print(f"{i}: FAILED {e}", flush=True)
    with ThreadPoolExecutor(max_workers=2) as ex:      # x3 requests each = 6 CDS jobs at once
        list(ex.map(one, inits))
