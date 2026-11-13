#!/usr/bin/env python3
"""Perturbed ERA5 starting states for the 220-member hindcast (RUNBOOK_20261105 section 4.2).

Member k (2..11) = control (member 1, ARCO + ecWAM, era5_state_pkl.py)
                   + (ERA5 EDA member k-2  -  mean of the 10 EDA members)
for each of the two state times (D-6 h, D) separately.

    EDA         CDS reanalysis-era5-{pressure,single}-levels, product_type ensemble_members,
                10 members, native 0.5 deg, 3-hourly
    perturbed   z t u v w q on the 14 levels (no q_10), 2t 2d 10u 10v msl sp skt tcw,
                sd stl1 stl2 swvl1 swvl2
    NOT         lsm, z (surface), slor, sdor, and the 12 wave fields (no wave EDA here)
    regrid      0.5 -> 0.25 deg bilinear in numpy (earthkit-regrid has no 0.5 -> N320
                matrix), then the same 0.25 -> N320 to_n320 as the control
    bounds      q, tcw, sd >= 0; swvl in [0, 0.766]; NaN where the control is NaN

CDS requests go through api_polite.polite("cds"): ONE at a time per machine, a gap between
them, and every target is checked before requesting, so a rerun never repeats one. Four
requests per year (2 datasets x 2 times), ~1 GB per year. Never run fetch on several
machines at once.

    era5_eda_perturb.py probe                      one tiny SL request: which fields exist
    era5_eda_perturb.py fetch --mmdd 1105 --years 2025-2006       GRIBs into BC_CACHE/eda/
    era5_eda_perturb.py build --mmdd 1105 --year 2025 --control PKL --out DIR [--members 2-11]
    era5_eda_perturb.py spread --mmdd 1105 --year 2025 --ifs-dir DIR   spread vs IFS ENS
"""

import argparse
import datetime as dt
import json
import os
import pickle
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.environ.get("EA_AIFS_ROOT", os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO_ROOT, "s3_grib_pkl"))
sys.path.insert(0, os.path.join(REPO_ROOT, "fp16FahamuAIFSv2", "run-pre50r1-dates"))
from api_polite import polite  # noqa: E402

CACHE = os.environ.get("BC_CACHE", "/tank/projects/bc-aifs")
N_EDA = 10
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50, 10]
PL = {"geopotential": "z", "temperature": "t", "u_component_of_wind": "u",
      "v_component_of_wind": "v", "vertical_velocity": "w", "specific_humidity": "q"}
SL = {"10m_u_component_of_wind": "10u", "10m_v_component_of_wind": "10v",
      "2m_dewpoint_temperature": "2d", "2m_temperature": "2t",
      "mean_sea_level_pressure": "msl", "skin_temperature": "skt",
      "surface_pressure": "sp", "total_column_water": "tcw", "snow_depth": "sd",
      "soil_temperature_level_1": "stl1", "soil_temperature_level_2": "stl2",
      "volumetric_soil_water_layer_1": "swvl1", "volumetric_soil_water_layer_2": "swvl2"}
NONNEG = ("tcw", "sd")
SWVL_MAX = 0.766                   # ERA5 soil porosity upper bound (organic soil)


# ------------------------------------------------------------------------------ CDS
def cds_job(dataset, inputs, target, tries=4):
    """One CDS request -> target GRIB, through polite("cds"). Skips if target exists."""
    import fetch_era5_l10 as F
    if os.path.exists(target) and os.path.getsize(target) > 0:
        return target
    os.makedirs(os.path.dirname(target), exist_ok=True)
    tok = F.token()
    body = {"inputs": dict(inputs, data_format="grib", download_format="unarchived")}
    for attempt in range(tries):
        try:
            with polite("cds"):
                t0 = time.time()
                job = F._req(f"{F.CDS}/retrieve/v1/processes/{dataset}/execute",
                             json.dumps(body).encode(), tok)
                jid = job["jobID"]
                while True:
                    st = F._req(f"{F.CDS}/retrieve/v1/jobs/{jid}", None, tok)["status"]
                    if st == "successful":
                        break
                    if st in ("failed", "dismissed"):
                        raise RuntimeError(f"CDS job {jid} {st}")
                    if time.time() - t0 > 4 * 3600:
                        raise RuntimeError(f"CDS job {jid} still {st} after 4 h")
                    time.sleep(30)                  # gentle polling
                t_q = time.time() - t0
                url = F._req(f"{F.CDS}/retrieve/v1/jobs/{jid}/results", None, tok)["asset"]["value"]["href"]
                tmp = target + ".part"
                urllib.request.urlretrieve(url, tmp)
                os.replace(tmp, target)
            mb = os.path.getsize(target) / 1e6
            print(f"  CDS {os.path.basename(target)}: {mb:.1f} MB, queue+run {t_q:.0f} s, "
                  f"total {time.time()-t0:.0f} s", flush=True)
            return target
        except Exception as e:
            print(f"  CDS {os.path.basename(target)} attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(120 * (attempt + 1))
    raise RuntimeError(f"gave up on {target}")


def state_times(year, mmdd):
    t1 = dt.datetime.strptime(f"{year}{mmdd}00", "%Y%m%d%H")
    return [t1 - dt.timedelta(hours=6), t1]


def eda_paths(year, mmdd):
    """{(kind, time): grib path} for the 4 requests of one year."""
    wd = os.path.join(CACHE, "eda", f"{year}{mmdd}")
    return {(k, t): os.path.join(wd, f"{k}_{t:%Y%m%d%H}.grib")
            for k in ("pl", "sl") for t in state_times(year, mmdd)}


def fetch_year(year, mmdd):
    for (kind, t), path in eda_paths(year, mmdd).items():
        base = dict(product_type=["ensemble_members"], year=[f"{t:%Y}"], month=[f"{t:%m}"],
                    day=[f"{t:%d}"], time=[f"{t:%H}:00"])
        if kind == "pl":
            cds_job("reanalysis-era5-pressure-levels",
                    dict(base, variable=list(PL), pressure_level=[str(l) for l in LEVELS]), path)
        else:
            cds_job("reanalysis-era5-single-levels", dict(base, variable=list(SL)), path)


# ------------------------------------------------------------------------------ GRIB
def read_eda(path):
    """{(field, member): (361, 720) float64} from one EDA GRIB (north->south, lon 0->359.5)."""
    import eccodes
    out = {}
    with open(path, "rb") as fh:
        while True:
            gid = eccodes.codes_grib_new_from_file(fh)
            if gid is None:
                break
            try:
                sn = eccodes.codes_get(gid, "shortName")
                lev = eccodes.codes_get(gid, "level")
                num = eccodes.codes_get(gid, "number")
                ni, nj = eccodes.codes_get(gid, "Ni"), eccodes.codes_get(gid, "Nj")
                eccodes.codes_set(gid, "missingValue", 9.999e20)
                v = np.asarray(eccodes.codes_get_values(gid), np.float64).reshape(nj, ni)
                v[v == 9.999e20] = np.nan
                lat0 = eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees")
                lon0 = eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees")
            finally:
                eccodes.codes_release(gid)
            if lat0 < 0:
                v = v[::-1]
            if lon0 not in (0, 0.0):
                raise RuntimeError(f"{path}: grid starts at lon {lon0}, expected 0")
            if sn in ("z", "t", "u", "v", "w", "q") and lev in LEVELS:
                name = f"{sn}_{lev}"
            else:
                name = sn
            out[(name, num)] = v
    return out


def up2(a):
    """0.5 deg (361, 720) -> 0.25 deg (721, 1440), bilinear, periodic in longitude."""
    ny, nx = a.shape
    b = np.empty((2 * ny - 1, nx), a.dtype)
    b[::2] = a
    b[1::2] = 0.5 * (a[:-1] + a[1:])
    c = np.empty((b.shape[0], 2 * nx), a.dtype)
    c[:, ::2] = b
    c[:, 1::2] = 0.5 * (b + np.roll(b, -1, axis=1))
    return c


def perturbations(year, mmdd, names):
    """{name: (N_EDA, 2, 542080)} = EDA member - EDA mean, on N320, for both state times."""
    from s3_grib_pkl_input_aifsens_v2_proto import to_n320
    to_n320(np.zeros((721, 1440)))                  # load the matrix once, before threads
    paths = eda_paths(year, mmdd)
    out = {n: np.zeros((N_EDA, 2, 542080), np.float32) for n in names}   # 4 GB, not 8
    for it, t in enumerate(state_times(year, mmdd)):
        raw = {}
        for kind in ("pl", "sl"):
            raw.update(read_eda(paths[(kind, t)]))
        for n in names:
            mem = np.stack([raw[(n, m)] for m in range(N_EDA)])
            if np.isnan(mem).all():
                raise RuntimeError(f"{n} at {t}: all NaN in the EDA GRIB")
            p = np.nan_to_num(mem - np.nanmean(mem, axis=0), nan=0.0)   # sea in land fields -> 0
            for m in range(N_EDA):
                out[n][m, it] = to_n320(up2(p[m]))
    return out


def perturbed_names():
    pl = [f"{s}_{l}" for s in PL.values() for l in LEVELS if not (s == "q" and l == 10)]
    return pl + list(SL.values())


def build(year, mmdd, control_pkl, out_dir, members, inflate=1.0):
    ctl = pickle.load(open(control_pkl, "rb"))
    names = perturbed_names()
    missing = [n for n in names if n not in ctl["fields"]]
    if missing:
        raise RuntimeError(f"control pkl lacks {missing}")
    t0 = time.time()
    P = perturbations(year, mmdd, names)
    print(f"  perturbations for {len(names)} fields x {N_EDA} members: {time.time()-t0:.0f} s",
          flush=True)
    os.makedirs(out_dir, exist_ok=True)
    outs = []
    for k in members:
        f = {n: np.array(a, copy=True) for n, a in ctl["fields"].items()}
        for n in names:
            x = f[n] + inflate * P[n][(k - 2) % N_EDA]   # 12-21 reuse EDA 0-9 (A/B tests)
            x[np.isnan(f[n])] = np.nan                     # keep the control's NaN (sea)
            if n.startswith("q_") or n in NONNEG:
                x = np.where(np.isnan(x), x, np.maximum(x, 0.0))
            if n.startswith("swvl"):
                x = np.where(np.isnan(x), x, np.clip(x, 0.0, SWVL_MAX))
            f[n] = x
        state = dict(date=ctl["date"], fields=f,
                     source=f"{ctl.get('source', 'ERA5')} + ERA5 EDA member {(k-2) % N_EDA} perturbation"
                            + (f" x {inflate:g}" if inflate != 1.0 else ""))
        out = os.path.join(out_dir, f"input_state_member_{k:03d}.pkl")
        with open(out, "wb") as fh:
            pickle.dump(state, fh)
        outs.append(out)
        print(f"  member {k:03d} -> {out}", flush=True)
    return outs


# ------------------------------------------------------------------------------ checks
def probe():
    """One tiny single-level request (1x1 deg box, one time): which SL fields exist."""
    path = os.path.join(CACHE, "eda", "probe_sl_2025110500.grib")
    cds_job("reanalysis-era5-single-levels",
            dict(product_type=["ensemble_members"], year=["2025"], month=["11"], day=["05"],
                 time=["00:00"], variable=list(SL), area=[1, 36, 0, 37]), path)
    import eccodes
    seen = {}
    with open(path, "rb") as fh:
        while (gid := eccodes.codes_grib_new_from_file(fh)) is not None:
            sn, num = eccodes.codes_get(gid, "shortName"), eccodes.codes_get(gid, "number")
            vals = eccodes.codes_get_values(gid)
            seen.setdefault(sn, []).append((num, float(np.mean(vals))))
            eccodes.codes_release(gid)
    for v in SL.values():
        got = seen.get(v, [])
        print(f"  {v:6s} members {len(got):2d}  values {[round(x, 4) for _, x in sorted(got)][:4]}")
    print(f"  missing: {[v for v in SL.values() if v not in seen] or 'none'}; "
          f"extra: {[k for k in seen if k not in SL.values()] or 'none'}")


def spread(year, mmdd, ifs_dir, n_ifs=10):
    """Global rms spread of the 10 EDA perturbations vs the IFS ENS pf spread (ratio per field)."""
    names = perturbed_names()
    P = perturbations(year, mmdd, names)
    load = lambda p: pickle.load(open(p, "rb"))["fields"]
    pf = [load(os.path.join(ifs_dir, f)) for f in sorted(os.listdir(ifs_dir))[:n_ifs]]
    print(f"{'field':8s} {'EDA sd':>10s} {'IFS sd':>10s} {'ratio':>6s}")
    rows = []
    for n in names:
        e = np.sqrt(np.nanmean(np.var(P[n][:, 1], axis=0, ddof=1)))
        i = np.sqrt(np.nanmean(np.var([p[n][1] for p in pf], axis=0, ddof=1)))
        rows.append((n, e, i, e / i if i > 0 else np.nan))
    for n, e, i, r in rows:
        print(f"{n:8s} {e:10.4g} {i:10.4g} {r:6.2f}")
    r = np.array([x[3] for x in rows])
    print(f"median ratio {np.nanmedian(r):.2f}, range {np.nanmin(r):.2f}-{np.nanmax(r):.2f}")
    return rows


def years_of(spec):
    a, _, b = spec.partition("-")
    a, b = int(a), int(b or a)
    return list(range(a, b - 1, -1)) if a >= b else list(range(a, b + 1))


def members_of(spec):
    a, _, b = spec.partition("-")
    return list(range(int(a), int(b or a) + 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("probe")
    p = sp.add_parser("fetch"); p.add_argument("--mmdd", required=True); p.add_argument("--years", required=True)
    p = sp.add_parser("build")
    p.add_argument("--mmdd", required=True); p.add_argument("--year", type=int, required=True)
    p.add_argument("--control", required=True); p.add_argument("--out", required=True)
    p.add_argument("--members", default="2-11")
    p.add_argument("--inflate", type=float, default=1.0,
                   help="scale the EDA perturbations (ERA5 EDA spread is ~0.6 of IFS ENS, "
                        "measured 2026-11-13)")
    p = sp.add_parser("spread")
    p.add_argument("--mmdd", required=True); p.add_argument("--year", type=int, required=True)
    p.add_argument("--ifs-dir", required=True)
    a = ap.parse_args()
    if a.cmd == "probe":
        probe()
    elif a.cmd == "fetch":
        for y in years_of(a.years):          # strictly one year, one request after another
            print(f"[{y}{a.mmdd}] EDA", flush=True)
            fetch_year(y, a.mmdd)
    elif a.cmd == "build":
        build(a.year, a.mmdd, a.control, a.out, members_of(a.members), a.inflate)
    elif a.cmd == "spread":
        spread(a.year, a.mmdd, a.ifs_dir)
