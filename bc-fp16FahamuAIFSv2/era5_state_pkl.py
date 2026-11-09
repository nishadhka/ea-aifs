#!/usr/bin/env python3
"""ERA5 starting state for AIFS-ENS v2 (route C, study plan section 2.2), in the pkl
format run_local_icechunk_v2.py reads: dict(date=datetime, fields={name: (2, 542080)})
for [t-6h, t], the same 112 fields as the IFS open-data builder.

    100 fields   ARCO-ERA5 full_37-1h-0p25deg-chunk-1.zarr-v3 (ERA5, and ERA5T for the
                 last ~3 months), regridded 0.25 deg -> N320 with the same to_n320 the IFS
                 builder uses (s3_grib_pkl_input_aifsens_v2_proto.py)
    12 fields    the AIFS-training ecWAM Cy50r1 wave hindcast forced by ERA5
                 (ikp6/ikg0/itul/j5rf by year) from MARS, on N320 directly: 6 period bands,
                 swh, mwp, cos/sin mwd, cdww, wmb. One source for all wave fields, the
                 wave model the network was trained on (ERA5's own waves are Cy41r2)

Differences from the IFS route, by construction:
    z_<lev>      ERA5 geopotential is already m2 s-2 (IFS open data gives gh, times g)
    constants    lsm, z, slor, sdor from ERA5 (what the model was pretrained on)
    wave mask    ecWAM's land mask (29.3 % of N320) is aligned to the operational one
                 (30.3 %) from a reference IFS pkl, as fetch_j1r2.py --mask-like does

This builds the unperturbed control. The 10 EDA perturbations (CDS) are a later step.

    era5_state_pkl.py --date 20261029_0000 --out DIR [--member 1]
    era5_state_pkl.py --date 20261029_0000 --out DIR --compare-ifs IFS_INPUT_STATES_DIR
"""

import argparse
import datetime as dt
import os
import pickle
import sys
import time

import numpy as np
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/ezra/ea-aifs/s3_grib_pkl")
sys.path.insert(0, "/home/ezra/ea-aifs/fp16FahamuAIFSv2/run-pre50r1-dates")

ARCO = ("https://storage.googleapis.com/gcp-public-data-arco-era5/ar/"
        "full_37-1h-0p25deg-chunk-1.zarr-v3")
CACHE = "/tank/projects/bc-aifs"
REF_WAVE_MASK = f"{CACHE}/ref_wave_land_n320.npy"   # True = land in the operational waef

SFC = {"10u": "10m_u_component_of_wind", "10v": "10m_v_component_of_wind",
       "2d": "2m_dewpoint_temperature", "2t": "2m_temperature",
       "msl": "mean_sea_level_pressure", "skt": "skin_temperature",
       "sp": "surface_pressure", "tcw": "total_column_water", "sd": "snow_depth",
       "stl1": "soil_temperature_level_1", "stl2": "soil_temperature_level_2",
       "swvl1": "volumetric_soil_water_layer_1", "swvl2": "volumetric_soil_water_layer_2",
       "lsm": "land_sea_mask", "z": "geopotential_at_surface",
       "slor": "slope_of_sub_gridscale_orography", "sdor": "standard_deviation_of_orography"}
PL = {"z": "geopotential", "t": "temperature", "u": "u_component_of_wind",
      "v": "v_component_of_wind", "w": "vertical_velocity", "q": "specific_humidity"}
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50, 10]


def arco_fields(times):
    """{name: (2, 542080)} for the 100 atmosphere/land fields at the two times."""
    import dask
    from s3_grib_pkl_input_aifsens_v2_proto import to_n320, MAX_WORKERS  # noqa: F401
    from concurrent.futures import ThreadPoolExecutor

    ds = xr.open_zarr(ARCO, chunks={"time": 1})
    tasks = {}
    for nm, var in SFC.items():
        tasks[nm] = ds[var].sel(time=times)
    for nm, var in PL.items():
        for lev in LEVELS:
            if nm == "q" and lev == 10:          # v2 drops q_10
                continue
            tasks[f"{nm}_{lev}"] = ds[var].sel(time=times, level=lev)
    t0 = time.time()
    with dask.config.set(scheduler="threads", num_workers=32):
        raw = dask.compute(tasks)[0]
    print(f"  ARCO: {len(raw)} fields x 2 times in {time.time()-t0:.0f}s")
    for nm, a in raw.items():
        if not np.isfinite(a.values).all():
            raise RuntimeError(f"{nm}: NaN in ARCO at {times} (ERA5T not out yet?)")

    def regrid(item):
        nm, a = item
        return nm, np.stack([to_n320(np.asarray(a.values[i], np.float64)) for i in (0, 1)])

    t0 = time.time()
    # one call first, alone: earthkit-regrid fetches/loads the 0.25->N320 matrix on first
    # use, and 8 threads doing that at once failed with "No matrix found" (20251105 run)
    to_n320(np.zeros((721, 1440)))
    with ThreadPoolExecutor(max_workers=8) as ex:
        out = dict(ex.map(regrid, raw.items()))
    print(f"  regrid to N320: {time.time()-t0:.0f}s")
    return out


def wave_fields(t1, workdir):
    import fetch_j1r2 as J
    d = t1.strftime("%Y%m%d")
    ev = J.expver_for(d)
    print(f"  waves: {ev} (ERA5-forced ecWAM Cy50r1) from MARS")
    f = J.build_state_fields(d, t1.strftime("%H"), workdir, all_wave=True, expver=ev)
    if os.path.exists(REF_WAVE_MASK):
        ref = np.load(REF_WAVE_MASK)
        # reconcile_mask wants the state shape (2, N), like the swh field it reads from a pkl
        J.reconcile_mask(f, np.where(np.broadcast_to(ref, (2, ref.size)), np.nan, 0.0))
    else:
        print(f"  WARNING: no {REF_WAVE_MASK}; ecWAM land mask left as is")
    return f, ev


WAVE_NAMES = ["h1012", "h1214", "h1417", "h1721", "h2125", "h2530",
              "swh", "mwp", "cos_mwd", "sin_mwd", "cdww", "wmb"]


def build(date, out_dir, member=1, workdir=None, waves_from=None):
    """waves_from: an existing pkl to take the 12 wave fields from instead of MARS --
    for checking the ARCO part on a date the ERA5-forced wave hindcast does not cover
    (j5rf stops between 2026-09-01 and 2026-10-01)."""
    t1 = dt.datetime.strptime(date, "%Y%m%d_%H%M")
    t0 = t1 - dt.timedelta(hours=6)
    workdir = workdir or os.path.join(CACHE, "mars_waves")
    os.makedirs(workdir, exist_ok=True)
    print(f"ERA5 control state {t0:%Y-%m-%d %H}z + {t1:%Y-%m-%d %H}z")
    cache = os.path.join(CACHE, "arco_states", f"{date}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        fields = {k: z[k] for k in z.files}
        print(f"  ARCO: {len(fields)} fields from cache {cache}")
    else:
        fields = arco_fields([t0, t1])
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, **fields)
    if waves_from:
        src = pickle.load(open(waves_from, "rb"))["fields"]
        wav, ev = {k: np.array(src[k]) for k in WAVE_NAMES}, f"pkl:{os.path.basename(waves_from)}"
        print(f"  waves: 12 fields copied from {waves_from} (check mode, not ERA5)")
    else:
        wav, ev = wave_fields(t1, workdir)
    fields.update(wav)
    mask = np.equal(fields["lsm"][0], 0)        # exactly as the IFS builder
    for f in ("sd", "swvl1", "swvl2"):           # as the IFS builder: NaN over sea
        fields[f][:, mask] = np.nan

    from s3_grib_pkl_input_aifsens_v2_proto import expected_field_names
    exp = expected_field_names()
    missing = [f for f in exp if f not in fields]
    extra = [f for f in fields if f not in exp]
    if missing or extra:
        raise RuntimeError(f"field set mismatch: missing {missing}, extra {extra}")
    state = dict(date=t1, fields=fields, source=f"ERA5 (ARCO) + ecWAM {ev}")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"input_state_member_{member:03d}.pkl")
    with open(out, "wb") as fh:
        pickle.dump(state, fh)
    print(f"  112/112 fields -> {out}")
    return out


def compare(era5_pkl, ifs_dir, n_pf=10):
    """ERA5 state against the IFS pf members of the same date: how far is the ERA5
    analysis from the IFS analysis, in units of the IFS ensemble spread?"""
    load = lambda p: pickle.load(open(p, "rb"))["fields"]
    e = load(era5_pkl)
    pf = [load(os.path.join(ifs_dir, f"input_state_member_{m:03d}.pkl"))
          for m in range(1, n_pf + 1)]
    print(f"{'field':10s} {'ERA5-IFS mean':>14s} {'rms(ERA5-IFSmean)':>18s} "
          f"{'IFS spread':>11s} {'ratio':>6s}  nan ERA5/IFS")
    rows = []
    for k in sorted(e):
        pm = np.mean([p[k] for p in pf], 0)
        sp = np.sqrt(np.nanmean(np.var([p[k] for p in pf], 0)))
        d = e[k] - pm
        rms = np.sqrt(np.nanmean(d ** 2))
        r = rms / sp if sp > 0 else np.inf
        nan_e, nan_i = np.isnan(e[k]).mean(), np.isnan(pf[0][k]).mean()
        rows.append((k, np.nanmean(d), rms, sp, r, nan_e, nan_i))
    for k, b, rms, sp, r, ne, ni in sorted(rows, key=lambda x: -x[4] if np.isfinite(x[4]) else 0):
        print(f"{k:10s} {b:+14.4g} {rms:18.4g} {sp:11.4g} {r:6.2f}  {ne:.3f}/{ni:.3f}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYYMMDD_HHMM")
    ap.add_argument("--out", required=True)
    ap.add_argument("--member", type=int, default=1)
    ap.add_argument("--compare-ifs", default=None, help="IFS input_states dir, same date")
    ap.add_argument("--waves-from", default=None,
                    help="take the 12 wave fields from this pkl (check mode)")
    a = ap.parse_args()
    p = build(a.date, a.out, a.member, waves_from=a.waves_from)
    if a.compare_ifs:
        compare(p, a.compare_ifs)
