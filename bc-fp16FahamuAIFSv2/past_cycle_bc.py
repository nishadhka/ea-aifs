#!/usr/bin/env python3
"""Bias correction of the AI-WQ tas / mslp / pr product from past real-time cycles.

Interim method until the same-date hindcast exists (README §1-3). Each past cycle whose
two verification weeks have ERA5 (AI-WQ weekly obs) gives one forecast-minus-obs map per
variable and lead week. Their mean, smoothed in space, is removed from every member of a
new cycle before the quintile counting -- the same counting 3b does
(shared/ensemble_quintile_analysis_cli.py: calculate_grid_quintiles).

    tas, mslp   additive:        x' = x - a * b(week, lat, lon),  b = mean_c(ensmean_c - obs_c)
    pr          multiplicative:  x' = x / r^a,  r = smooth(sum_c ensmean_c) / smooth(sum_c obs_c),
                                 clipped to [1/3, 3]

The smoothing width sigma (grid points of 1.5 deg) and the strength a are chosen per
variable by leave-one-cycle-out RPSS, scored with the competition's own code
(AI_WQ_package.forecast_evaluation). With so few cycles that choice is itself fitted on
the same data, so the LOO numbers are mildly optimistic.

Subcommands
    aggregate  CYCLE...      weekly member aggregates -> CACHE/weekly/<cycle>.npz
    evaluate                 LOO RPSS raw vs corrected over the verifiable cycles
    apply --date YYYYMMDD    corrected submission file for one cycle

Aggregation mirrors 3b exactly: week1 = time chunks 0,1 (24 steps), week2 = chunks 2,3,4
(36 steps); tas/msl are step means, tp is the sum * 1000 * 7 / days (mm per 7 days).
"""

import argparse
import datetime as dt
import glob
import json
import os
import sys

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CACHE = "/tank/projects/bc-aifs"
RUN = "/tank/projects/aifs-run"
ENV = "/home/ezra/ea-aifs/fp16FahamuAIFSv2/.env"

# where each cycle's per-member 1.5 deg files live. 0709 and 0813 were regridded by
# older routes (eval rerun; via O96) -- kept, but flagged in the output.
NC_DIRS = {
    "20260709": f"{RUN}/eval_20260709/nc_1p5deg",
    "20260813": f"{RUN}/20260813_0000/nc_1p5deg_via_o96",
}
VARS = ["tas", "mslp", "pr"]
RAW = {"tas": "2t", "mslp": "msl", "pr": "tp"}
QVAR = {"tas": "2t_quintiles", "mslp": "msl_quintiles", "pr": "tp_quintiles"}
WEEKS = [([0, 1], 24), ([2, 3, 4], 36)]
SIGMAS = [0, 1, 2, 4]
STRENGTHS = [0.5, 1.0]
PR_CLIP = 3.0


def nc_dir(cycle):
    return NC_DIRS.get(cycle, f"{RUN}/{cycle}_0000/nc_1p5deg")


def verification_dates(cycle):
    d0 = dt.datetime.strptime(cycle, "%Y%m%d").date()
    return [(d0 + dt.timedelta(days=k)).strftime("%Y%m%d") for k in (18, 25)]


# ------------------------------------------------------------------ aggregation
def aggregate(cycle):
    out = os.path.join(CACHE, "weekly", f"{cycle}.npz")
    if os.path.exists(out):
        print(f"{cycle}: cached")
        return
    files = sorted(glob.glob(os.path.join(nc_dir(cycle), "aifs_ensemble_forecast_1p5deg_member*.nc")))
    if not files:
        raise SystemExit(f"{cycle}: no member files in {nc_dir(cycle)}")
    # float64: in float32 ~10 cells per field land on the other side of a quintile edge vs 3b
    arr = {v: np.empty((2, len(files), 121, 240), np.float64) for v in VARS}
    for m, f in enumerate(files):
        with xr.open_dataset(f) as d:
            for v in VARS:
                x = d[RAW[v]].isel(member=0)
                for w, (chunks, nsteps) in enumerate(WEEKS):
                    s = sum(x.isel(time=t).sum("step", skipna=True) for t in chunks).values
                    arr[v][w, m] = s * 1000 * 7.0 / (nsteps * 6 / 24) if v == "pr" else s / nsteps
        print(f"  {cycle} member {m+1}/{len(files)}", end="\r")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, **arr, files=np.array([os.path.basename(f) for f in files]))
    print(f"{cycle}: {len(files)} members -> {out}")


def load_weekly(cycle):
    z = np.load(os.path.join(CACHE, "weekly", f"{cycle}.npz"))
    return {v: z[v] for v in VARS}


# ------------------------------------------------------------------ AI-WQ data
_R = _E = _PW = None


def aiwq():
    global _R, _E, _PW
    if _R is None:
        from dotenv import load_dotenv
        load_dotenv(ENV)
        _PW = os.environ["AIWQ_PASSWORD"]
        os.makedirs(os.path.join(CACHE, "aiwq"), exist_ok=True)  # FTP writes here; no auto-create
        from AI_WQ_package import retrieve_evaluation_data as R, forecast_evaluation as E
        _R, _E = R, E
    return _R, _E, _PW


def _quiet(fn, *a, **k):
    """AI_WQ_package prints a lot; keep the log readable."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def obs(wdate, var):
    """AI-WQ weekly obs; after 2026-08-24 the AI-WQ archive has none, so fall back to
    era5_weekly.py's ARCO-ERA5T build (identical to AI-WQ's to 0.0002 K / 0.02 Pa /
    0.002 mm rms on 3 overlap weeks). None if neither exists yet."""
    R, _, pw = aiwq()
    try:
        return _quiet(R.retrieve_weekly_obs, wdate, var, pw,
                      local_destination=os.path.join(CACHE, "aiwq")).squeeze(drop=True)
    except Exception:
        # a failed FTP RETR leaves a 0-byte file behind; do not let it look like data
        for f in glob.glob(os.path.join(CACHE, "aiwq", f"{var}_obs_*_{wdate}.nc")):
            if os.path.getsize(f) == 0:
                os.remove(f)
    import era5_weekly
    return era5_weekly.load(wdate, var)


def clim(wdate, var):
    R, _, pw = aiwq()
    return _quiet(R.retrieve_20yr_quantile_clim, wdate, var, pw,
                  local_destination=os.path.join(CACHE, "aiwq")).squeeze(drop=True)


def lsm():
    R, _, pw = aiwq()
    return _quiet(R.retrieve_land_sea_mask, pw, local_destination=os.path.join(CACHE, "aiwq"))


# ------------------------------------------------------------------ correction
def smooth(a, sigma):
    if sigma == 0:
        return a
    # periodic in longitude, edge-held at the poles
    return gaussian_filter(a, sigma=sigma, mode=("nearest", "wrap"))


def fit(pairs, var, sigma):
    """pairs: list of (ensmean, obs) 2-d arrays for one variable and week."""
    fc = np.stack([p[0] for p in pairs])
    ob = np.stack([p[1] for p in pairs])
    if var == "pr":
        num = smooth(np.nanmean(fc, 0), sigma) + 1.0   # +1 mm/week keeps deserts finite
        den = smooth(np.nanmean(ob, 0), sigma) + 1.0
        return np.clip(num / den, 1 / PR_CLIP, PR_CLIP)
    return smooth(np.nanmean(fc - ob, 0), sigma)


def correct(members, var, table, strength):
    if var == "pr":
        return members / table[None] ** strength
    return members - strength * table[None]


def quintile_probs(members, edges):
    """Exactly 3b's calculate_grid_quintiles: lower-closed bins against the 4 edges."""
    e = np.asarray(edges)
    n = members.shape[0]
    p = np.empty((5,) + members.shape[1:])
    p[0] = (members < e[0]).sum(0) / n
    for k in range(1, 4):
        p[k] = ((members >= e[k - 1]) & (members < e[k])).sum(0) / n
    p[4] = (members >= e[3]).sum(0) / n
    return p


def rpss(probs, o, c, var, mask):
    _, E, _ = aiwq()
    obs_pbs = _quiet(E.conditional_obs_probs, o, c).transpose("quantile", "latitude", "longitude")
    fc = xr.DataArray(probs, dims=("quantile", "latitude", "longitude"),
                      coords=dict(quantile=np.arange(5), latitude=o.latitude,
                                  longitude=o.longitude))
    return float(_quiet(E.work_out_RPSS, fc, obs_pbs, var, mask).values)


# ------------------------------------------------------------------ evaluate
def verifiable(cycles):
    """cycle -> {(var, week): obs} for every week whose ERA5 obs exist."""
    out = {}
    for c in cycles:
        for w, wd in enumerate(verification_dates(c)):
            for v in VARS:
                o = obs(wd, v)
                if o is not None:
                    out.setdefault(c, {})[(v, w)] = o
    return out


def evaluate(args):
    cycles = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{CACHE}/weekly/*.npz"))
    vo = verifiable(cycles)
    print("verifiable (cycle: weeks):",
          {c: sorted({w + 1 for (_, w) in d}) for c, d in vo.items()})
    mask = lsm()
    data = {c: load_weekly(c) for c in vo}
    rows = []
    for v in VARS:
        for w in (0, 1):
            held = [c for c in vo if (v, w) in vo[c]]
            if len(held) < 3:
                print(f"{v} week{w+1}: only {len(held)} verifiable cycles, skipped")
                continue
            for c in held:
                wd = verification_dates(c)[w]
                o, cl = vo[c][(v, w)], clim(wd, v)
                mem = data[c][v][w]
                rec = dict(var=v, week=w + 1, cycle=c, n_train=len(held) - 1,
                           raw=rpss(quintile_probs(mem, cl.values), o, cl, v, mask))
                train = [(np.nanmean(data[t][v][w], 0), vo[t][(v, w)].values)
                         for t in held if t != c]
                for s in SIGMAS:
                    tab = fit(train, v, s)
                    for a in STRENGTHS:
                        p = quintile_probs(correct(mem, v, tab, a), cl.values)
                        rec[f"s{s}_a{a}"] = rpss(p, o, cl, v, mask)
                rows.append(rec)
                best = max((k for k in rec if k.startswith("s")), key=lambda k: rec[k])
                print(f"{v:4s} wk{w+1} {c} raw {rec['raw']:+.4f}  best {best} {rec[best]:+.4f}")
    with open(os.path.join(CACHE, "loo_rpss.json"), "w") as fh:
        json.dump(rows, fh, indent=1)

    print("\nmean LOO RPSS by setting (both weeks pooled)")
    keys = ["raw"] + [f"s{s}_a{a}" for s in SIGMAS for a in STRENGTHS]
    choice = {}
    for v in VARS:
        r = [x for x in rows if x["var"] == v]
        if not r:
            continue
        means = {k: float(np.mean([x[k] for x in r])) for k in keys}
        best = max(keys[1:], key=means.get)
        choice[v] = best if means[best] > means["raw"] else "raw"
        print(f"{v:4s} n={len(r):2d}  " + "  ".join(f"{k} {means[k]:+.4f}" for k in keys))
        print(f"     -> {choice[v]}  (raw {means['raw']:+.4f}, chosen {means[choice[v]]:+.4f}; "
              f"per-cycle wins {sum(x[choice[v]] > x['raw'] for x in r)}/{len(r)})")
    with open(os.path.join(CACHE, "choice.json"), "w") as fh:
        json.dump(choice, fh, indent=1)
    print(f"\nwrote {CACHE}/loo_rpss.json and choice.json")


# ------------------------------------------------------------------ apply
def apply(args):
    choice = json.load(open(os.path.join(CACHE, "choice.json")))
    cycles = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{CACHE}/weekly/*.npz")
                    if os.path.basename(f)[:-4] != args.date)
    vo = verifiable(cycles)
    data = {c: load_weekly(c) for c in vo}
    tgt = load_weekly(args.date)
    wds = verification_dates(args.date)
    out, tables, meta = {}, {}, {}
    for v in VARS:
        probs = []
        for w in (0, 1):
            cl = clim(wds[w], v)
            mem = tgt[v][w]
            setting = choice.get(v, "raw")
            train = [c for c in vo if (v, w) in vo[c]]
            if setting != "raw":
                s, a = setting[1:].split("_a")
                tab = fit([(np.nanmean(data[c][v][w], 0), vo[c][(v, w)].values) for c in train],
                          v, int(s))
                mem = correct(mem, v, tab, float(a))
                tables[f"{v}_week{w+1}"] = tab
            meta[f"{v}_week{w+1}"] = f"{setting}; trained on {','.join(train)}"
            probs.append(quintile_probs(mem, cl.values))
        out[QVAR[v]] = xr.DataArray(
            np.stack(probs), dims=("time_week", "quintile", "latitude", "longitude"),
            coords=dict(quintile=[0.2, 0.4, 0.6, 0.8, 1.0], latitude=cl.latitude,
                        longitude=cl.longitude, week=("time_week", ["week1", "week2"])))
    ds = xr.Dataset(out)
    ds.attrs.update(title="Ensemble Forecast Quintile Probabilities, bias-corrected from past cycles",
                    forecast_date=args.date, method="bc-fp16FahamuAIFSv2/past_cycle_bc.py",
                    processing_date=str(np.datetime64("now")),
                    n_members=int(tgt["tas"].shape[1]), **meta)
    d = os.path.join(RUN, f"{args.date}_0000", "aiwq_bc")
    os.makedirs(d, exist_ok=True)
    f = os.path.join(d, f"ensemble_quintile_probabilities_{args.date}_bc.nc")
    ds.to_netcdf(f)
    if tables:
        np.savez(os.path.join(d, f"bias_tables_{args.date}.npz"), **tables)
    for k, m in meta.items():
        print(f"{k:12s} {m}")
    print(f"wrote {f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("aggregate")
    a.add_argument("cycles", nargs="+")
    sp.add_parser("evaluate")
    p = sp.add_parser("apply")
    p.add_argument("--date", required=True)
    args = ap.parse_args()
    if args.cmd == "aggregate":
        for c in args.cycles:
            aggregate(c)
    elif args.cmd == "evaluate":
        evaluate(args)
    else:
        apply(args)


if __name__ == "__main__":
    sys.exit(main())
