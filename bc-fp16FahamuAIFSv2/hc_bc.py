#!/usr/bin/env python3
"""Bias correction of the AI-WQ tas / mslp / pr product from the same-date hindcast.

The hindcast (hc_date.sh) gives, for one calendar date, one ERA5-started member per
year 2006-2025, regridded exactly as 3a does. Each is aggregated exactly as 3b does
(past_cycle_bc.aggregate) and verified against ERA5 weekly means for the same leads
(era5_weekly.py: AI-WQ's definitions, any start weekday). The correction form is the
one past_cycle_bc.py uses (additive tas/mslp, ratio pr, smoothing sigma, strength a);
only the training set changes: 20 years of the same date instead of a few recent cycles.

With one member per year the leave-one-year-out check is on area-weighted MSE of the
weekly field (land-only for tas and pr, as AI-WQ scores them), not RPSS: it asks
whether the bias is stable enough from year to year to remove.

    hc_bc.py obs      1105 2006 2025      ERA5 weekly for each year's +18 d and +25 d
    hc_bc.py evaluate 1105 2006 2025      LOO over years -> CACHE/hc_choice_1105.json
    hc_bc.py apply    --date 20261105     -> aifs-run/D_0000/aiwq_bc_hc/..._bchc.nc
"""

import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import era5_weekly as W  # noqa: E402
import past_cycle_bc as B  # noqa: E402

S = B.CACHE
LAT = np.linspace(90, -90, 121)
SIGMAS, STRENGTHS = B.SIGMAS, [0.5, 0.75, 1.0]


def leads(year, md):
    d0 = dt.date(year, int(md[:2]), int(md[2:]))
    return [(d0 + dt.timedelta(days=k)).strftime("%Y%m%d") for k in (18, 25)]


def tag(year, md):
    return f"hc{year}{md}"


def load_year(year, md):
    t = tag(year, md)
    B.NC_DIRS[t] = f"{S}/hc_nc/{year}{md}_0000"
    if not os.path.exists(f"{S}/weekly/{t}.npz"):
        if not os.path.isdir(B.NC_DIRS[t]):
            return None
        B.aggregate(t)
    return B.load_weekly(t)


def weights(var):
    w = np.cos(np.radians(LAT))[:, None] * np.ones((1, 240))
    if var in ("tas", "pr"):
        lsm = np.asarray(B.lsm().squeeze().values) > 0.5
        w = w * lsm
    return w


def mse(a, o, w):
    m = np.isfinite(a) & np.isfinite(o)
    return float(np.sum(((a - o) ** 2 * w)[m]) / np.sum(w[m]))


def fit_table(train, var, setting):
    """setting 's<sigma>_a<a>' = plain fit (past_cycle_bc.fit), or 'sig_a<a>' = the
    unsmoothed fit kept only where the training years' bias is significant
    (two-sided t-test, p < 0.05; pr tested on log((fc+1)/(obs+1))), else no correction.
    Returns (table, strength)."""
    from scipy import stats
    if setting.startswith("sig_a"):
        f = np.stack([a for a, _ in train]); o = np.stack([b for _, b in train])
        d = np.log((f + 1) / (o + 1)) if var == "pr" else f - o
        t = d.mean(0) / (d.std(0, ddof=1) / np.sqrt(len(train)))
        keep = np.abs(t) > stats.t.ppf(0.975, len(train) - 1)
        tab = np.where(keep, B.fit(train, var, 0), 1.0 if var == "pr" else 0.0)
        return tab, float(setting[5:])
    s, a = setting[1:].split("_a")
    return B.fit(train, var, int(s)), float(a)


def cmd_obs(md, y0, y1):
    days = sorted({d for y in range(y0, y1 + 1) for d in leads(y, md)})
    W.build(days)


def collect(md, y0, y1):
    yrs, fc, ob = [], {}, {}
    for y in range(y0, y1 + 1):
        h = load_year(y, md)
        if h is None:
            print(f"  {y}: no hindcast member yet, left out")
            continue
        o = {(v, k): W.load(leads(y, md)[k], v) for v in B.VARS for k in (0, 1)}
        if any(x is None for x in o.values()):
            print(f"  {y}: ERA5 weekly missing (run: hc_bc.py obs), left out")
            continue
        yrs.append(y)
        for v in B.VARS:
            for k in (0, 1):
                fc[(y, v, k)] = np.nanmean(h[v][k], 0)    # one member: the member itself
                ob[(y, v, k)] = o[(v, k)].values
    return yrs, fc, ob


def cmd_evaluate(md, y0, y1):
    yrs, fc, ob = collect(md, y0, y1)
    print(f"hindcast {md}: {len(yrs)} years {yrs[0]}-{yrs[-1]}" if yrs else "no years")
    if len(yrs) < 5:
        raise SystemExit("need at least 5 years")
    res, choice = {}, {}
    for v in B.VARS:
        w = weights(v)
        for k in (0, 1):
            keys = (["raw"] + [f"s{s}_a{a}" for s in SIGMAS for a in STRENGTHS]
                    + [f"sig_a{a}" for a in STRENGTHS])
            scores = {kk: [] for kk in keys}
            for y in yrs:
                train = [(fc[(t, v, k)], ob[(t, v, k)]) for t in yrs if t != y]
                x, o = fc[(y, v, k)][None], ob[(y, v, k)]
                scores["raw"].append(mse(x[0], o, w))
                for kk in keys[1:]:
                    tab, a = fit_table(train, v, kk)
                    scores[kk].append(mse(B.correct(x, v, tab, a)[0], o, w))
            m = {kk: float(np.mean(sc)) for kk, sc in scores.items()}
            best = min(keys[1:], key=m.get)
            wins = sum(c < r for c, r in zip(scores[best], scores["raw"]))
            res[f"{v}_wk{k+3}"] = dict(mse=m, best=best, wins=wins, n=len(yrs))
            print(f"{v:4s} wk{k+3}: raw rmse {np.sqrt(m['raw']):9.4g}  best {best:9s} "
                  f"rmse {np.sqrt(m[best]):9.4g}  ({100*(1-m[best]/m['raw']):+5.1f}% MSE, "
                  f"years improved {wins}/{len(yrs)})")
        # one setting per variable over both weeks, as past_cycle_bc does
        pooled = {kk: np.mean([res[f"{v}_wk{k+3}"]["mse"][kk] / res[f"{v}_wk{k+3}"]["mse"]["raw"]
                               for k in (0, 1)]) for kk in res[f"{v}_wk3"]["mse"]}
        best = min((kk for kk in pooled if kk != "raw"), key=pooled.get)
        choice[v] = best if pooled[best] < 1 else "raw"
        print(f"     -> {v}: {choice[v]} (pooled MSE ratio {pooled[choice[v]]:.3f})")
    json.dump(dict(choice=choice, years=yrs, results=res),
              open(f"{S}/hc_choice_{md}.json", "w"), indent=1)
    print(f"wrote {S}/hc_choice_{md}.json")


def cmd_apply(date, setting_all=None):
    md = date[4:8]
    cfg = json.load(open(f"{S}/hc_choice_{md}.json"))
    if setting_all:
        cfg["choice"] = {v: setting_all for v in B.VARS}
    yrs, fc, ob = collect(md, min(cfg["years"]), max(cfg["years"]))
    tgt = B.load_weekly(date)
    wds = B.verification_dates(date)
    out, tables, meta = {}, {}, {}
    for v in B.VARS:
        probs = []
        for k in (0, 1):
            cl = B.clim(wds[k], v)
            mem = tgt[v][k]
            setting = cfg["choice"][v]
            if setting != "raw":
                tab, a = fit_table([(fc[(y, v, k)], ob[(y, v, k)]) for y in yrs], v, setting)
                mem = B.correct(mem, v, tab, a)
                tables[f"{v}_week{k+1}"] = tab
            meta[f"{v}_week{k+1}"] = f"{setting}; hindcast {md} {yrs[0]}-{yrs[-1]} ({len(yrs)} yrs)"
            probs.append(B.quintile_probs(mem, cl.values))
        out[B.QVAR[v]] = xr.DataArray(
            np.stack(probs), dims=("time_week", "quintile", "latitude", "longitude"),
            coords=dict(quintile=[0.2, 0.4, 0.6, 0.8, 1.0], latitude=cl.latitude,
                        longitude=cl.longitude, week=("time_week", ["week1", "week2"])))
    ds = xr.Dataset(out)
    ds.attrs.update(title="Ensemble Forecast Quintile Probabilities, bias-corrected from the "
                          "same-date hindcast", forecast_date=date,
                    method="bc-fp16FahamuAIFSv2/hc_bc.py", n_members=int(tgt["tas"].shape[1]),
                    processing_date=str(np.datetime64("now")), **meta)
    d = os.path.join(B.RUN, f"{date}_0000", "aiwq_bc_hc")
    os.makedirs(d, exist_ok=True)
    f = os.path.join(d, f"ensemble_quintile_probabilities_{date}_bchc.nc")
    ds.to_netcdf(f)
    np.savez(os.path.join(d, f"bias_tables_hc_{date}.npz"), **tables)
    for k, m in meta.items():
        print(f"{k:12s} {m}")
    print(f"wrote {f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    for c in ("obs", "evaluate"):
        p = sp.add_parser(c)
        p.add_argument("md"); p.add_argument("y0", type=int); p.add_argument("y1", type=int)
    p = sp.add_parser("apply"); p.add_argument("--date", required=True)
    p.add_argument("--setting", default=None,
                   help="use this setting for all variables instead of the evaluate choice, "
                        "e.g. sig_a0.5")
    a = ap.parse_args()
    if a.cmd == "obs":
        cmd_obs(a.md, a.y0, a.y1)
    elif a.cmd == "evaluate":
        cmd_evaluate(a.md, a.y0, a.y1)
    else:
        cmd_apply(a.date, a.setting)
