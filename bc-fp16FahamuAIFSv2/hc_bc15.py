#!/usr/bin/env python3
"""Hindcast calibration for tropical-storm days, MJO and the Africa / Indian Ocean indices
(README 0.2, variables 4-15). tas / mslp / pr stay in hc_bc.py.

Model side, per hindcast year: hc_nc/<YYYY><MMDD>_0000/{ts_days.nc, mjo_probs.nc,
indices.npz} (hc_member.sh). Truth side:
    TS days   IBTrACS via AI_WQ_package.TS_processing.compute_single_week_numTSdays,
              weeks starting D+18 and D+25 (the weeks ts_days.py counts)       global
    MJO       AI-WQ MJO_DAILY_<YYYY>.nc, RMM1/RMM2 for D..D+33                   global
    indices   truth_idx/<YYYYMMDD>.npz (era5_truth_indices.py, same compute())   domain

Corrections, each checked leave-one-year-out:
    TS days   tercile edges from the model's own 20-year counts ("model climate"), with
              AI-WQ's binning (ts_days.tercile_probs); the observed edges stay the target
    MJO       per lead-day bin, a linear map model (vpm1, vpm2) -> observed (RMM1, RMM2)
              with intercept, fitted over the training years (corrects amplitude, phase
              offset and mean-state bias); categories by mjo_index.phase_from_pcs
    indices   weekly mean bias removed; tercile probabilities from model-climate edges
    maps      per-cell bias, significance-masked (p < 0.05), half strength, as hc_bc.py

    hc_bc15.py evaluate 1105 2006 2025      -> CACHE/hc15_eval_1105.json + .md
    hc_bc15.py apply --date 20261105        -> aifs-run/D_0000/aiwq_bc_hc/*_bchc.nc
"""

import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.environ.get("EA_AIFS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "fp16FahamuAIFSv2", "ts-mjo"))
from ts_days import tercile_probs  # noqa: E402
from mjo_index import phase_from_pcs  # noqa: E402

S = os.environ.get("BC_CACHE", "/tank/projects/bc-aifs")
RUN = os.environ.get("BC_RUN", "/tank/projects/aifs-run")
IBTRACS = os.environ.get("HC_IBTRACS", "/tank/projects/ibtracs/IBTrACS.ALL.v04r01.nc")
MJO_OBS = os.path.join(os.environ.get("HC_MJO_OBS_DIR", "/tank/projects/era5_vpm_clim/aiwq"),
                       "MJO_DAILY_{y}.nc")
BASINS = ["ATL", "NWP", "SWIO", "SEIO"]
WEEKS = {"week3": slice(18, 25), "week4": slice(25, 32)}     # lead days 19-25, 26-32
MJO_BINS = [(1, 7), (8, 14), (15, 18), (19, 25), (26, 33)]   # lead-day bins for the map
GHA = (-12, 18, 21, 52)
_IB = None


def init_of(year, md):
    return f"{year}{md}"


# ------------------------------------------------------------------ truth
def ts_truth(init):
    """(2 weeks, 4 basins) observed storm days, cached."""
    global _IB
    f = f"{S}/truth_ts/{init}.npz"
    if os.path.exists(f):
        return np.load(f)["counts"]
    from AI_WQ_package import TS_processing as T
    import contextlib, io
    if _IB is None:
        _IB = xr.open_dataset(IBTRACS)
    d0 = dt.datetime.strptime(init, "%Y%m%d")
    out = []
    for k in (18, 25):
        ds = (d0 + dt.timedelta(days=k)).strftime("%Y%m%d")
        with contextlib.redirect_stdout(io.StringIO()):
            da = T.compute_single_week_numTSdays(ds, _IB)
        out.append(np.asarray(da.sel(basin=BASINS).values if "basin" in da.dims else da.values).ravel()[:4])
    out = np.asarray(out, int)
    os.makedirs(os.path.dirname(f), exist_ok=True)
    np.savez(f, counts=out)
    return out


def mjo_truth(init):
    """(34, 2) observed RMM1, RMM2 for D .. D+33."""
    d0 = dt.datetime.strptime(init, "%Y%m%d")
    days = [d0 + dt.timedelta(days=k) for k in range(34)]
    parts = []
    for y in sorted({d.year for d in days}):
        o = xr.open_dataset(MJO_OBS.format(y=y))
        parts.append(o[["RMM1", "RMM2"]])
    o = xr.concat(parts, "time").sel(time=[np.datetime64(d) for d in days])
    return np.stack([o.RMM1.values, o.RMM2.values], 1)


def idx_truth(init):
    f = f"{S}/truth_idx/{init}.npz"
    return np.load(f) if os.path.exists(f) else None


# ------------------------------------------------------------------ model
class Members:
    """indices.npz of k members of one year: ['daily_x'] -> member mean, .mem('daily_x')
    -> (k, ...), constant grids (dom_lat/lon/land) from the first member."""
    def __init__(self, files):
        self.z = [np.load(f) for f in files]
        self.files = self.z[0].files

    def mem(self, key):
        return np.stack([np.asarray(z[key], float) for z in self.z])

    def __getitem__(self, key):
        if key.startswith("dom_"):
            return self.z[0][key]
        return np.nanmean(self.mem(key), 0)


def member_dirs(init, base=None):
    """Colab bundle: hc15/<init>/m<NNN>/ (11 members); GPU box: hc_nc/<init>_0000/ (1)."""
    d = f"{S}/hc15/{init}"
    if base is None and os.path.isdir(d):
        return sorted(os.path.join(d, m) for m in os.listdir(d)
                      if os.path.exists(os.path.join(d, m, "indices.npz")))
    d = base or f"{S}/hc_nc/{init}_0000"
    return [d] if os.path.exists(f"{d}/DONE") else []


def model(init, base=None):
    """ts (k,2,4) storm days, mjo (k,34,2) VPM PCs, idx Members -- k members of one year."""
    ds_ = member_dirs(init, base)
    if not ds_:
        return None
    ts = np.stack([xr.open_dataset(f"{d}/ts_days.nc")["storm_days"].sel(basin=BASINS).values[0]
                   for d in ds_])
    mjo = []
    for d in ds_:
        mj = xr.open_dataset(f"{d}/mjo_probs.nc")
        mjo.append(np.stack([mj.vpm1.values[0], mj.vpm2.values[0]], 1))
    return dict(ts=ts, mjo=np.stack(mjo), idx=Members([f"{d}/indices.npz" for d in ds_]))


def collect(md, y0, y1, need_ts=True):
    yrs, M, T = [], {}, {}
    for y in range(y0, y1 + 1):
        i = init_of(y, md)
        m, t = model(i), idx_truth(i)
        if m is None:
            print(f"  {y}: no model, left out")
            continue
        if t is None:      # TS and MJO do not need it; indices and maps are skipped
            print(f"  {y}: no index truth (TS/MJO only)")
        yrs.append(y)
        M[y] = m
        T[y] = dict(ts=ts_truth(i) if need_ts else None, mjo=mjo_truth(i), idx=t)
    return yrs, M, T


# ------------------------------------------------------------------ helpers
def terciles(x):
    return np.percentile(np.asarray(x, float), [100 / 3, 200 / 3])


def cat(x, lo, hi):
    """AI-WQ binning: 0 below (x<lo), 1 near (lo<=x<hi), 2 above (x>=hi)."""
    return np.where(x < lo, 0, np.where(x >= hi, 2, 1))


def rps3(p, obs_cat):
    o = np.zeros(3); o[obs_cat] = 1
    return float(np.sum((np.cumsum(p) - np.cumsum(o)) ** 2))


def mjo_fit(Xm, Yo):
    """least squares Yo (n,2) ~ [Xm, 1] (n,3) -> (3,2)"""
    A = np.c_[Xm, np.ones(len(Xm))]
    coef, *_ = np.linalg.lstsq(A, Yo, rcond=None)
    return coef


def mjo_apply(coef_bins, X):
    """X (34,2) or (n,34,2) -> mapped, per lead-day bin; day 0 left as is."""
    X = np.asarray(X, float)
    out = X.copy()
    for (a, b), coef in coef_bins.items():
        sl = (Ellipsis, slice(a, b + 1), slice(None))
        xs = X[sl]
        out[sl] = np.c_[xs.reshape(-1, 2), np.ones(xs.reshape(-1, 2).shape[0])].dot(coef).reshape(xs.shape)
    return out


def nn_map(src_lat, src_lon, dst_lat, dst_lon):
    """index of the nearest src point for every dst point (great-circle via 3-D chords)."""
    from scipy.spatial import cKDTree
    v = lambda la, lo: np.c_[np.cos(np.radians(la)) * np.cos(np.radians(lo)),
                             np.cos(np.radians(la)) * np.sin(np.radians(lo)), np.sin(np.radians(la))]
    return cKDTree(v(src_lat, src_lon)).query(v(dst_lat, dst_lon))[1]


# ------------------------------------------------------------------ evaluate
def evaluate(md, y0, y1):
    yrs, M, T = collect(md, y0, y1)
    n = len(yrs)
    print(f"hindcast {md}: {n} years with model and truth: {yrs[0]}-{yrs[-1]}")
    R = dict(years=yrs, ts={}, mjo={}, idx={}, maps={})
    lines = [f"# 15-variable hindcast evaluation, {md}, {yrs[0]}-{yrs[-1]} ({n} years, {M[yrs[0]]['ts'].shape[0]} members/yr)", ""]

    # ---- TS days
    lines += ["## Tropical-storm days (IBTrACS truth, global)", "",
              "| basin, week | obs mean | model mean | raw: hit / RPS | model-climate terciles: hit / RPS | scoreable years |",
              "|---|---|---|---|---|---|"]
    for wi, wk in enumerate(("week3", "week4")):
        for bi, b in enumerate(BASINS):
            mm = [np.asarray(M[y]["ts"][:, wi, bi], float) for y in yrs]   # members per year
            m = np.array([x.mean() for x in mm])
            o = np.array([T[y]["ts"][wi, bi] for y in yrs], float)
            hr = hm = rr = rm = 0.0; ns = 0
            for j in range(n):
                tr = np.arange(n) != j
                lo, hi = terciles(o[tr])
                if lo == hi:
                    continue                                 # AI-WQ: not scoreable
                oc = int(cat(o[j], lo, hi))
                p_raw, _ = tercile_probs(mm[j], lo, hi)
                mlo, mhi = terciles(np.concatenate([mm[t] for t in np.where(tr)[0]]))
                p_mc, _ = tercile_probs(mm[j], mlo, mhi) if mlo != mhi else (p_raw, None)
                hr += p_raw[oc]; hm += p_mc[oc]; rr += rps3(p_raw, oc); rm += rps3(p_mc, oc); ns += 1
            key = f"{b}_{wk}"
            R["ts"][key] = dict(obs_mean=o.mean(), model_mean=m.mean(), n_scoreable=ns,
                                hit_raw=hr / max(ns, 1), hit_mc=hm / max(ns, 1),
                                rps_raw=rr / max(ns, 1), rps_mc=rm / max(ns, 1))
            r = R["ts"][key]
            lines.append(f"| {b} {wk} | {r['obs_mean']:.1f} | {r['model_mean']:.1f} | "
                         f"{r['hit_raw']:.2f} / {r['rps_raw']:.3f} | {r['hit_mc']:.2f} / {r['rps_mc']:.3f} | {ns}/{n} |")

    # ---- MJO
    lines += ["", "## MJO (AI-WQ RMM truth, global)", "",
              "| lead days | obs amplitude | model amplitude | raw: category hit / RMM error | mapped: category hit / RMM error |",
              "|---|---|---|---|---|"]
    Xm = np.stack([M[y]["mjo"] for y in yrs])       # (n, k, 34, 2): every member
    Yo = np.stack([T[y]["mjo"] for y in yrs])       # (n, 34, 2)
    k = Xm.shape[1]
    Yk = np.repeat(Yo[:, None], k, 1)               # each member paired with its year's truth
    for (a, b) in MJO_BINS:
        hr = hm = er = em = 0.0; cnt = 0
        for j in range(n):
            tr = np.arange(n) != j
            coef = mjo_fit(Xm[tr, :, a:b + 1].reshape(-1, 2), Yk[tr, :, a:b + 1].reshape(-1, 2))
            for mi in range(k):
                x, y_ = Xm[j, mi, a:b + 1], Yo[j, a:b + 1]
                xm = np.c_[x, np.ones(len(x))].dot(coef)
                po, _ = phase_from_pcs(y_[:, 0], y_[:, 1])
                pr_, _ = phase_from_pcs(x[:, 0], x[:, 1])
                pm, _ = phase_from_pcs(xm[:, 0], xm[:, 1])
                hr += (pr_ == po).sum(); hm += (pm == po).sum()
                er += np.sum((x - y_) ** 2); em += np.sum((xm - y_) ** 2); cnt += len(x)
        amp_o = np.hypot(*Yo[:, a:b + 1].reshape(-1, 2).T).mean()
        amp_m = np.hypot(*Xm[:, :, a:b + 1].reshape(-1, 2).T).mean()
        R["mjo"][f"{a}-{b}"] = dict(amp_obs=amp_o, amp_model=amp_m, hit_raw=hr / cnt, hit_map=hm / cnt,
                                    err_raw=np.sqrt(er / cnt), err_map=np.sqrt(em / cnt))
        r = R["mjo"][f"{a}-{b}"]
        lines.append(f"| {a}-{b} | {amp_o:.2f} | {amp_m:.2f} | {r['hit_raw']:.2f} / {r['err_raw']:.2f} | "
                     f"{r['hit_map']:.2f} / {r['err_map']:.2f} |")

    if any(T[y]["idx"] is None for y in yrs):
        lines += ["", "Indices and weekly maps: skipped (no index truth for every year)."]
        js = json.loads(json.dumps(R, default=float))
        json.dump(js, open(f"{S}/hc15_eval_{md}.json", "w"), indent=1)
        open(f"{S}/hc15_eval_{md}.md", "w").write("\n".join(lines) + "\n")
        print("\n".join(lines))
        return
    # ---- indices (weekly means of the daily series)
    names = sorted(k[6:] for k in M[yrs[0]]["idx"].files if k.startswith("daily_"))
    lines += ["", "## Indices, weekly means (ERA5 truth, same definitions)", "",
              "| index | week | bias (model−ERA5) | t | rmse raw → corrected | tercile hit raw → model-climate |",
              "|---|---|---|---|---|---|"]
    for nm in names:
        for wk, sl in WEEKS.items():
            m = np.array([np.nanmean(M[y]["idx"][f"daily_{nm}"][sl]) for y in yrs])
            o = np.array([np.nanmean(T[y]["idx"][f"daily_{nm}"][sl]) for y in yrs])
            d = m - o
            t = d.mean() / (d.std(ddof=1) / np.sqrt(n))
            er = ec = hr = hm = 0.0
            for j in range(n):
                tr = np.arange(n) != j
                b = d[tr].mean()
                er += (m[j] - o[j]) ** 2; ec += (m[j] - b - o[j]) ** 2
                lo, hi = terciles(o[tr]); oc = cat(o[j], lo, hi)
                hr += cat(m[j], lo, hi) == oc
                pool = np.concatenate([np.nanmean(M[yrs[t]]["idx"].mem(f"daily_{nm}")[:, sl], 1)
                                       for t in np.where(tr)[0]])
                mlo, mhi = terciles(pool); hm += cat(m[j], mlo, mhi) == oc
            R["idx"][f"{nm}_{wk}"] = dict(bias=d.mean(), t=t, rmse_raw=np.sqrt(er / n), rmse_corr=np.sqrt(ec / n),
                                          hit_raw=hr / n, hit_mc=hm / n, obs_sd=o.std(ddof=1))
            r = R["idx"][f"{nm}_{wk}"]
            lines.append(f"| {nm} | {wk} | {r['bias']:+.4g} | {t:+.1f} | {r['rmse_raw']:.4g} → {r['rmse_corr']:.4g} | "
                         f"{r['hit_raw']:.2f} → {r['hit_mc']:.2f} |")

    # ---- weekly maps (model O96 cells -> nearest, on the ERA5 1 deg points)
    t0 = T[yrs[0]]["idx"]
    lat1, lon1 = t0["dom_lat"], t0["dom_lon"]
    land1 = t0["dom_land"]
    gha = (lat1 >= GHA[0]) & (lat1 <= GHA[1]) & (lon1 >= GHA[2]) & (lon1 <= GHA[3]) & land1
    nn = nn_map(M[yrs[0]]["idx"]["dom_lat"], M[yrs[0]]["idx"]["dom_lon"], lat1, lon1)
    from scipy import stats
    tcrit = stats.t.ppf(0.975, n - 2)
    lines += ["", "## Weekly maps, per-cell bias (significance-masked, half strength)", "",
              "| field | week | MSE change, domain | MSE change, Greater Horn land | cells corrected |",
              "|---|---|---|---|---|"]
    land_only = {"swvl1", "swvl2", "tmax", "twmax", "tp"}
    for f in ("swvl1", "swvl2", "tmax", "twmax", "tcw", "swh", "mwp", "skt", "tp"):
        for wi, wk in enumerate(("week3", "week4")):
            m = np.stack([M[y]["idx"][f"week_{f}"][wi][nn] for y in yrs]).astype(float)
            o = np.stack([T[y]["idx"][f"week_{f}"][wi] for y in yrs]).astype(float)
            dom = np.isfinite(m).all(0) & np.isfinite(o).all(0) & (land1 if f in land_only else True)
            raw = cor = rawg = corg = 0.0; ncorr = 0
            for j in range(n):
                tr = np.arange(n) != j
                d = m[tr] - o[tr]
                b = d.mean(0); tt = b / (d.std(0, ddof=1) / np.sqrt(n - 1) + 1e-12)
                tab = np.where(np.abs(tt) > tcrit, 0.5 * b, 0.0)
                ncorr += (np.abs(tt[dom]) > tcrit).mean()
                e0 = (m[j] - o[j]) ** 2; e1 = (m[j] - tab - o[j]) ** 2
                raw += e0[dom].mean(); cor += e1[dom].mean()
                g = dom & gha
                rawg += e0[g].mean(); corg += e1[g].mean()
            R["maps"][f"{f}_{wk}"] = dict(mse_change_domain=cor / raw - 1, mse_change_gha=corg / rawg - 1,
                                          frac_corrected=ncorr / n)
            r = R["maps"][f"{f}_{wk}"]
            lines.append(f"| {f} | {wk} | {100*r['mse_change_domain']:+.1f}% | {100*r['mse_change_gha']:+.1f}% | "
                         f"{100*r['frac_corrected']:.0f}% |")

    js = json.loads(json.dumps(R, default=float))
    json.dump(js, open(f"{S}/hc15_eval_{md}.json", "w"), indent=1)
    open(f"{S}/hc15_eval_{md}.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {S}/hc15_eval_{md}.json and .md")


# ------------------------------------------------------------------ apply
def rt_members(date):
    """Colab real-time members: rt15/<date>/m<NNN>/ in the bundle, else [] (GPU box layout)."""
    d = f"{S}/rt15/{date}"
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, m) for m in os.listdir(d)
                  if os.path.exists(os.path.join(d, m, "ts_days.nc")))


def apply(date, hindcast_md=None):
    md = hindcast_md or date[4:8]
    ev = json.load(open(f"{S}/hc15_eval_{md}.json"))
    yrs = ev["years"]
    _, M, T = collect(md, min(yrs), max(yrs), need_ts=False)   # apply needs no TS truth
    base = f"{RUN}/{date}_0000"
    rt = f"{base}/products15"
    out = f"{base}/aiwq_bc_hc"
    os.makedirs(out, exist_ok=True)
    n = len(yrs)

    # ---- TS: model-climate tercile edges from the 20 hindcast years
    rtm = rt_members(date)
    raw = xr.open_dataset(f"{rtm[0]}/ts_days.nc") if rtm else xr.open_dataset(f"{rt}/ts_days.nc")
    cnt = (np.stack([xr.open_dataset(f"{d}/ts_days.nc")["storm_days"].sel(basin=BASINS).values[0]
                     for d in rtm]) if rtm
           else raw["storm_days"].sel(basin=BASINS).values)              # (51,2,4)
    probs = np.zeros((2, 4, 3)); bounds = np.zeros((2, 4, 2)); degen = []
    for wi in range(2):
        for bi, b in enumerate(BASINS):
            mlo, mhi = terciles(np.concatenate([M[y]["ts"][:, wi, bi] for y in yrs]))
            p, dg = tercile_probs(cnt[:, wi, bi], mlo, mhi)
            if dg:
                degen.append(f"{b} week{wi+3}")
            probs[wi, bi], bounds[wi, bi] = p, (mlo, mhi)
    ds = raw.isel(member=[0]).copy() if "member" in raw.dims else raw.copy()
    ds["TS_tercile_probability"] = (("week", "basin", "category"), probs)
    ds["tercile_bounds"] = (("week", "basin", "bound"), bounds)
    ds.attrs.update(tercile_source=f"MODEL climate: {n}-year same-date hindcast {md} "
                                   f"({yrs[0]}-{yrs[-1]}), bc-fp16FahamuAIFSv2/hc_bc15.py",
                    degenerate_model_terciles=", ".join(degen) or "none")
    f_ts = f"{out}/ts_days_probs_{date}_bchc.nc"
    ds.to_netcdf(f_ts)

    # ---- MJO: per lead-bin map fitted on all hindcast years
    Xm = np.stack([M[y]["mjo"] for y in yrs]); Yo = np.stack([T[y]["mjo"] for y in yrs])
    Yk = np.repeat(Yo[:, None], Xm.shape[1], 1)
    coefs = {(a, b): mjo_fit(Xm[:, :, a:b + 1].reshape(-1, 2), Yk[:, :, a:b + 1].reshape(-1, 2))
             for a, b in MJO_BINS}
    if rtm:
        mjs = [xr.open_dataset(f"{d}/mjo_probs.nc") for d in rtm]
        mr = xr.concat(mjs, "member") if "member" in mjs[0].dims else mjs[0]
        X = np.stack([np.concatenate([m.vpm1.values for m in mjs]),
                      np.concatenate([m.vpm2.values for m in mjs])], -1)  # (51,34,2)
    else:
        mr = xr.open_dataset(f"{rt}/mjo_probs.nc")
        X = np.stack([mr.vpm1.values, mr.vpm2.values], -1)                 # (51,34,2)
    Xc = mjo_apply(coefs, X)
    ph, amp = phase_from_pcs(Xc[..., 0], Xc[..., 1])
    pp = np.stack([(ph == k).mean(0) for k in range(9)], 1)                # (34,9)
    md_ = mr.copy()
    for v in [v for v in md_.data_vars if "member" in md_[v].dims and md_[v].sizes["member"] != X.shape[0]]:
        md_ = md_.drop_vars(v)
    md_["MJO_phase_probability"] = (("day", "MJO_phase"), pp)
    md_["vpm1"] = (("member", "day"), Xc[..., 0]); md_["vpm2"] = (("member", "day"), Xc[..., 1])
    md_["amplitude"] = (("member", "day"), amp); md_["phase"] = (("member", "day"), ph)
    md_.attrs.update(calibration=f"per lead-day bin linear map onto AI-WQ RMM, {n}-year hindcast "
                                 f"{md}; bins {MJO_BINS}; bc-fp16FahamuAIFSv2/hc_bc15.py")
    f_mjo = f"{out}/mjo_probs_{date}_bchc.nc"
    md_.to_netcdf(f_mjo)

    # ---- indices: corrected weekly ensemble means and model-climate tercile probabilities
    if not ev["idx"]:
        for f in (f_ts, f_mjo):
            print(f"wrote {f}")
        print("indices: skipped (no index truth)")
        return
    if rtm:
        E = [np.load(f"{d}/indices.npz") for d in rtm]
    else:
        mem = sorted(f for f in os.listdir(rt) if f.startswith("indices_m"))
        E = [np.load(f"{rt}/{f}") for f in mem]
    names = sorted(k[6:] for k in E[0].files if k.startswith("daily_"))
    rows = []
    vals = np.zeros((len(names), 2, 3)); pr3 = np.zeros((len(names), 2, 3)); obs_t = np.zeros((len(names), 2, 2))
    for ii, nm in enumerate(names):
        for wi, (wk, sl) in enumerate(WEEKS.items()):
            x = np.array([np.nanmean(e[f"daily_{nm}"][sl]) for e in E])                 # 51 members
            m = np.array([np.nanmean(M[y]["idx"][f"daily_{nm}"][sl]) for y in yrs])
            o = np.array([np.nanmean(T[y]["idx"][f"daily_{nm}"][sl]) for y in yrs])
            b = (m - o).mean()
            r = ev["idx"][f"{nm}_{wk}"]
            use = r["rmse_corr"] < r["rmse_raw"]
            mlo, mhi = terciles(np.concatenate([np.nanmean(M[y]["idx"].mem(f"daily_{nm}")[:, sl], 1)
                                                for y in yrs]))
            p, _ = tercile_probs(x, mlo, mhi)
            vals[ii, wi] = (x.mean(), x.mean() - (b if use else 0.0), b)
            pr3[ii, wi] = p
            obs_t[ii, wi] = terciles(o)
    ids = xr.Dataset(
        {"ens_mean_raw": (("index", "week"), vals[..., 0]),
         "ens_mean_corrected": (("index", "week"), vals[..., 1]),
         "hindcast_bias": (("index", "week"), vals[..., 2]),
         "tercile_probability": (("index", "week", "category"), pr3),
         "obs_terciles": (("index", "week", "bound"), obs_t)},
        coords=dict(index=names, week=list(WEEKS), category=["below", "near", "above"],
                    bound=["lower", "upper"]),
        attrs=dict(forecast_date=date, n_members=len(E),
                   method=f"weekly means of lead days 19-25 / 26-32; bias removed where it lowered "
                          f"leave-one-year-out rmse; tercile probabilities against the model's own "
                          f"{n}-year climate; bc-fp16FahamuAIFSv2/hc_bc15.py"))
    f_idx = f"{out}/indices_{date}_bchc.nc"
    ids.to_netcdf(f_idx)
    for f in (f_ts, f_mjo, f_idx):
        print(f"wrote {f}")
    if degen:
        print(f"  TS: model terciles degenerate (not scoreable by AI-WQ's rule) for {degen}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("evaluate"); p.add_argument("md"); p.add_argument("y0", type=int); p.add_argument("y1", type=int)
    p = sp.add_parser("apply"); p.add_argument("--date", required=True)
    p.add_argument("--hindcast-md", default=None, help="MMDD of the hindcast to correct with")
    a = ap.parse_args()
    evaluate(a.md, a.y0, a.y1) if a.cmd == "evaluate" else apply(a.date, a.hindcast_md)
