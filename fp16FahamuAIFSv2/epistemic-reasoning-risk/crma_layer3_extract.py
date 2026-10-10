"""AIFS-ENS cycle -> CRMA layer 3, so the ~256 GB cycle store can be purged.

    python crma_layer3_extract.py \
        --store /tank/projects/aifs-run/20261001_0000/icechunk_o96 --cycle 20261001 \
        --crma-repo /path/to/crma --out /tank/projects/crma_layer3

DRAFT (2026-10-10): written where /tank is not mounted, syntax-checked only.
Run it on one complete cycle and check the manifest before trusting it, or
before wiring it into the purge guard.

Why: the products are ~1e-5 of the store (CYCLE_DATA_INVENTORY.md, "Storage
arithmetic").  CRMA's evidence is the same kind of product.  Once layer 3 is
written, the O96 corpus is no longer the only source of what CRMA needs.

What it writes, per cycle, into <out>/<cycle>/ -- OUTSIDE aifs-run so
cleanup_aifs_run.py cannot reach it, like the MJO band series:

  evidence.parquet     the CRMA long layout, shared with the medium range and
                       S2S (crma/hazards/dryspell/s2s_features.py):
                       init_time, window, region_id, region_type, feature, node,
                       member, value
                       windows D1-10 D11-20 D21-30 D31-33 (lead hours 0-792)
                       region types:
                         hydrobasin  55 lev04 basins, read through a 1.5-deg
                                     buffer (the S2S 3x3 neighbourhood)
                         synoptic    regions.SYNOPTIC_LOCI
                         global      regions.GLOBAL_LOCI
                         ocean       regions.OCEAN_LOCI -- only if the store has sst
  tp_basins.npz        daily basin rainfall per member (mm/day), day 0..32
  regional_compact.npz the regime box and the ten gridded drivers per member
                       and window, on the S2S 1.5-deg regional grid, so the IFS
                       regime catalogues (crma/hazards/dryspell/regional_layer.py)
                       apply directly
  manifest.json        rows, members, windows, variables used and missing,
                       sha256 of each file -- the purge guard keys on it

Node features use the S2S names (s2s_features.FEATURES), so an AIFS row and an
S2S row are the same quantity:
  q850 q700 uq850 vq850 mfx850 slp_mean slp_max vo850 div850 div200 z500
  cape (NaN: absent in AIFS) lapse_850_500 dpd2m t2d tp
plus AIFS-only features the S2S store lacks:
  w700           omega, Pa/s, negative = ascent (vertical node)
  cp_frac        cp/tp, convective fraction
  swvl1          top-layer soil moisture (antecedent: dry spell, drought)
  ro             runoff (flow)

Derivatives use ts-mjo/grid_ops.ReducedGaussianGrid on the native O96 rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ts-mjo"))
from grid_ops import ReducedGaussianGrid, cyclonic  # noqa: E402

WINDOWS = {"D1-10": (0, 240), "D11-20": (240, 480), "D21-30": (480, 720), "D31-33": (720, 792)}
FEATURES = ["q850", "q700", "uq850", "vq850", "mfx850", "slp_mean", "slp_max", "vo850",
            "div850", "div200", "z500", "cape", "lapse_850_500", "dpd2m", "t2d", "tp"]
EXTRA = ["w700", "cp_frac", "swvl1", "ro"]
NODE_OF = {"q850": "moisture", "q700": "moisture", "uq850": "transport", "vq850": "transport",
           "mfx850": "transport", "slp_mean": "disturbance", "slp_max": "disturbance",
           "vo850": "disturbance", "div850": "vertical", "div200": "vertical", "z500": "vertical",
           "w700": "vertical", "cape": "convective", "lapse_850_500": "convective",
           "dpd2m": "surface_humidity", "t2d": "surface_humidity", "tp": "precipitation",
           "cp_frac": "precipitation", "swvl1": "antecedent", "ro": "antecedent"}
G = 9.80665
NEEDED = ["q_850", "q_700", "u_850", "v_850", "u_200", "v_200", "z_500", "t_850", "t_500",
          "w_700", "msl", "2t", "2d", "tp", "cp", "swvl1", "ro"]


def lead_hours(ds):
    """The lead-hour coordinate, whatever the store calls it."""
    for name in ("step", "lead_time", "time"):
        if name in ds.coords and ds[name].size > 1:
            v = ds[name].values
            if np.issubdtype(v.dtype, np.timedelta64):
                return name, (v / np.timedelta64(1, "h")).astype(int)
            if np.issubdtype(v.dtype, np.datetime64):
                return name, ((v - v[0]) / np.timedelta64(1, "h")).astype(int) + 6   # first step is 6 h
            return name, v.astype(int)
    raise SystemExit("no lead dimension found -- inspect the store")


def region_weights(lat, lon, crma):
    """Point weights (n_region, n_point) for basins (buffered) and every locus."""
    sys.path.insert(0, os.path.join(crma, "bn-evidence"))
    sys.path.insert(0, os.path.join(crma, "medium-range-forecast", "4-bn-preprocess"))
    import ens
    import regions as reg
    import shapely
    lon180 = ((lon + 180) % 360) - 180
    b = reg.basins_for_countries(4, iso3=reg.ICPAC_11, domain=ens.DOMAIN, min_area_km2=5_000.0)
    out_ids, out_type, rows = [], [], []
    cw = np.cos(np.deg2rad(lat))
    for rid, geom in zip(b.region_id, b.geometry):
        g = geom.buffer(1.5)                                   # the S2S 3x3 neighbourhood, ~1.5 deg
        m = shapely.contains_xy(g, lon180, lat)
        if m.any():
            out_ids.append(rid); out_type.append("hydrobasin"); rows.append(np.where(m, cw, 0.0))
    for boxes, rtype in ((reg.SYNOPTIC_LOCI, "synoptic"), (reg.GLOBAL_LOCI, "global"),
                         (reg.OCEAN_LOCI, "ocean")):
        for name, bx in boxes.items():
            m = ((lat >= bx["lat"][0]) & (lat <= bx["lat"][1])
                 & (lon180 >= bx["lon"][0]) & (lon180 <= bx["lon"][1]))
            if m.any():
                out_ids.append(name); out_type.append(rtype); rows.append(np.where(m, cw, 0.0))
    W = np.stack(rows).astype(np.float32)
    W /= W.sum(1, keepdims=True)
    return out_ids, out_type, W


def nearest_index(lat, lon, glat, glon):
    """Nearest O96 point for every cell of a regular (glat, glon) grid -- for the
    regional compact fields, which must sit on the S2S 1.5-deg grid."""
    lon180 = ((lon + 180) % 360) - 180
    xyz = np.c_[np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lon180)),
                np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(lon180)), np.sin(np.deg2rad(lat))]
    LA, LO = np.meshgrid(glat, glon, indexing="ij")
    q = np.c_[np.cos(np.deg2rad(LA.ravel())) * np.cos(np.deg2rad(LO.ravel())),
              np.cos(np.deg2rad(LA.ravel())) * np.sin(np.deg2rad(LO.ravel())), np.sin(np.deg2rad(LA.ravel()))]
    idx = np.empty(len(q), int)
    for s in range(0, len(q), 2000):                           # chunked argmax of dot product
        idx[s:s + 2000] = (q[s:s + 2000] @ xyz.T).argmax(1)
    return idx.reshape(LA.shape)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--cycle", required=True)
    ap.add_argument("--crma-repo", required=True)
    ap.add_argument("--out", default="/tank/projects/crma_layer3")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    import icechunk
    sys.path.insert(0, os.path.join(a.crma_repo, "hazards", "dryspell"))
    from regional_situation import drivers, regime_inputs      # one definition with S2S

    ds = xr.open_zarr(icechunk.Repository.open(icechunk.local_filesystem_storage(a.store))
                      .readonly_session("main").store, consolidated=False, chunks={})
    lat, lon = ds["latitude"].values, ds["longitude"].values
    ldim, lh = lead_hours(ds)
    members = ds["member"].values
    have = [v for v in NEEDED if v in ds]
    missing = [v for v in NEEDED if v not in ds] + ["cape (absent in AIFS)", "olr (absent)"]
    print(f"{a.cycle}: {len(members)} members, {len(lh)} steps {lh.min()}-{lh.max()} h, "
          f"{lat.size} points | missing {missing}", flush=True)
    grid = ReducedGaussianGrid(lat, lon)
    ids, rtypes, W = region_weights(lat, lon, a.crma_repo)
    # the S2S regional grid (s2s_features.REGION at 1.5 deg), for the compact fields
    glat = np.arange(30.0, -40.0 - 1e-9, -1.5); glon = np.arange(10.0, 90.0, 1.5)
    nidx = nearest_index(lat, lon, glat, glon)
    init = pd.Timestamp(a.cycle)

    def get(v, m):
        return ds[v].sel(member=m).transpose(ldim, ...).values.astype(np.float32) if v in ds else None

    def one(m):
        f = {v: get(v, m) for v in have}                        # (step, point)
        tp = f.get("tp")
        if tp is not None and np.nanmax(tp) < 1.0:             # metres per 6 h -> mm
            tp = tp * 1000.0
        q8 = f["q_850"] * 1000.0
        u8, v8 = f["u_850"], f["v_850"]
        fld = {"q850": q8, "q700": f["q_700"] * 1000.0, "uq850": u8 * q8, "vq850": v8 * q8,
               "mfx850": np.hypot(u8, v8) * q8, "slp": f["msl"] / 100.0,
               "vo850": cyclonic(grid.relative_vorticity(u8, v8), lat) * 1e5,
               "div850": grid.divergence(u8, v8) * 1e5,
               "div200": grid.divergence(f["u_200"], f["v_200"]) * 1e5,
               "z500": f["z_500"] / G, "cape": np.full_like(q8, np.nan),
               "lapse_850_500": f["t_850"] - f["t_500"], "dpd2m": f["2t"] - f["2d"], "t2d": f["2d"],
               "tp": tp * 4.0 if tp is not None else None,      # 6-h interval -> mm/day rate
               "w700": f.get("w_700"),
               "cp": f.get("cp"),
               "swvl1": f.get("swvl1"), "ro": f.get("ro")}
        recs = []
        for wname, (h0, h1) in WINDOWS.items():
            sel = (lh > h0) & (lh <= h1)
            if not sel.any():
                continue
            for fe, x in fld.items():
                if x is None:
                    continue
                if fe == "slp":
                    daily = x[sel].reshape(-1, 4, x.shape[1]).mean(1) if sel.sum() % 4 == 0 else x[sel]
                    rb = daily @ W.T                                    # (day, region)
                    for name, val in (("slp_mean", rb.mean(0)), ("slp_max", rb.max(0))):
                        recs.append((wname, name, val))
                    continue
                if fe == "cp":
                    # convective fraction = window-total cp / window-total tp, per region
                    # (a mean of point ratios is undefined where it does not rain)
                    if f.get("tp") is not None:
                        num = np.nansum(x[sel], 0) @ W.T
                        den = np.nansum(f["tp"][sel], 0) @ W.T
                        recs.append((wname, "cp_frac", np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)))
                    continue
                recs.append((wname, fe, np.nanmean(x[sel], 0) @ W.T))
        ev = pd.DataFrame([{"window": w, "feature": fe, "region_id": rid, "region_type": rt,
                            "value": float(v)}
                           for w, fe, vals in recs for rid, rt, v in zip(ids, rtypes, vals)])
        ev["member"] = int(m)
        # daily basin rainfall
        basins = [i for i, t in enumerate(rtypes) if t == "hydrobasin"]
        tpd = None
        if tp is not None:
            nd = len(lh) // 4
            tpd = tp[:nd * 4].reshape(nd, 4, -1).sum(1) @ W[basins].T     # (day, basin) mm/day
        # regional compact, on the S2S 1.5-deg grid, window means
        reg_vars = {"u850": u8, "v850": v8, "q850": q8, "slp": f["msl"] / 100.0,
                    "olr": np.full_like(u8, np.nan), "u200": f["u_200"], "z500": f["z_500"] / G,
                    "tp": tp * 4.0 if tp is not None else np.full_like(u8, np.nan)}
        Fw = []
        for wname, (h0, h1) in list(WINDOWS.items())[:3]:
            sel = (lh > h0) & (lh <= h1)
            Fw.append(np.stack([np.nanmean(reg_vars[v][sel], 0)[nidx] for v in
                                ["u850", "v850", "q850", "slp", "olr", "u200", "z500", "tp"]]))
        Fw = np.stack(Fw)[None]                                   # (1, window, var, lat, lon)
        vars8 = ["u850", "v850", "q850", "slp", "olr", "u200", "z500", "tp"]
        return m, ev, tpd, regime_inputs(Fw, glat, glon, vars8)[0], drivers(Fw, glat, glon, vars8)[0]

    od = os.path.join(a.out, a.cycle); os.makedirs(od, exist_ok=True)
    evs, tps, Xs, Ds = [], {}, {}, {}
    with ThreadPoolExecutor(a.workers) as ex:
        for m, ev, tpd, X, D in ex.map(one, members):
            evs.append(ev); tps[m] = tpd; Xs[m] = X; Ds[m] = D
            print(f"  member {m} done", flush=True)
    E = pd.concat(evs, ignore_index=True)
    E.insert(0, "init_time", init)
    E["node"] = E.feature.map(NODE_OF).fillna("disturbance")
    for c in ("window", "region_id", "region_type", "feature", "node"):
        E[c] = E[c].astype("category")
    p_ev = os.path.join(od, "evidence.parquet"); E.to_parquet(p_ev, index=False)
    ms = sorted(tps)
    bids = [i for i, t in zip(ids, rtypes) if t == "hydrobasin"]
    p_tp = os.path.join(od, "tp_basins.npz")
    np.savez_compressed(p_tp, tp=np.stack([tps[m] for m in ms]).astype(np.float32),
                        members=np.array(ms), ids=np.array(bids), init=str(init.date()))
    p_rc = os.path.join(od, "regional_compact.npz")
    np.savez_compressed(p_rc, X=np.stack([Xs[m] for m in ms])[None].astype(np.float16),
                        DRV=np.stack([Ds[m] for m in ms])[None].astype(np.float32),
                        inits=np.array([init.to_datetime64()]).astype("datetime64[D]"),
                        windows=np.array(["D1-10", "D11-20", "D21-30"]), lat=glat, lon=glon)
    man = {"cycle": a.cycle, "store": a.store, "members": len(ms), "steps": int(len(lh)),
           "windows": list(WINDOWS), "rows": int(len(E)), "regions": len(ids),
           "variables_used": have, "missing": missing, "draft": True,
           "files": {os.path.basename(p): {"bytes": os.path.getsize(p), "sha256": sha(p)}
                     for p in (p_ev, p_tp, p_rc)}}
    json.dump(man, open(os.path.join(od, "manifest.json"), "w"), indent=1)
    print(f"wrote {od}: {len(E):,} evidence rows, {len(ms)} members -- manifest.json")


if __name__ == "__main__":
    main()
