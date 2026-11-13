#!/usr/bin/env python3
"""A/B test of the EDA perturbation inflation (RUNBOOK_20261105 section 4.2): weeks 3-4
ensemble spread of two member sets of the same year, from the 1.5 deg products on
source.coop, with the same weekly aggregation as hc_bc.py / past_cycle_bc.py.

    hc_ab_spread.py --mmdd 1105 --year 2025 --a 1-11 --b 1,12-21
        set A = control + raw EDA (x1.0), set B = control + EDA x1.6

Prints the area-weighted rms spread (std over members, ddof=1) per variable and week,
globally and over East Africa, and the ratio B / A.
"""

import argparse
import os
import sys

import numpy as np
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import sc_io  # noqa: E402
import past_cycle_bc as B  # noqa: E402  (WEEKS, RAW, VARS: the bias-correction definitions)

LAT = np.linspace(90, -90, 121)
LON = np.arange(240) * 1.5
EA = (-12, 15, 22, 52)                    # lat0, lat1, lon0, lon1: East Africa box
NAMES = {0: "week 3 (D+18..24)", 1: "week 4 (D+25..31)"}


def members(spec):
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def weekly(mmdd, year, m, work):
    key = sc_io.product_prefix(mmdd, year, m) + "aifs_ensemble_forecast_1p5deg_member001.nc"
    path = os.path.join(work, f"{year}_m{m:03d}.nc")
    if not os.path.exists(path):
        sc_io.get(key, path)
    out = {}
    with xr.open_dataset(path) as d:
        for v in B.VARS:
            x = d[B.RAW[v]].isel(member=0)
            for w, (chunks, nsteps) in enumerate(B.WEEKS):
                s = sum(x.isel(time=t).sum("step", skipna=True) for t in chunks).values
                out[(v, w)] = s * 1000 * 7.0 / (nsteps * 6 / 24) if v == "pr" else s / nsteps
    return out


def rms_spread(stack, box=None):
    sd = np.std(stack, axis=0, ddof=1)
    w = np.cos(np.radians(LAT))[:, None] * np.ones((1, 240))
    if box:
        la0, la1, lo0, lo1 = box
        w = w * ((LAT[:, None] >= la0) & (LAT[:, None] <= la1)
                 & (LON[None, :] >= lo0) & (LON[None, :] <= lo1))
    ok = np.isfinite(sd)
    return float(np.sqrt(np.sum(w[ok] * sd[ok] ** 2) / np.sum(w[ok])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmdd", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--a", required=True, help="members of set A, e.g. 1-11")
    ap.add_argument("--b", required=True, help="members of set B, e.g. 1,12-21")
    ap.add_argument("--work", default="/content/ab")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    sets = {"A": members(a.a), "B": members(a.b)}
    data = {s: [weekly(a.mmdd, a.year, m, a.work) for m in ms] for s, ms in sets.items()}
    units = {"tas": "K", "mslp": "Pa", "pr": "mm/week"}
    print(f"{a.year}{a.mmdd}: A = members {a.a} ({len(sets['A'])}), B = members {a.b} ({len(sets['B'])})")
    print(f"{'var':5s} {'week':18s} {'region':6s} {'A sd':>9s} {'B sd':>9s} {'B/A':>6s}")
    for v in B.VARS:
        for w in (0, 1):
            for rname, box in (("global", None), ("EA", EA)):
                sa = rms_spread(np.stack([d[(v, w)] for d in data["A"]]), box)
                sb = rms_spread(np.stack([d[(v, w)] for d in data["B"]]), box)
                print(f"{v:5s} {NAMES[w]:18s} {rname:6s} {sa:9.4g} {sb:9.4g} {sb/sa:6.2f}   [{units[v]}]")
