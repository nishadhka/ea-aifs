#!/usr/bin/env python3
"""Africa / Indian Ocean indices (README section 0.2, variables 6-15) from one member of
an O96 Icechunk store. The same code serves the hindcast members and the real-time
forecast, so both are extracted identically.

Reads the 132 six-hourly steps (6-792 h). A lead day d (1..33) is steps 4(d-1)..4d-1,
i.e. lead hours (24(d-1), 24d]. Weeks 3 and 4 are lead days 19-25 and 26-32 (AI-WQ's
forecast periods 1 and 2).

Output (.npz):
    daily_<name>     (33,)   one value per lead day
    week_<field>     (2, n)  weekly means (tp: weekly sum, mm) on the domain cells
    dom_lat, dom_lon (n,)    O96 cells inside the Africa-Indian Ocean domain
    meta             json    init, member, store, definitions

    hc_extract.py --store DIR --member 0 --init 20251105 --out indices.npz
"""

import argparse
import json
import os

import numpy as np

LSM_1P5 = "/tank/projects/bc-aifs/aiwq/land_sea_mask_1pt5DEG.nc"
NDAYS = 33
WEEKS = {"week3": range(19, 26), "week4": range(26, 33)}
DOMAIN = (-40, 40, -25, 120)                  # lat0, lat1, lon0, lon1 (deg E, -180..180)
GHA = (-12, 18, 21, 52)                       # Greater Horn of Africa

# name: (field or derived, box (lat0, lat1, lon0, lon1), mask: None/'land'/'sea')
BOXES = {
    "iod_w_skt":        ("skt",   (-10, 10, 50, 70),    "sea"),
    "iod_e_skt":        ("skt",   (-10, 0, 90, 110),    "sea"),
    "io_u850":          ("u_850", (-5, 5, 70, 90),      None),
    "somali_u925":      ("u_925", (-5, 10, 40, 55),     None),
    "somali_v925":      ("v_925", (-5, 10, 40, 55),     None),
    "somali_speed925":  ("spd925", (-5, 10, 40, 55),    None),
    "gha_swvl1":        ("swvl1", GHA,                  "land"),
    "gha_swvl2":        ("swvl2", GHA,                  "land"),
    "mascarene_msl":    ("msl",   (-35, -20, 40, 90),   None),
    "tej_u150":         ("u_150", (0, 15, 40, 80),      None),
    "tej_u200":         ("u_200", (0, 15, 40, 80),      None),
    "aej_u600":         ("u_600", (10, 15, -20, 20),    None),
    "aej_u700":         ("u_700", (10, 15, -20, 20),    None),
    "wio_swh":          ("swh",   (-12, 12, 39, 60),    "sea"),
    "wio_mwp":          ("mwp",   (-12, 12, 39, 60),    "sea"),
    "ea_tcw":           ("tcw",   (-5, 15, 30, 52),     None),
    "ea_qu850":         ("qu850", (-5, 10, 40, 52),     None),
    "ea_qv850":         ("qv850", (-5, 10, 40, 52),     None),
    "gha_tmax":         ("tmax",  GHA,                  "land"),
    "gha_twmax":        ("twmax", GHA,                  "land"),
}
ITCZ_SECTORS = {"africa": (-20, 55), "indian_ocean": (55, 100), "east_africa": (30, 52)}
WEEKLY_MAPS = ["swvl1", "swvl2", "tmax", "twmax", "tcw", "swh", "mwp", "skt", "tp"]


def open_store(path, tag=None):
    import icechunk
    import zarr
    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(path))
    s = repo.readonly_session(tag=tag) if tag else repo.readonly_session(branch="main")
    return zarr.open_group(s.store, mode="r")


def in_box(lat, lon180, b):
    lat0, lat1, lon0, lon1 = b
    return (lat >= lat0) & (lat <= lat1) & (lon180 >= lon0) & (lon180 <= lon1)


def land_mask(lat, lon180):
    """AI-WQ's 1.5 deg land-sea mask, nearest cell (the mask the scores use)."""
    import xarray as xr
    m = xr.open_dataset(LSM_1P5)["lsm"].squeeze().values > 0.5
    i = np.clip(np.rint((90 - lat) / 1.5).astype(int), 0, 120)
    j = np.rint(np.mod(lon180, 360) / 1.5).astype(int) % 240
    return m[i, j]


def wet_bulb_c(t_k, td_k):
    """Stull (2011) wet-bulb temperature (deg C) from 2 m temperature and dewpoint."""
    t, td = t_k - 273.15, td_k - 273.15
    es = lambda x: 6.112 * np.exp(17.62 * x / (243.12 + x))
    rh = np.clip(100 * es(td) / es(t), 1, 100)
    return (t * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) + np.arctan(t + rh)
            - np.arctan(rh - 1.676331) + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
            - 4.686035)


def daily(x, how="mean"):
    """(132, n) six-hourly -> (33, n) per lead day."""
    x = x[: 4 * NDAYS].reshape(NDAYS, 4, -1)
    return {"mean": np.nanmean, "sum": np.nansum, "max": np.nanmax}[how](x, axis=1)


# six-hourly fields the indices need; tp is the accumulation over each 6 h step (m)
FIELDS = ("skt", "u_850", "u_925", "v_925", "swvl1", "swvl2", "msl", "u_150", "u_200",
          "u_600", "u_700", "swh", "mwp", "tcw", "q_850", "v_850", "2t", "2d", "tp")


def extract(store, member, init, tag=None):
    """Read one member of an O96 store and compute the indices."""
    g = open_store(store, tag)
    lat = np.asarray(g["latitude"][:], np.float64)
    lon180 = ((np.asarray(g["longitude"][:], np.float64) + 180) % 360) - 180
    nt = g["msl"].shape[1]
    if nt < 4 * NDAYS:
        raise SystemExit(f"{store}: {nt} steps, need {4 * NDAYS} (write all 6-792 h)")
    six = {v: np.asarray(g[v][member, : 4 * NDAYS, :], np.float32) for v in FIELDS}
    if not np.isfinite(six["msl"]).all():
        raise SystemExit(f"{store}: NaN in msl - not every step was written")
    out = compute(six, lat, lon180, land_mask(lat, lon180))
    out["meta"] = meta(init, member, os.path.abspath(store), tag)
    return out


def compute(six, lat, lon180, land):
    """Indices from six-hourly arrays (132, n) on any set of points (model O96 cells or
    an ERA5 grid). Step i is lead hour 6(i+1); tp is the accumulation over that step.
    era5_truth_indices.py calls this same function, so model and truth share one
    definition."""
    day = {v: daily(six[v]) for v in ("skt", "u_850", "u_925", "v_925", "swvl1", "swvl2",
                                       "msl", "u_150", "u_200", "u_600", "u_700", "swh",
                                       "mwp", "tcw")}
    day["spd925"] = daily(np.hypot(six["u_925"], six["v_925"]))
    day["qu850"] = daily(six["q_850"] * six["u_850"])
    day["qv850"] = daily(six["q_850"] * six["v_850"])
    day["tmax"] = daily(six["2t"], "max") - 273.15
    day["twmax"] = daily(wet_bulb_c(six["2t"], six["2d"]), "max")
    day["tp"] = daily(six["tp"], "sum") * 1000.0  # m per 6 h -> mm per day

    out = {}
    for name, (fld, box, mask) in BOXES.items():
        sel = in_box(lat, lon180, box)
        if mask == "land":
            sel &= land
        elif mask == "sea":
            sel &= ~land
        out[f"daily_{name}"] = np.nanmean(day[fld][:, sel], axis=1)
    out["daily_dmi"] = out["daily_iod_w_skt"] - out["daily_iod_e_skt"]

    band = (lat >= -20) & (lat <= 20)
    for sec, (lo0, lo1) in ITCZ_SECTORS.items():
        sel = band & (lon180 >= lo0) & (lon180 <= lo1)
        w = np.clip(day["tp"][:, sel], 0, None)
        out[f"daily_itcz_lat_{sec}"] = (w * lat[sel]).sum(1) / np.maximum(w.sum(1), 1e-9)

    dom = in_box(lat, lon180, DOMAIN)
    out["dom_lat"], out["dom_lon"] = lat[dom], lon180[dom]
    out["dom_land"] = land[dom]
    for f in WEEKLY_MAPS:
        wk = []
        for days in WEEKS.values():
            idx = np.array(list(days)) - 1
            x = day[f][idx][:, dom]
            wk.append(np.nansum(x, 0) if f == "tp" else np.nanmean(x, 0))
        out[f"week_{f}"] = np.stack(wk).astype(np.float32)
    return out


def meta(init, member, source, tag=None):
    return np.array(json.dumps(dict(
        init=init, member=int(member), store=source, tag=tag,
        lead_days=f"1..{NDAYS}", weeks={k: [min(v), max(v)] for k, v in WEEKS.items()},
        boxes={k: dict(field=f, box=b, mask=m) for k, (f, b, m) in BOXES.items()},
        itcz_sectors=ITCZ_SECTORS, domain=DOMAIN,
        notes="tmax/twmax: max of the four 6-hourly values; twmax by Stull (2011); "
              "tp daily sums in mm; skt/tmax/twmax deg C except skt (K)")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True, help="O96 icechunk store")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--member", type=int, default=0, help="0-based member index")
    ap.add_argument("--init", required=True, help="YYYYMMDD")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = extract(a.store, a.member, a.init, a.tag)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **res)
    print(f"{a.out}: {sum(1 for k in res if k.startswith('daily_'))} daily series, "
          f"{sum(1 for k in res if k.startswith('week_'))} weekly maps on "
          f"{res['dom_lat'].size} cells, {os.path.getsize(a.out)/1e6:.2f} MB")
