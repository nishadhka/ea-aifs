"""Tropical-storm-days forecast + tercile probabilities from the Icechunk store.

AI-WQ defines the *observed* diagnostic from IBTrACS
(`training_data.html#tropical-storm-days-data-processing`):
  - a day is a storm day in a basin if >=1 three-hourly record has usa_wind >= 17 m/s
  - multiple records of the same storm on one day count once
  - daily flags summed over a 7-day window -> storm days per basin per week
and evaluates a *forecast* as tercile probabilities per basin, scored by RPSS
(`forecast_evaluation.html#tropical-storm-days-ts`).

IBTrACS is observation-only, so a forecast must detect storms in the model
fields. This module applies the same day/week logic to a detection built from
what the store carries:

    10 m wind speed = sqrt(10u^2 + 10v^2) >= --wind-threshold  (17 m/s)
      AND closed low:  msl   <= basin median msl - --msl-drop  (5 hPa)
      AND warm core:   t_300 >= same-latitude median + --warm-core (0.5 K)
      AND cyclonic:    zeta850 >= --vorticity                   (1e-4 1/s)

all evaluated at the store's 6-hourly steps (AI-WQ uses 3-hourly observations;
6-hourly can only miss short-lived qualifying periods, never add them).

`vo` is not an AIFS-ENS-2.0 output, so the vorticity is derived from `u_850`/
`v_850` directly on the reduced Gaussian grid (see `grid_ops.py`) - ~1.5 s per
member, versus ~26 min if every field were regridded to differentiate it.

*** DETECTION CAVEAT ***
Still a point-wise detector, **not a tracker**: no track continuity, no storm
identity, and N320 under-resolves TC peak winds. Measured on cycle 20260730
(valid mid/late August), each filter suppresses the austral-winter basins while
leaving the genuinely active North-West Pacific alone - the right physical
signature - but SWIO does not fall to the ~0 that August climatology implies,
so the thresholds are **not yet calibrated**. Treat the absolute counts as
uncalibrated: the tercile probabilities are only meaningful against a
climatology built with this *same* detector (`--tercile-clim`).
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np

import store_io as sio
from grid_ops import ReducedGaussianGrid, cyclonic

WIND_THRESHOLD = 17.0     # m/s, AI-WQ / IBTrACS tropical-storm strength
MSL_DROP = 500.0          # Pa below the basin daily median = "closed low"
WARM_CORE_K = 0.5         # K above the same-latitude reference at 300 hPa
VORTICITY = 1.0e-4        # 1/s, cyclonic 850 hPa relative vorticity at the core
LAT_BIN = 2.0             # deg, for the latitude-banded temperature reference


def _lat_banded_reference(field, lat_sel, bin_deg=LAT_BIN):
    """Per-step median of `field` within each latitude band, broadcast back.

    The 0-40 deg basin boxes span tropics to midlatitudes, where the upper-level
    temperature gradient is large; comparing a candidate to the whole-basin
    median would call every midlatitude low 'cold' or every tropical system
    'warm'. Referencing within +-1 deg of the candidate's own latitude removes
    that gradient so the anomaly reflects the core, not the latitude.
    """
    bins = np.floor(lat_sel / bin_deg).astype(int)
    ref = np.empty_like(field)
    for b in np.unique(bins):
        cols = bins == b
        ref[:, cols] = np.nanmedian(field[:, cols], axis=1, keepdims=True)
    return ref


def storm_days_for_member(g, m, steps, times, masks, wind_thr, msl_drop,
                          warm_core_k=WARM_CORE_K, lat=None,
                          grid=None, vort_thr=VORTICITY):
    """{basin: {date: bool}} for one member."""
    u = sio.read_member_window(g, "10u", m, steps)
    v = sio.read_member_window(g, "10v", m, steps)
    p = sio.read_member_window(g, "msl", m, steps)
    spd = np.sqrt(u * u + v * v)
    del u, v
    t300 = sio.read_member_window(g, "t_300", m, steps) if warm_core_k else None

    czeta = None
    if vort_thr and grid is not None:
        u850 = sio.read_member_window(g, "u_850", m, steps)
        v850 = sio.read_member_window(g, "v_850", m, steps)
        # `vo` is not an output variable; derive it on the reduced Gaussian grid
        czeta = cyclonic(grid.relative_vorticity(u850, v850), lat)
        del u850, v850

    out = {}
    for name, mask in masks.items():
        s_b = spd[:, mask]
        p_b = p[:, mask]
        # per-step basin reference pressure -> "closed low" is relative, so the
        # test does not depend on the basin's mean seasonal pressure level
        ref = np.nanmedian(p_b, axis=1, keepdims=True)
        hit = (s_b >= wind_thr) & (p_b <= ref - msl_drop)
        if warm_core_k:
            t_b = t300[:, mask]
            t_ref = _lat_banded_reference(t_b, lat[mask])
            hit &= (t_b - t_ref) >= warm_core_k
        if czeta is not None:
            # sign-corrected, so the same threshold means "cyclonic" in both
            # hemispheres; NaN at the polar rows never passes
            hit &= np.nan_to_num(czeta[:, mask], nan=-1.0) >= vort_thr
        step_hit = hit.any(axis=1)
        by_day = {}
        for (day, pos) in sio.group_steps_by_day(times):
            by_day[day] = bool(step_hit[pos].any())
        out[name] = by_day
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--tag")
    ap.add_argument("--init", required=True, help="cycle init YYYYMMDD")
    ap.add_argument("--members", type=int, default=None)
    ap.add_argument("--wind-threshold", type=float, default=WIND_THRESHOLD)
    ap.add_argument("--msl-drop", type=float, default=MSL_DROP,
                    help="Pa below basin median msl required (default 500 = 5 hPa)")
    ap.add_argument("--vorticity", type=float, default=VORTICITY,
                    help="cyclonic 850 hPa relative vorticity required at the "
                         "core [1/s]; 0 disables (default 1e-4). Derived on the "
                         "reduced Gaussian grid - `vo` is not a model output")
    ap.add_argument("--warm-core", type=float, default=WARM_CORE_K,
                    help="K above the same-latitude 300 hPa reference; 0 disables "
                         "the warm-core test (default 0.5)")
    ap.add_argument("--tercile-clim",
                    help=".npz with `bounds` (basin, 2) lower/upper tercile "
                         "boundaries built with THIS detector; without it the "
                         "ensemble's own terciles are used and flagged")
    ap.add_argument("--out", default="ts_days_probs.nc")
    args = ap.parse_args()

    init = dt.datetime.strptime(args.init, "%Y%m%d")
    g = sio.open_store(args.store, args.tag)
    lat, lon180, _ = sio.coords(g)
    steps = sio.written_steps(g)
    times = sio.valid_times(g, init, steps)
    nmem = args.members or sio.n_members(g)
    masks = sio.basin_masks(lat, lon180)
    grid = ReducedGaussianGrid(lat, np.asarray(g["longitude"][:])) if args.vorticity else None

    days = sio.group_steps_by_day(times)
    weeks = sio.seven_day_windows(days)
    basins = list(sio.BASINS)

    print(f"TS days | init {init:%Y-%m-%d} | members {nmem} | "
          f"{times[0]:%Y-%m-%d} .. {times[-1]:%Y-%m-%d} | "
          f"wind>={args.wind_threshold} m/s, msl<=median-{args.msl_drop:.0f} Pa"
          + (f", warm core>=+{args.warm_core} K @300hPa" if args.warm_core else ", NO warm-core")
          + (f", cyclonic zeta850>={args.vorticity:.1e} 1/s" if args.vorticity else ", NO vorticity test"))
    print(f"  {len(days)} days -> {len(weeks)} full 7-day windows")
    for b in basins:
        print(f"    basin {b}: {masks[b].sum()} N320 points")

    counts = np.zeros((nmem, len(weeks), len(basins)), dtype=int)
    for m in range(nmem):
        sd = storm_days_for_member(g, m, steps, times, masks,
                                   args.wind_threshold, args.msl_drop,
                                   args.warm_core, lat, grid, args.vorticity)
        for wi, wk in enumerate(weeks):
            for bi, b in enumerate(basins):
                counts[m, wi, bi] = sum(1 for d in wk if sd[b].get(d, False))
        if (m + 1) % 10 == 0 or m == nmem - 1:
            print(f"    member {m+1}/{nmem} done")

    # tercile probabilities
    if args.tercile_clim:
        bounds = np.load(args.tercile_clim)["bounds"]      # (basin, 2)
        clim_src = args.tercile_clim
    else:
        bounds = np.stack([
            np.nanpercentile(counts[:, :, bi], [100 / 3, 200 / 3])
            for bi in range(len(basins))
        ])
        clim_src = "ENSEMBLE_SELF (uncalibrated - see caveat)"
        print("\n  !! no --tercile-clim: using the ensemble's own terciles, so the "
              "probabilities are self-referential and NOT submission-grade.")

    probs = np.zeros((len(weeks), len(basins), 3))
    for wi in range(len(weeks)):
        for bi in range(len(basins)):
            c = counts[:, wi, bi]
            lo, hi = bounds[bi]
            probs[wi, bi, 0] = np.mean(c <= lo)
            probs[wi, bi, 1] = np.mean((c > lo) & (c <= hi))
            probs[wi, bi, 2] = np.mean(c > hi)

    import xarray as xr
    ds = xr.Dataset(
        {
            "TS_tercile_probability": (("week", "basin", "category"), probs),
            "storm_days": (("member", "week", "basin"), counts),
            "tercile_bounds": (("basin", "bound"), np.asarray(bounds, float)),
        },
        coords={
            "week": [str(w[0]) for w in weeks],
            "basin": basins,
            "category": ["below", "near", "above"],
            "member": np.arange(nmem),
            "bound": ["lower", "upper"],
        },
        attrs={
            "detector": ("10m wind + relative msl minimum + 300 hPa warm core + "
                         "cyclonic 850 hPa vorticity (no track continuity)"),
            "warm_core_k": args.warm_core,
            "vorticity_threshold_s-1": args.vorticity,
            "vorticity_source": "derived from u_850/v_850 on the N320 reduced Gaussian grid",
            "wind_threshold_ms": args.wind_threshold,
            "msl_drop_pa": args.msl_drop,
            "step_hours": 6,
            "ibtracs_native_step_hours": 3,
            "tercile_source": clim_src,
            "source_store": str(args.store),
            "cycle_init": args.init,
        },
    )
    ds.to_netcdf(args.out)
    print(f"\n  wrote {args.out}")
    for wi, wk in enumerate(weeks):
        print(f"  week from {wk[0]}:")
        for bi, b in enumerate(basins):
            c = counts[:, wi, bi]
            print(f"    {b:<5} mean {c.mean():4.1f} d  range {c.min()}-{c.max()}  "
                  f"P(below/near/above) = {probs[wi,bi,0]:.2f}/"
                  f"{probs[wi,bi,1]:.2f}/{probs[wi,bi,2]:.2f}")


if __name__ == "__main__":
    main()
