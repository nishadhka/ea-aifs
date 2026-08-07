"""MJO RMM-style index + 9-category phase probabilities from the Icechunk store.

Follows AI-WQ `training_data.html#mjo-data-processing` as far as this store allows:

  1. cosine-weighted meridional mean over +-15 deg  -> 144 longitude points (2.5 deg)
  2. daily means from the 6-hourly steps
  3. day-of-year climatology removed                        [needs --clim]
  4. preceding 120-day rolling mean removed                 [needs --lowfreq]
  5. Wheeler & Hendon (2004) normalisation factors
  6. projection onto the observed combined EOFs             [needs --eofs]
  7. RMM1/RMM2 -> amplitude, phase (0-8; 0 when amplitude < 1)

then converts the 50 members into probabilities over the 9 AI-WQ categories
(`forecast_evaluation.html#mjo-phase-probability-forecasts`).

*** OLR LIMITATION - READ THIS ***
The canonical RMM combines OLR, U850 and U200. This store carries **no `ttr`**
(top net longwave) - only the surface fluxes `ssrd`/`strd` - so OLR cannot be
derived from it. Two supported modes:

  --olr-source none  (default) : wind-only projection using the U850/U200 rows
      of the EOFs. This is NOT the official RMM; it is a wind-only proxy and is
      labelled as such in the output (`index_kind = "wind_only_proxy"`).
  --olr-source FILE            : supply OLR on the same daily/longitude grid
      (e.g. from ERA5 `ttr`, converted with olr = -ttr/3600) for the full
      3-field projection (`index_kind = "rmm3"`).

Add `ttr` to the inference output variable list to close this properly.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

import numpy as np

import store_io as sio

# Wheeler & Hendon (2004) normalisation factors (1979-2001 observed)
NORM = {"olr": 15.1, "u850": 1.81, "u200": 4.81}
N_LON_BINS = 144           # 2.5 deg
LAT_BAND = 15.0


def meridional_band(values, lat, lon180, lat_band=LAT_BAND, nbins=N_LON_BINS):
    """(nsteps, npts) -> (nsteps, 144) cosine-weighted mean over +-lat_band."""
    m = np.abs(lat) <= lat_band
    v = values[:, m]
    la, lo = lat[m], lon180[m]
    w = np.cos(np.deg2rad(la))
    # bin edges on -180..180
    idx = np.floor((lo + 180.0) / (360.0 / nbins)).astype(int)
    idx = np.clip(idx, 0, nbins - 1)
    out = np.full((v.shape[0], nbins), np.nan, dtype=np.float64)
    wsum = np.bincount(idx, weights=w, minlength=nbins)
    for k in range(v.shape[0]):
        num = np.bincount(idx, weights=w * np.nan_to_num(v[k]), minlength=nbins)
        out[k] = np.where(wsum > 0, num / np.maximum(wsum, 1e-12), np.nan)
    return out


def daily_mean(band, times):
    """(nsteps, nlon) 6-hourly -> (ndays, nlon) daily means + the dates."""
    days = sio.group_steps_by_day(times)
    out = np.stack([np.nanmean(band[pos], axis=0) for _, pos in days])
    return out, [d for d, _ in days]


def phase_from_pcs(rmm1, rmm2):
    """AI-WQ 9 categories: phase 1-8 by octant, 0 when amplitude < 1."""
    amp = np.sqrt(rmm1 ** 2 + rmm2 ** 2)
    ang = np.arctan2(rmm2, rmm1)                 # -pi..pi
    oct_ = np.floor((ang + np.pi) / (np.pi / 4.0)).astype(int) % 8
    phase = oct_ + 1
    phase = np.where(amp < 1.0, 0, phase)
    return phase.astype(int), amp


def load_eofs(path):
    """Combined EOFs. Expected .npz with `eof1`,`eof2` of shape (nvar*144,)
    ordered [olr, u850, u200], or (3, 144) blocks; plus optional `sd1`,`sd2`
    (observed RMM standard deviations)."""
    z = np.load(path)
    e1, e2 = np.asarray(z["eof1"], float), np.asarray(z["eof2"], float)
    sd1 = float(z["sd1"]) if "sd1" in z else 1.0
    sd2 = float(z["sd2"]) if "sd2" in z else 1.0
    if e1.ndim == 2:
        e1, e2 = e1.reshape(-1), e2.reshape(-1)
    return e1, e2, sd1, sd2


def project(fields, eof1, eof2, sd1, sd2, use_olr):
    """fields: dict name -> (ndays, 144) normalised anomalies. Returns RMM1, RMM2."""
    order = ["olr", "u850", "u200"] if use_olr else ["u850", "u200"]
    x = np.concatenate([fields[k] for k in order], axis=1)      # (ndays, nvar*144)
    if not use_olr and eof1.size == 3 * N_LON_BINS:
        # drop the OLR block from the EOFs to match the wind-only state vector
        eof1 = eof1[N_LON_BINS:]
        eof2 = eof2[N_LON_BINS:]
    if x.shape[1] != eof1.size:
        raise ValueError(f"state vector {x.shape[1]} != EOF length {eof1.size}")
    return (x @ eof1) / sd1, (x @ eof2) / sd2


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--tag")
    ap.add_argument("--init", required=True, help="cycle init YYYYMMDD")
    ap.add_argument("--members", type=int, default=None, help="limit members (debug)")
    ap.add_argument("--clim", help=".npz DOY climatology on the (144,) band grid")
    ap.add_argument("--lowfreq", help=".npz preceding-120-day mean on the band grid")
    ap.add_argument("--eofs", help=".npz combined EOFs (see load_eofs)")
    ap.add_argument("--olr-source", default="none",
                    help="'none' (wind-only proxy) or a .npz with daily OLR band data")
    ap.add_argument("--out", default="mjo_probs.nc")
    ap.add_argument("--dump-bands", help="write the normalised band anomalies here (.npz)")
    args = ap.parse_args()

    init = dt.datetime.strptime(args.init, "%Y%m%d")
    g = sio.open_store(args.store, args.tag)
    lat, lon180, _ = sio.coords(g)
    steps = sio.written_steps(g)
    times = sio.valid_times(g, init, steps)
    nmem = args.members or sio.n_members(g)
    use_olr = args.olr_source != "none"
    kind = "rmm3" if use_olr else "wind_only_proxy"

    print(f"MJO index | init {init:%Y-%m-%d} | members {nmem} | steps {len(steps)} "
          f"({times[0]:%Y-%m-%d} .. {times[-1]:%Y-%m-%d}) | kind={kind}")
    if not use_olr:
        print("  !! no OLR: 'ttr' is absent from this store. Producing a WIND-ONLY "
              "proxy, not the official RMM. See module docstring.")

    clim = np.load(args.clim) if args.clim else None
    lowf = np.load(args.lowfreq)["mean"] if args.lowfreq else None
    olr_src = np.load(args.olr_source) if use_olr else None

    all_phase, all_amp, all_pcs, bands_dump = [], [], [], {}
    for m in range(nmem):
        fields = {}
        for var, key in (("u_850", "u850"), ("u_200", "u200")):
            raw = sio.read_member_window(g, var, m, steps)
            band = meridional_band(raw, lat, lon180)
            daily, dates = daily_mean(band, times)
            if clim is not None:                      # step 3: DOY climatology
                doy = np.array([d.timetuple().tm_yday for d in dates])
                daily = daily - clim[key][doy - 1]
            if lowf is not None:                      # step 4: 120-day low-freq
                daily = daily - lowf
            fields[key] = daily / NORM[key]           # step 5
        if use_olr:
            fields["olr"] = np.asarray(olr_src["olr"], float) / NORM["olr"]

        if m == 0:
            bands_dump = {k: v for k, v in fields.items()}
            bands_dump["dates"] = np.array([str(d) for d in dates])

        if args.eofs:
            e1, e2, sd1, sd2 = load_eofs(args.eofs)
            r1, r2 = project(fields, e1, e2, sd1, sd2, use_olr)
            ph, amp = phase_from_pcs(r1, r2)
            all_pcs.append(np.stack([r1, r2]))
            all_phase.append(ph)
            all_amp.append(amp)

    if args.dump_bands:
        np.savez(args.dump_bands, **bands_dump)
        print(f"  wrote normalised band anomalies -> {args.dump_bands}")

    if not args.eofs:
        print("\n  No --eofs given: stopped after the normalised band anomalies "
              "(steps 1-5). Supply the observed combined EOFs to get RMM/phases.")
        return

    phase = np.stack(all_phase)                  # (member, day)
    amp = np.stack(all_amp)
    pcs = np.stack(all_pcs)                      # (member, 2, day)

    # 9-category probabilities, per 7-day window (AI-WQ weeks)
    days = sio.group_steps_by_day(times)
    weeks = sio.seven_day_windows(days)
    date_list = [d for d, _ in days]
    probs = np.zeros((len(weeks), 9))
    for wi, wk in enumerate(weeks):
        cols = [date_list.index(d) for d in wk]
        # a member's weekly phase = its modal phase over the window
        for mi in range(phase.shape[0]):
            vals, cnt = np.unique(phase[mi, cols], return_counts=True)
            probs[wi, vals[np.argmax(cnt)]] += 1
        probs[wi] /= phase.shape[0]

    import xarray as xr
    ds = xr.Dataset(
        {
            "MJO_phase_probability": (("week", "MJO_phase"), probs),
            "rmm1": (("member", "day"), pcs[:, 0, :]),
            "rmm2": (("member", "day"), pcs[:, 1, :]),
            "amplitude": (("member", "day"), amp),
            "phase": (("member", "day"), phase),
        },
        coords={
            "week": [str(w[0]) for w in weeks],
            "MJO_phase": np.arange(9),
            "member": np.arange(phase.shape[0]),
            "day": [str(d) for d in date_list],
        },
        attrs={
            "index_kind": kind,
            "olr_available": str(use_olr),
            "note": ("wind-only proxy: store has no ttr/OLR" if not use_olr
                     else "full 3-field RMM"),
            "doy_climatology_removed": str(clim is not None),
            "lowfreq_120day_removed": str(lowf is not None),
            "norm_factors": json.dumps(NORM),
            "source_store": str(args.store),
            "cycle_init": args.init,
        },
    )
    ds.to_netcdf(args.out)
    print(f"\n  wrote {args.out}")
    for wi, wk in enumerate(weeks):
        top = np.argsort(probs[wi])[::-1][:3]
        print(f"  week from {wk[0]}: " +
              ", ".join(f"phase{p}={probs[wi, p]:.2f}" for p in top))


if __name__ == "__main__":
    main()
