"""Sanity check for the control input pkl built from HRES oper/wave 0h.

The control is unperturbed, so for every field and both time steps it should sit
inside the perturbed ensemble: its global-mean difference from the pf mean should
be small compared with the pf spread. A unit or level mix-up (e.g. gh vs z, the
wave land mask, a wrong step) shows up as a difference of many spreads.

Usage: check_control_pkl.py <input_states dir> <control member number>
Exit 1 if any field is off by more than LIMIT pf standard deviations.
"""
import os
import pickle
import sys

import numpy as np

LIMIT = 5.0
N_PF = 10  # perturbed members to compare against; enough for a mean and spread

d, ctl = sys.argv[1], int(sys.argv[2])
load = lambda m: pickle.load(open(os.path.join(d, f"input_state_member_{m:03d}.pkl"), "rb"))["fields"]
c = load(ctl)
pf = [load(m) for m in range(1, N_PF + 1)]

bad = []
for k in sorted(c):
    cg = np.nanmean(c[k], axis=1)                          # (2,) global means per time step
    pg = np.array([np.nanmean(p[k], axis=1) for p in pf])  # (N_PF, 2)
    mu, sd = pg.mean(0), pg.std(0)
    # rms difference over the grid, scaled by the pf rms spread
    rms_c = np.sqrt(np.nanmean((c[k] - np.mean([p[k] for p in pf], 0)) ** 2))
    rms_p = np.sqrt(np.nanmean(np.var([p[k] for p in pf], 0)))
    z = np.abs(cg - mu) / np.where(sd > 0, sd, np.nan)
    r = rms_c / rms_p if rms_p > 0 else 0.0
    nan_c, nan_p = np.isnan(c[k]).mean(), np.isnan(pf[0][k]).mean()
    # A global-mean offset only counts if it is also physically non-trivial (>0.1% of
    # the field mean): for sd the pf global means agree to 3e-5, so the control's
    # 0.014% lower snow (20261105) scored 5.6 "spreads" while being 0.14 mm w.e.
    rel = np.nanmax(np.abs(cg - mu) / np.maximum(np.abs(mu), 1e-12))
    with np.errstate(all="ignore"):
        zmax = np.nanmax(z) if np.isfinite(z).any() else 0.0
    flag = (zmax > LIMIT and rel > 1e-3) or r > LIMIT or abs(nan_c - nan_p) > 0.01
    if flag:
        bad.append(k)
    if flag or k in ("2t", "msl", "tp", "z_500", "t_850", "swh", "sd", "q_850"):
        print(f"{k:10s} mean z={zmax:6.2f}  rms ratio={r:5.2f}  "
              f"nan ctl/pf={nan_c:.3f}/{nan_p:.3f}{'  <-- BAD' if flag else ''}")

print(f"{len(c)} fields, {len(bad)} flagged: {bad}")
sys.exit(1 if bad else 0)
