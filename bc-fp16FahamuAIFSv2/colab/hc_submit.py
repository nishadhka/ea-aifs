#!/usr/bin/env python3
"""Submit the bias-corrected AI-WQ entry (all five variables) under a separate model name
(RUNBOOK_20261112 section 4). The raw entry stays under AIWQ_MODEL_NAME_FP16.

    tas, mslp, pr   ensemble_quintile_probabilities_<D>_bchc.nc (hc_bc.py apply), weeks 1-2
    TS              ts_days_probs_<D>_bchc.nc (hc_bc15.py apply): the hindcast-calibrated
                    tercile probabilities, periods 1-2; out-of-season basins 1/3 (the same
                    month rule as fp16FahamuAIFSv2/submit_ts_mjo_cli.py)
    MJO             mjo_probs_<D>_bchc.nc: phase probabilities at init +7/14/21/28 days

    hc_submit.py --date 20261112 --model bc-fp16FahamuAIFSv2 --dir RUN/D_0000/aiwq_bc_hc \\
                 --env-file /root/.aiwq.env [--variables tas mslp pr TS MJO] [--dry-run]
"""

import argparse
import datetime as dt
import os
import sys

import numpy as np
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "fp16FahamuAIFSv2"))
import submit_ts_mjo_cli as TM  # noqa: E402  (BASINS, active_basins, build_mjo, load_env)

QVAR = {"tas": "2t_quintiles", "mslp": "msl_quintiles", "pr": "tp_quintiles"}


def gridded(qfile, var, period):
    ds = xr.open_dataset(qfile)
    a = ds[QVAR[var]].isel(time_week=list(ds.week.values).index(f"week{period}")).values
    assert a.shape == (5, 121, 240), a.shape
    assert np.all((a >= 0) & (a <= 1)) and np.allclose(a.sum(0), 1, atol=1e-6)
    return a


def ts(ts_nc, date, period):
    ds = xr.open_dataset(ts_nc)
    p = ds["TS_tercile_probability"].values[period - 1]             # (basin 4, category 3)
    valid = dt.datetime.strptime(date, "%Y%m%d") + dt.timedelta(days=18 if period == 1 else 25)
    act = TM.active_basins(valid.month)
    arr = np.full((3, 4), 1.0 / 3.0)
    for j in act:
        arr[:, j] = p[j]
    lines = [f"    {b:5s} P(below,near,above) = {arr[0,j]:.2f},{arr[1,j]:.2f},{arr[2,j]:.2f}"
             + ("" if j in act else "  [inactive -> 1/3]") for j, b in enumerate(TM.BASINS)]
    return arr, valid.strftime("%Y-%m-%d"), lines


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dir", required=True, help="folder with the *_bchc.nc files")
    ap.add_argument("--env-file", required=True)
    ap.add_argument("--variables", nargs="+", default=["tas", "mslp", "pr", "TS", "MJO"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    TM.load_env(a.env_file)
    team = os.environ.get("AIWQ_TEAM_NAME") or os.environ.get("IWQ_TEAM_NAME")
    pw = (os.environ.get("AIWQ_ECBOX_TOKEN") or os.environ.get("ecbox")
          or os.environ["AIWQ_PASSWORD"])
    print(f"team={team}  model={a.model}  init={a.date}  vars={a.variables}")

    jobs = []
    q = os.path.join(a.dir, f"ensemble_quintile_probabilities_{a.date}_bchc.nc")
    for v in ("tas", "mslp", "pr"):
        if v in a.variables:
            for period in (1, 2):
                arr = gridded(q, v, period)
                print(f"{v} week{period}: shape {arr.shape}, mean P per quintile "
                      f"{np.round(np.nanmean(arr, (1, 2)), 3)}")
                jobs.append((v, str(period), arr))
    if "TS" in a.variables:
        f = os.path.join(a.dir, f"ts_days_probs_{a.date}_bchc.nc")
        for period in (1, 2):
            arr, valid, lines = ts(f, a.date, period)
            print(f"TS period {period} (valid week {valid})\n" + "\n".join(lines))
            jobs.append(("TS", str(period), arr))
    if "MJO" in a.variables:
        arr, vd = TM.build_mjo(os.path.join(a.dir, f"mjo_probs_{a.date}_bchc.nc"))
        print(f"MJO valid {', '.join(vd)}: P(inactive) {np.round(arr[0], 3)}, "
              f"modal {[int(np.argmax(arr[:, j])) for j in range(4)]}")
        jobs.append(("MJO", "1", arr))

    if a.dry_run:
        print(f"\nDRY RUN: {len(jobs)} submissions prepared, nothing sent.")
        sys.exit(0)
    from AI_WQ_package import forecast_submission as FS
    ok = 0
    for var, period, arr in jobs:
        try:
            da = FS.AI_WQ_create_empty_dataarray(var, a.date, period, team, a.model, pw)
            da.values = arr if var in QVAR else da.values * 0 + arr
            FS.AI_WQ_forecast_submission(da, var, a.date, period, team, a.model, pw)
            print(f"  submitted {var} period {period}")
            ok += 1
        except Exception as e:
            print(f"  FAILED {var} period {period}: {e}")
    print(f"\n{ok}/{len(jobs)} submitted")
    sys.exit(0 if ok == len(jobs) else 1)
