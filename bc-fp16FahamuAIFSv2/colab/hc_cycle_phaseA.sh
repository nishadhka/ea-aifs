#!/usr/bin/env bash
# Phase A of one weekly cycle on a Colab CPU runtime (RUNBOOK_20261119 routine):
#   51 real-time pkls (IBTrACS-free, open data)   ||   20 control pkls (ARCO + MARS, one request at a time)
#   200 perturbed pkls, built for each year as soon as its control is on source.coop (CDS one at a time)
#   truth: ERA5 weekly (D+18, D+25 for 2006-2025), IBTrACS, official TS terciles for both weeks,
#          index truth after the EDA requests -- every product saved to source.coop when made
# Resumable: everything already on source.coop is skipped, so a relaunch on a new runtime
# (Colab ends CPU runtimes without warning) continues where the last one stopped.
#
#   bash hc_cycle_phaseA.sh MMDD YYYYMMDD          e.g. bash hc_cycle_phaseA.sh 1126 20261126
#   log: /content/cycle_<MMDD>.log ("### ... PHASE A DONE", "### ... CYCLE A DONE")
set -u
MD=$1; RT=$2
BC=/content/ea-aifs/bc-fp16FahamuAIFSv2
export BC_CACHE=/content/bc CDSAPI_RC=/root/.cdsapirc HC_LSM_1P5=/content/aux/land_sea_mask_1pt5DEG.nc
say(){ echo "### $(date -u +%T) $*"; }
SAVE() { sub=$1; shift; python3 -c "import sys,os; sys.path.insert(0,'$BC/colab'); import sc_io
for p in sys.argv[2:]: sc_io.put(p, sc_io.BASE+'/hc-products/$MD/'+sys.argv[1]+'/'+os.path.basename(p))" "$sub" "$@" && say "saved $sub ($#)"; }
WEEKS=$(python3 -c "
import datetime as dt
print(' '.join((dt.date(y,int('$MD'[:2]),int('$MD'[2:]))+dt.timedelta(days=k)).strftime('%Y%m%d') for y in range(2025,2005,-1) for k in (18,25)))")
TSW=$(python3 -c "
import datetime as dt
d=dt.date(int('$RT'[:4]),int('$RT'[4:6]),int('$RT'[6:])); print(' '.join((d+dt.timedelta(days=k)).strftime('%Y%m%d') for k in (18,25)))")
mkdir -p /content/bc /content/states /content/ibtracs /content/ts_terciles
cd $BC/colab
say "start $MD: real-time $RT || controls 2025-2006 || perturbed as controls land"
python3 -u hc_phaseA.py --realtime $RT --jobs 4 > /content/rt_$MD.log 2>&1 &
P_RT=$!
( for pass in $(seq 1 120); do
    python3 -u hc_phaseA.py --mmdd $MD --years 2025-2006 --jobs 2 --perturbed 2-11 --inflate 1.0 >> /content/pert_$MD.log 2>&1 \
      && { say "PERTURBED DONE"; break; }
    sleep 600
  done ) &
P_PERT=$!
python3 -u hc_phaseA.py --mmdd $MD --years 2025-2006 --jobs 3 > /content/ctl_$MD.log 2>&1
python3 -u hc_phaseA.py --mmdd $MD --years 2025-2006 --jobs 2 >> /content/ctl_$MD.log 2>&1   # retry failed years
say "controls done: $(grep -c 'built\|on source.coop' /content/ctl_$MD.log) years ok"
wait $P_RT; say "real-time: $(tail -1 /content/rt_$MD.log)"
cd $BC
python3 -u era5_weekly.py build $WEEKS > /content/weekly_$MD.log 2>&1 && say "ERA5 weekly built" && SAVE truth/era5_weekly /content/bc/era5_weekly/*.nc
curl -sS -o /content/ibtracs/IBTrACS.ALL.v04r01.nc https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/netcdf/IBTrACS.ALL.v04r01.nc && say "ibtracs ok"
python3 - $TSW <<'EOF'
import contextlib, io, glob, os, sys, xarray as xr
sys.path.insert(0, "/content/ea-aifs/bc-fp16FahamuAIFSv2/colab"); import sc_io
from AI_WQ_package import TS_processing as T
ib = xr.open_dataset("/content/ibtracs/IBTrACS.ALL.v04r01.nc")
for d in sys.argv[1:]:
    f = f"/content/ts_terciles/TS_20yrCLIM_WEEKLYTSDAYS_terciles_{d}.nc"
    if not os.path.exists(f):
        with contextlib.redirect_stdout(io.StringIO()):
            T.compute_20yr_TStercile_climatology(d, ib, "/content/ts_terciles")
    print("TS terciles", d, flush=True)
EOF
SAVE ts_terciles /content/ts_terciles/*.nc
wait $P_PERT; say "PHASE A DONE"
say "index truth (CDS, one request at a time)"
python3 -u era5_truth_indices.py $MD 2006 2025 > /content/idx_$MD.log 2>&1; SAVE truth/truth_idx /content/bc/truth_idx/*.npz
say "CYCLE A DONE"
