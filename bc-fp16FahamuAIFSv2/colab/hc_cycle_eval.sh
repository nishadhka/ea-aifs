#!/usr/bin/env bash
# Phase C of one weekly cycle on a Colab CPU runtime (RUNBOOK_20261119): precompute TS truth
# while the GPUs run, wait for all hindcast members, then bundle, evaluate, apply and dry-run
# the submission. Products saved to source.coop (hc-products/<MMDD>/bc_out_<TAG>/).
#
#   bash hc_cycle_eval.sh MMDD YYYYMMDD [Y0 Y1 TAG]     e.g. bash hc_cycle_eval.sh 1126 20261126
#   needs: /root/.aiwq.env, /content/aiwq_mjo/MJO_DAILY_*.nc, /content/ibtracs, /content/ts_terciles,
#          /content/bc/era5_weekly (+ /content/bc/truth_idx) -- hc_cycle_phaseA.sh leaves these
#   log: "### ... C <TAG> DONE"; submit afterwards with colab/hc_submit.py + submit_ts_mjo_cli.py
set -u
MD=$1; RT=$2; Y0=${3:-2006}; Y1=${4:-2025}; TAG=${5:-y20}
N=$(( (Y1 - Y0 + 1) * 11 ))
BC=/content/ea-aifs/bc-fp16FahamuAIFSv2
say(){ echo "### $(date -u +%T) $*"; }
SAVE() { sub=$1; shift; python3 -c "import sys,os; sys.path.insert(0,'$BC/colab'); import sc_io
for p in sys.argv[2:]: sc_io.put(p, sc_io.BASE+'/hc-products/$MD/'+sys.argv[1]+'/'+os.path.basename(p))" "$sub" "$@" && say "saved $sub ($#)"; }
cd $BC
say "TS truth $Y0-$Y1 (IBTrACS) while members run"
BC_CACHE=/content/bc HC_IBTRACS=/content/ibtracs/IBTrACS.ALL.v04r01.nc python3 -c "
import hc_bc15 as H
for y in range($Y0, $Y1 + 1): H.ts_truth(f'{y}$MD')" && say "TS truth done"
cd colab
until python3 -c "
import sys, re; import sc_io
d=[k for k,_ in sc_io.ls(sc_io.BASE+'/hc-products/$MD/') if k.endswith('/DONE')]
n=sum($Y0<=int(re.search(r'/$MD/(\d{4})/',k).group(1))<=$Y1 for k in d)
sys.exit(0 if n>=$N else 1)" 2>/dev/null; do sleep 120; done
say "hindcast $Y0-$Y1 complete ($N members); real-time must be complete too"
until python3 -c "
import sys; import sc_io
d=[k for k,_ in sc_io.ls(sc_io.BASE+'/hc-products/$MD/${RT:0:4}/') if k.endswith('/DONE')]
sys.exit(0 if len(d)>=51 else 1)" 2>/dev/null; do sleep 120; done
OUTB=/content/bundle${MD}_$TAG
say "bundle $Y0-$Y1 + real-time"
BC_CACHE=/content/bc python3 -u hc_bundle.py --mmdd $MD --years $Y0-$Y1 --rt-year ${RT:0:4} --out $OUTB --no-upload || { say "FAILED bundle"; exit 1; }
ln -sfn /content/bc/era5_weekly $OUTB/era5_weekly
[ -d /content/bc/truth_idx ] && ln -sfn /content/bc/truth_idx $OUTB/truth_idx
[ -d /content/bc/truth_ts ] && ln -sfn /content/bc/truth_ts $OUTB/truth_ts
cd $BC
export BC_CACHE=$OUTB BC_RUN=/content/c${MD}_$TAG BC_ENV=/root/.aiwq.env HC_LSM_1P5=/content/aux/land_sea_mask_1pt5DEG.nc HC_IBTRACS=/content/ibtracs/IBTrACS.ALL.v04r01.nc HC_MJO_OBS_DIR=/content/aiwq_mjo
mkdir -p $BC_RUN
say "C1 tas/mslp/pr evaluate";  python3 -u hc_bc.py evaluate $MD $Y0 $Y1 || say "FAILED hc_bc evaluate"
say "C2 tas/mslp/pr apply";     python3 -u hc_bc.py apply --date $RT || say "FAILED hc_bc apply"
say "C1 TS/MJO/indices evaluate"; python3 -u hc_bc15.py evaluate $MD $Y0 $Y1 || say "FAILED hc_bc15 evaluate"
say "C2 TS/MJO/indices apply";  python3 -u hc_bc15.py apply --date $RT || say "FAILED hc_bc15 apply"
OUT=$BC_RUN/${RT}_0000/aiwq_bc_hc
python3 -c "
import glob, xarray as xr
ms = sorted(glob.glob('$OUTB/rt15/$RT/m*/ts_days.nc'))
ds = xr.concat([xr.open_dataset(f)[['storm_days']] for f in ms], 'member')
ds.to_netcdf('$OUT/ts_days_raw_${RT}_colab51.nc'); print('raw TS', dict(ds.sizes))"
SAVE bc_out_$TAG $OUT/* $OUTB/hc_choice_$MD.json $OUTB/hc15_eval_$MD.json $OUTB/hc15_eval_$MD.md
say "dry-run"
python3 -u colab/hc_submit.py --date $RT --model bcfp16FahamuAIFSv2 --dir $OUT --env-file /root/.aiwq.env --variables tas mslp pr MJO --dry-run
cd $BC/../fp16FahamuAIFSv2 && AIWQ_TEAM_NAME=$(sed -n "s/^AIWQ_TEAM_NAME=//p;s/^IWQ_TEAM_NAME=//p" /root/.aiwq.env | head -1) AIWQ_MODEL_NAME_V2=bcfp16FahamuAIFSv2 python3 -u submit_ts_mjo_cli.py --date $RT --variables TS --ts-nc $OUT/ts_days_raw_${RT}_colab51.nc --tercile-dir /content/ts_terciles --env-file /root/.aiwq.env --dry-run
say "C $TAG DONE"
