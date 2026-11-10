#!/usr/bin/env bash
# One hindcast member: ERA5 state pkl -> 792 h rollout -> 1.5 deg weekly-window NetCDF.
# Keeps only the 1.5 deg file (~200 MB raw, ~40 MB on zfs) and deletes both stores.
#
#   hc_member.sh <DATE YYYYMMDD_0000> <STATE_DIR with input_state_member_001.pkl> <OUT_DIR>
#
# The O96 store gets a single step (792 h) because run_local_icechunk_v2.py needs one;
# everything used is in the N320 sidecar: msl, tp, 2t over 432-792 h, exactly the window
# 3a reads. ~0.3 GB peak per member.
set -u
DT=$1; STATE=$2; OUT=$3
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
REPO=/home/ezra/ea-aifs/fp16FahamuAIFSv2
WORK=/tank/projects/bc-aifs/hc_work/$DT
export HF_HOME=/tank/projects/hf_cache
mkdir -p "$WORK" "$OUT"
cd "$REPO" || exit 1
say(){ echo "### $(date -u +%H:%M:%S) $DT $*"; }

say "rollout 792 h"
$PY -u run_local_icechunk_v2.py --date $DT --members 1 --n-members 1 --lead-time 792 \
    --input-dir "$STATE" \
    --grid o96 --store "$WORK/icechunk_o96" --write-hours 792-792 \
    --native-store "$WORK/icechunk_n320" --native-vars msl,tp,2t \
    --native-write-hours 432-792 \
    --commit-every 1 --float-size f4 > "$WORK/rollout.log" 2>&1
grep -q "1 written" "$WORK/rollout.log" || { say "ABORT rollout"; tail -20 "$WORK/rollout.log"; exit 1; }

say "regrid to 1.5 deg (3a)"
$PY -u ../shared/aifs_n320_grib_1p5defg_nc_cli.py --date $DT --members 1 --v2 \
    --no-upload --source icechunk --icechunk-store "$WORK/icechunk_n320" \
    --source-grid n320 --icechunk-tag cycle-$DT --output-dir "$OUT" > "$WORK/3a.log" 2>&1
grep -q "Successful: 1/1" "$WORK/3a.log" || { say "ABORT 3a"; tail -20 "$WORK/3a.log"; exit 1; }

du -sh "$WORK"/icechunk_* | sed "s/^/### store size /"
rm -rf "$WORK"/icechunk_o96 "$WORK"/icechunk_n320
say "done -> $OUT"
