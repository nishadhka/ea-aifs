#!/usr/bin/env bash
# One hindcast member: ERA5 state pkl -> 792 h rollout -> all 15 products (README 0.2)
# -> delete both stores. Keeps ~45 MB per member.
#
#   hc_member.sh <DATE YYYYMMDD_0000> <STATE_DIR with input_state_member_001.pkl> <OUT_DIR>
#
# Stores are the real-time chain's tier B, so hindcast and real time are processed alike:
#   O96 main store    all 132 steps (6-792 h), all fields     -> MJO (5), indices 6-15
#   N320 sidecar      432-792 h, the 10 downstream fields      -> tas/mslp/pr (1-3), TS days (4)
# Peak ~4.3 GB for one member, deleted at the end.
#
# Products in OUT_DIR:
#   aifs_ensemble_forecast_1p5deg_member001.nc   3a, weeks 3-4 tas/mslp/pr      (1-3)
#   ts_days.nc                                   storm days per basin and week   (4)
#   mjo_bands.npz, mjo_probs.nc                  daily MJO bands, probabilities  (5)
#   indices.npz                                  hc_extract.py                   (6-15)
#   DONE                                         written last; hc_date.sh checks it
set -u
DT=$1; STATE=$2; OUT=$3
D=${DT%_*}
# Defaults are the GPU box; Colab (colab/hc_gpu_worker.py) overrides them by environment.
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${HC_PY:-/tank/projects/micromamba/envs/aifs-gpu/bin/python}
REPO=${HC_REPO:-/home/ezra/ea-aifs/fp16FahamuAIFSv2}
WORK=${HC_WORK:-/tank/projects/bc-aifs/hc_work}/$DT
VPM=${HC_VPM:-/tank/projects/era5_vpm_clim}
export HF_HOME=${HF_HOME:-/tank/projects/hf_cache}
mkdir -p "$WORK" "$OUT"
cd "$REPO" || exit 1
say(){ echo "### $(date -u +%H:%M:%S) $DT $*"; }
fail(){ say "ABORT $1"; tail -15 "$WORK/$2"; exit 1; }

say "rollout 792 h (O96 all steps + N320 sidecar 432-792 h)"
$PY -u run_local_icechunk_v2.py --date $DT --members 1 --n-members 1 --lead-time 792 \
    --input-dir "$STATE" \
    --grid o96 --store "$WORK/icechunk_o96" \
    --native-store "$WORK/icechunk_n320" \
    --native-vars msl,tp,2t,10u,10v,t_200,t_300,t_500,u_850,v_850 \
    --native-write-hours 432-792 \
    --commit-every 1 --float-size f4 > "$WORK/rollout.log" 2>&1
grep -q "1 written" "$WORK/rollout.log" || fail rollout rollout.log

say "products: 3a | TS days | MJO | indices"
$PY -u ../shared/aifs_n320_grib_1p5defg_nc_cli.py --date $DT --members 1 --v2 \
    --no-upload --source icechunk --icechunk-store "$WORK/icechunk_n320" \
    --source-grid n320 --icechunk-tag cycle-$DT --output-dir "$OUT" > "$WORK/3a.log" 2>&1
grep -q "Successful: 1/1" "$WORK/3a.log" || fail 3a 3a.log

$PY -u ts-mjo/ts_days.py --store "$WORK/icechunk_n320" --tag cycle-$DT --init $D \
    --out "$OUT/ts_days.nc" > "$WORK/ts.log" 2>&1 || fail ts_days ts.log

# HC_OPTIONAL=1 (Colab): skip MJO / indices when their inputs are missing instead of aborting
if [ -f $VPM/vpm_clim_1991_2020.npz ] || [ "${HC_OPTIONAL:-0}" != 1 ]; then
$PY -u ts-mjo/vpm_index.py --store "$WORK/icechunk_o96" --init $D \
    --clim $VPM/vpm_clim_1991_2020.npz --eofs $VPM/rmm_basis.npz \
    --dump-bands "$OUT/mjo_bands.npz" --out "$OUT/mjo_probs.nc" > "$WORK/mjo.log" 2>&1 \
    || fail vpm_index mjo.log
else say "SKIP vpm_index: no $VPM/vpm_clim_1991_2020.npz"; fi

if [ -f "${HC_LSM_1P5:-/tank/projects/bc-aifs/aiwq/land_sea_mask_1pt5DEG.nc}" ] || [ "${HC_OPTIONAL:-0}" != 1 ]; then
$PY -u "$HERE/hc_extract.py" --store "$WORK/icechunk_o96" --member 0 --init $D \
    --out "$OUT/indices.npz" > "$WORK/extract.log" 2>&1 || fail hc_extract extract.log
else say "SKIP hc_extract: no AI-WQ 1.5 deg land-sea mask"; fi

du -sh "$WORK"/icechunk_* | sed "s/^/### store size /"
rm -rf "$WORK"/icechunk_o96 "$WORK"/icechunk_n320
cp "$WORK"/*.log "$OUT/" 2>/dev/null
date -u +%FT%TZ > "$OUT/DONE"
say "done -> $OUT ($(du -sh "$OUT" | cut -f1))"
