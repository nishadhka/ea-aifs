#!/usr/bin/env bash
# Control-only hindcast for one calendar date: one ERA5-started member per year.
# Loops over the years and runs any year whose state pkl exists and whose 1.5 deg file
# does not, in any order, so slow MARS/tape years do not block the others. Keeps the
# 1.5 deg file and deletes the pkl. Stops when every year is done, or when nothing new
# has appeared for IDLE_MAX seconds (default 6 h).
#
#   hc_date.sh MMDD [FIRST_YEAR] [LAST_YEAR]      e.g.  hc_date.sh 1105 2006 2025
#
# (A first version waited at most 1 h per year in fixed order and skipped 2023 and 2022
#  whose states arrived after 3 h and 7 h of tape waits, 2026-11-09.)
set -u
MD=$1; Y0=${2:-2006}; Y1=${3:-2025}
IDLE_MAX=${IDLE_MAX:-21600}
S=/tank/projects/bc-aifs
HERE=$(cd "$(dirname "$0")" && pwd)
say(){ echo "### $(date -u +%H:%M) $*"; }
idle=0
while :; do
  todo=0; ran=0
  for y in $(seq $Y1 -1 $Y0); do
    D=${y}${MD}_0000; OUT=$S/hc_nc/$D
    [ -f "$OUT/DONE" ] && continue     # all 15 products written (hc_member.sh)
    todo=$((todo+1))
    P=$S/era5_states/$D/input_state_member_001.pkl
    [ -f "$P" ] || continue
    sleep 5   # the builder writes the pkl in one go; let the write finish
    if "$HERE/hc_member.sh" "$D" "$S/era5_states/$D" "$OUT"; then
      rm -f "$P"; ran=$((ran+1)); todo=$((todo-1))
    else
      say "$D: member failed; state kept for a retry"
    fi
  done
  [ $todo -eq 0 ] && { say "hindcast $MD: all years $Y0-$Y1 done"; exit 0; }
  if [ $ran -gt 0 ]; then idle=0; else sleep 60; idle=$((idle+60)); fi
  [ $idle -ge $IDLE_MAX ] && { say "hindcast $MD: stopping, $todo years still without a state after $((IDLE_MAX/3600)) h idle"; exit 1; }
done
