#!/usr/bin/env bash
# Stop each GPU runtime as soon as its worker has exited (COLAB_PLAN.md section 10.4: on
# 12 Nov two L4s idled ~4.6 h each after their last member, 19 CU). Run on the gateway:
#
#   colab/hc_watch_stop.sh MMDD hcB hcC hcD        (checks every 5 min; exits when all stopped)
#
# A runtime is stopped when its worker has started (/content/worker_<MMDD>.log exists) and no
# hc_gpu_worker.py process is left -- whether the worker finished or crashed (on 2026-11-20 three
# workers crashed without a summary line and the old check left their L4s idle ~10 h). The source.coop credentials are re-pushed to every live
# runtime on each pass (they last <= 12 h; workers re-read /root/.sc.env).
MD=$1; shift
HERE=$(cd "$(dirname "$0")" && pwd)
left=("$@")
while [ ${#left[@]} -gt 0 ]; do
  next=()
  for n in "${left[@]}"; do
    ls_out=$(colab sessions 2>&1)
    if [ $? -ne 0 ] || echo "$ls_out" | grep -q -E "ColabRequestError|Bad Gateway"; then
      next+=("$n"); continue                                # API hiccup: unknown, not gone
    fi
    if ! echo "$ls_out" | grep -q "\[$n\]"; then
      echo "$(date -u +%H:%M) $n: gone"; continue
    fi
    st=$(timeout 60 "$HERE/hc_colab.sh" sh "$n" \
         "test -f /content/worker_$MD.log && echo SUMMARY; ps aux | grep -q '[h]c_gpu_worker.py' && echo ALIVE" 2>/dev/null)
    if [ $? -ne 0 ]; then next+=("$n"); continue; fi          # ssh slot busy: try next pass
    if echo "$st" | grep -q SUMMARY && ! echo "$st" | grep -q ALIVE; then
      summ=$(timeout 60 "$HERE/hc_colab.sh" sh "$n" "grep -E '^WORKER|FAILED|Error' /content/worker_$MD.log | tail -2" 2>/dev/null)
      timeout 90 colab stop -s "$n" >/dev/null 2>&1
      echo "$(date -u +%H:%M) $n: worker exited, runtime stopped | $summ"
    else
      timeout 60 "$HERE/hc_colab.sh" creds "$n" >/dev/null 2>&1
      next+=("$n")
    fi
  done
  left=("${next[@]}")
  [ ${#left[@]} -gt 0 ] && sleep 300
done
echo "$(date -u +%H:%M) all GPU runtimes stopped; $(colab usage 2>/dev/null | grep -E 'rate' | tr -s ' ')"
