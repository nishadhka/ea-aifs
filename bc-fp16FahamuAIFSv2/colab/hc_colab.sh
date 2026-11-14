#!/usr/bin/env bash
# Local driver for the Colab hindcast (COLAB_PLAN.md). Runs on the gateway box, which has
# the `colab` and `source-coop` CLIs logged in.
#
#   hc_colab.sh up    NAME [cpu|L4|A100|...]  create the runtime and write its ssh config
#   hc_colab.sh push  NAME                    code tarball (+ ~/.ecmwfapirc on CPU runtimes)
#   hc_colab.sh creds NAME                    source.coop S3 credentials -> /root/.sc.env (600)
#   hc_colab.sh sh    NAME 'command'          run a command over ssh
#   hc_colab.sh phaseA NAME MMDD YEARS [JOBS] [--perturbed 2-11]  build pkls -> source.coop (background)
#   hc_colab.sh realtime NAME YYYYMMDD [JOBS] 51 real-time IFS open-data pkls -> <MMDD>/<YYYY>/ (background)
#   hc_colab.sh setup NAME                    GPU env + checkpoint (background, log /content/setup.log)
#   hc_colab.sh worker NAME MMDD [ARGS...]    GPU worker (background, log /content/worker_MMDD.log)
#   hc_colab.sh stop  NAME                    stop the runtime; prints sessions and usage
#
# Colab allows ONE ssh connection per runtime: every command here connects, runs, and
# disconnects; long jobs run under nohup on the VM. Credentials travel only over ssh stdin,
# never in `colab exec` code (which the Colab CLI records in ~/.config/colab-cli/history/).
set -eu
CMD=$1; NAME=$2; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
CFGDIR=${HC_CFGDIR:-$HOME/.cache/hc_colab}; mkdir -p "$CFGDIR"; chmod 700 "$CFGDIR"
CFG=$CFGDIR/$NAME.cfg
KEY=${KEY:-$HOME/.ssh/colab_only}
BC=/content/ea-aifs/bc-fp16FahamuAIFSv2
SSH="ssh -F $CFG $NAME"

write_cfg() { cat >"$CFG" <<CFG
Host $NAME
  ProxyCommand $(command -v colab) --auth oauth2 ssh --proxy-mode -s $NAME -i $KEY
  User root
  IdentityFile $KEY
  IdentitiesOnly yes
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
  LogLevel ERROR
  ServerAliveInterval 30
  ServerAliveCountMax 6
CFG
}

# setsid -f: fully detached, so ssh returns at once (with plain nohup & the channel stayed open)
bg() { ssh -n -F "$CFG" "$NAME" "cd $BC/colab && setsid -f nohup $1 > $2 2>&1 < /dev/null; echo started: $2"; }

# `colab ssh` CREATES a runtime if NAME is gone (Colab ends idle or long-lived runtimes):
# refuse to connect unless the session exists (a status check on 2026-11-14 started one).
if [ "$CMD" != up ] && [ "$CMD" != stop ]; then
  colab sessions 2>/dev/null | grep -q "\[$NAME\]" || { echo "no Colab session '$NAME' (ended?); run: hc_colab.sh up $NAME ..."; exit 3; }
fi

case $CMD in
  up)
    ACC=${1:-cpu}
    if [ "$ACC" = cpu ]; then colab new -s "$NAME" --high-mem; else colab new -s "$NAME" --gpu "$ACC"; fi
    write_cfg; colab usage 2>&1 | grep -E 'rate|balance' ;;
  push)
    T=$(mktemp --suffix=.tgz)
    ( cd "$ROOT" && tar czf "$T" --exclude='*.png' --exclude='*.txt' --exclude='__pycache__' \
      --exclude='loadtest_artifacts' --exclude='dev-test' \
      --exclude='epistemic-reasoning-risk' --exclude='.ecmwfapirc' --exclude='.cdsapirc' --exclude='.env' --exclude='*.pkl' \
      s3_grib_pkl shared/aifs_n320_grib_1p5defg_nc_cli.py fp16FahamuAIFSv2/*.py \
      fp16FahamuAIFSv2/ts-mjo fp16FahamuAIFSv2/run-pre50r1-dates fp16FahamuAIFSv2/O96-icechunk-store/*.py bc-fp16FahamuAIFSv2 )
    scp -q -F "$CFG" "$T" "$NAME":/content/code.tgz
    $SSH 'mkdir -p /content/ea-aifs && tar xzf /content/code.tgz -C /content/ea-aifs && echo code pushed' </dev/null
    rm -f "$T"
    RC=$HERE/../.ecmwfapirc; [ -f "$RC" ] || RC=$HOME/.ecmwfapirc
    if [ -f "$RC" ]; then $SSH 'umask 077; cat > /root/.ecmwfapirc' < "$RC"; echo "MARS key pushed"; fi
    # MJO and indices inputs (vpm_clim_1991_2020.npz, rmm_basis.npz, land_sea_mask_1pt5DEG.nc),
    # from the GPU box (aifs_20261105_bundle.zip inputs_for_colab/) -> /content/aux/
    if ls "$HERE"/../colab_aux/*.n* >/dev/null 2>&1; then
      $SSH 'mkdir -p /content/aux' </dev/null
      scp -q -F "$CFG" "$HERE"/../colab_aux/* "$NAME":/content/aux/ && echo "aux pushed: $(ls "$HERE"/../colab_aux | tr '\n' ' ')"
    else echo "WARNING: no colab_aux/: MJO and indices will be skipped"; fi
    RC=$HERE/../.cdsapirc; [ -f "$RC" ] || RC=$HOME/.cdsapirc
    if [ -f "$RC" ]; then $SSH 'umask 077; cat > /root/.cdsapirc' < "$RC"; echo "CDS key pushed"; fi ;;
  creds)
    source-coop creds --format env >/dev/null 2>&1 || { echo "source-coop creds failed: log in again"; exit 1; }
    source-coop creds --format env 2>/dev/null | $SSH 'umask 077; cat > /root/.sc.env; echo "source.coop credentials in /root/.sc.env"'
    source-coop creds 2>/dev/null | python3 -c "import json,sys; print('expire', json.load(sys.stdin)['Expiration'])" ;;
  sh)     $SSH "$1" </dev/null ;;
  phaseA)
    $SSH "mkdir -p /content/bc; test -f /content/bc/ref_wave_land_n320.npy || echo 'WARNING: no /content/bc/ref_wave_land_n320.npy (control builds only; see RUNBOOK_20261105.md section 4.1)'" </dev/null
    M=$1 Y=$2 J=${3:-4}; shift 3 2>/dev/null || shift $#
    bg "env BC_CACHE=/content/bc CDSAPI_RC=/root/.cdsapirc python3 -u hc_phaseA.py --mmdd $M --years $Y --jobs $J $*" /content/phaseA_$M${*:+_pert}.log ;;
  realtime) bg "env BC_CACHE=/content/bc python3 -u hc_phaseA.py --realtime $1 --jobs ${2:-4}" /content/realtime_$1.log ;;
  setup)  bg "bash hc_gpu_setup.sh" /content/setup.log ;;
  worker) M=$1; shift; bg "/usr/bin/python3 -u hc_gpu_worker.py --mmdd $M $*" /content/worker_$M.log ;;
  stop)   colab stop -s "$NAME" || true; sleep 2; colab sessions 2>&1 | tail -3; colab usage 2>&1 | grep -E 'rate|balance' ;;
  *) echo "unknown command $CMD"; exit 2 ;;
esac
