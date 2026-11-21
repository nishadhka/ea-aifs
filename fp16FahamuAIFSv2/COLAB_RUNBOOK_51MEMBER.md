# Quick start: a weekly 51-member AIFS-ENS 2.0 submission on Google Colab

Produce the weekly AI Weather Quest submission — nine files — on Google Colab GPUs instead of
a local machine, driven from a laptop through the Colab CLI.

**Scope.** 51 real-time ensemble members only: no hindcast, no bias correction. The
220-member same-date hindcast is a separate workflow with its own plan and runbooks.

**Generic by design.** Nothing below names a bucket, an endpoint, a storage provider or an
absolute path on any particular machine. Set the placeholders in §1.1 once and the commands
follow. Everything marked **measured** is from real runs on an L4; **unverified** has not been run.

---

## 0. Choose the GPU before anything else

| GPU | Memory | flash-attn 2 | Per member | Verdict |
|---|---|---|---|---|
| **L4** | 22.5 GB | yes | **9.4 min measured** | **use this** |
| A100 | 40/80 GB | yes | ~3–5 min est. | cheaper per member only below ~2.7 min |
| **T4** | 15 GB | **no** | unknown | **blocked — §6** |

The AIFS-ENS 2.0 checkpoint ships `attention_implementation: flash_attention`, which needs an
Ampere GPU (SM 8.0+). A T4 is Turing (SM 7.5). §6 gives the fallback and why it is still
expected to fail.

One weekly cycle on L4: **~15 compute units ≈ $1.5**, about 2.5 h wall on three runtimes.

---

## 1. Set up the laptop

The laptop computes nothing. It holds credentials, starts and stops runtimes, reads logs, and
makes the final submission.

### 1.1 Placeholders used throughout

Put these in a file you source before each cycle. Every command below uses them, so this is
the only place anything is machine- or account-specific.

```bash
# --- object storage (any S3-compatible bucket: see 1.4) -----------------------
export OBJ_BUCKET=<your-bucket>
export OBJ_ENDPOINT=<https://storage.googleapis.com | https://s3.<region>.amazonaws.com | ...>
export OBJ_PREFIX=<a prefix inside the bucket, e.g. aifs/weekly>
export OBJ_CRED_FILE=<path on the runtime holding the AWS_* variables, mode 600>

# --- local ------------------------------------------------------------------
export REPO=<your clone of the repository>
export PY=<python interpreter of the AIFS environment>   # torch 2.7, anemoi 0.8.3, icechunk 2.1
export RUN=<local working directory for cycle outputs>

# --- competition ------------------------------------------------------------
export TEAM=<your registered team name>
export MODEL=<your registered model name>

# --- cycle ------------------------------------------------------------------
export CYCLE=<YYYYMMDD>        # the Thursday 00z init
export MMDD=${CYCLE:4:4}
export YEAR=${CYCLE:0:4}
```

Requirements: **Linux or macOS** (the Colab CLI is not published for Windows — use WSL2),
Python 3.10+, `git`, `ssh`, `curl`, and a Google account with **Colab Pro**. The free tier
will not hold a GPU runtime long enough.

### 1.2 Install the Colab CLI

```bash
pip install --upgrade google-colab-cli
colab --version
colab auth login        # opens a browser; interactive, needs the account owner
colab usage             # prints burn rate and compute-unit balance
```

`colab usage` printing a balance is the test that login worked. If `colab new --help` does not
mention `--gpu`, you have a different package of the same name.

### 1.3 An ssh key used only for Colab

```bash
ssh-keygen -t ed25519 -f ~/.ssh/colab_only -N ''
```

The driver script expects this path (override with `KEY=`). Keep it separate from your normal
key: it is installed into each runtime's `authorized_keys`.

### 1.4 Object storage — why it is required, and what will do

**You cannot avoid a bucket.** Two different runtime types have to exchange ~50 GB:

| What | Size | Written by | Read by |
|---|---|---|---|
| 51 starting states, one per member | **0.97 GB each ≈ 50 GB** | the CPU runtime | the GPU runtimes |
| per-member products | ~31 MB each ≈ 1.6 GB | the GPU runtimes | the laptop |

Colab runtimes are ephemeral and isolated: they have an idle limit and a ~24 h lifetime, they
cannot see each other's disks, and anything left only on a VM is lost when it ends. The bucket
is the only shared, durable place in the chain. 50 GB is also far more than is comfortable to
push over a laptop link twice, which is the other reason it goes to cloud storage directly.

**Any S3-compatible object store works** — the transfer layer is `boto3` with an explicit
`endpoint_url`:

| Provider | `OBJ_ENDPOINT` | Credentials |
|---|---|---|
| **Google Cloud Storage** | `https://storage.googleapis.com` | HMAC key for a service account, used as `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` |
| **AWS S3** | `https://s3.<region>.amazonaws.com` | IAM access key, or temporary STS credentials |
| MinIO / R2 / other | your endpoint | that provider's S3 keys |

The bucket needs no special features: plain object storage, one prefix, no versioning or
lifecycle rules required. Add a lifecycle rule deleting the starting-state prefix after a few
days if you want the transient 50 GB cleaned up automatically.

**Two changes to point the helper at your bucket.** The storage module currently pins three
constants and authenticates reads anonymously:

1. **The three constants** — bucket, endpoint and prefix are module-level literals. Make them
   read `OBJ_BUCKET`, `OBJ_ENDPOINT` and `OBJ_PREFIX` from the environment (a three-line
   change), or edit them once. The credential-file path is already environment-driven via
   `SC_CRED_FILE`; point it at `OBJ_CRED_FILE`.
2. **Reads are unsigned.** `size`, `ls` and `get` build an anonymous client, which only works
   for a publicly readable bucket. **For a private bucket, make reads use the authenticated
   client too** — otherwise every download fails with an access error that looks like a
   missing object.

The key layout is defined by the helper's own functions, so use those rather than typing keys:
one function returns a member's starting-state key, another a member's product prefix. Nothing
else in the pipeline cares how they are spelled.

### 1.5 Credentials the pipeline needs

```bash
git clone <repo> && cd <repo>
```

| Credential | For | Needed for 51 real-time members? |
|---|---|---|
| Object-store keys | the bucket | **yes** |
| AI-WQ submission + retrieval | the competition | **yes** (§1.6) |
| MARS / ECMWF Web API | reanalysis waves | no — open data is anonymous |
| Copernicus CDS | reanalysis ensemble | no — hindcast only |

None of these is ever committed; the repository ignores the credential family
case-insensitively.

### 1.6 The AI-WQ credential trap

```
AIWQ_TEAM_NAME=<your team name, as registered>
AIWQ_MODEL_NAME_V2=<your model name, as registered>
AIWQ_PASSWORD=<FTP password, for RETRIEVAL>
ecbox=<ECBox token, for SUBMISSION>
```

Both the team and the model must already be registered for the competition, under exactly
these spellings. An unregistered model name is rejected at upload with a message naming the
**team** rather than the model, which is misleading — check the model name first. The scripts
read `AIWQ_MODEL_NAME_V2` and fall back to `AIWQ_MODEL_NAME_FP16`, so either key works.

> **Submission and retrieval use different credentials.** Retrieval (tercile climatologies) is
> FTP with `AIWQ_PASSWORD`. Submission is ECBox with the `ecbox` token. The wrong one returns
> `403 "Not a Branca token"`, which reads like a path error and is not one. Precedence is
> `AIWQ_ECBOX_TOKEN or ecbox or AIWQ_PASSWORD`.

The environment file is found with `find_dotenv(usecwd=True)`, so **run the submission steps
from the model directory**.

### 1.7 Check the laptop is ready

```bash
colab usage                        # balance prints
ls ~/.ssh/colab_only               # key exists
$PY -c "import torch, anemoi.inference, icechunk; print('env ok')"
```

---

## 2. The Colab CLI, and its four traps

| Command | Does |
|---|---|
| `colab new -s NAME --gpu L4` | start a GPU runtime |
| `colab new -s NAME --high-mem` | start a CPU high-RAM runtime |
| `colab sessions` | list live runtimes |
| `colab usage` | burn rate and balance |
| `colab stop -s NAME` | stop a runtime |

The driver script in the repository wraps these:

```
hc_colab.sh up       NAME [cpu|L4|A100]   create the runtime, write its ssh config
hc_colab.sh push     NAME                 code tarball and any auxiliary inputs
hc_colab.sh creds    NAME                 object-store credentials -> OBJ_CRED_FILE (mode 600)
hc_colab.sh realtime NAME YYYYMMDD [JOBS] build the 51 starting states (background)
hc_colab.sh setup    NAME                 GPU environment + checkpoint (background)
hc_colab.sh worker   NAME MMDD [ARGS]     the rollout worker (background)
hc_colab.sh sh       NAME 'cmd'           one command over ssh
hc_colab.sh stop     NAME                 stop, then print sessions and usage
```

`creds` is the one provider-specific step: it mints short-lived credentials with a vendor CLI
and pipes them over ssh. **Replace that single line with however your provider issues keys** —
`gcloud`, `aws sts`, or a static key written once. The file must contain
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and optionally `AWS_SESSION_TOKEN`. It travels
over ssh stdin only, never inside code passed to the Colab CLI, which records its own command
history.

**Trap 1 — one ssh connection per runtime.** Colab allows a single connection. Every driver
call connects, runs and disconnects; long jobs launch detached and are watched by reading their
log. Do not hold an interactive shell open while a job runs.

**Trap 2 — connecting *creates* a runtime** if the named one is gone, and it bills. The driver
refuses to connect unless `colab sessions` already lists the name.

**Trap 3 — runtimes die on their own.** Idle limit, plus a ~24 h lifetime. A CPU runtime once
ended at ~24 h and took the only copy of its outputs with it. **Put every product in the
bucket as soon as it is made.** Never leave the sole copy on a VM.

**Trap 4 — idle GPUs are the largest avoidable cost.** On one 271-member cycle, 19.4 CU — a
quarter of the total — went on runtimes left up after their workers exited. A later cycle lost
~10 h of three idle runtimes a different way: transient object-store 5xx responses killed all
three workers, and nothing was watching for a worker that had *crashed* rather than finished.
Use the watcher (§4.2): it stops a runtime whose worker has exited for either reason, and
downloads retry transient 5xx rather than failing the member. **Never leave GPU runtimes
unwatched**, and check `colab usage` reads zero before you walk away.

---

## 3. What runs where

```
LAPTOP        credentials, start/stop runtimes, read logs, final submission

COLAB CPU     51 starting states from IFS open data
  (high-mem)    members 1-50 = open-data perturbed forecasts
                member  51   = unperturbed control from the deterministic 0 h analysis
              -> bucket: one starting-state object per member, 0.97 GB each

COLAB L4 x N  for each state without a completion marker: claim it, download, then
                792 h rollout, native grid all steps + a sidecar for the late leads
                regrid to 1.5 deg | storm days | MJO bands | indices
                delete both stores (3.1 GB peak per member)
              -> bucket: ~31 MB of products per member, marker written last

LAPTOP        aggregate the 51 per-member files -> quintiles -> submit -> verify
```

The open-data ensemble has **no control member**, which is why member 51 is built from the
deterministic 0 h analysis — the same substitution the local chain makes.

---

## 4. The weekly run

The AI-WQ window closes the **Sunday after** the Thursday init, 23:59 UTC.

### 4.1 Phase A — the 51 starting states (CPU runtime)

```bash
cd $REPO
B=<path to the colab driver directory>

$B/hc_colab.sh up    cpu1 cpu
$B/hc_colab.sh push  cpu1
$B/hc_colab.sh creds cpu1
$B/hc_colab.sh sh    cpu1 'uv pip install --system -q earthkit-regrid==0.5.1 eccodes gribberish \
                            obstore ecmwf-opendata boto3 zarr fsspec aiohttp requests xarray netcdf4'

$B/hc_colab.sh realtime cpu1 $CYCLE 4        # 4 at a time, detached; writes its own log
```

Watch it, and do not start GPUs until it is done:

```bash
$B/hc_colab.sh sh cpu1 "tail -5 /content/realtime_$CYCLE.log"
```

**Acceptance: 51 objects, each 971 413 699 bytes**, 112 fields of shape `(2, 542080)`.

```bash
$B/hc_colab.sh sh cpu1 "cd <colab dir> && python3 -c \"
import sc_io
keys = [sc_io.input_key('$MMDD', '$YEAR', m) for m in range(1, 52)]
sz = [sc_io.size(k) for k in keys]
print(sum(s == 971413699 for s in sz), 'of 51 at the expected size')
print('missing:', [i+1 for i, s in enumerate(sz) if s is None])\""
```

Then **stop the CPU runtime** — it is not needed again:

```bash
$B/hc_colab.sh stop cpu1
```

> Start GPU runtimes **only after Phase A completes.** Starting them early once burned ~18 CU
> waiting for inputs.

### 4.2 Phase B — 51 rollouts (GPU runtimes)

One runtime is simplest; three cost the same and take a third of the wall time. The worker is
claim-and-skip, so any number run together and a killed one is retaken after its claim expires.

```bash
for n in gpu1 gpu2 gpu3; do
  $B/hc_colab.sh up    $n L4
  $B/hc_colab.sh push  $n
  $B/hc_colab.sh creds $n
  $B/hc_colab.sh setup $n       # ~80 s: python 3.12 venv, torch 2.7.0+cu126, anemoi 0.8.3,
done                            # flash-attn 2.7.4, icechunk 2.1.0, model checkpoint
```

Wait for the setup success line on each, then start the workers:

```bash
for n in gpu1 gpu2 gpu3; do $B/hc_colab.sh sh $n 'tail -2 /content/setup.log'; done

for n in gpu1 gpu2 gpu3; do $B/hc_colab.sh worker $n $MMDD --years $YEAR --members 1-51; done
# stops each runtime whose worker has exited (finished OR crashed), and refreshes
# credentials on each pass
$B/hc_watch_stop.sh $MMDD gpu1 gpu2 gpu3
```

`--years $YEAR` is what restricts the worker to the real-time set; without it, it would also
pick up hindcast members if any share the same `MMDD`.

**Measured per member on L4:** 8.4–9.0 min end to end — state download 7 s, rollout 487 s
(~3.6 s/step), extraction 34 s, upload 32 s. Peak GPU memory **11 634 MiB of 23 034**.

**Acceptance: 51 completion markers.**

```bash
$B/hc_colab.sh sh gpu1 "cd <colab dir> && python3 -c \"
import sc_io
done = [m for m in range(1, 52)
        if sc_io.size(sc_io.product_prefix('$MMDD', '$YEAR', m) + 'DONE') is not None]
print(len(done), 'of 51 complete'); print('missing:', [m for m in range(1,52) if m not in done])\""
```

Then confirm **everything is stopped and the burn rate is zero** — the step that costs money
when skipped:

```bash
colab sessions; colab usage
```

### 4.3 Phase C — quintiles (laptop)

Download the per-member 1.5° files into `$RUN/$CYCLE_0000/nc_1p5deg/`, keeping the member
number in each filename, plus each member's storm-days and MJO files for the merge in §4.4.

The aggregation must reproduce the local quintile step exactly: **week 1 = time chunks 0,1
(24 steps), week 2 = chunks 2,3,4 (36 steps); temperature and pressure are step means,
precipitation is the sum × 1000 × 7 / days.** The repository's aggregation helper already
implements that definition, and its binning helper does the quintile counting against the
official climatology:

```bash
cd <aggregation helper directory>
BC_RUN=$RUN BC_CACHE=<a cache directory> $PY - <<'PY'
import os, numpy as np, xarray as xr, past_cycle_bc as B
date = os.environ["CYCLE"]
B.aggregate(date)                       # -> (2 weeks, 51 members, 121, 240)
mem  = B.load_weekly(date)
wds  = B.verification_dates(date)       # init+18, init+25
out  = {}
for v in B.VARS:
    probs, cl = [], None
    for k in (0, 1):
        cl = B.clim(wds[k], v)
        probs.append(B.quintile_probs(mem[v][k], cl.values))
    out[B.QVAR[v]] = xr.DataArray(
        np.stack(probs), dims=("time_week", "quintile", "latitude", "longitude"),
        coords=dict(quintile=[0.2, 0.4, 0.6, 0.8, 1.0], latitude=cl.latitude,
                    longitude=cl.longitude, week=("time_week", ["week1", "week2"])))
ds = xr.Dataset(out)
ds.attrs.update(title="Ensemble Forecast Quintile Probabilities", forecast_date=date,
                n_members=int(mem["tas"].shape[1]), method="Colab 51-member, raw")
f = os.path.join(os.environ["RUN"], f"{date}_0000", "aiwq",
                 f"ensemble_quintile_probabilities_{date}_v2.nc")
os.makedirs(os.path.dirname(f), exist_ok=True)
ds.to_netcdf(f); print("wrote", f, ds[B.QVAR['tas']].shape)
PY
```

> **This aggregator has not been run.** Its parts are all in use elsewhere, but assembling an
> *uncorrected* submission file from them is new: the local pipeline reads the Icechunk store
> directly, and the Colab worker deletes every store after extraction, so there is no store to
> read. **Validate it once before trusting it** against a cycle that was run both ways: compare
> with that cycle's locally produced quintile file. Expect small differences from the model's
> noise draw, not structural ones.

Validate the file whatever produced it:

```bash
$PY -I -c "
import xarray as xr, numpy as np, os
d = xr.open_dataset(os.path.expandvars('\$RUN/\$CYCLE_0000/aiwq/ensemble_quintile_probabilities_\$CYCLE_v2.nc'))
for v in d.data_vars:
    a = d[v].values                      # (time_week, quintile, lat, lon)
    print(v, a.shape, 'NaN', int(np.isnan(a).sum()),
          'range %.3f-%.3f' % (a.min(), a.max()),
          'quintile sum min/max %.6f/%.6f' % (a.sum(axis=1).min(), a.sum(axis=1).max()))"
```

Expect shape `(2, 5, 121, 240)` per variable, no NaN, every value in [0, 1], and quintile sums
of 1.000000 everywhere.

### 4.4 Submit

```bash
cd <the model directory in your clone>   # so the environment file is found

# tercile climatologies FIRST: the storm-days script run without them falls back to
# self-referential terciles, and its printed probabilities are then NOT the forecast
$PY - <<'PY'
import os, datetime as dt
from AI_WQ_package import retrieve_evaluation_data as red
d0 = dt.datetime.strptime(os.environ["CYCLE"], "%Y%m%d").date()
for k in (18, 25):
    red.retrieve_20yr_quantile_clim((d0 + dt.timedelta(days=k)).strftime("%Y%m%d"), "TS",
        os.environ["AIWQ_PASSWORD"],
        local_destination=os.path.join(os.environ["RUN"], os.environ["CYCLE"] + "_0000", "ts_terciles"))
PY

$PY -u ../shared/forecast_submission_cli.py --date $CYCLE --v2 \
    --output-dir $RUN/${CYCLE}_0000/aiwq

# storm days and MJO: merge the 51 per-member files first, then submit with --fill-rule bounds
# (scoring is keyed on the published tercile bounds, not the calendar month)
$PY -u submit_ts_mjo_cli.py --date $CYCLE --fill-rule bounds \
    --ts-nc <merged storm-days file> --tercile-dir $RUN/${CYCLE}_0000/ts_terciles \
    --mjo-nc <merged MJO file> --dry-run          # then without --dry-run
```

### 4.5 Verify against the server, never against the logs

```bash
$PY check_submissions_paged.py --date $CYCLE      # expect "9/9 present"
```

**Do not use the packaged checker.** It lists the remote directory once and ignores the
continuation token, while the server pages at 100 entries — so it reports false ABSENT for
every late-sorting name. Uppercase names sort early and are reliable; the lowercase gridded
ones are not. Use the paginated check.

---

## 5. Cost and wall time for 51 members

Unit costs **measured**: 0.285 CU per member (8.8 min on L4 at 1.94 CU/h); 80 s of setup per
runtime; CPU high-RAM at 0.26 CU/h; ~$0.10 per compute unit.

| | Runtimes | GPU hours | CU | ≈ $ | Wall |
|---|---|---|---|---|---|
| Phase A, 51 states | 1 CPU | — | ~0.2 | ~$0.02 | ~25–35 min |
| Phase B | 1 L4 | 7.5 | 14.6 | $1.5 | **7.5 h** |
| Phase B | 2 L4 | 7.5 | 14.7 | $1.5 | **3.8 h** |
| Phase B | 3 L4 | 7.5 | 14.8 | $1.5 | **2.5 h** |
| Phase C + submit | laptop | — | — | — | ~30 min |
| **Total** | | | **~15** | **~$1.5** | **~3.5 h on 3 L4** |

Parallel runtimes cost the same and finish sooner, so use one only if Colab will not grant
three. Bucket usage: **~50 GB transient** (delete after Phase B) and ~1.6 GB of products to keep.

Storage costs, if your provider charges: holding 50 GB for a week is cents, but **reading it
back is the item to watch** — the GPU runtimes download the full ~50 GB once per cycle, so a
provider with per-GB egress can cost more than the GPUs. Prefer a bucket in the same cloud as
the runtimes; Colab assigns regions you do not control, so egress is not always free even then.

---

## 6. T4 — what to change, and why it is still expected to fail

**Unverified throughout.** Keep this separate from the working path.

### 6.1 The two blockers

**Blocker 1 — flash-attn 2 does not support Turing.** It needs SM 8.0+; a T4 is SM 7.5. The
setup script ends with a live flash-attention call on a CUDA half-precision tensor, so it fails
at setup. That check is doing its job: the alternative is a failure hours later inside a rollout.

**Blocker 2 — the fallback needs a dense mask this model cannot afford.** The model library
offers PyTorch scaled dot-product attention as an alternative, and the checkpoint is compatible
in every respect but one:

| Checkpoint setting | Value | Under SDPA |
|---|---|---|
| rotary embeddings | absent | fine — the wrapper takes no rotary argument |
| ALiBi slopes | off | fine |
| softcap | 0.0 | disabled anyway |
| qk-norm | on | applied outside the wrapper |
| **attention window** | **1120** | **builds a dense `seq_len × seq_len` boolean mask** |

The processor is a 16-layer, 8-head transformer on a **reduced-Gaussian hidden mesh of 40,320
nodes**. So the window mask alone is **40,320² = 1.63 GB**, and passing a dense mask forces
SDPA off its fused backends onto the math backend, which materialises the attention matrix
instead of fusing it. Against a measured 11.6 GB peak with flash-attention, on a 15 GB card,
with chunking as the only relief — it is very likely to run out of memory.

### 6.2 If you want to try anyway

1. Run the GPU setup with the flash-attention install and its verification removed.
2. Override the attention implementation to scaled dot-product attention at model load. **No
   such override exists today** — the rollout script does not read one. Adding it means
   patching the processor configuration on the loaded checkpoint before the model is built.
   Write that as a small explicit change; do not edit the checkpoint.
3. Run **one short member first** (a 72 h lead time), never a full 792 h member.

### 6.3 Stop rule

| Outcome of the 72 h test | Do |
|---|---|
| out of memory | stop; use L4. This is the expected outcome |
| runs, but > 15 min | stop; a full member would exceed ~2.5 h and cost more than L4 |
| runs in < 15 min | compare one **full** member's products against an L4 member from the same starting state before trusting any of it |

A T4's lower hourly rate cannot win from behind: it only beats L4's measured 0.285 CU per
member by finishing one in under ~9.5 min — roughly flash-attention speed, on a card that
cannot run flash-attention.

---

## 7. Differences from a local run

| | Local | Colab, 51 members |
|---|---|---|
| Starting states | built locally | same script, on a Colab CPU runtime |
| Stores | kept, ~210 GB per cycle | deleted per member; ~31 MB of products kept |
| Quintile step | reads the Icechunk store directly | aggregated from 51 per-member 1.5° files (§4.3) |
| Disk | ~250 GB per cycle locally | ~50 GB transient in the bucket |
| Cost | electricity | ~$1.5 of compute units |
| Noise draw | — | **different** — same model and states, different random draws |

That last row matters if both are ever submitted: they are different forecasts and are not
interchangeable for evaluation.
