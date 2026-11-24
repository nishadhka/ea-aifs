# Running the hindcast on Google Colab (plan and test, 2026-11-11)

Plan for moving the hindcast (README §0.2, up to 220 members per date) from this box to
Google Colab, driven from the terminal with the Colab CLI and paid with Colab Pro compute
units. **Status (2026-11-12): tests 1–4 passed on L4, measured numbers in §9.** The 20 control
pkls for 5 Nov are on source.coop and one full member ran end to end on a Colab L4 in
9.4 min (0.30 CU). §5 still lists the routines and the original estimates.

## 1. Why

| Problem on this box | Effect |
|---|---|
| The link from this building to Google Cloud Storage is ~1.4 MB/s, whatever the number of readers (measured, `../fp16FahamuAIFSv2/ts-mjo/make_colab_bundle.py`) | one ARCO-ERA5 starting state (1.34 GB) takes ~7 min; ERA5 verification for one date (17 GB) takes hours |
| One GPU | 220 members per date is ~18 h in series, and it must not collide with the Thursday real-time run |

ARCO-ERA5 lives in GCS. Building the starting states inside Google's cloud reads it at
cloud speed and sends nothing large over the slow link.

## 2. Colab CLI

Google's official CLI, released June 2026: `googlecolab/google-colab-cli`.

| Command | Does |
|---|---|
| `colab new --gpu {T4,L4,G4,A100,H100}` | start a runtime |
| `colab run [--gpu L4] script.py` | start a runtime, run the script, tear it down. `--timeout` is an idle limit, so a job that keeps printing stays alive |
| `colab exec -s NAME -f notebook.ipynb` | run a notebook's cells on a runtime |
| `colab upload`, `colab download`, `colab stop` | files and shutdown |

Linux and macOS only. Check the repo's docs for the current syntax before scripting it;
a third-party project with the same name uses different commands.

## 3. Design

```
PHASE A  CPU runtime, no GPU                       reads ARCO inside GCP
  for each year Y:
    era5_state_pkl.py   control state: ARCO (100 fields) + MARS ecWAM waves (12 fields)
    (+ EDA)             10 perturbed states from CDS ensemble members (code not written yet)
    -> gs://<bucket>/hc/<MMDD>/<YYYY>/input_state_member_NNN.pkl      0.97 GB each
    era5_weekly.py / truth fetchers -> gs://<bucket>/hc/<MMDD>/truth/

PHASE B  GPU runtime(s), L4 or A100, one or several in parallel
  bootstrap once per runtime: pip env (torch 2.7, anemoi-inference 0.8.3, flash-attn 2.7.4
                              wheel matching torch), HF checkpoint ecmwf/aifs-ens-2.0
  loop until no pkl is left:
    claim the next pkl without a DONE marker
    run_local_icechunk_v2.py -> stores on the runtime's local disk (3.1 GB per member)
    3a | ts_days.py | vpm_index.py | hc_extract.py      (the hc_member.sh steps)
    push products -> gs://<bucket>/hc/<MMDD>/products/ (Icechunk product store)
    write gs://.../DONE marker, delete local stores and the pkl

LOCAL  hc_bc.py evaluate / apply (reads the product store); Thursday real-time run
```

**Push products, not full stores.** The full stores exist only to be extracted:

| | Per member | 220 members (one date) |
|---|---|---|
| Full stores (O96 132 steps + N320 sidecar) | 3.1 GB | ~680 GB to upload, store and read back |
| Products (1.5° weekly file, storm days, MJO bands, indices) | 41 MB, or ~1.3 MB without the 1.5° file | 9 GB, or ~0.3 GB |

## 4. GPU choice and cost

The model's measured peak GPU memory is ~11.8 GB (FP16, 16 inference chunks;
`../fp16FahamuAIFSv2/LOCAL_GPU_RUN.md`).

| Colab GPU | Memory | Fits 11.8 GB? | flash-attn 2 | Time per 792 h member | Compute units per hour (measured by others) |
|---|---|---|---|---|---|
| T4 | 15–16 GB | yes, barely | **no** (Turing; flash-attn 2 needs Ampere or newer) | probably will not run as configured; slow if it does | ~1.2–1.8 |
| **L4** | 22.5 GB | yes | yes | **not measured: ~7–22 min** (Cloud Run L4 ran v1 at ~10 s per step; L4 has about half this box's memory bandwidth) | ~1.7–5 |
| A100 | 40/80 GB | yes | yes | likely ~3–5 min (estimate) | ~5–15 |
| this box, RTX 5000 Ada | 30 GB | yes | yes | **3.3 min (measured)** | electricity only |

Cost per date for 220 members, at ~$0.10 per compute unit (100 for $9.99). Estimates:

| Where | GPU hours | Compute units | Cost |
|---|---|---|---|
| Colab L4 | ~26–81 | ~45–400 | ~$5–40 |
| Colab A100 | ~11–18 | ~55–270 | ~$6–27 |
| This box | ~18 | — | ~$1.3–2 electricity |
| Phase A (CPU builds) | ~2–3 h of CPU runtime | small | ≲ $1 |
| Bucket: 214 GB of pkls kept ~1 week, then deleted | | | ~$1 |

All 52 dates at 220 members: ~$250–2,000 on L4, ~$300–1,400 on A100. The L4 range is wide
because its speed for this model is not measured; the test in §6 fixes that.

## 5. Routines and expected durations, by ensemble size

### 5.1 Three ensemble sizes per date

So far the hindcast is **control only**: one ERA5-started member per year (README §0.1–0.2).
Adding EDA-perturbed members gives what a single member cannot:

| Size per date | Runs | What it supports |
|---|---|---|
| **Control only** (built for 5 Nov) | 20 = 20 years × 1 | mean bias per lead; model-climate tercile edges (20 values); MJO amplitude/phase map. Leave-one-out checks are deterministic (error, hit rate) |
| **Control + 4 EDA** | 100 = 20 × 5 | adds **spread vs error**, i.e. whether the real-time ensemble is over- or under-confident (README: effective ensemble size ~1.3–1.9 over East Africa); first probabilistic (RPSS) leave-one-out check |
| **Control + 10 EDA** | 220 = 20 × 11 | the ECMWF-style reforecast size: steadier bias, tercile edges from 220 values, full reliability calibration |

The year-to-year weather is shared by the members of one year, so extra members mainly reduce
the member-noise part of the bias estimate, not the year-to-year part.

### 5.2 Routines, in order, per date D

| # | Routine | Script | Status | Runs where |
|---|---|---|---|---|
| A1 | Control starting state per year: ARCO (100 fields) + MARS ecWAM waves (12 fields) | `era5_state_pkl.py`, `prefetch_waves.py` | built | CPU (Colab Phase A) |
| A2 | EDA members per year from CDS (`ensemble_members`, 0.5°, 3-hourly), 100 fields × 2 times | to write: `era5_eda_perturb.py` | **not built** | CPU |
| A3 | Perturbed states: control + (EDA member − EDA mean), regridded to N320; waves unperturbed | to write: `era5_eda_perturb.py` | **not built** | CPU |
| A4 | Truth: ERA5 weekly tas/mslp/pr (ARCO), index truth (CDS, clipped), IBTrACS and AI-WQ MJO (on disk) | `era5_weekly.py`, `era5_truth_indices.py`, `hc_bc15.py` | built | CPU |
| B1 | Rollout, 792 h: O96 all steps + N320 sidecar 432–792 h | `run_local_icechunk_v2.py` via `hc_member.sh` | built | GPU (Colab Phase B) |
| B2 | Extraction: 3a, storm days, MJO bands, 10 indices; delete stores | `hc_member.sh`, `hc_extract.py` | built | CPU on the GPU runtime |
| B3 | Claim/skip worker, products to the bucket | to write: `colab_gpu_worker.py` | **not built** | GPU runtime |
| C1 | Evaluation (leave-one-year-out) and bias tables | `hc_bc.py`, `hc_bc15.py` | built | local |
| C2 | Apply to the Thursday real-time forecast | `hc_bc.py apply`, `hc_bc15.py apply` | built | local |

### 5.3 Measured on this box (5 Nov, control only)

| Routine | Measured |
|---|---|
| A1 control state | ARCO fetch 410–470 s (link ~1.4–3 MB/s); total ~16 min per year with waves and writing; MARS waves seconds when on disk, up to ~3 h on tape; frequent network failures needed retries |
| A4 tas/mslp/pr truth | 432 MB per week from ARCO, 40 weeks per date |
| A4 index truth | ~6.5 min and ~90–106 MB per year from CDS (clipped, 1°) |
| B1 + B2 | **5.3 min per member** (rollout 4 min 50 s with all O96 steps; extraction 30 s); stores 3.1 GB peak; 41–42 MB kept |
| C1 | tas/mslp/pr evaluation ~5 min; variables 4–15 a few minutes |

### 5.4 Expected durations per date

GPU time per member: this box 5.3 min (measured); Colab L4 ~7–22 min and A100 ~3–5 min
(estimates until test 4 of §6). Compute units at ~1.7–5 per hour for L4 and ~5–15 for A100
(measured by others), ~$0.10 per unit.

| | Control only (20) | Control + 4 EDA (100) | Control + 10 EDA (220) |
|---|---|---|---|
| **A1** control states | local ~5.3 h; Colab ~0.3–1 h (estimate) + any tape wait | same as 20 (the control is shared) | same as 20 |
| **A2–A3** EDA + perturbed states | — | CDS ~2–5 h for 20 years (queue-bound, same anywhere); building ~1 min per state | same CDS volume (~20 GB); ~200 states to build |
| **A4** truth | local ~1–2 h (17 GB ARCO + CDS); Colab ~1 h (CDS-bound) | same | same |
| **B** on this box | **1.8 h** | **8.8 h** | **19.4 h** |
| **B** on 1 Colab L4 | 2.3–7.3 h, 4–37 CU (~$0.4–4) | 12–37 h, 20–183 CU (~$2–18) | 26–81 h, 44–403 CU (~$4–40) |
| **B** on 1 Colab A100 | 1–1.7 h, 5–25 CU (~$0.5–2.5) | 5–8 h, 25–125 CU (~$2.5–12.5) | 11–18 h, 55–275 CU (~$5.5–27.5) |
| **B** on 3 Colab runtimes at once | ÷3 wall time, same CU | ÷3 | ÷3 |
| **C** evaluate + apply | ~10 min local | ~15 min | ~20 min |
| Bucket, transient | ~20 GB of pkls | ~100 GB | ~214 GB |
| Kept per date | ~0.85 GB | ~4.2 GB | ~9.2 GB (or ~0.3 GB without the 1.5° files) |

All 52 dates once (the one-off build; afterwards each date adds only that year's members):

| | Control only | Control + 4 EDA | Control + 10 EDA |
|---|---|---|---|
| Runs | 1 040 | 5 200 | 11 440 |
| This box, GPU | ~3.8 days | ~19 days | ~42 days |
| Colab A100, compute units | ~260–1 300 | ~1 300–6 500 | ~2 900–14 300 |
| Colab L4, compute units | ~200–1 900 | ~1 000–9 500 | ~2 300–21 000 |

### 5.5 Week before a Thursday start, with Colab

| Day | Control only | Control + 4 EDA | Control + 10 EDA |
|---|---|---|---|
| Mon | A1 + A4 on a Colab CPU runtime (~1–2 h); prefetch MARS waves | + A2 CDS requests queued | + A2 |
| Tue | B on one GPU runtime: done within ~2–7 h | A3, then B on 2–3 runtimes (~4–12 h wall) | A3, then B on 3 runtimes (~4–27 h wall) |
| Wed | C1 locally (~10 min) | B continues if needed; C1 | B continues; C1 |
| Thu | real-time run locally; C2 applies the tables | same | same |

The control-only size fits a single Monday–Tuesday. 100 runs fit Tuesday–Wednesday on 2–3
runtimes. 220 fit only if the L4 turns out fast or A100 runtimes are available; otherwise
build those dates ahead of time, once, and reuse them every year.

## 6. Test plan (run first)

| # | Test | Measures | Pass mark |
|---|---|---|---|
| 1 | Install the CLI here, log in, `colab run` a hello-world on CPU and on `--gpu L4` | CLI works from this box; runtime start time | runs and returns output |
| 2 | **Phase A on a CPU runtime:** build `20241105_0000` with `era5_state_pkl.py`, write the pkl to the bucket | ARCO fetch time inside GCP (here: 410–470 s); MARS access from Colab; GCS write time | pkl identical in fields and shapes to the local build of the same date (112 fields, (2, 542080)); max abs difference 0 |
| 3 | **Bootstrap on L4:** install the env and flash-attn wheel, fetch the checkpoint | setup time per runtime | `run_local_icechunk_v2.py --lead-time 72` on that pkl finishes |
| 4 | **One full member on L4**, then the same on A100 (and try T4) | s per step, peak GPU memory, compute units per hour (Colab resource panel), wall time per member including extraction | products match a local run statistically: the members differ by the model's random draws, so compare weekly means and indices against the 20-year spread, not value by value |
| 5 | **Resumable worker:** 3 members in one runtime, kill it after the first, restart | claim/skip logic, DONE markers | members 2–3 run once, member 1 is skipped |

Decision after test 4:

| L4 time per member | Choice |
|---|---|
| ≲ 8 min | Colab L4 for the one-off 52-week build (several runtimes in parallel); this box for the weekly steady state |
| 8–15 min | A100 for the build if its cost per member is lower; otherwise this box when free plus L4 for overflow |
| ≳ 15 min | keep GPU work local; use Colab only for Phase A |

## 7. Colab limits to design around

- **The GPU type is not guaranteed;** runtimes have lifetime and idle limits; Pro runs only a
  few runtimes at once. Hence the claim-and-skip worker and DONE markers in the bucket.
- **Setup per runtime** (pip env, flash-attn wheel matched to torch 2.7 with the C++11 ABI,
  checkpoint download) is perhaps 5–10 min. Process many members per runtime.
- **Credentials** (CDS token, ECMWF Web API key, GCS service account, `HF_HOME` cache) are
  passed at run time through Colab secrets or `colab upload`. Never into the repo or the
  bucket.
- **Icechunk on GCS** uses `icechunk.gcs_storage`; one writer per store, or per member
  branch, to avoid commit conflicts between parallel runtimes.
- **Same code everywhere.** Phase A and B run the repo's own scripts (`era5_state_pkl.py`,
  `run_local_icechunk_v2.py`, `hc_extract.py`, ...), installed from a checkout or a generated
  bundle like `make_colab_bundle.py` does, so cloud and local results cannot drift apart.

## 8. Needed before test 1

- The Colab CLI installed and logged in on this box (the login is interactive and needs the
  account owner).
- The bucket: the existing `gs://aifs-aiquest-us-20251127` (service account
  `coiled-data.json`) or a new one.
- Approval to spend Pro compute units on tests 1–5 (under an hour of GPU).

## Sources

- [googlecolab/google-colab-cli: run command doc](https://github.com/googlecolab/google-colab-cli/blob/main/docs/05_run_command.md)
- [MarkTechPost: Google's new Colab CLI](https://www.marktechpost.com/2026/06/06/googles-new-colab-cli-lets-developers-and-ai-agents-run-python-on-remote-colab-gpus-and-tpus-from-the-terminal/)
- [ai-tldr.dev: Colab CLI](https://ai-tldr.dev/tools/google-colab-cli/)
- [ChatForest: Colab CLI builder guide](https://chatforest.com/builders-log/google-colab-cli-gpu-tpu-terminal-agent-builder-guide/)
- [Common Behavior Data: Colab CLI tips](https://x.com/CommonBehavior_/article/2094814274422243496)
- [note.com: Trying out Google Colab CLI](https://note.com/npaka/n/n3ad5e2331e00?hl=en)
- [McCormick: Colab GPUs features and pricing](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/)
- [varlog-info: Colab compute unit consumption](https://varlog-info.pages.dev/colab-credit-consumption-comparison-2024/)

## 9. Measured on Colab, 2026-11-12 (5 Nov, control)

Scripts in `colab/`: `hc_colab.sh` (local driver: up / push / creds / phaseA / setup /
worker / stop), `hc_phaseA.py` (build pkls → source.coop), `hc_gpu_setup.sh` (GPU env),
`hc_gpu_worker.py` (claim-and-skip worker around the unchanged `hc_member.sh`), `sc_io.py`
(source.coop I/O). `era5_state_pkl.py`, `hc_member.sh` and `hc_extract.py` take their paths
from the environment, with the GPU box paths as defaults.

### 9.1 Where things live

| | Path |
|---|---|
| Input pkls (temporary, delete after the members run) | `s3://e4drr-project/forecasts/fp16FahamuAIFSv2/hc-inputs-tmp/<MMDD>/<YYYY>/input_state_member_<NNN>.pkl` |
| Products, DONE marker last | `s3://e4drr-project/forecasts/fp16FahamuAIFSv2/hc-products/<MMDD>/<YYYY>/m<NNN>/` |
| Endpoint | `https://data.source.coop` (reads anonymous; the repository is public) |

Credentials: source.coop S3 credentials from `source-coop creds`, the MARS key from
`.ecmwfapirc`. Both are piped over ssh into `/root/` on the runtime (mode 600), never into
`colab exec` code (the Colab CLI records that in `~/.config/colab-cli/history/`). The S3
credentials last ≤ 12 h: re-run `hc_colab.sh creds NAME` before they expire on long runs.

### 9.2 Phase A (CPU high-RAM runtime, 0.26 CU/h)

| | Colab | GPU box (§5.3) |
|---|---|---|
| ARCO, 100 fields × 2 times | **13 s** | 410–470 s |
| regrid to N320 | 7 s | |
| MARS ecWAM waves (2 × 8.8 MB, disk, not tape) | ~50 s incl. queue | seconds to 3 h |
| **one state, total** | **80–145 s** | ~16 min |
| upload 971 MB to source.coop | 16–35 s | |
| anonymous download to a Colab VM | 6–7 s | |
| 20 states, 4 at a time | **~12 min wall, < 0.1 CU** | ~5.3 h |

All 20 pkls (2006–2025) are 971 413 699 bytes, 112 fields of (2, 542080), float64, wave
land 30.3 %. The reference wave mask (`ref_wave_land_n320.npy`) was rebuilt on Colab from
the 20261029 IFS open-data member 1, as in the runbook (land 30.29 %).

Two fixes were needed: (1) `arco_fields` no longer uses dask, because recent dask materialises
the 1.3 M-chunk graph and was OOM-killed at 52 GB; it now reads with xarray lazy indexing, one
read per (variable, time) with all 14 levels from the one 37-level chunk; (2) whole-upload
retries, after one HTTP 520 from source.coop in 20 uploads.

Not yet checked: the pass mark of test 2 (pkl identical to the GPU box build of the same
date). The GPU box pkls were deleted after the run; rebuild one there, or compare the 3a
files (§9.3).

### 9.3 Phase B (L4, 1.94 CU/h)

| | Measured |
|---|---|
| runtime start | 4 s |
| bootstrap: uv venv Python 3.12, torch 2.7.0+cu126, anemoi 0.8.3, flash-attn 2.7.4.post1, icechunk 2.1.0, checkpoint | **80 s** (pip 40, flash-attn 19, checkpoint 21) |
| pkl download | 7 s |
| rollout 792 h (O96 all steps + N320 sidecar 432–792 h), incl. model load | **487 s** (~3.6 s per step; GPU box 290 s) |
| extraction: 3a + TS days | 34 s |
| upload products | 32 s |
| **one member, wall** | **9.4 min = 0.30 CU ≈ $0.03** |
| peak GPU memory | 11 634 MiB of 23 034 |
| stores before deletion | O96 2.0 GB, N320 1.0 GB |

MJO (`vpm_index.py`) and the indices (`hc_extract.py`) were **skipped**: their inputs
(`vpm_clim_1991_2020.npz`, `rmm_basis.npz`, the AI-WQ `land_sea_mask_1pt5DEG.nc`) are
only on the GPU box. Copy them to the runtime's `/content/aux/` to run all 15 products.

The 1.5° file is written as uncompressed float64: 209 MB (the "41 MB" in §3 is the size on
the GPU box's compressed filesystem). The worker now applies lossless zlib before upload.

### 9.4 Cost from the measured L4 numbers (0.30 CU per member, ~$0.10 per CU)

| | Members | L4 hours | CU | Cost | Wall time, 1 / 3 runtimes |
|---|---|---|---|---|---|
| Control only, one date | 20 | 3.1 | 6 | $0.6 | 3.1 h / 1.1 h |
| Control + 4 EDA, one date | 100 | 15.7 | 30 | $3 | 16 h / 5.3 h |
| **Control + 10 EDA, one date** | **220** | **34** | **67** | **$6.7** | **34 h / 11.5 h** |
| All 52 dates × 220 | 11 440 | 1 790 | 3 480 | $350 | |
| Phase A per date (CPU) | | | < 1 | < $0.1 | |

Decision (§6): L4 at 9.4 min per member is at the edge of the 8–15 min band. An A100
(6.73 CU/h) would be cheaper per member only below ~2.7 min per member. **L4 is the GPU to
use.** The balance on 2026-11-12, 462.9 CU, covers about 6 dates at 220 members.

Before the 220-member run: the 200 perturbed pkls need `era5_eda_perturb.py` (A2–A3, not
written). The 19 control years not run here already exist from the GPU box run (§5.3).

### 9.5 Commands, as run

```bash
B=bc-fp16FahamuAIFSv2/colab
$B/hc_colab.sh up hcA cpu;  $B/hc_colab.sh push hcA;  $B/hc_colab.sh creds hcA
# once per runtime: reference wave mask from the IFS 20261029 open-data member 1
#   (s3_grib_pkl_input_aifsens_v2_proto.py --date 20261029_0000 --member 1 --source gcs,
#    then np.save('/content/bc/ref_wave_land_n320.npy', ~isfinite(swh[1])))
$B/hc_colab.sh phaseA hcA 1105 2025-2006 4        # log /content/phaseA_1105.log
$B/hc_colab.sh stop hcA

$B/hc_colab.sh up hcB L4;  $B/hc_colab.sh push hcB;  $B/hc_colab.sh creds hcB
$B/hc_colab.sh setup hcB                          # log /content/setup.log, "SETUP OK"
$B/hc_colab.sh worker hcB 1105 --years 2025 --members 1 --max 1   # log /content/worker_1105.log
$B/hc_colab.sh stop hcB
```

Always end with `stop` and check `colab usage` shows 0.00/hr: five CPU sessions left
running from 5 to 11 Nov cost ~130 CU.

## 10. Measured cost of a full cycle, and the cost model (12 Nov 2026)

From the 12 Nov cycle (`RUNBOOK_20261112.md`): 220 hindcast + 51 real-time members on Colab,
starting states on source.coop, corrected entry submitted as `bcfp16FahamuAIFSv2`. Compute units
(CU) at ~$0.10 each (100 for $9.99). Balance 454.31 → 325.95 = **128.4 CU ≈ $12.8**.

### 10.1 Where the compute units went

| Item | Hours | CU | ≈ $ |
|---|---|---|---|
| 3 × L4 billed (2026-11-14 15:44 → 11-15 12:30 UTC, 20.8 h each) | 62.5 GPU-h | 121.2 | 12.1 |
| ↳ running members: 271 × 8.8 min (incl. pkl download and product upload) | 39.8 | 77.2 | 7.7 |
| ↳ idle after the last member: two L4s ~4.6 h each, the third ~0.8 h | 10.0 | 19.4 | 1.9 |
| ↳ idle during the run: waiting for hindcast pkls (MARS tape) and, on one L4, ~3.2 h for 2 stale member claims to expire (3 h claim TTL) | 12.7 | 24.6 | 2.5 |
| CPU runtime 1: pkls, EDA, truth (ended by Colab at ~24 h) | ~24 | 6.3 | 0.6 |
| CPU runtime 2: truth rebuild, bundle, evaluation, upload | ~3.9 | 1.0 | 0.1 |
| **Total** | | **128.4** | **12.8** |

Without the idle time: **~80 CU (~$8)** — 77 GPU + ~2 CPU (~8.5 h of CPU actually needed).
Project Colab spend to 2026-11-18: 138 CU (128 for 12 Nov, ~10 for the 5 Nov tests and the
inflation A/B test).

### 10.2 Storage on source.coop (measured 2026-11-17)

| Path | Size | Kind |
|---|---|---|
| `hc-inputs-tmp/1112/` | **283.5 GB** | 271 pkls × 0.97 GB = 263 GB + EDA GRIBs 20.3 GB; transient |
| `hc-products/1112/` | 8.9 GB | 271 members × ~31 MB + truth + bundle; keep |
| `hc-inputs-tmp/1105/` | 230.1 GB | paused 5 Nov run; transient |
| `hc-products/1105/`, `hc-products-abtest/` | 0.8 GB | keep |
| **Total** | **~523 GB** | |

source.coop is free for this today. If it were paid object storage:

- storage of one cycle's inputs for a week: 284 GB ≈ **$1.5** (S3 Standard, $0.023/GB-month);
- **reading them back dominates**: the L4s read 263 GB once; at S3 internet egress ($0.09/GB)
  that is **~$24 per cycle, three times the GPU cost**; a GCS bucket costs $0.02–0.08/GB
  depending on the region Colab assigns (Taiwan, Singapore and us-west4 were seen), ~$5–21;
- kept products: 9 GB per cycle → ~460 GB per year (~$10/month on S3 by the end of the year);
  the compact bundle alone (~0.4 GB per cycle) → ~20 GB per year.

### 10.3 Cost model for bc-fp16FahamuAIFSv2

Unit costs measured: **0.285 CU per member** (8.8 min on L4 at 1.94 CU/h); ~2 CU of CPU per
cycle (~8.5 h at 0.26 CU/h: pkls, EDA, truth, bundle, evaluation); 80 s of setup per L4.

| Scenario | Members per cycle | CU per cycle | $ per cycle | $ per year (52) | Wall time |
|---|---|---|---|---|---|
| **A. As run for 12 Nov:** full hindcast every week | 271 | ~80 | ~$8 | ~$420 | ~14 h on 3 L4 + ~5 h pkls + ~4 h evaluation |
| **B. Archive once:** 52 dates × 220 hindcast members | 11 440 once | ~3 330 once | ~$330 once | — | ~5 weeks of 3 L4 (part-time) |
| ↳ then weekly: 51 real-time + 11 new-year members (rolls the 20-year window) | 62 | ~20 | ~$2 | ~$100 | ~3 h on 3 L4 |
| ↳ or weekly with the real-time members from the GPU box: 11 members | 11 | ~5 | ~$0.5 | ~$25 | ~1 h |

Transient storage: A ~284 GB per cycle for a few days; B ~13 GB per week (62 pkls), the one-off
build ~284 GB per date if pkls are deleted as each date completes. Building the perturbed
members on the GPU runtime from the control + EDA files instead of storing them would cut the
inputs from ~284 GB to ~40 GB per date, for ~+6 CU per cycle (~1 min of CPU per member on L4).

### 10.4 Avoidable cost, and the fixes

| Cause | Cost on 12 Nov | Fix |
|---|---|---|
| L4 runtimes left on after their worker exited | 19.4 CU | stop each runtime as soon as its worker log shows `WORKER done` |
| stale CLAIM of a member waited out for up to 3 h | ~6 CU | worker claim TTL lowered from 3 h to 45 min (a member takes ~9 min) |
| L4s started before the hindcast pkls existed (MARS tape) | ~18 CU | start the L4s when the real-time and first hindcast years are on source.coop, or start one L4 first |
| CPU runtime ended at ~24 h with truth only on its disk | index truth (~120 CDS requests) lost | every product saved to source.coop when made |

### 10.5 Idle compute on the 12 Nov and 19 Nov cycles (estimated 2026-11-24)

From the member timings (`timing.json`), session start/stop times and the balance history.
12 Nov is measured; 19 Nov is estimated (CPU-runtime lifetimes and the exact crash times are
not recorded), ±10 CU.

| | 12 Nov | 19 Nov |
|---|---|---|
| Total spent | 128.4 CU | 182.0 CU (325.95 → 143.93) |
| L4 GPU-hours billed | 62.5 h | ~89 h |
| L4 hours running members | 39.8 h | ~40.5 h (271 + a few re-runs) |
| **Idle L4** | **22.7 GPU-h ≈ 44 CU** | **~49 GPU-h ≈ 95 CU** |
| Idle CPU runtime | ~5 CU | ~5 CU |
| **Idle total** | **~49 CU (~$5)** | **~100 CU (~$10)** |

Over the two cycles ~150 CU (~$15) went on idle runtimes, about as much as the ~157 CU of
useful GPU work.

| Cause | 12 Nov | 19 Nov | Status |
|---|---|---|---|
| L4s left on after their last member | 19 CU | — | fixed: `hc_watch_stop.sh` |
| all three workers crashed on a source.coop 520/525 (no retry); watcher waited for a summary line a crashed worker never writes | — | ~62 CU | fixed (d7d1348): `sc_io.get` retries, worker survives a failed member, watcher stops on any worker exit |
| waiting for pkls (MARS tape), stale 3 h claims, setup | ~18 CU | ~21 CU | claim TTL 45 min; start L4s when pkls exist |
| **gaps between members: picking checked each candidate's DONE marker with its own request** — ~1 min per member early on, ~5 min when ~200 members were done | ~7 CU | ~13 CU | **fixed 2026-11-24**: one listing of `hc-products/<MMDD>/` per pick (`status()` in `hc_gpu_worker.py`) |

Measured for the fix, on the finished 19 Nov products from the gateway: one listing of all
271 DONE markers took 4.8 s; the old per-member checks took 1.0 s each, ~276 s per pick.
The remaining avoidable cost per cycle is mainly L4s started before their pkls exist.
