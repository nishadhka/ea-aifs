# Running the hindcast on Google Colab (plan and test, 2026-11-11)

Plan for moving the hindcast (README §0.2, up to 220 members per date) from this box to
Google Colab, driven from the terminal with the Colab CLI and paid with Colab Pro compute
units. **Status: plan. Nothing has run on Colab yet.** §5 is the test to run first.

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
because its speed for this model is not measured; the test in §5 fixes that.

## 5. Test plan (run first)

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

## 6. Colab limits to design around

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

## 7. Needed before test 1

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
