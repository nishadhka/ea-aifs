# Run log — 20261112 (AIFS-ENS 2.0, tier B, 51 members)

**9/9 submitted and verified, on the last day of the window.** Three things make this cycle
worth reading: **the AI-WQ step failed for ~44 h because the wrong one of the two passwords in
`.env` was used** (found from the ECMWF forum, fixed, and then 3c took 5 minutes instead of
3 h 20 m); **the chain ran in three parts**; and **TS terciles were built locally with AI-WQ's
own code**, after checking it reproduces a published file exactly.

*Written 2026-11-20 from the cycle's logs. Second cycle with 51 members (see 20261105).*

---

## ⏱ End-to-end timing

| Step | What | Wall time | Output |
|------|------|-----------|--------|
| — | disk purge, 20260924 stores (`icechunk_o96`, `icechunk_n320_aiwq`, `input_states`) | 11-13 09:0x | free 62 GB → **304 GB**; MJO bands, CRMA layer 3, `nc_1p5deg`, `aiwq/`, TS/MJO files confirmed kept first |
| **0** | open-data pre-check | 11-13, seconds | **RUNNABLE** — 9/9 sfc, 2/2 sol, 6/6 pl, 14/14 levels, 11/11 wave |
| **1a** | pkls, members 1–50 | 11-13 09:03 → 09:32, **28.9 min** (**35.3 s/member**) | 50/50 (first launch at 09:02 with 50 members was stopped after 2 pkls to add member 51) |
| **1b** | member 51, the control (HRES `oper`/`wave` 0 h) | 09:32 → 09:33, 36.8 s | 112/112 fields |
| — | control check (`check_control_pkl.py`) | 09:33 | **0 flagged** (2t 0.49, msl 0.40, z_500 0.34, swh 1.14 of the pf spread) |
| **2** | tier-B rollout | 11-13 09:33 → 15:57, **6 h 24 m** | **51 written, 0 skipped, 0 failed** |
| **3a** | regrid 432–792 h → 1.5° | 15:57 → 16:06, 9.1 m | **51/51**, 0 failed |
| **3b** | quintiles, 1st try | 16:06 → 16:14 | **ABORT**: quintile edges for 20261130 / 20261207 not downloadable |
| **TS** | tracker, local TS terciles | 11-13 ~16:3x, 5 m | 8.3 tracks/member |
| **MJO** | VPM → RMM basis | 11-13 ~16:4x, 4 m | bands preserved |
| — | AI-WQ access lost, retried every 20 min | 11-14 15:39 → 11-15 12:00 | **61 tries, all failed** — wrong password (below) |
| **3b** | quintiles, with the `ecbox` password | 11-15 12:34 → 12:38 | edges downloaded on try 1; 6 arrays, validated |
| **3c** | submit tas/mslp/pr | 11-15 12:38 → 12:43, **5 m** | **Successful: 6, Failed: 0**, all "within competition time window" |
| **TS+MJO submit** | 3 files | 12:43 → 12:45 | **3/3** |
| **verify** | `check_submissions_paged.py` | 12:45 | **9/9 present** (414 files, 5 pages) |

Stores: **O96** (`CAKVSDQBM39ZWBXCM3D0`) **+ N320** (`AEZCVTMB91QJJP7SD080`), both tagged
`cycle-20261112_0000`, 51/51 members. Window closed 2026-11-15 23:59 UTC — **~11 h of
margin** at the end, after ~44 h lost.

**Step 2 was slow** (6 h 24 m against 4 h 44 m for 51 members on 20261105; mean 441 s per
member, member 51 took 1 126 s) because CPU-heavy work on the 20261105 stores ran alongside
it: TS days, MJO, `hc_extract.py` for 51 members, a 1.7 GB zip, and two TS-tercile builds.

---

## ⚠ The AI-WQ failure: two passwords in `.env`, the wrong one used

`fp16FahamuAIFSv2/.env` holds **two** AI-WQ passwords:

| key | what | state |
|---|---|---|
| `ecbox` (lowercase) | the ECBox password AI-WQ emailed on **2026-08-11** | **current** |
| `AIWQ_PASSWORD` | the old FTP password | the FTP was **fully closed 2026-09-04** (forum, below). It still answered until about 2026-11-12, then dropped every connection (`EOFError`) |

`forecast_submission_cli.py`, `submit_ts_mjo_cli.py` and `check_submissions_paged.py` read
`ecbox` first (`AIWQ_ECBOX_TOKEN` → `ecbox` → `AIWQ_PASSWORD`), which is why every earlier
submission worked. **`ensemble_quintile_analysis_cli.py` (3b) reads `AIWQ_PASSWORD`**, and so
did the chain's own download steps. While the FTP still answered, 3b got its quintile edges
over FTP; once it stopped, 3b fell back to ECBox with the old password and got
`403 "Not a Branca token"` — the signature of the wrong password, not an outage.

It was first read as an AI-WQ outage, then as a credentials change needing a new token. The
ECMWF forum thread
[Update your AI Weather Quest Python environment and review the new Terms and Conditions](https://forum.ecmwf.int/t/update-your-ai-weather-quest-python-environment-and-review-the-new-terms-and-conditions/15249)
settled it: new password emailed 2026-08-11, FTP closed 2026-09-04, package 3.32 stops calling
FTP. Team Fahamu is in that thread (2026-08-18). The `ecbox` value worked at the first try.

**The fix in the chain** (`chain_20261112_part3.sh`): export `AIWQ_PASSWORD` as the `ecbox`
value for every child process, read from `.env` and never printed:

```bash
export AIWQ_PASSWORD="$($PY -c 'from dotenv import dotenv_values; print(dotenv_values("/home/ezra/ea-aifs/fp16FahamuAIFSv2/.env")["ecbox"])')"
```

**Side effect: 3c took 5 minutes, not 3 h 20 m.** The 3.29 package tries the FTP before ECBox
at several points; each attempt used to hang until a timeout. With the FTP refusing at once,
only the ECBox calls remain. Package 3.32+ removes the FTP calls entirely; this cycle kept
3.29, the version that had worked, on deadline day.

**Rule for later cycles:** any AI-WQ download or submission uses the `ecbox` password. A 403
"Not a Branca token" means the wrong password.

---

## TS terciles built locally — with AI-WQ's own code

The tercile download failed for the same reason. `retrieve_20yr_quantile_clim` also has a trap:
it **opens the target file for writing before the transfer**, so a failed download leaves a
0-byte file in place of any existing one.

AI-WQ's package builds the terciles itself:
`TS_processing.compute_20yr_TStercile_climatology(date, ibtracs, savedir)` on the local
IBTrACS v04r01. **Checked first: it reproduces the published 20261123 file exactly**
(`[[0,0,0,0],[0,3,0,0]]`, identical). Built for 20261130 and 20261207 into
`/tank/projects/bc-aifs/ts_terciles_local/`.

| week | ATL | NWP | SWIO | SEIO |
|---|---|---|---|---|
| 2026-11-30 | 0 / 0 | 0 / 1 | 0 / 0 | 0 / 0 |
| 2026-12-07 | 0 / 0 | 0 / 1 | **0 / 1** | **0 / 1** |

---

## The chain in three parts

| part | script | ran | why it ended |
|---|---|---|---|
| 1 | `/tank/projects/chain_20261112.sh` | 11-13 09:03 → 11:12 | stopped (SIGKILL to the shell only) during Step 2, so the tercile step could get a fallback. **The rollout kept running** — killing the parent shell does not kill the child |
| 2 | `/tank/projects/chain_20261112_part2.sh` | 11:12 → 16:14 | waited for the rollout, ran 3a, **aborted at 3b** (quintile edges) |
| 3 | `/tank/projects/chain_20261112_part3.sh` | 11-14 15:39 → 11-15 12:45 | retried AI-WQ every 20 min; with the `ecbox` password: 3b → validate → TS+MJO dry run → 3c → TS+MJO submit → 9/9 |

The first launch at 11-13 09:02 (50 members) was stopped after 2 pkls, on the user's request
to include member 51, and restarted at 09:03 with `--skip-existing`.

TS days and MJO were run by hand on 11-13 after part 2's abort, with the local terciles.

---

## Commands (as run)

```bash
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
BASE=/tank/projects/aifs-run/20261112_0000

# disk: purge the 20260924 stores (authorised); products confirmed kept first
rm -rf /tank/projects/aifs-run/20260924_0000/{icechunk_o96,icechunk_n320_aiwq,input_states}

cd run-pre50r1-dates && $PY check_open_data_inputs.py --date 20261112      # RUNNABLE

# part 1: Step 1 (pf 1-50, then control 51) -> control check -> Step 2
bash /tank/projects/chain_20261112.sh > /tank/projects/chain_20261112.log 2>&1
#   s3_grib_pkl ... --date 20261112_0000 --members 1-50 --source gcs --skip-existing
#   s3_grib_pkl ... --members 51 --control-member 51 --source gcs --skip-existing
#   check_control_pkl.py $BASE/input_states 51
#   run_local_icechunk_v2.py --members 1-51 --n-members 51 --lead-time 792 --grid o96 \
#       --store $BASE/icechunk_o96 --native-store $BASE/icechunk_n320_aiwq \
#       --native-vars msl,tp,2t,10u,10v,t_200,t_300,t_500,u_850,v_850 \
#       --native-write-hours 432-792 --commit-every 1 --float-size f4 --skip-existing

# part 2: wait for the rollout -> 3a -> 3b (aborted: no quintile edges)
bash /tank/projects/chain_20261112_part2.sh > /tank/projects/chain_20261112_part2.log 2>&1

# TS terciles with AI-WQ's builder (checked against the published 20261123 file first)
$PY - <<'EOF'
import xarray as xr
from AI_WQ_package import TS_processing as T
ib = xr.open_dataset('/tank/projects/ibtracs/IBTrACS.ALL.v04r01.nc')
for d in ('20261130', '20261207'):
    T.compute_20yr_TStercile_climatology(d, ib, '/tank/projects/bc-aifs/ts_terciles_local')
EOF
cp /tank/projects/bc-aifs/ts_terciles_local/TS_*_2026113[0]*.nc \
   /tank/projects/bc-aifs/ts_terciles_local/TS_*_20261207.nc $BASE/ts_terciles/

$PY -u ts-mjo/ts_days.py --store $BASE/icechunk_n320_aiwq --tag cycle-20261112_0000 \
    --init 20261112 --aiwq-tercile-dir $BASE/ts_terciles \
    --out $BASE/ts_days_probs_20261112_tracked.nc
$PY -u ts-mjo/vpm_index.py --store $BASE/icechunk_o96 --init 20261112 \
    --clim /tank/projects/era5_vpm_clim/vpm_clim_1991_2020.npz \
    --eofs /tank/projects/era5_vpm_clim/rmm_basis.npz \
    --dump-bands /tank/projects/mjo_model_clim/bands_20261112.npz \
    --out $BASE/mjo_probs_20261112.nc

# part 3: AIWQ_PASSWORD := the ecbox value; quintile edges -> 3b -> validate -> dry run
#         -> 3c -> TS+MJO submit -> verify
bash /tank/projects/chain_20261112_part3.sh > /tank/projects/chain_20261112_part3.log 2>&1
#   ensemble_quintile_analysis_cli.py --date 20261112 --v2 --local-nc-dir $BASE/nc_1p5deg --work-dir $BASE/aiwq
#   submit_ts_mjo_cli.py --date 20261112 --fill-rule bounds ... --dry-run
#   ../shared/forecast_submission_cli.py --date 20261112 --v2 --output-dir $BASE/aiwq
#   submit_ts_mjo_cli.py --date 20261112 --fill-rule bounds \
#       --ts-nc $BASE/ts_days_probs_20261112_tracked.nc --tercile-dir $BASE/ts_terciles \
#       --mjo-nc $BASE/mjo_probs_20261112.nc
#   check_submissions_paged.py --date 20261112
```

Validation before 3c: 6 arrays `(2, 5, 121, 240)`, NaN 0, ⊂ [0,1], Σ = 1; weeks
2026-11-30 / 2026-12-07. Global mean P(Q5) for `2t` **0.422** (3b's own unweighted grid
mean, comparable with 20261029's 0.423 and 20261105's 0.412); latitude-weighted it is
0.527 / 0.526 for weeks 3 / 4.

---

## TS days — first scored week for SWIO and SEIO this season

8.3 tracks per member. What went up (`--fill-rule bounds`):

| week | basin | bounds | mean | below | near | above | note |
|---|---|---|---|---|---|---|---|
| 2026-11-30 | ATL | 0 / 0 | 1.1 d | 0.33 | 0.33 | 0.33 | degenerate → uniform, masked |
| | **NWP** | 0 / 1 | 2.8 d | 0.00 | 0.47 | **0.53** | scored |
| | SWIO | 0 / 0 | 0.3 d | 0.33 | 0.33 | 0.33 | degenerate → uniform, masked |
| | SEIO | 0 / 0 | 0.2 d | 0.33 | 0.33 | 0.33 | degenerate → uniform, masked |
| 2026-12-07 | ATL | 0 / 0 | 0.7 d | 0.33 | 0.33 | 0.33 | degenerate → uniform, masked |
| | **NWP** | 0 / 1 | 0.6 d | 0.00 | **0.71** | 0.29 | **off-table but scoreable** |
| | **SWIO** | 0 / 1 | 0.6 d | 0.00 | **0.78** | 0.22 | scored |
| | **SEIO** | 0 / 1 | 0.1 d | 0.00 | **0.94** | 0.06 | scored |

**Four scored columns**, the most yet. The week of 7 Dec is the first with real SWIO/SEIO
bounds this season, and AI-WQ's month table switches to SWIO/SEIO for December, so NWP is
now the off-table column, scored by its bounds.

## MJO — the submitted phase-8 run ends

| lag | valid | modal category |
|---|---|---|
| +7 | 2026-11-19 | **7** |
| +14 | 2026-11-26 | 8 |
| +21 | 2026-12-03 | 8 |
| +28 | 2026-12-10 | **1** (P(inactive) 0.098) |

The first *submitted* modal sequence that is not 8/8/8/8 since 20261015. 20261105 (made
after its window, not submitted) already ended on phase 1: 8/8/8/1, with `P(inactive)`
0.02 / 0.235 at +21 / +28. The forecast's own per-field σ is **1.26–1.43×** the basis
(1.39–1.56× on 20261029): still over-amplified, `MJO_METHOD.md` §5 unchanged.

Bands at `/tank/projects/mjo_model_clim/bands_20261112.npz`.

---

## Server metadata for the nine

```
MJO_20261112_p1_Fahamu_fp16FahamuAIFSv2.nc        11,061 B   2026-11-15T12:45:51Z
TS_20261112_p1_Fahamu_fp16FahamuAIFSv2.nc         12,029 B   2026-11-15T12:44:02Z
TS_20261112_p2_Fahamu_fp16FahamuAIFSv2.nc         12,029 B   2026-11-15T12:44:50Z
mslp_20261112_p1_Fahamu_fp16FahamuAIFSv2.nc    1,182,087 B   2026-11-15T12:39:03Z
mslp_20261112_p2_Fahamu_fp16FahamuAIFSv2.nc    1,182,087 B   2026-11-15T12:39:51Z
pr_20261112_p1_Fahamu_fp16FahamuAIFSv2.nc      1,182,081 B   2026-11-15T12:40:49Z
pr_20261112_p2_Fahamu_fp16FahamuAIFSv2.nc      1,182,081 B   2026-11-15T12:41:37Z
tas_20261112_p1_Fahamu_fp16FahamuAIFSv2.nc     1,182,123 B   2026-11-15T12:42:26Z
tas_20261112_p2_Fahamu_fp16FahamuAIFSv2.nc     1,182,123 B   2026-11-15T12:43:14Z
```

---

## Related

- [`run_commands_20261105.md`](run_commands_20261105.md) — the first 51-member cycle, and the TS/MJO it missed.
- [`run_commands_20261029.md`](run_commands_20261029.md) — the `--fill-rule bounds` decision.
- [`fp16FahamuAIFSv2_20261112.txt`](fp16FahamuAIFSv2_20261112.txt) — the assembled logs of this cycle.
- [`ts-mjo/TS_STORM_DAYS.md`](ts-mjo/TS_STORM_DAYS.md), [`ts-mjo/MJO_METHOD.md`](ts-mjo/MJO_METHOD.md).
