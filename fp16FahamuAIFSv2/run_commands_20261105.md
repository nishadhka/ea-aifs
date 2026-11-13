# Run log — 20261105 (AIFS-ENS 2.0, tier B, 51 members)

**6/9 submitted and verified: tas, mslp, pr. TS and MJO were not run or submitted.** Two
things make this cycle different from every earlier one: **51 members instead of 50** (a
51st, unperturbed control member was added), and **the cycle ran two days late**,
after a disk purge, so 3c went in about 7 h before the window closed.

*Written 2026-11-13 from the cycle's logs; the cycle ran 2026-11-07 → 11-08.*

---

## ⏱ End-to-end timing

| Step | What | Wall time | Output |
|------|------|-----------|--------|
| **0** | open-data pre-check | **not run** | — |
| — | disk purge, 20260917 stores (`icechunk_o96`, `icechunk_n320_aiwq`, `input_states`) | 11-07 | free 67.5 GB → **323 GB**; `nc_1p5deg`, `aiwq/`, TS/MJO files kept |
| **1a** | control pkl, member 51 (HRES `oper`/`wave` 0 h) | 11-07, **39.5 s** | 112/112 fields, verify OK |
| **1b** | pkl creation, members 1–50 | 11-07 19:27 → 20:01, **34 min** (~41 s/member) | **51/51** pkls |
| — | control sanity check (`check_control_pkl.py`) | 20:01 → 20:02 | **ABORT on `sd`** (false alarm, below); check fixed, chain rerun 20:04 |
| **2** | tier-B rollout | 11-07 20:04 → 11-08 00:48, **4 h 44 m** | **51 written, 0 skipped, 0 failed** |
| **3a** | regrid 432–792 h → 1.5° | 00:48 → 00:57, 9.1 m | **51/51**, 0 failed |
| **3b** | quintiles | 00:57 → 00:59 | 6 arrays from **51** NetCDF files |
| — | `input_states` deleted by the chain after Step 2 | 00:59 | ~50 GB freed |
| **3c** | submit tas/mslp/pr | 11-08 13:57 → 17:17, **3 h 20 m** | **Successful: 6, Failed: 0**, all "within competition time window" |
| **TS** | tracker | **not run** | — |
| **MJO** | VPM → RMM | **not run** | no `bands_20261105.npz` |
| **verify** | `check_submissions_paged.py` (run 2026-11-13) | seconds | **6/9 present** (338 files, 4 pages); TS p1, TS p2, MJO p1 missing |

Stores: **O96 165 GB + N320 53 GB**, both tagged `cycle-20261105_0000`. Window closed
2026-11-08 23:59 UTC — **~6.7 h of margin** for 3c.

The chain ran as a harness-tracked background job (`/tank/projects/chain_20261105.sh`, log
`/tank/projects/chain_20261105.log`), so its end was noticed at once, both after the abort and
after 3b.

---

## Member 51: the unperturbed control

Every earlier cycle used members 1–50, the perturbed members (`pf`) from the IFS open-data
`enfo` files. **This cycle adds member 51, the unperturbed control.**

| | |
|---|---|
| Why | to match the hindcast design in `../bc-fp16FahamuAIFSv2/` (control + perturbed members) |
| Problem | open data publishes **no ENS control (`cf`) at step 0**: the `enfo-ef` and `waef-ef` files hold only the 50 `pf` members (8 850 = 50 × 177 records, checked 20261104 18z and 20261105 00z) |
| Source used | the HRES files at step 0: `oper` (atmosphere, land) and `wave`. They carry every field the `pf` build reads (checked against both indexes). Since IFS cycle 48r1, HRES and the ENS control share the 9 km resolution and the same analysis |
| Numbered 51, not 0 | the run script and 3a index members as `member − 1` from 1..N |
| Code | `s3_grib_pkl/s3_grib_pkl_input_aifsens_v2_proto.py --control-member 51` (commit `999003c`): member 51 reads `oper`/`wave`, looked up with `number=None` |
| Check | `../bc-fp16FahamuAIFSv2/check_control_pkl.py` compares the control with 10 `pf` members field by field and gates Step 2 |

**Control sanity check: one false alarm.** The first run aborted on `sd` (snow depth): the
control's global mean was 0.99227 against 0.99241 for the `pf` members, a 0.014 % (0.14 mm
water equivalent) difference that scored 5.6 "spreads" only because the `pf` members agree on
global-mean snow to ±0.00003. Gridpoint rms was 0.75 of the `pf` spread, the mask identical.
The check now ignores global-mean offsets below 0.1 % of the field; everything passed
(2t 0.49, msl 0.42, z_500 0.36, swh 1.12 of the spread).

**Member 51 in the forecast.** Weeks 3–4 global means of member 51 against the 50 `pf`
members: 2t +0.86, msl −0.38, tp −1.34 standard deviations. Inside the ensemble.

**Consequences.**
- Probabilities move in steps of 1/51, not 1/50. The control is the unperturbed analysis, not
  one more random draw, but it is counted with equal weight. The effect is small (1 of 51), but
  20261105 is not strictly comparable with the 50-member cycles before it.
- AI-WQ accepted the file without complaint (format checks only).
- **Decision for 20261112 onward: 51 members, or back to 50 for continuity?**

---

## Commands (as run)

```bash
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
BASE=/tank/projects/aifs-run/20261105_0000

# disk: purge the 20260917 stores, keep products (authorised 2026-11-07)
rm -rf /tank/projects/aifs-run/20260917_0000/{icechunk_o96,icechunk_n320_aiwq,input_states}

# 1a. the control, member 51, from HRES oper/wave 0 h
mkdir -p $BASE/input_states
cd /home/ezra/ea-aifs
$PY -u s3_grib_pkl/s3_grib_pkl_input_aifsens_v2_proto.py --date 20261105_0000 \
    --members 51 --control-member 51 --source gcs --out $BASE/input_states

# 1b → 2 → 3a → 3b, chained (51 members; symlink convention for proto_ names;
#   control check before Step 2; input_states deleted after Step 2)
bash /tank/projects/chain_20261105.sh > /tank/projects/chain_20261105.log 2>&1
#   inside: s3_grib_pkl ... --members 1-50 --source gcs --skip-existing
#           check_control_pkl.py $BASE/input_states 51
#           run_local_icechunk_v2.py --members 1-51 --n-members 51 --lead-time 792 \
#               --grid o96 --store $BASE/icechunk_o96 --native-store $BASE/icechunk_n320_aiwq \
#               --native-vars msl,tp,2t,10u,10v,t_200,t_300,t_500,u_850,v_850 \
#               --native-write-hours 432-792 --commit-every 1 --float-size f4 --skip-existing
#           aifs_n320_grib_1p5defg_nc_cli.py --members 1-51 ...      (3a)
#           ensemble_quintile_analysis_cli.py --date 20261105 --v2 ... (3b)

# 3c. dry run, then live
cd /home/ezra/ea-aifs/fp16FahamuAIFSv2
$PY -u ../shared/forecast_submission_cli.py --date 20261105 --v2 --output-dir $BASE/aiwq --dry-run
$PY -u ../shared/forecast_submission_cli.py --date 20261105 --v2 --output-dir $BASE/aiwq \
    > /tank/projects/step3c_1105_live.log 2>&1

# TS and MJO: NOT run this cycle (see below)

$PY check_submissions_paged.py --date 20261105      # 6/9
```

Validation before 3c: 6 arrays `(2, 5, 121, 240)`, NaN 0, Σ = 1 everywhere (to 1e-15);
`prepare_aiwq_submission` read all 6 at `(5, 121, 240)`; dry run picked
`aiwq/ensemble_quintile_probabilities_20261105_v2.nc`, team Fahamu, model fp16FahamuAIFSv2.
Weeks 2026-11-23 / 2026-11-30. Global mean P(Q5) for `2t` **0.514** (week 3) and **0.511**
(week 4) — a stronger warm shift than 20261029's 0.423.

---

## TS days and MJO — not run, not submitted

The chain stops at 3b by design; TS, MJO and their submission were never started this cycle.
No `ts_days_probs_20261105_tracked.nc`, no `mjo_probs_20261105.nc`, no
`/tank/projects/mjo_model_clim/bands_20261105.npz`. The window has closed, so the three
files cannot be submitted now. The stores still exist, so the products (and the MJO bands for
the model-climatology set) can be made afterwards for the record:

```bash
$PY -u ts-mjo/ts_days.py --store $BASE/icechunk_n320_aiwq --tag cycle-20261105_0000 \
    --init 20261105 --out $BASE/ts_days_probs_20261105_tracked.nc
$PY -u ts-mjo/vpm_index.py --store $BASE/icechunk_o96 --init 20261105 \
    --clim /tank/projects/era5_vpm_clim/vpm_clim_1991_2020.npz \
    --eofs /tank/projects/era5_vpm_clim/rmm_basis.npz \
    --dump-bands /tank/projects/mjo_model_clim/bands_20261105.npz \
    --out $BASE/mjo_probs_20261105.nc
```

**Rule for later cycles:** the chain's "CHAIN COMPLETE through 3b" is not the end of the
cycle; TS and MJO must follow before the window closes.

---

## Also in this cycle folder (not submitted)

| Path | What |
|---|---|
| `aiwq_bc/ensemble_quintile_probabilities_20261105_bc.nc` | tas/mslp/pr corrected with the bias of past real-time cycles (`../bc-fp16FahamuAIFSv2/past_cycle_bc.py`). Superseded |
| `aiwq_bc_hc/ensemble_quintile_probabilities_20261105_bchc.nc` | corrected from the 20-year same-date control hindcast (`../bc-fp16FahamuAIFSv2/hc_bc.py`, setting `sig_a0.5`). Changes the modal quintile in 3–4 % of cells |
| `products15/` | per-member `indices_mNNN.npz` (Africa / Indian Ocean indices from `hc_extract.py`), created 2026-11-12 06:41, outside this cycle's chain |

---

## ⚠ Disk — 20261112 cannot run without a purge

After this cycle: 104 GB free. Since then 28 GB of operational IFS (2024-09-23 → 12-31) went to
`/home/ezra/IFS-Forecasts/2024/` (same pool), leaving **~62 GB** against ~260 GB for a
51-member cycle. A purge must be authorised before Step 2. Note that the study plan
(`../paper/AIFS_ENS_V2_S2S_EXTREMES_STUDY_PLAN.md` §3.1) asks to keep `input_states` and the
stores of 20260917 → 20261029 for RQ1; the 20260917 purge above already broke that for one
cycle (its IFS pkls can be refetched from open data, its run redone in ~5 h).

---

## Related

- [`run_commands_20261029.md`](run_commands_20261029.md) — the previous cycle (50 members, 9/9).
- [`../bc-fp16FahamuAIFSv2/README.md`](../bc-fp16FahamuAIFSv2/README.md) — why the control member, the hindcast and the corrected files.
- [`../bc-fp16FahamuAIFSv2/runbook_20261111.md`](../bc-fp16FahamuAIFSv2/runbook_20261111.md) — the 20-year hindcast for 5 Nov.
- [`ts-mjo/MJO_METHOD.md`](ts-mjo/MJO_METHOD.md), [`ts-mjo/TS_STORM_DAYS.md`](ts-mjo/TS_STORM_DAYS.md) — the products not run this cycle.
