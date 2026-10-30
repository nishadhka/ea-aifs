# Run log — 20261029 (AIFS-ENS 2.0, tier B)

9/9 submitted and verified. Two things make this cycle worth reading: **Step 1 ran 3.5× faster
than any previous cycle**, and the open question left standing on 20261022 — whether AI-WQ's
basin activation is keyed on the calendar month or on the published bounds — was **resolved
from the scoring code and the answer changed what we submit**.

---

## ⏱ End-to-end timing

| Step | What | Wall time | Output |
|------|------|-----------|--------|
| **0** | open-data pre-check | seconds | **RUNNABLE** — 9/9 sfc, 2/2 sol, 6/6 pl, 14/14 levels, 11/11 wave |
| **1** | pkl creation | 10-29 13:39 → 14:13, **33.8 min** (**39.8 s/member**) | 38 GB, **50/50** |
| **2** | tier-B rollout | 14:13 → 18:53, **4 h 40 m** | **50 written, 0 skipped, 0 failed** |
| **3a** | regrid 432–792 h → 1.5° | 18:53 → 19:02, 8.3 m | 50/50, 0 failed |
| **3b** | quintiles | 19:02 → 19:04 | 6 arrays |
| **3c** | submit tas/mslp/pr | 19:05 → 22:24, 3 h 19 m | **Successful: 6, Failed: 0** |
| **TS** | tracker on the N320 sidecar | 19:05 → 19:10, 5 m | 8.7 tracks/member |
| **MJO** | VPM → RMM-fitted basis | 19:05 → 19:09, 4 m | bands preserved |
| **TS+MJO submit** | 3 files | 10-30 05:4x → 07:22 | **3/3** |
| **verify** | `check_submissions_paged.py` | seconds | **9/9 present** (68 files, 1 page) |

Stores: **O96 158 GB** (`WB3SB9PQ94A0JWB2XVWG`) **+ N320 51 GB** (`PVT5E251QFEC1FKCS2E0`),
both tagged `cycle-20261029_0000`. Window closed 2026-11-01 23:59 UTC — **~41 h of margin**.

The chain was launched with a **harness-tracked waiter armed in the same breath**, per the rule
20261022 forced, and 3b's completion was noticed within seconds instead of 41 hours later.

### Step 1 at 39.8 s/member

| cycle | s/member | Step 1 wall |
|---|---|---|
| 20261015 | 187.4 | 2 h 37 m |
| 20261022 | 138.1 | 1 h 56 m |
| **20261029** | **39.8** | **33.8 min** |

Verified genuine rather than a skipped-work artefact: all 50 members logged
`verify: 112/112 fields, shapes={(2, 542080)}, missing=[], extra=[] -> OK`, and the output is
50 files × 971 MB = 38 GB, the same as every other cycle. Nothing in the fetch code changed,
so this is GCS-side throughput.

---

## The basin-activation question, resolved: bounds, not the month

20261022 left this open and flagged it for resolution "before the December cycles". It is
resolved now, from AI-WQ's own package in
`site-packages/AI_WQ_package/`:

1. **`forecast_evaluation.calculate_RPS_TS` applies no month mask.** Its only mask is
   `nan_quantile_mask = obs_pbs.isnull().all(dim=obs_q_name)`.
2. Those NaNs originate solely in `conditional_obs_probs`, which blanks a basin-week when
   `all_equal = (quintile_bounds.min(dim=q) == quintile_bounds.max(dim=q))`.
3. So **scoring is keyed on the published bounds**. The month table exists only in
   `check_fc_submission.py` lines 249–259, and there it is used for
   `np.allclose(summed_values[active_mask], 1.0, atol=0.2)` — *active columns only*. A real
   forecast in an off-table column cannot be rejected by the checker.
4. `calculate_RPSS_TS` uses uniform 1/3 as its own climatological baseline, so a uniform 1/3
   submission in a scored column earns **RPSS exactly 0**.

Our `build_ts` gated on `active_basins(month)` — months 6–11 → ATL/NWP, 12/1/2 → SWIO/SEIO.
The correct gate is `lower != upper`.

### What changed in the code

`submit_ts_mjo_cli.py` gains `--fill-rule {month,bounds}`, **default `month`** so no past
behaviour changes silently. `bounds` forecasts every non-degenerate basin-week and annotates
each column: `[off-table but scoreable]` or `[degenerate bounds -> uniform 1/3]`.

Run this cycle with **`--fill-rule bounds`** on the user's decision.

### What it changed in the submission

One scored column: **SWIO at 2026-11-16**, bounds `0/2`.

| | under `month` | under `bounds` |
|---|---|---|
| SWIO 2026-11-16 | 0.33 / 0.33 / 0.33 | **0.00 / 0.84 / 0.16** |

That is the forecast 20261022 threw away in its SWIO equivalent, now submitted. Everything
else that scores is identical. The ATL columns move from `0/0/1` to uniform 1/3, but both
bounds are `0/0`, so the scorer NaNs them either way — cosmetic.

**First basin-week this pipeline has submitted outside AI-WQ's month table.**

---

## Commands (as run)

```bash
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
BASE=/tank/projects/aifs-run/20261029_0000

cd run-pre50r1-dates && $PY check_open_data_inputs.py --date 20261029

mkdir -p $BASE/input_states
$PY -u s3_grib_pkl/s3_grib_pkl_input_aifsens_v2_proto.py \
    --date 20261029_0000 --members 1-50 --source gcs \
    --out $BASE/input_states --skip-existing

bash /tank/projects/chain_20261029.sh     # rename -> 2 -> 3a -> 3b, COMPUTE ONLY
# ...with a tracked waiter armed alongside it, NOT a bare nohup:
until grep -qE "CHAIN COMPLETE through 3b|ABORT" /tank/projects/chain_20261029.log; do sleep 60; done

# terciles BEFORE ts_days.py -- see the 20261015 trap
$PY - <<'EOF2'
import os
from AI_WQ_package import retrieve_evaluation_data as red
for d in ("20261116", "20261123"):
    red.retrieve_20yr_quantile_clim(d, "TS", os.environ["AIWQ_PASSWORD"],
                                    local_destination=f"{os.environ['BASE']}/ts_terciles")
EOF2

$PY -u ../shared/forecast_submission_cli.py --date 20261029 --v2 --output-dir $BASE/aiwq
$PY -u ts-mjo/ts_days.py --store $BASE/icechunk_n320_aiwq --tag cycle-20261029_0000 \
    --init 20261029 --aiwq-tercile-dir $BASE/ts_terciles \
    --out $BASE/ts_days_probs_20261029_tracked.nc
$PY -u ts-mjo/vpm_index.py --store $BASE/icechunk_o96 --init 20261029 \
    --clim /tank/projects/era5_vpm_clim/vpm_clim_1991_2020.npz \
    --eofs /tank/projects/era5_vpm_clim/rmm_basis.npz \
    --dump-bands /tank/projects/mjo_model_clim/bands_20261029.npz \
    --out $BASE/mjo_probs_20261029.nc
$PY -u submit_ts_mjo_cli.py --date 20261029 --fill-rule bounds \
    --ts-nc $BASE/ts_days_probs_20261029_tracked.nc --tercile-dir $BASE/ts_terciles \
    --mjo-nc $BASE/mjo_probs_20261029.nc --dry-run     # then without
$PY check_submissions_paged.py --date 20261029
```

Validation before 3c: 6 arrays `(2, 5, 121, 240)`, NaN 0, ⊂ [0,1], Σ = 1.000000; weeks
2026-11-16 / 2026-11-23. Global mean P(Q5) for `2t` **0.423** — the warm shift, a little below
20261022's 0.441.

---

## TS days — NWP still the only basin with real bounds, and SWIO gets scored

8.7 tracks per member. Official bounds and what went up:

| week | basin | bounds | mean | below | near | above | note |
|---|---|---|---|---|---|---|---|
| 2026-11-16 | ATL | 0 / 0 | 0.6 d | 0.33 | 0.33 | 0.33 | degenerate → uniform, masked |
| | **NWP** | 0 / 2 | 4.7 d | 0.00 | 0.24 | **0.76** | scored |
| | **SWIO** | **0 / 2** | 0.5 d | 0.00 | **0.84** | 0.16 | **off-table but scoreable** |
| | SEIO | 0 / 0 | 0.1 d | 0.33 | 0.33 | 0.33 | degenerate → uniform, masked |
| 2026-11-23 | ATL | 0 / 0 | 0.5 d | 0.33 | 0.33 | 0.33 | degenerate → uniform, masked |
| | **NWP** | 0 / 3 | 2.6 d | 0.00 | **0.56** | 0.44 | scored |
| | SWIO | 0 / 0 | 0.3 d | 0.33 | 0.33 | 0.33 | degenerate → uniform, masked |
| | SEIO | 0 / 0 | 0.1 d | 0.33 | 0.33 | 0.33 | degenerate → uniform, masked |

Three scored columns this cycle, against two in every earlier cycle. NWP leans **above** again
at week 1 (0.76 on a 4.7-day mean against an upper bound of 2) and shifts to **near** at week 2
— the detector over-count that `TS_STORM_DAYS.md` documents is doing part of that work, and the
published bounds measure the bias rather than correct it.

`ts_days.py` printed its own degeneracy warning naming all five `0/0` basin-weeks, and
`--aiwq-tercile-dir` was passed, so its probabilities match the submission exactly.

## MJO — phase 8 for a third straight cycle, and the first hint of propagation

| lag | valid | modal phase | members in modal | mean amp | P(amp<1) |
|---|---|---|---|---|---|
| +7 | 2026-11-05 | 8 | 47/50 | 2.12 | 0.000 |
| +14 | 2026-11-12 | 8 | 48/50 | 2.87 | 0.000 |
| +21 | 2026-11-19 | 8 | 45/50 | 3.39 | 0.000 |
| +28 | 2026-11-26 | 8 | **29/50** (17 in phase 1) | 2.60 | 0.000 |

`P(inactive)` is **0.00 at all four lags** — flatter than 20261022's 0.009 and the most
concentrated submission yet. Overall mean amplitude **2.61** against an observed 1.30, and
the forecast's own per-field σ runs **1.39–1.56×** the basis. §5 unchanged: the concentration
is an over-amplified field, **not confidence**. Submitted on standing instruction, flagged in
the dry-run.

The one new thing is the +28 spread — 29/50 in phase 8 against 17 in phase 1. Do not read that
as eastward propagation: the 34-day window cannot resolve it, and `MJO_METHOD.md` is explicit
that forecast phase speeds in this product are not evidence of MJO presence.

Bands at `/tank/projects/mjo_model_clim/bands_20261029.npz` — **thirteenth cycle** in the set.

---

## ⚠ Disk — the next cycle cannot run without a purge

**74 GB free** against ~210 GB for Step 2. The 20261105 cycle needs a purge authorised before
Step 2, not during it. Candidates are the oldest cycles whose MJO bands are already extracted
to `/tank/projects/mjo_model_clim/` — **confirm the bands exist before purging any store**,
because the store is the irreplaceable part and `cleanup_aifs_run.py` will not warn about it.

`20260514` must not be purged: its source.coop upload is incomplete (68,979 / 805,110 objects)
and deleting the local copy strands it permanently.

---

## Related

- [`run_commands_20261022.md`](run_commands_20261022.md) — where the SWIO question was raised.
- [`run_commands_20261015.md`](run_commands_20261015.md) — the tercile trap.
- [`run_commands_20261008.md`](run_commands_20261008.md) — the paginated checker.
- [`run_commands_20261001.md`](run_commands_20261001.md) — the first half of the verify rule.
- [`ts-mjo/TS_STORM_DAYS.md`](ts-mjo/TS_STORM_DAYS.md) — detector over-count, basin activation.
- [`ts-mjo/MJO_METHOD.md`](ts-mjo/MJO_METHOD.md) — §5, the model climatology blocker.
