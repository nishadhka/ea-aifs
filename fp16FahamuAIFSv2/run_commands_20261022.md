# Run log — 20261022 (AIFS-ENS 2.0, tier B)

9/9 submitted and verified, with **41 hours lost between 3b and 3c because nothing was
watching the chain**. The compute was clean; the handoff was not. This is the 20261001 failure
in a new shape and the reason the rule now has a second half.

---

## ⏱ End-to-end timing

| Step | What | Wall time | Output |
|------|------|-----------|--------|
| **0** | open-data pre-check | seconds | **RUNNABLE** — 9/9 sfc, 2/2 sol, 6/6 pl, 14/14 levels, 11/11 wave |
| **1** | pkl creation | 10-23 07:30 → 09:26, **1 h 56 m** (138.1 s/member) | 39 GB, **50/50** |
| **2** | tier-B rollout | 09:26 → 14:12, **4 h 46 m** | **50 written, 0 skipped, 0 failed** |
| **3a** | regrid 432–792 h → 1.5° | 14:12 → 14:20 | 50/50 |
| **3b** | quintiles | 14:20 → 14:22 | 6 arrays |
| — | **nothing running** | **10-23 14:22 → 10-25 07:40, 41 h 18 m** | **—** |
| **3c** | submit tas/mslp/pr | 10-25 07:40 → 08:54, 1 h 14 m | **6/6, 0 failed** |
| **TS** | tracker on the N320 sidecar | ~4 m | 9.5 tracks/member |
| **MJO** | VPM → RMM-fitted basis | ~2 m | bands preserved |
| **TS+MJO submit** | 3 files, **concurrent with 3c** | 07:50 → 08:29, 39 m | **3/3** |
| **verify** | `check_submissions_paged.py` | seconds | **9/9 present** (272 files, 3 pages) |

Stores: **O96 158 GB** (`1K1WB5NQZ9B5NRC32PGG`) **+ N320 51 GB** (`KJRS6VNTK624N1ETHE8G`),
both tagged `cycle-20261022_0000`. Disk **354 GB free** after.

Step 1 was the fastest yet at 138.1 s/member, against 187.4 s on 20261015. Window closed
2026-10-25 23:59 UTC; the last file landed with **15 h to spare** — margin that existed only
because the deadline was three days out.

---

## The 41-hour gap: a chain nobody was watching

`chain_20261022.sh` finished 3b at **10-23 14:22** and stopped there by design — compute is
chained, submission is not. Then nothing happened until **10-25 07:40**, when the user asked
for status.

The chain was launched with

```bash
nohup bash /tank/projects/chain_20261022.sh > /tank/projects/chain_20261022.log 2>&1 &
```

`nohup ... &` detaches the job from the harness, so **no completion notification was ever
going to arrive**, and the stated intention to "report when the chain finishes" rested on
nothing. On 20261015 the same launch pattern happened to work only because that chain was
waited on with a foreground `until grep -q` loop, which *is* harness-tracked.

### The rule, both halves

20261001 established: **verify a submission against the server, never against its own log** —
after a chained submitter died silently. 20261008 qualified the instrument: use a *paginated*
check.

This cycle adds the part that was missing: **a compute chain must be waited on by something
the harness tracks, or its completion is invisible.** Either

```bash
# harness-tracked wait -> notification on exit
until grep -qE "CHAIN COMPLETE|ABORT" /tank/projects/chain_<DATE>.log; do sleep 30; done
```

run as a tracked background command, or the chain launched as one directly. A bare
`nohup ... &` for the chain is fine *only* if a tracked waiter is armed in the same breath.

The failure mode is quiet in the worst way: every log says success, the products are all on
disk and correct, and the only thing missing is anyone looking.

---

## Commands (as run)

```bash
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
BASE=/tank/projects/aifs-run/20261022_0000

cd run-pre50r1-dates && $PY check_open_data_inputs.py --date 20261022

# disk: 95 GB free, Step 2 needs ~210 GB -> purge authorised for two cycles
$PY cleanup_aifs_run.py --date 20260903 --yes      # 255.3 GB
$PY cleanup_aifs_run.py --date 20260910 --yes      # 259.9 GB   -> 566 GB free

mkdir -p $BASE/input_states
$PY -u s3_grib_pkl/s3_grib_pkl_input_aifsens_v2_proto.py \
    --date 20261022_0000 --members 1-50 --source gcs \
    --out $BASE/input_states --skip-existing

bash /tank/projects/chain_20261022.sh     # rename -> 2 -> 3a -> 3b, COMPUTE ONLY

# terciles BEFORE ts_days.py -- see the 20261015 trap
$PY - <<'EOF'
from AI_WQ_package import retrieve_evaluation_data as red
for d in ("20261109", "20261116"):
    red.retrieve_20yr_quantile_clim(d, "TS", os.environ["AIWQ_PASSWORD"],
                                    local_destination=f"{BASE}/ts_terciles")
EOF

$PY -u ../shared/forecast_submission_cli.py --date 20261022 --v2 --output-dir $BASE/aiwq
$PY -u ts-mjo/ts_days.py --store $BASE/icechunk_n320_aiwq --tag cycle-20261022_0000 \
    --init 20261022 --aiwq-tercile-dir $BASE/ts_terciles \
    --out $BASE/ts_days_probs_20261022_tracked.nc
$PY -u ts-mjo/vpm_index.py --store $BASE/icechunk_o96 --init 20261022 \
    --clim /tank/projects/era5_vpm_clim/vpm_clim_1991_2020.npz \
    --eofs /tank/projects/era5_vpm_clim/rmm_basis.npz \
    --dump-bands /tank/projects/mjo_model_clim/bands_20261022.npz \
    --out $BASE/mjo_probs_20261022.nc
$PY -u submit_ts_mjo_cli.py --date 20261022 \
    --ts-nc $BASE/ts_days_probs_20261022_tracked.nc --tercile-dir $BASE/ts_terciles \
    --mjo-nc $BASE/mjo_probs_20261022.nc --dry-run     # then without
$PY check_submissions_paged.py --date 20261022
```

**`--aiwq-tercile-dir` was passed this cycle**, fixing the 20261015 trap: `ts_days.py` emitted
no self-referential warning and its printed probabilities match the submission exactly.

**TS+MJO was submitted concurrently with 3c** on the user's instruction, rather than after it.
No contention: 3 files in 39 min while 3c was mid-upload, and both finished clean. The
sequential ordering used on 20260924–20261015 is not required.

Validation before 3c: 6 arrays `(2, 5, 121, 240)`, NaN 0, ⊂ [0,1], Σ = 1.000000; weeks
2026-11-09 / 2026-11-16. Global mean P(Q5) for `2t` **0.441** — unchanged warm shift.

---

## TS days — ATL's season ends, and SWIO gets real bounds

| week | ATL mean | NWP mean |
|---|---|---|
| 2026-11-09 | 0.78 d | 1.72 d |
| 2026-11-16 | 1.12 d | 3.56 d |

9.5 tracks per member. Official bounds:

| week | ATL | NWP | SWIO | SEIO |
|---|---|---|---|---|
| 2026-11-09 | 0 / 1 | 0 / 3 | 0/0 | 0/0 |
| 2026-11-16 | **0 / 0** | 0 / 2 | **0 / 2** | 0/0 |

Submitted:

| period | basin | below | near | above |
|---|---|---|---|---|
| 1 | ATL | 0.00 | 0.62 | 0.38 |
| | NWP | 0.00 | 0.72 | 0.28 |
| 2 | ATL | 0.00 | **0.00** | **1.00** |
| | NWP | 0.00 | 0.46 | 0.54 |

**ATL week 2 has degenerate bounds `0/0`** — lower == upper, so AI-WQ's
`conditional_obs_probs` masks the observation to NaN (`all_equal`) and the basin-week is not
scoreable. The 1.00 in "above" is arithmetic that keeps Σ = 1, **not a forecast**. ATL is
still listed active for month 11, so this is the Atlantic season ending inside an active
month, not an inactive-basin fill.

### Open question: SWIO at 2026-11-16 has non-degenerate bounds and we discarded the forecast

AI-WQ published **SWIO 0/2** for week 2 — the first non-degenerate SWIO bounds seen — and
`ts_days.py` produced a real forecast for it, P(below, near, above) = 0.00, 0.94, 0.06.
`submit_ts_mjo_cli.py` applies the activation table (SWIO/SEIO active in months 12/1/2 only),
so 2026-11-16 counts as inactive and the column was overwritten with uniform 1/3.

**If AI-WQ scores that column, a real forecast was thrown away.** Submitted as-is rather than
changing activation logic on the submission deadline, which was the right call under time
pressure and is the wrong state to leave standing. **Resolve before the December cycles**, when
SWIO goes properly active: check whether a published non-degenerate bound, rather than the
month, is the actual activation signal.

## MJO — phase 8 at every one of 34 days

Modal category **8 on all 34 days**, so 8 → 8 → 8 → 8 at init +7/14/21/28. Flatter even than
20261015, which dipped to 7 on days 6–7.

Mean amplitude **2.59** against an observed 1.30; P(amp < 1) **0.009** against 0.37; the
forecast's own per-field σ runs 1.34–1.56× the basis. §5 unchanged — the concentration is an
over-amplified field, **not confidence**. Submitted on standing instruction, flagged in the
dry-run.

Bands at `/tank/projects/mjo_model_clim/bands_20261022.npz` — twelfth cycle in the set.

---

## Disk

95 GB free at the start, against ~210 GB for Step 2. Purged **20260903 + 20260910** on
explicit authorisation: **515.2 GB reclaimed**, 95 → 566 GB. Both cycles kept their
`ts_days_probs_*.nc`, `mjo_probs_*.nc` and 7 `aiwq/*.nc`; their MJO bands were already outside
`aifs-run` and untouched. Ten cycles of bands remain — **always confirm bands are extracted
before purging a store.**

---

## Related

- [`run_commands_20261015.md`](run_commands_20261015.md) — the tercile trap fixed here.
- [`run_commands_20261008.md`](run_commands_20261008.md) — the paginated checker.
- [`run_commands_20261001.md`](run_commands_20261001.md) — the first half of the verify rule.
- [`ts-mjo/TS_STORM_DAYS.md`](ts-mjo/TS_STORM_DAYS.md) — detector over-count, basin activation.
- [`ts-mjo/MJO_METHOD.md`](ts-mjo/MJO_METHOD.md) — §5, the model climatology blocker.
