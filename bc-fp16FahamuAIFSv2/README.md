# bc-fp16FahamuAIFSv2 — bias-corrected and calibrated AIFS-ENS v2

**Status: plan only (2026-11-05).** Nothing here is built or run yet.

A second model entry alongside the raw `../fp16FahamuAIFSv2/`. Same checkpoint
(`ecmwf/aifs-ens-2.0`), same 50-member real-time run each Thursday. The difference: each
real-time product is calibrated against a **same-date hindcast** run from ERA5 starting states.
The raw entry keeps running unchanged, so the two can be compared cycle by cycle.

Design background: `../paper/AIFS_ENS_V2_S2S_EXTREMES_STUDY_PLAN.md` §4.4–4.8 (weekly hindcast,
product calibration, what to keep) and `../paper/DATA_AVAILABILITY.md` (ERA5 starting states,
including the 6 wave bands).

---

## 1. What runs each week

| | Real time (shared with the raw entry) | Hindcast (new) |
|---|---|---|
| Starting states | IFS open data, today's date | ERA5: ARCO control + 10 ERA5 EDA members, same calendar date in 2006–2025 |
| Runs | 50 | **20 years × 11 members = 220** |
| Lead | 792 h | 792 h |
| Purpose | the forecast | the model's own climatology for that date and lead |

**The 220 runs are in addition to the 50, not a replacement.** The 50 real-time members are
calibrated against the 220 hindcast members.

Each calendar date's hindcast is **reusable**: 5 Nov 2027 needs only the one new year (11 runs)
on top of 5 Nov 2006–2025. After a one-off build of ~52 calendar weeks, the weekly cost drops to
11 runs.

## 2. Cost of the 220 hindcast runs

Measured per member (from `../fp16FahamuAIFSv2/CYCLE_DATA_INVENTORY.md` and the 20261029 cycle):
input pkl **0.97 GB**; tier-B output (O96 corpus + N320 sidecar) **~5.1 GB**; full N320 corpus
**~11.7 GB**; GPU time **~5.5 min** per 792 h member on the RTX 5000 Ada.

A real-time cycle today is **~256 GB** (46 GB pkls, 150 GB O96, 50 GB N320 sidecar, 10 GB
NetCDF), not 1.5 TB. A full-N320 corpus would be ~583 GB per 50 members.

### 2.1 If the 220 runs were stored like a real-time cycle

| | Per member | × 220 |
|---|---|---|
| Input pkls | 0.97 GB | **213 GB** |
| Tier-B output (O96 + N320 sidecar) | ~5.1 GB | **~1.1 TB** |
| Full N320 output | ~11.7 GB | **~2.6 TB** |

**This cannot be done on this box.** On 2026-11-05, `zfs list` shows the `tank` pool at
2.68 TB used with **71.8 GB free** (`tank/projects` 1.76 TB, `tank/home/ezra` 938 GB). That is
already less than one real-time cycle.

### 2.2 Streamed: build, run, extract, delete

Do not stage all 220 pkls, and do not keep any hindcast store:

```
for each hindcast year (20 per calendar date):
    build 11 ERA5 starting-state pkls           ~10.7 GB
    for each member:
        run 792 h on N320 → temporary store       ≤ ~5 GB (one member at a time)
        extract products (§3) → keep ~3 MB
        delete the temporary store
    delete the 11 pkls
```

| | Size |
|---|---|
| Peak disk (one year of pkls + one member's temporary store) | **~16 GB**, ~30 GB if the next year's pkls are prebuilt |
| Kept per member (products) | ~3 MB |
| Kept per weekly hindcast (220 members) | **~0.7 GB** |
| Kept for all 52 calendar weeks × 20 years | **~35 GB**, reusable every year |

Downloads per hindcast year are transient: the ARCO control (106 fields × 2 times at 0.25°),
10 CDS EDA members, and one MARS wave-band request (~10 min from tape). The exact EDA volume
is not checked yet (`DATA_AVAILABILITY.md` §5).

### 2.3 Compute

| | GPU-hours | Local wall time | Cloud Run L4 (~$0.17/member) |
|---|---|---|---|
| Real time, 50 members | ~4.6 | ~4.6 h | ~$8.50 |
| Hindcast, 220 members, every week | **~20** | ~20 h | **~$37/week** |
| Hindcast every other week, pooled ±7 days | ~10/week averaged | — | ~$19/week |
| One-off build of all 52 weeks | ~1 050 | ~44 days local | ~$1 950 |
| Steady state after the build (11 new runs per week) | ~1 | ~1 h | ~$1.90/week |

Running at O96 instead of N320 saves no compute: the model always computes on N320, and O96 is
only a write grid.

## 3. Products kept from each hindcast member

All extracted from N320 at run time, the same way as for the real-time run:

| Product | Calibration it feeds |
|---|---|
| AI-WQ 1.5° weekly tas, mslp, pr (N320 → 1.5° directly, the submission route) | Model-climate quintile edges, Dirichlet shrinkage |
| Tropical storm tracks (`ts-mjo/ts_tracks.py` detector on N320) | Per-basin, per-week model terciles of storm days |
| MJO band means: χ200, u850, u200, daily, ±15° | Hindcast climatology per lead day, replacing ERA5 in `MJO_METHOD.md` step 5 |
| CRMA layer 3 (`crma_layer3_extract.py`) | AIFS's own climatology for the BN states |

No O96 or N320 field archive for the hindcast. Refitting reads only these products
(study plan §4.7).

## 4. All the improvement steps, ranked

These are the steps suggested earlier in the analysis, ranked by expected gain per unit of
effort. This folder implements steps 1–3; the rest follow.

| # | Step | Needs new GPU runs? | Notes |
|---|---|---|---|
| 0 | **Measure first:** rescore all completed cycles (~18) with the competition scorer, for the Greater Horn only, with reliability diagrams pooled across cycles, and test simple shrinkage towards climatology (one weight per variable, leave-one-cycle-out) | No | If shrinkage helps, the raw ensemble is overconfident and calibration pays even before the hindcast exists |
| 1 | **Model climatology (E4) from the same-date hindcast** | Yes (§2) | `epistemic-reasoning-risk/CROSS_CYCLE_FINDINGS.md` §5 already calls E4 "the clear priority" |
| 2 | **Calibration against that climatology (SPQR-style):** lead-dependent bias removal over a window of nearby dates; quintile edges from the model climatology (or quantile mapping to ERA5); Dirichlet shrinkage, P = Dirichlet(α·0.2 + w·counts), with w and α learned per region and lead, as in `cpt_build.py` | No | The measured ESS of ~1.3–1.9 over East Africa predicts w ≪ 50. Fit on 11-member subsamples; check on real-time cycles subsampled to 11 |
| 3 | **Calibrate the MJO and tropical storm products** against the hindcast | No (uses step 1) | Fixes MJO amplitude 2.84 vs 1.30 observed (20260924) and the NW Pacific storm over-count |
| 4 | **Reduce draw noise:** more members, or two stochastic draws per member | Yes | A different draw flips the modal quintile in 28–40% of cells (`RPSS_LEADERBOARD_DISCREPANCY.md`). Matters for a stable operational product |
| 5 | **A small multi-model crowd:** open ECMWF ENS fields plus freely available real-time S2S systems (e.g. NCEP CFSv2, ECCC), each bias-corrected, combined by Bayesian model averaging weighted by verified skill per region and lead; C3S seasonal as a low-weight prior for weeks 3–4 | No AIFS runs | SPQR's ECCC/NCEP mixing step (`../crowd-fahamu/vd.txt`). Real-time availability of each system not yet checked |
| 6 | **Regime-dependent calibration through the Bayesian network:** index the step-2 parameters by regime (k = 2 circulation regime, Somali jet strength, IOD phase), i.e. learn P(ERA5 quintile \| AIFS quintile, regime) from the hindcast pairs | No | Gives the BN the matched forecast/observation pairs `NETWORK_DIRECTION.md` says Phase 3 needs |
| 7 | **CHIRPS/IMERG calibration as a separate East Africa rainfall track** | No | Not for AI-WQ, which scores against ERA5 |
| — | Not recommended now: taking weights from the network into a BN, or Bayesian deep learning on the weights. Later research: decoder fine-tuning or LoRA on long rollouts to reduce drift at days 15–30 | — | Spread is not the S2S problem; bias, drift and correlated members are |

## 5. Not built yet

- [ ] Disk: free space for even one real-time cycle (71.8 GB free on 2026-11-05)
- [ ] ERA5 starting-state builder (ARCO + CDS EDA + MARS wave bands) writing the pkl format
      `run_local_icechunk_v2.py` reads
- [ ] Full-ERA5 donor test (study plan §2.4), the gate for everything here
- [ ] Per-member run → extract → delete loop (the run script writes one store per cycle today)
- [ ] Product extractors callable per member (AI-WQ 1.5°, TS tracks, MJO band, CRMA layer 3)
- [ ] Calibration code and correction-table layout per calendar week
- [ ] Step 0 rescoring of the completed cycles
- [ ] Submission naming for the second entry, and whether AI-WQ allows a second model per team
