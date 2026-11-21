# Study plan — can AIFS-ENS v2 anticipate Greater Horn of Africa droughts and floods at sub-seasonal range?

**Written:** 2026-11-05 · **Status:** plan, nothing run yet · **Owner:** Fahamu / ICPAC (`fp16FahamuAIFSv2`)

**Working title:** *Detecting Greater Horn of Africa hydro-climatic extremes with a machine-learned
ensemble beyond its 15-day design range: AIFS-ENS v2 hindcasts, calibration and 23 GHACOF events.*

This document is the full plan for the research article. It is written so each claim can be
checked against the repo. Where a number comes from an earlier measurement, the source file is
named.

---

## 0. Why this study, in four facts

1. **The raw ensemble already scores well.** `fp16FahamuAIFSv2` submits uncalibrated 50-member
   quintile counts to the AI Weather Quest and places in the top ten. Cycle 20260709 week 1 scored
   +0.071 / +0.055 / +0.106 (tas / mslp / pr), reproduced locally to three decimals
   (`O96-icechunk-store/forecast-evaluation/RPSS_LEADERBOARD_DISCREPANCY.md`). A global leaderboard
   score is not evidence of regional skill, reliability or extreme-event detection.
2. **The model runs well outside its design range.** The model card says AIFS-ENS v2 was built
   for 15-day forecasts. We run it to 33 days (792 h).
3. **Every historical event is inside the training data.** Pretraining used ERA5 1979–2022.
   Fine-tuning used IFS operational analysis 2018–2024. All 22 GHACOF events (2016–2024) are in
   sample. Only Nairobi March 2026 and the real-time 2026 cycles are out of sample.
4. **Real time and hindcast cannot share initial conditions.** Real time uses IFS ENS from ECMWF
   open data, which can fully run v2 only from 2026-05-13 (Cy50r1). A 20-year hindcast must start
   from ERA5. ECMWF does the same with its own extended-range reforecasts: ERA5 starts for the
   reforecasts, operational analysis for real time.

---

## 1. Research questions

| | Question | Main output |
|---|---|---|
| **RQ1** | Initial-condition sensitivity: do ERA5-started forecasts behave like IFS-started ones at days 18–33? | Paired comparison on cycles 20260917–20261029 (§3) |
| **RQ2** | Drift and calibration: how large is the lead-dependent bias over the Greater Horn at days 15–33, and does hindcast calibration improve RPSS and reliability? | 20-year hindcast climatology, calibrated and raw scores (§4) |
| **RQ3** | Extreme detection: over all years, does AIFS separate extreme-quintile weeks from other weeks, and at what lead? The 23 events illustrate hits and misses | ROC, SEDI, reliability, event figures (§5) |
| **RQ4** | Training overlap: is skill different in-sample (2005–2024) and out of sample (2025–2026)? | Skill split by period (§6) |
| **RQ5** | Drivers: in hits and misses, does AIFS get the IOD, ENSO, MJO and Somali-jet state right? | BN diagnostics from `epistemic-reasoning-risk/` (§7) |

---

## 2. The three initial-condition routes

| Route | Period | Source | Status |
|---|---|---|---|
| **A. Operational** | 2026-05-13 → | ECMWF open data, IFS ENS, 50 perturbed members, all 112 fields | In production every Thursday |
| **B. Hybrid pre-50r1** | ~2025-01-16 → 2026-05-12 | Open data for 99 fields, plus 13–14 donor fields: 8 wave fields from `j1r2`, 10 hPa from ERA5, `sd` from ERA5 | **Built and validated** in `../fp16FahamuAIFSv2/run-pre50r1-dates/` |
| **C. Full ERA5** | 1979 → present | ERA5 for every field | **New: needed for the 20-year hindcast** |

### 2.1 What route B already established (`run-pre50r1-dates/README.md`)

- Before Cy50r1, 13 fields are missing from open data: 6 period-band wave heights, `cdww`, `wmb`,
  and the 10 hPa level. For 2025 dates `sd` is missing too (14 fields).
- **The 10 hPa level matters most.** Carrying 50 hPa up to 10 hPa puts the day-5 forecast at
  ~500 % of the ensemble scale. ERA5's 10 hPa costs **14.8 %**.
- **Waves from `j1r2` cost 17.7 %.** Both donors together cost **20.4 % of the ensemble scale,
  5.6× the non-determinism floor** at day 5. That passes the folder's own decision rule (≲ 40 %,
  ≲ 10× floor).
- **At days 18–33 a single-member comparison tells you nothing.** Two identical runs with the
  same seed diverge to 85–97 % of the ensemble spread because of FP16/cuDNN non-determinism.
  Sensitivity must be measured at short lead, or as a whole distribution with many members.
- Before Cy50r1 the wave ensemble has **no member spread at t=0**, so a deterministic wave donor
  loses nothing. The deterministic ERA5 10 hPa level does remove stratospheric spread, and that
  cost is still unmeasured.
- Measurement harness already in place: `wave_sensitivity.py` + `analyse_sensitivity.py`, with
  same-seed pairing, a duplicate control (the floor) and a positive control (`l10copy`).

### 2.2 Route C: building a full ERA5 starting state, field by field

| Field group | ERA5 source | Notes |
|---|---|---|
**The control state comes entirely from ARCO, with no CDS queue.** Checked 2026-11-05:
`gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3` holds 277 variables at
0.25°, hourly, on all 37 pressure levels. That includes every field below except the 6 wave
period bands. **ARCO holds no ERA5 ensemble members** (its `ar/`, `co/` and `raw/` trees are the
single high-resolution reanalysis only), so the 10 perturbation members must come from CDS.

**ERA5 cloud stores compared (checked 2026-11-05):**

| | ARCO `ar/full_37-1h-0p25deg-chunk-1.zarr-v3` (GCS) | Earthmover `earthmover-icechunk-era5/icechunkV2` (S3, Icechunk) |
|---|---|---|
| Period | 1940-01-01 → **2026-06-30 final, 2026-10-30 ERA5T** (updated daily) | 1940-01-01 → **2026-06-30** (updated quarterly; last on 2026-10-12) |
| Grid, time step | 0.25°, hourly | 0.25°, hourly. Groups `single`/`pressure`/`500hPa` × `spatial` (chunk = 1 time step) / `temporal` |
| Pressure levels | 37, **including 10 hPa** | **13 (50–1000 hPa), no 10 hPa** |
| Surface, soil, `sd`, `tcw`, constants | all | all (39 single-level variables plus `lsm z_sfc slor sdor`) |
| Waves (`swh mwp mwd cdww wmb`) | yes | **none** |
| Ensemble (EDA) members | none | none |
| Access | plain Zarr over HTTPS | needs the `pcodec` codec (`numcodecs[pcodec]`), which the `aifs-gpu` env lacks |
| AIFS v2 fields covered | 106 of 112 (missing only the 6 wave bands) | 95 of 112 (also missing the 5 fields at 10 hPa and all 6 other wave fields: `swh mwp cos_mwd sin_mwd cdww wmb`) |

**Use ARCO as the single source for the control state.** It is the only store that covers the
Sep–Oct 2026 RQ1 cycles (Earthmover stops at 2026-06-30), the 10 hPa level and the wave fields.
One source means one pipeline. Earthmover is a fallback and a cross-check for the 95 fields it
holds, since both are converted from the same CDS GRIB. The 10 EDA members come from CDS either
way.

| Field group | ERA5 source | Notes |
|---|---|---|
| Surface (`10u 10v 2d 2t msl skt sp tcw`), soil (`stl1 stl2 swvl1 swvl2`), `sd`, constants | ARCO `ar/full_37…v3` (`snow_depth`, `soil_temperature_level_1/2`, `volumetric_soil_water_layer_1/2` are all present). ARCO `co/single-level-reanalysis.zarr-v2` is also available on **native N320, point for point the AIFS grid** (`ts-mjo/ERA5_VPM_CLOUD_BRIEF.md` §4.2) | Check that `sd` units (water equivalent) match the IFS field |
| Pressure levels `t u v w z q` × 14 levels (including 10 hPa) | ARCO `ar/full_37…v3` (37 levels include all 14; `vertical_velocity` present), regridded 0.25° → N320 | Reuse the grid-order check from `fetch_era5_l10.py --validate` |
| `swh mwp mwd` | ARCO `ar/full_37…v3` (`significant_height_of_combined_wind_waves_and_swell`, `mean_wave_period`, `mean_wave_direction`) | Older wave model (Cy41r2) than the ecWAM Cy50r1 the model was trained on. Measure the difference in input space on the `j1r2` overlap (2024-05 → 2026-06) |
| `cdww`, `wmb` | ARCO `ar/full_37…v3` (`coefficient_of_drag_with_waves`, `model_bathymetry`) | Apply the land-sea mask repair from `fetch_j1r2.py` |
| **6 period-band wave heights `h1012…h2530`** | **Not in ERA5.** `j1r2` covers only 2024-05-02 → 2026-06-20 | **The one real gap, see §2.3** |
| Perturbations | **CDS only:** ERA5 EDA, 10 members (`product_type: ensemble_members`, ~0.5°, 3-hourly) | Perturbation = EDA member − EDA mean, added to the ARCO control. Control + 10 = **11 members**, matching ECMWF reforecasts. In week 1, confirm that every needed variable (including `w` and 10 hPa) is published for the EDA members |

### 2.3 The wave-band gap for years before 2024

> **Update 2026-11-05: the gap is closed by published data.** The ECMWF experiment catalogue
> (<https://apps.ecmwf.int/ifs-experiments/>) lists **the whole AIFS-training wave hindcast**,
> not just `j1r2`: Tco1279 (9 km), ecWAM CY50R1, forced by corrected ERA5, 6-hourly, with the six
> period bands (140114–140119), `cdww` (140233) and `wmb` (140219) on every page checked.
>
> | expver | Period | Forcing | Use in this study |
> |---|---|---|---|
> | `ikxa` | 1979–1989 | ERA5 (no wave assimilation) | not needed |
> | `ikp7` | 1990–1999 | ERA5 + altimeter assimilation | not needed |
> | **`ikp6`** | **2000–2015** | ERA5 + altimeter assimilation | **hindcast years 2005–2015** |
> | **`ikg0`** | **2016–2023** | ERA5 + altimeter assimilation | **hindcast years 2016–2023** |
> | **`itul`** | **2024–2025** | ERA5 + altimeter assimilation | **2024, plus out-of-sample 2025** |
> | `j5rf` | 2026 | ERA5 + altimeter assimilation | out-of-sample 2026, RQ1 consistency |
> | `ikfz` / `ikdi` | 2016–2018 / 2019–2025 | operational winds | alternative |
> | `j6bd` / `j6bf`, `j1j2` / `j1j3`, `j1r2` | 2016–2026 | operational winds (± ORAS6 currents) | `j1r2` = the existing pre-50r1 donor |
>
> These are the same model, grid and parameters as `j1r2`, so `fetch_j1r2.py` needs only an
> `--expver` option. The ERA5-forced series also matches the ERA5 starting state, which suits
> route C better than the operational-wind series. Same `.ecmwfapirc` key, `class=rd`. Large parts
> are likely on tape, so pull whole days per request and allow time for queues.
>
> Retrieval test: **2011-01-06 from `ikp6` succeeded**: bands, `cdww` and `wmb` on N320
> (542 080 points), physical values, 615 s per request (tape). Other years were not tested; the
> exploration was stopped on request. Details and the download scope (event dates only) are in
> `DATA_AVAILABILITY.md`.
>
> **Options 1–4 below become fallbacks** if retrieval fails or is too slow. Option 5 (asking
> ECMWF) is no longer needed.
>
> **Also in the catalogue: `j339`, AIFS-SUBS reforecasts.** ECMWF's own sub-seasonal AIFS: a
> 20-year reforecast, 10 members, 46 days, daily 00 UTC, started from **ERA5 recentred on the
> EDA** (preprint arXiv:2607.05100, DOI 10.21957/3pt9-xe41). It is the closest published precedent
> for our route C (ERA5 control + EDA perturbations), and **an ML benchmark for RQ3**. Add it to
> §8 and the related-work section.

ECMWF's training wave hindcast (1979–2025, ecWAM, ~9 km) is described in the AIFS ocean paper.
Before the catalogue search, only `j1r2` (from 2024-05) was known. The `run-pre50r1-dates` workaround, the `j1r2` donor,
therefore covers only **2024-05-02 → 2026-06-20**. That handles the 2024 tail of the hindcast and
not 2005–2023.

**What the folder's donor work gives us:** `j1r2` is the real target. It holds the exact fields
AIFS v2 was trained on, at 9 km, on N320, and it overlaps ERA5 for about two years. That overlap
makes it a **training and validation set**, not just a donor.

**What ARCO `ar/full_37…v3` holds** besides `swh mwp mwd cdww wmb` (checked 2026-11-05): wind-sea
height, period and direction; total-swell height and period; **three swell partitions**, each with
height, mean period and direction; peak period; mean periods from the first and second moments;
zero-crossing period; spectral directional width, peakedness, skewness and kurtosis. These describe
the shape of the 1D spectrum, which is what the period bands slice up.

**Options, in recommended order:**

1. **Band emulator trained on the `j1r2` overlap (recommended).** Rebuild the six bands from the
   ARCO partition fields, in two steps:
   - *Physical first guess:* treat each partition (wind sea, swell 1–3) as a parametric 1D
     spectrum (JONSWAP-like, peak period from its mean period, total energy = (Hs/4)²). Integrate
     each over the band edges (10, 12, 14, 17, 21, 25, 30 s) and sum the partitions:
     `Hs_band = 4·√(Σ_partitions E_band)`. This keeps the physics: bands shrink with period,
     there is no energy without swell, and the bands never exceed the total.
   - *Statistical correction:* fit a model of the `j1r2` band minus the first guess, per band,
     on the 2024-05 → 2026-06 overlap (e.g. gradient boosting on the partition fields plus
     latitude and season). Hold out whole months for validation. This also absorbs the difference
     between ERA5's older wave model at 0.5° and ecWAM Cy50r1 at 9 km.
   - Regrid to N320 and apply the land-sea mask repair from `fetch_j1r2.py`.
2. **`j1r2` calendar-day climatology** of the 6 bands (mean by day of year over the overlap).
   Cheap baseline: realistic magnitudes and season, but not the actual sea state.
3. **NaN fill.** Already measured: at the floor at days 18–33 (one-sided test), ~40 % of the
   ensemble scale at day 5 (`run-pre50r1-dates` §3 Result 2).
4. **ERA5 2D wave spectra** (CDS, tape-backed). The folder (§2.6) ruled this out for a single
   date because `j1r2` existed. Here it is the reference check: pull spectra for 3–5 dates
   **before 2024** and confirm the emulator's bands against bands integrated from the full
   spectra, where `j1r2` cannot help.
5. **Ask ECMWF for the training hindcast** (1979–2025). Supersedes everything above if granted.
   Send the request in week 1.

**How to choose (same two-stage test as the folder):**

| Stage | Test | Pass mark |
|---|---|---|
| Input space (no GPU) | Emulator and climatology against `j1r2` on held-out months: rms and r per band | Near `j1r2`'s own agreement with operational analysis (5.7–14.8 % rms, r ≥ 0.99; `run-pre50r1-dates` §2.4) |
| Forecast space, day 5 | `wave_sensitivity.py` on 20260604 (real bands available), same seed, new modes `waveemul`, `waveclim`, `wavenan` against `control` | ≲ 40 % of the ensemble scale, ≲ 10× floor. Ideally near `wavej1r2`'s 17.7 % |

Keep the option closest to the floor and report the measured cost in the paper. At days 18–33 all
wave treatments so far sat at the non-determinism floor, so this choice matters most for the
short-lead consistency test (RQ1). For the S2S product it is a second-order effect, but the paper
has to show that rather than assume it.

**Cost:** about 2–3 days of CPU work. Pull the overlap from ARCO (6-hourly, ~2 years) and `j1r2`
from MARS (about 2 minutes per 52-field day, some of it on tape), fit, validate, then run 6 short
GPU rollouts for the day-5 test.

### 2.4 The full-ERA5 donor test (gate for everything downstream)

The same experiment as `run-pre50r1-dates` §5, extended to **all 112 fields from ERA5**:

```bash
# date with real operational fields AND j1r2 coverage, so every donor can be scored
D=20260604
$PY wave_sensitivity.py --out RESULTS_era5full_short --input-dir $INPUT/$D \
    --lead-time 120 --write-hours 6-120 --seeds 0,1 \
    --modes control,controlB,l10copy,era5full,era5full_waveclim,era5full_wavenan
```

(`era5full*` are new modes; the script must be extended.)

| If `era5full` at day 5 is… | Then |
|---|---|
| ≲ 40 % of the ensemble scale, ≲ 10× floor | Route C is defensible. Proceed |
| 50–150 % | Usable only as an ensemble. Quantify the lost spread before building the hindcast on it |
| ≳ 300 % | Route C fails. Fall back to route B for 2025–2026 only, and rethink the hindcast |

Note: the model was **pretrained on ERA5**, so full-ERA5 states are probably closer to what it
learned from than the mixed states in route B. This test checks that claim instead of assuming it.

---

## 3. RQ1 — the September–October 2026 cycles as a paired experiment

### 3.1 What is on disk (checked 2026-11-05)

| Cycle | Full stores (`icechunk_o96`, `icechunk_n320_aiwq`) | IFS input pkls | Products (`aiwq/`, TS, MJO) |
|---|---|---|---|
| 20260903, 20260910 | **purged** | purged | kept (quintiles, TS, MJO, climatology) |
| **20260917, 20260924, 20261001, 20261008, 20261015, 20261022, 20261029** | **present** | **present** (50 members) | kept |

**Seven complete real-time cycles exist with IFS starting states.** Each can be rerun from ERA5
for the same date, giving IFS-started and ERA5-started forecasts for identical dates. They are
**out of sample** (after the 2024 training cut-off) and verify in the **OND 2026 short rains**.

> **Retention action, before anything else:** exclude 20260917 → 20261029 from
> `cleanup_aifs_run.py` until RQ1 is done. At ~256 GB per cycle this is ~1.8 TB. If disk forces
> a choice, keep `input_states/`, `icechunk_n320_aiwq/`, `nc_1p5deg/` and `aiwq/`, and drop
> `icechunk_o96/` last (it is needed for RQ5 drivers).

For 20260903 and 20260910 a comparison is still possible at product level: the submitted quintile
files are kept, and the IFS starting state can be refetched from the open-data mirrors if they
still hold those dates (check with `check_open_data_inputs.py`).

### 3.2 Design

For each of the 7 cycles:

1. **Short-lead leg (day 5, member 1, seeds 0/1).** Paired same-seed test as in
   `run-pre50r1-dates` §5: IFS start against ERA5 start. Measures the input offset in forecast
   space while the floor is still small (~3 % of the ensemble scale).
2. **Product leg (days 18–33).** ERA5-started ensemble with 11 members (control + 10 EDA), and
   one with 50 members using the IFS perturbation structure if feasible. Compare as
   **distributions**:
   - quintile probabilities and modal-quintile agreement for tas, mslp and pr
   - weekly-mean bias and spread
   - East Africa (ICPAC box) and global
   - RPSS of both against ERA5 once the weeks verify (all 7 cycles verify by ~2026-12-01).
3. **Draw-noise reference.** A second IFS-started stochastic draw for 1–2 cycles, so the ERA5
   versus IFS difference can be read against the difference between two draws of the same
   start. Earlier measurement: a new draw flips the modal quintile in 28–40 % of cells.

**Cost:** 7 cycles × (11 + 50) members × ~5.5 min ≈ 39 GPU-hours, plus the short legs (~1 h).

**Result for the paper:** "ERA5-started hindcasts are a valid proxy for real-time IFS-started
forecasts at days 18–33, within X of the draw noise." Without this, RQ2–RQ4 cannot be
interpreted.

---

## 4. RQ2 — 20-year hindcast climatology and calibration

### 4.1 Hindcast design

| Choice | Value | Reason |
|---|---|---|
| Years | 2005–2024 (20, in sample) plus 2025–2026 (out of sample) | 20 years matches ECMWF reforecasts. The extra two years serve RQ4 |
| Calendar dates | Same day of year as the Thursday real-time cycle | One model climatology per cycle |
| Members | **11 (control + 10 EDA), decided 2026-11-05.** Not 5 | Matches ECMWF reforecasts, so the IFS benchmark is like for like. Uses every ERA5 EDA member. With 5 members, probabilities step in 0.2s, so the extreme-quintile ROC has only 6 points and SEDI and reliability are too coarse for detection claims. See §4.3 |
| Lead | 792 h | AI-WQ week 3 and week 4 windows |
| Output kept | Weekly means of tas, mslp and pr at 1.5°, HydroBASINS level-3 and admin-1 means, BN diagnostics (jets, regimes, highs) | A few MB per date, not ~256 GB. What to keep: §4.7 |

**Compute** (~5.5 min per 792 h member on the RTX 5000 Ada; 6.6 min per 960 h measured in
`LOCAL_GPU_RUN.md`):

| Version | Runs | GPU-hours | Local wall time | Cloud Run L4 (~$0.17/member) |
|---|---|---|---|---|
| Hindcasts run each week alongside real time | 20 × 11 = 220 per cycle | ~20 per week | fits beside the ~4.6 h real-time run | ~$37 per week |
| **Paper, 4-week version:** 12 dates at the event-season onsets (MAM, JJAS, OND) × 20 yr × 11 members | 2 640 | ~240 | ~10 days | ~$450, 1–2 days with a burst |
| Paper, reduced: 24 semi-monthly dates in event seasons × 20 yr × 11 members | 5 280 | ~480 | ~20 days | ~$900 |
| Paper, full: 48 weekly dates × 20 yr × 11 members | 10 560 | ~970 | ~40 days | ~$1 800 |

**To fit the month, cut calendar dates, not members.** Fewer dates narrows the seasons covered;
fewer members weakens every probability the paper reports.

**EDA download from CDS is the likely bottleneck, not the GPU.** The control comes from ARCO with
no queue. The 10 EDA members per starting time (14 pressure levels × 6 parameters × 2 timesteps,
plus surface fields) must come from CDS. Batch requests by month and start in week 1.

### 4.2 Calibration methods (compared, not assumed)

All fitted with leave-one-year-out cross-validation, per variable, lead week and region:

1. **Raw:** the current submission (member counts against ERA5 climatology quintiles).
2. **Mean bias correction:** subtract the lead-dependent hindcast bias, using a ±2-week window of
   calendar dates (as in SPQR, `vd.txt`).
3. **Model-climate quintiles:** quintile edges from the hindcast's own climatology (as in the
   Lu Peng hindcast-threshold branch, `lp.txt`).
4. **Quantile mapping:** model CDF to ERA5 CDF.
5. **Dirichlet shrinkage towards climatology:** P = Dirichlet(α·0.2 + w·counts), with the
   effective member count w and α learned from verification. This follows the `cpt_build.py`
   ESS/α₀ approach. The measured ESS of ~1.3–1.9 over East Africa
   (`epistemic-reasoning-risk/CROSS_CYCLE_FINDINGS.md`) predicts that w will come out well below 50.

Scores: RPSS (AI_WQ_package), reliability diagrams, ROC for each quintile, and spread–error
ratio. Globally, for the Greater Horn, and for each target basin.

### 4.3 Ensemble-size caveat

Hindcasts have 11 members, real time has 50. Fit the calibration on hindcast
subsamples, and check it on real-time cycles subsampled to the same size before applying it at 50.

### 4.4 Operational version: the weekly hindcast run alongside real time

Added 2026-11-05. Exploration only: nothing below is built.

For each Thursday cycle (e.g. 20261105):

1. **Real time:** 50/51 members from IFS starting states, as now.
2. **Hindcast:** the same calendar date (5 Nov) in each of 2006–2025, 11 members each
   (ARCO control + 10 EDA), so 20 × 11 = **220 runs**.
3. **Calibrate** each real-time product against the same product from those 220 runs (§4.5).

**The hindcast for a calendar date is reusable.** Once 5 Nov 2006–2025 exists, the 5 Nov
2027 cycle needs only one new year (11 runs). So the full set is a one-off build of ~52 calendar
weeks, plus a small top-up each year.

**Running at O96 does not save compute.** AIFS-ENS v2 always computes on N320, the grid the
checkpoint was trained on. O96 is only a write grid (`run_local_icechunk_v2.py`: "N320 is the
native model grid; O96 is the archive grid"). A member costs ~5.5 GPU-minutes whatever is
written. 220 runs ≈ **20 GPU-h per week**, plus ~4.7 h for real time.

**Compute levers, in order:**
- Run the hindcast every other week and pool ±7 days of calendar dates (~10 GPU-h/week
  averaged, same 20-year climatology).
- Keep 11 members (§4.1). Cut how often the hindcast runs, never the member count.
- For the paper, only the event-season dates (§4.1 table).
- Prefetch the wave bands a week ahead: ~10 min per tape request (`DATA_AVAILABILITY.md` §2.1).

**Storage is where the saving is:** keep run-time products only, ~3 MB per member (§4.7).

**Keep the regrid route identical.** AI-WQ fields for the hindcast must go N320 → 1.5°, exactly
as the submission does. If the hindcast went N320 → O96 → 1.5° while real time goes N320 → 1.5°,
the route difference would end up inside the "bias". The O96-vs-N320 RPSS gap is still
unresolved (`O96-icechunk-store/forecast-evaluation/RPSS_LEADERBOARD_DISCREPANCY.md`). O96 is
kept only for the paper's event-year field archive (§4.7.3).

### 4.5 Calibrate products, not input fields

Compute each product (quintile counts, TC counts, RMM phase, basin features) from **raw** model
output, the same way for hindcast and real time. Then calibrate the product against its own
20-year hindcast distribution. Do not quantile-map u, v, q separately and derive products
afterwards: that breaks physical consistency, and vorticity, moisture flux and χ200 come out
worse than from raw fields. QM also does not fix spread.

| Product | Calibration against the hindcast | Status / blocker it removes |
|---|---|---|
| **AI-WQ tas, mslp, pr quintiles** | Quintile edges from the model hindcast per grid point × lead week, pooled ±7 days; then Dirichlet shrinkage (§4.2). Equivalent to QM for quintile probabilities, because the mapping is monotone | Submitted raw today |
| **MJO phase** (`ts-mjo/MJO_METHOD.md`) | Replace step 5 (ERA5 calendar-day climatology) with the hindcast climatology per lead day; then calibrate the 9-category phase probabilities on the hindcast (amplitude decay with lead) | Mean amplitude 2.84 vs 1.30 observed on 20260924. The §5 leave-one-out test already shows the fix works |
| **Tropical storm days** (`ts-mjo/TS_STORM_DAYS.md`) | Run the same N320 detector on all 220 hindcast members. Bin real-time counts against **per-basin, per-week model terciles**, not IBTrACS terciles | Known NW Pacific over-count. The doc notes detector and IBTrACS bounds only interchange after bias correction |
| **CRMA basin features** (layer 3) | None needed: CRMA discretises each feature against the model's own climatology, which is its bias correction. The hindcast supplies that climatology for AIFS | Today AIFS has no per-model climatology (e.g. tp 22–32% drier than IFS, `CRMA_LAYER3.md`) |
| **Event-study fields** (rainfall and temperature maps, basin totals vs CHIRPS/IMERG) | `xsdba` empirical QM, grouped by lead week, not calendar time | Needed for §5 figures only |

**TC tracking must run on N320 at run time.** At O96 (~112 km) tropical cyclones are smoothed
out and counts change. Keep only the tracks (kB per member), never fields for later re-tracking.

**Sample size:** 220 samples (~660 with ±7-day pooling) is enough for quintiles and terciles,
but noisy beyond about the 95th percentile. The 11-vs-50 caveat (§4.3, and `vd.txt`) applies to
every product.

### 4.6 Tools: xclim and xsdba

| Use | Tool | Variables |
|---|---|---|
| AI-WQ quintiles, TC terciles, MJO climatology | Own code (counts against hindcast edges). No library needed | 3 + derived indices |
| Gridded event-study fields | **`xsdba`** (the bias-adjustment part of xclim, formerly `xclim.sdba`, now a separate package): `EmpiricalQuantileMapping` with lead-week grouping | tp, 2t (maybe 2d) |
| Drought and flood indicators: SPI, consecutive dry days, R95p, dry spells | **`xclim` indicators** on the corrected tp/2t | derived |
| Circulation fields (u, v, q, z, χ200, vorticity) | **Not bias-corrected** | — |

`xsdba` is designed for calendar-time climate series. With forecasts, treat each lead week as a
separate group and pool start dates. That needs a small adapter, not a fork.

### 4.7 Pipeline and what to keep

Revised 2026-11-05: the weekly O96 archive is dropped from the operational design.

```
each member (hindcast or real time), at run time on N320:
  ├─ AI-WQ: N320 → 1.5° weekly tas/mslp/pr (raw)          same route as the submission
  ├─ TS:    N320 detector → tracks
  ├─ MJO:   N320 → 1.5° χ200/u850/u200 → ±15° band, daily
  ├─ CRMA:  crma_layer3_extract (55 basins, loci, regional 1.5°)
  └─ delete the full store
     (paper event starting dates only: also write the O96 field archive, §4.7.3)

weekly, after real time:
  calibrate each product against the hindcast products → submit
  save the correction tables (quintile edges, TS terciles, MJO climatology)
```

#### 4.7.1 What a later year reuses: products, not fields

When the same calendar date comes round a year later (e.g. 5 Nov 2027), refitting needs only the
**products** saved from the earlier hindcast members, and every one of them was extracted from
N320 at run time:

| Saved per hindcast member | Extracted from | Used for |
|---|---|---|
| AI-WQ 1.5° weekly tas/mslp/pr | N320 → 1.5° directly (the submission route) | quintile edges, shrinkage |
| Tropical storm tracks | N320 detector | per-basin storm-day terciles |
| MJO band means (χ200, u850, u200) | N320 → 1.5° | model climatology per lead day |
| CRMA layer 3 | the run output | the model's own climatology for the BN states |

The new year's 11 members are run, the same products are extracted from N320 and appended, and
the calibration is refitted. Nothing is regridded back to N320 and no O96 field is read, so there
is no circularity. **About 3 MB per member, ~0.7 GB per calendar date for 20 years, ~35 GB for
all 52 calendar weeks.**

Archive **raw products plus correction tables**, not corrected products. Corrections change every
year as the hindcast grows and can be regenerated in minutes; corrected values cannot be
un-corrected.

#### 4.7.2 Why there is no operational O96 archive

- **Calibration never reads it** (§4.7.1).
- **A new product later is the only use**, and even then only if hindcast *and* real time both
  compute it from O96. That is the route-consistency problem of §4.4 again. It cannot serve
  anything needing N320 or 6-hourly data, such as TC tracks.
- **Cost:** ~36 fields as daily means is ~190 MB per member (≈ 80–100 MB compressed), ~20 GB per
  weekly hindcast and ~1 TB for all calendar weeks. That is too much for a speculative use.

**If a new product is needed, add it to the extractor and rerun the affected dates.** That is
cleaner than relying on an archive grid.

#### 4.7.3 For the paper: products for reproducibility, plus O96 for the event years only

Bit-for-bit reproduction by rerunning the model is not possible: FP16 non-determinism reaches
85–97% of ensemble spread at days 18–33 (`run-pre50r1-dates/README.md`). Reproducibility therefore
rests on:

1. **Products, scores and correction tables** behind every figure and table. ~GB, deposited on
   Zenodo. These reproduce the paper exactly.
2. **Code, checkpoint ID and starting-state requests** (ARCO, CDS EDA, the MARS wave request in
   `DATA_AVAILABILITY.md`), so others can regenerate a statistically equivalent ensemble.
3. **Fields for event figures and RQ5 drivers, only for the event starting dates in the event
   year**, not all 20 climatology years:
   - **O96 global, daily means, ~36 fields** (below): ~2 GB per date for 11 members, so
     **~50–100 GB for ~24–48 event starting dates**. Covers drivers outside the region (jets,
     MJO and ENSO context). Recommended.
   - Optionally, **N320 over the ICPAC box** for tp and 2t only, for basin-scale rainfall maps.

**Fields in the event-year O96 archive: the CRMA 39-channel set** (`~/crma/DYNAMICAL_ENS_HYBRID.md`),
mapped onto AIFS output:

| | Channels | Count |
|---|---|---|
| In AIFS output directly | tp, msl, sp, 2t, 2d, 10u, 10v, skt, tcwv (AIFS `tcw`, includes condensate), ro, t850/700/500, gh500 (z/g), u & v at 925/850/700/500/200, q925/850/700/500, w700 | **29** |
| Derived, not stored | r850, r700 (from t, q); vo850/700/500, d850/700/200 (from u, v on the 1.5° grid, as `s2s_features.py` does) | **8** |
| Constant | lsm, from the checkpoint's static fields | 1 |
| Missing | **cape**. Could be approximated from 13-level t/q profiles, but treat it as absent, as layer 3 does | 1 |

Plus cp, swvl1, z200 and u/v/t at 50 hPa (stratospheric evidence; `u10`, the 10 hPa wind, is
unreliable, see `CRMA_LAYER3.md`): **~36 fields**. AIFS cannot supply OLR (the MJO uses χ200) or
SST (layer 3 uses `skt` over sea).

| Totals (estimates) | |
|---|---|
| Operational products, per weekly hindcast (220 members) | ~0.7 GB |
| Operational products, real time (51 members) | ~0.15 GB, against ~256 GB per cycle today |
| Operational products, all 52 calendar weeks × 20 years | ~35 GB, reusable every year |
| Paper: event-year O96 field archive | ~50–100 GB, one-off |

### 4.8 Not built yet

- [ ] Extraction at run time. The run script writes a full icechunk store per cycle. "Extract,
      then delete" per hindcast member needs a temporary per-member store or extraction hooks
      in the run loop.
- [ ] MJO step 5 switched to the hindcast climatology (`ts-mjo/mjo_submission.py`).
- [ ] TS model terciles from the hindcast (`ts-mjo/ts_days.py`, `--tercile-clim`).
- [ ] Quintile-edge and shrinkage tables per calendar week, and their storage layout.
- [ ] `xsdba` lead-week adapter for the event-study fields.

---

## 5. RQ3 — extreme-event detection, scored over all years

### 5.1 Why not score the events alone

Scoring only on observed disasters rewards a forecast that always predicts extremes: the
forecaster's dilemma (Lerch et al. 2017, *Statistical Science* 32(1)). Without false alarms a hit
rate means nothing. The events were chosen for their impacts and cluster in strongly forced seasons
(2019 positive IOD, La Niña 2020–22, El Niño with positive IOD 2023), which flatters skill. And
several disasters were not local rainfall extremes: riverine floods driven upstream, flash floods,
droughts made worse by conflict.

**So the events illustrate. They are not the sample.** Detection statistics come from the full
hindcast (§4), where event years and other years are scored the same way.

### 5.2 Event definition: observations first

- For each target basin or admin-1 area and its season, mark weeks (and dekads or months for
  drought) in the **driest or wettest quintile** in all 20+ years.
  - CHIRPS (1981→, 0.05°): drought
  - IMERG daily (2000→): floods
  - ERA5: the reference matching the model's own climate
- The GHACOF and EM-DAT disasters are a **labelled subset**. Report whether each one had a local
  rainfall extreme. Those that did not (e.g. Khartoum 2019, a river flood) are either moved to
  the upstream basin or reported as out of scope for a rainfall forecast.

### 5.3 Metrics

| Metric | Answers |
|---|---|
| Hit rate and false alarm rate, ROC area for P(extreme quintile) | Does it separate event weeks from other weeks? |
| SEDI (Ferro & Stephenson 2011) | Detection skill for rare events that doesn't degrade as events get rarer |
| Reliability diagram for P(extreme quintile) | When it says 40 %, does it happen 40 % of the time? |
| Brier skill score, threshold-weighted CRPS (Gneiting & Ranjan 2011) | Proper scores focused on the tail |
| Cases where the forecast predicted an extreme: frequency observed when P > 0.4 | Exposes the false alarms an events-only check hides |
| For each event: P(observed category) by lead, first lead with a signal, rank of the observation in the ensemble | The illustrative figures |

Each metric: raw against calibrated (§4.2), and AIFS against IFS extended-range reforecasts.

### 5.4 The event set (from `crma/scenario-sim/scenarios/`)

**Floods (11 GHACOF + 1 out of sample)**

| Event | Admin-1 | Season | Cluster | In training? |
|---|---|---|---|---|
| Uganda, May 2019 | Kampala | MAM | 2019 | yes (both) |
| Eritrea Highlands, Aug 2019 | Maekel | JJAS | 2019 | yes |
| Sudan Khartoum, Aug 2019 | Khartoum | JJAS (riverine) | 2019 | yes |
| South Sudan Upper Nile, Oct 2019 | Upper Nile | JJAS/OND (riverine) | 2019 | yes |
| Djibouti City, Nov 2019 | Djibouti City | OND (positive IOD) | 2019 | yes |
| Ethiopia Addis/Akaki, May 2021 | Addis Abeba | Belg | — | yes |
| Rwanda, May 2023 | Western | MAM | — | yes |
| Somalia Deyr, 2023 | Lower Shabelle | OND (El Niño + positive IOD) | 2023 | yes |
| Burundi, Apr 2024 | Bujumbura | MAM | 2024 | yes (fine-tune only) |
| Kenya Nairobi, Apr 2024 | Nairobi | MAM | 2024 | yes (fine-tune only) |
| Tanzania, Apr 2024 | Dar es Salaam | MAM | 2024 | yes (fine-tune only) |
| **Nairobi flash flood, Mar 2026** | Nairobi | MAM onset | — | **no (out of sample)** |

**Droughts (11 GHACOF)**

| Event | Admin-1 | Seasons | Cluster | In training? |
|---|---|---|---|---|
| Rwanda Akagera, 2016–17 | Eastern | MAM/OND | 2016–17 | pretraining only |
| Kenya ASAL, 2020–23 | Tana River | OND/MAM | Horn 2020–23 | yes |
| Somalia South-Central, 2020–23 | Bay | Gu/Deyr | Horn 2020–23 | yes |
| Eritrea Highlands, 2021–23 | Maekel | JJAS | 2021–23 | yes |
| Ethiopia Blue Nile, 2021–22 | Amhara | JJAS | 2021–22 | yes |
| Burundi, 2021–22 | Kirundo | OND/MAM | Horn 2020–23 | yes |
| South Sudan Upper Nile, 2021–23 | Upper Nile | JJAS | 2021–23 | yes |
| Tanzania Kagera, 2021–22 | Kagera | OND/MAM | Horn 2020–23 | yes |
| Djibouti, 2022 | Dikhil | — | Horn 2020–23 | yes |
| Sudan Kassala, 2022 | Kassala | JJAS | 2022 | yes |
| Uganda Karamoja, 2022 | Moroto | Apr–Oct | Horn 2020–23 | yes |

Seasons and window dates are to be confirmed from CHIRPS and the scenario JSONs in week 1.

**These events are not independent.** Seven droughts belong to the single 2020–23 Horn drought
(five consecutive failed seasons, OND 2020 → OND 2022), and five floods come from 2019. That is
about 6–8 independent cases, which is why the statistics rest on §4's full hindcast.

**What S2S can forecast.** Flash floods (Nairobi, Addis, Djibouti) cannot be forecast at weeks
3–4. Their target is the wet-week anomaly over the area during the window. Droughts span several
seasons while 33 days covers only part of one. Their target is dry spells and deficits within
each failed season.

### 5.5 Controls

For each event, add one or two **years without the event** in the same region and season
(near-normal or the opposite extreme), with forecasts shown side by side.

---

## 6. RQ4 — training-data overlap

Split all skill scores by period:

| Period | Status in training | Initial conditions |
|---|---|---|
| 2005–2017 | ERA5 pretraining only | Route C |
| 2018–2024 | Pretraining and IFS-analysis fine-tuning | Route C |
| **2025–2026** | **Out of sample** | Route C, plus the real-time route A cycles and route B |

Test whether the difference in skill exceeds what resampling years would produce (block
bootstrap by year). A small difference is publishable in itself. It would support the argument
that a 33-day autoregressive rollout of a model trained on 6-hour steps does not reproduce
memorised trajectories. Nairobi March 2026 and the OND 2026 cycles (§3) are the out-of-sample
showcase.

---

## 7. RQ5 — drivers, from the existing BN work

Reuse `epistemic-reasoning-risk/`: the k = 2 circulation regime, Somali and Turkana jets, TEJ,
Mascarene and St Helena ridge position, IVT. Add the IOD and ENSO state from SST and the MJO phase
from `ts-mjo/`. For each event and its controls, did the forecast carry the observed driver state?
This turns hits and misses into explanations, the "understanding" argument already in that
folder's README.

This needs the O96 corpus or the extracted CRMA layer 3, so keep the relevant drivers in the
hindcast output (§4.1).

---

## 8. Benchmarks

| Benchmark | Why | Source |
|---|---|---|
| Climatology | RPSS reference | AI_WQ_package and our own |
| **ECMWF extended-range reforecasts** (ERA5 starts, 11 members, 20 years) | A like-for-like physics model on the same events | S2S database / ECMWF data portal |
| ECMWF extended-range real time (2025–2026) | Like-for-like out of sample | S2S database (3-week delay, fine for research) |
| **AIFS-SUBS reforecasts** (`class=rd expver=j339`; 20 years, 10 members, 46 days, ERA5 recentred on EDA) | ECMWF's own sub-seasonal ML model: an ML-to-ML benchmark, and a precedent for ERA5+EDA starts | MARS research data (arXiv:2607.05100) |
| SEAS5 | Seasonal context for droughts | C3S |

---

## 9. Timeline

### 9.1 Four weeks: a preprint or conference-paper draft, reduced design

| Week | Dates | Work | Gate |
|---|---|---|---|
| 1 | 9–13 Nov | Protect the Sep–Oct cycles (§3.1). Generalise `fetch_era5_l10.py` into a full ERA5 starting-state builder. Full-ERA5 donor test (§2.4) and wave-band choice (§2.3). Start CDS downloads. Confirm event windows. Download IFS reforecasts. Write to ECMWF about the wave hindcast | §2.4 passes |
| 2 | 16–20 Nov | RQ1 product leg on 7 cycles. Hindcast (12 dates × 20 yr × 11 members), local plus cloud burst. Verification code | RQ1 within draw noise |
| 3 | 23–27 Nov | RQ2 calibration comparison. RQ3 scoring over all years plus event figures. RQ4 period split. RQ5 drivers for hits and misses. RQ1 verification as OND weeks arrive | — |
| 4 | 30 Nov–4 Dec | Figures, writing, internal review | — |

### 9.2 Journal-ready version: 2–3 months

Adds the full hindcast (48 dates × 11 members), the 50-member ERA5 ensemble for RQ1, the
ECMWF wave hindcast if granted, and a second season of out-of-sample cycles.

### 9.3 Risks

| Risk | Effect | Mitigation |
|---|---|---|
| CDS queue | Hindcast starts late | ARCO for surface fields, batch by month, start in week 1 |
| Full-ERA5 test fails | Route C unusable | Route B for 2025–2026. Rethink the 20-year climatology (e.g. lagged real-time cycles) |
| Wave-band fill costs too much | Hindcast quality | Fill with ERA5 spectra, or wait for ECMWF data |
| GPU shared with weekly AI-WQ cycles | Slower | Cloud Run burst for the hindcast |
| Disk (~1.8 TB held for RQ1) | Real-time cycles blocked | Keep only the subset in §3.1 |
| Events-only framing by reviewers | Rejection | §5 is built to avoid this |

---

## 10. Paper outline

1. Introduction: ML weather models at S2S range; why the Greater Horn; acceptance takes more than
   a skill score.
2. Data and model: AIFS-ENS v2, its 15-day design range and training periods; the three
   initial-condition routes; observations.
3. Methods: hindcast design, calibration, verification over all years, event definition,
   training-overlap test.
4. Results:
   - (a) RQ1 initial-condition consistency
   - (b) RQ2 drift and calibration
   - (c) RQ3 detection skill over all years
   - (d) event showcase with controls
   - (e) RQ4 in sample against out of sample
   - (f) RQ5 drivers
5. Discussion: operational implications for ICPAC and GHACOF; limits of the competition score.
6. Conclusions.

**Possible journals:** AMS *Artificial Intelligence for the Earth Systems*, *Weather and
Forecasting*, *npj Climate and Atmospheric Science*; *GRL* for a short version.

---

## 11. Open decisions

1. ~~Hindcast members: 5 or 11?~~ **Decided 2026-11-05: 11** (ARCO control + 10 ERA5 EDA
   members from CDS).
2. How many calendar dates fit the month: 12 (4-week version) or 24 (reduced)? This depends on
   the cloud budget.
3. Wave-band fill: climatology or NaN? Decide from §2.3's test.
4. Cloud budget for the hindcast burst.
5. Co-authors and roles.
6. Operational hindcast cadence: every week (~20 GPU-h) or every other week with ±7-day pooling
   (~10 GPU-h/week averaged)? See §4.4.

## 12. Related files

- `../fp16FahamuAIFSv2/run-pre50r1-dates/README.md`: donor fields, sensitivity harness, the
  non-determinism floor.
- `../fp16FahamuAIFSv2/CYCLE_DATA_INVENTORY.md`: what each cycle still holds.
- `../fp16FahamuAIFSv2/epistemic-reasoning-risk/`: BN diagnostics, ESS, regimes, jets.
- `../fp16FahamuAIFSv2/ts-mjo/ERA5_VPM_CLOUD_BRIEF.md`: ARCO-ERA5, native N320.
- `../fp16FahamuAIFSv2/O96-icechunk-store/forecast-evaluation/`: RPSS reproduction, draw noise.
- `../crowd-fahamu/lp.txt`, `../crowd-fahamu/vd.txt`: two AI-WQ teams' post-processing (SPQR,
  hindcast thresholds).
- `~/crma/scenario-sim/scenarios/*.json`: the 23 event definitions.
- `../fp16FahamuAIFSv2/ts-mjo/MJO_METHOD.md`, `TS_STORM_DAYS.md`: MJO and tropical-storm-days
  products, both waiting on a model climatology (§4.5).
- `../fp16FahamuAIFSv2/epistemic-reasoning-risk/CRMA_LAYER3.md`: layer-3 extraction and its gaps.
- `~/crma/DYNAMICAL_ENS_HYBRID.md`: the CRMA 39-channel design (§4.7).
