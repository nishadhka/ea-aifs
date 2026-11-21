# bc-fp16FahamuAIFSv2 — bias-corrected and calibrated AIFS-ENS v2

**Status (2026-11-11):** a 20-year **control** hindcast for 5 Nov is built and corrects
tas / mslp / pr (§0.1). The hindcast is being extended to **15 variables** (§0.2). The
220-member version (control + 10 EDA) is not built yet. The past-cycle correction (§0) is
superseded where a hindcast exists.

A second model entry alongside the raw `../fp16FahamuAIFSv2/`. Same checkpoint
(`ecmwf/aifs-ens-2.0`), same 50-member real-time run each Thursday. The difference: each
real-time product is calibrated against a **same-date hindcast** run from ERA5 starting states.
The raw entry keeps running unchanged, so the two can be compared cycle by cycle.

Design background: `../paper/AIFS_ENS_V2_S2S_EXTREMES_STUDY_PLAN.md` §4.4–4.8 (weekly hindcast,
product calibration, what to keep) and `../paper/DATA_AVAILABILITY.md` (ERA5 starting states,
including the 6 wave bands).

---

## Introduction: why bias-correct AIFS, and why this way

### Background

Since mid-2026 the Fahamu team has run **AIFS-ENS v2** (`ecmwf/aifs-ens-2.0`), ECMWF's
machine-learning ensemble, every Thursday on this box. It is started from the IFS
open-data analysis, run 792 h (33 days), and its week-3 and week-4 forecasts of tas, mslp
and pr go to the **AI Weather Quest (AI-WQ)** as quintile probabilities, with tropical-storm
days and MJO alongside. The same runs feed ICPAC's regional work on Greater Horn of Africa
droughts and floods at sub-seasonal range (study plan §0).

The submitted product is the **raw** ensemble: each member is counted against AI-WQ's
20-year ERA5 quintile edges, with no correction. Every operational sub-seasonal system
calibrates instead. ECMWF's extended-range ENS, NCEP CFSv2 and ECCC all run **reforecasts**
(hindcasts: the same model started from past dates) and correct the real-time forecast
against the model's own climate. ECMWF's sub-seasonal AIFS (AIFS-SUBS) is no exception: a
20-year, 10-member reforecast started from ERA5 recentred on the EDA (`j339`, study plan
§2.3). No such reforecast is published for AIFS-ENS v2, so the raw product carries
whatever the model gets systematically wrong.

### What the raw forecasts showed

**The raw product is already competitive.** AI-WQ leaderboard, SON 2026, period-aggregated
RPSS after competition week 8 (starts 13 Aug – 1 Oct, published 6 Nov 2026; from the
leaderboard's `response.json`):

| | Rank (of 31 models) | Average | tas | mslp | pr | ECMWF dynamical: average / tas | Leader (SPQRnow): average / tas |
|---|---|---|---|---|---|---|---|
| Week 3 (days 19–25) | #16 | 0.196 | **0.141** | **0.336** | 0.111 | 0.193 / 0.191 | 0.231 / 0.227 |
| Week 4 (days 26–32) | #17 | 0.161 | **0.063** | **0.343** | 0.078 | 0.168 / 0.161 | 0.206 / 0.196 |

It ranked in the top 10 through most of the period (week 3: #9–10 after weeks 3–6;
week 4: #5–8 after weeks 1, 2, 4, 5), then weekly scores of 0.112 / 0.053 in week 7 pulled it
to #16 / #17. In week 3 it beats ECMWF's dynamical model on average, and its week-4 **mslp is
the best of all models**. **tas is the weak variable**: 0.141 and 0.063, against 0.191 and 0.161
for ECMWF's dynamical model. Matching ECMWF on tas alone, holding mslp and pr fixed, would give
an average of ~0.213 / ~0.194, about #7 / #5 on the same table.

So the correction is aimed where the score is lost: **tas first, pr second; mslp is left raw
for AI-WQ** (the hindcast also finds mslp correction neutral globally, §0.1) and corrected only
for the regional products, where its Greater Horn bias matters.

| Finding | Evidence |
|---|---|
| **tas is the weakest variable** | leaderboard above. Locally, the raw leave-one-out RPSS over 7 cycles is tas +0.033, mslp +0.245, pr +0.078 (§0). It includes two summer cycles and is harsher than the official score, but it points the same way |
| **Week-to-week scores swing widely** | weekly variable-averaged RPSS ranges 0.053–0.260 across the 8 SON weeks (rank #4 to #28) |
| **The model drifts with lead.** A 6-hourly autoregressive model run for 132 steps accumulates bias, worst at weeks 3–4, exactly the AI-WQ weeks | lead-dependent bias maps in §0 and §0.1; 20-year hindcast for 5 Nov: significant tas bias over 20–24 % of land, mslp over the Greater Horn +0.8–0.9 hPa on ~2/3 of the region (§0.1) |
| **The MJO is too strong** | ensemble amplitude 2.84 vs 1.30 observed on 20260924 (§4, step 3) |
| **Tropical storms are over-counted** in the NW Pacific | §4, step 3 |
| **The ensemble is overconfident** over East Africa | effective ensemble size ~1.3–1.9 out of 50 (`epistemic-reasoning-risk/CROSS_CYCLE_FINDINGS.md`), which already called a model climatology "the clear priority" |
| **Different random draws flip the modal quintile** in 28–40 % of cells | `RPSS_LEADERBOARD_DISCREPANCY.md` |

The problem at S2S range is bias, drift and correlated members, not missing spread. That is
why correction comes before more members or new model work (ranking in §4).

### Why this path

1. **Correct the products, not the model.** The corrections are applied to what is
   submitted and used (weekly fields, storm counts, MJO indices, regional indices), not to
   the network or its input fields. Fine-tuning or Bayesian deep learning on the weights is
   later research (§4); calibration of products is cheap, reversible and standard practice.
2. **Use a same-date hindcast, because the bias depends on date and lead.** A first,
   quicker attempt used the bias seen in recent real-time cycles (§0). The 20-year
   hindcast then showed those tables barely match the model's real November bias (pattern
   correlation −0.18 to +0.29): with a few recent cycles, the "bias" is mostly that
   season's weather. Twenty years of the same calendar date average the weather out and
   leave the model's own error.
3. **Start the hindcast from ERA5 (route C), because no IFS analyses exist for 2006–2025
   in the form AIFS needs, and ERA5 is what AIFS was pretrained on.** The 106 atmosphere
   and land fields come from ARCO-ERA5. The 6 wave-period bands come from the ERA5-forced
   wave hindcast AIFS was trained on. The gate test shows an ERA5-started member is
   indistinguishable from the IFS-started members at weeks 3–4 (§0.1), so the hindcast
   measures the same model behaviour as the real-time run.
4. **It is affordable on this box.** One member takes 5.3 min of GPU and ~3 GB of
   temporary disk; nothing large is kept. A 20-member control hindcast per date is
   ~1.8 h of GPU, the full 220 members ~19 h, and each date is reused every later year
   (§0.2, §2).
5. **Extend beyond the competition where it serves the region.** The same hindcast
   calibrates 10 extra indices chosen for Africa and the Indian Ocean (IOD, ITCZ, Somali
   Jet, soil moisture, heat stress and others, §0.2) at no extra GPU cost.

The raw entry keeps running unchanged, so every week gives a paired comparison of raw and
corrected forecasts.

---

## 0. Interim: bias correction from past real-time cycles (built)

Until the hindcast exists, the tas / mslp / pr product is corrected with the bias seen in
earlier real-time cycles. Only each cycle's `nc_1p5deg` (2.1 GB) is needed, so the O96 and
N320 stores can be deleted once 3a has run.

| Script | Does |
|---|---|
| `past_cycle_bc.py aggregate` | weekly member values per cycle, exactly 3b's aggregation (float64: reproduces 3b's 20261105 file in every cell) |
| `era5_weekly.py build` | ERA5 weekly tas/mslp/pr from ARCO ERA5T, because the AI-WQ archive has no obs after week 20260824. Same definitions as AI-WQ: tas/mslp 6-hourly mean Mon 00–Sun 18, pr hourly sum Mon 01–Mon 00, every 6th 0.25° point. Matches AI-WQ to 0.0002 K / 0.02 Pa / 0.002 mm rms on 20260727, 0803, 0824 |
| `past_cycle_bc.py evaluate` | leave-one-cycle-out RPSS (AI-WQ's own scorer) for smoothing σ ∈ {0,1,2,4} grid points × strength a ∈ {0.5,1} |
| `past_cycle_bc.py apply --date D` | `aifs-run/D_0000/aiwq_bc/ensemble_quintile_probabilities_D_bc.nc` + the bias tables |

tas, mslp: x − a·b, b = mean over cycles of (ensemble mean − ERA5), per lead week and grid point.
pr: x / r^a, r = forecast/ERA5 ratio, limited to ⅓–3. Cache: `/tank/projects/bc-aifs/`.

Leave-one-out result, 7 cycles (0709, 0813, 0917, 0924, 1001 both weeks, 1008 week 1),
11 cycle-weeks per variable, computed 2026-11-08:

| Var | Raw RPSS | Corrected | Setting | Cycle-weeks improved |
|---|---|---|---|---|
| tas | +0.033 | **+0.111** | σ=0, a=0.5 | 9/11 |
| mslp | +0.245 | **+0.270** | σ=4, a=0.5 | 9/11 |
| pr | +0.078 | +0.084 | σ=0, a=0.5 | 7/11, within noise |

Caveats: the setting is chosen on the same cycles, so these numbers are a little
optimistic. Neighbouring cycles verify on neighbouring weeks, and persistent weather
anomalies make that look like model bias. 0709 and 0813 are summer cycles from older
regrid routes. Half strength wins everywhere, which fits a bias estimated from few cycles.

### 0.1 Same-date hindcast, control only: 5 Nov (built, 2026-11-10)

20 ERA5-started members, 5 Nov 2006–2025, one per year (`era5_state_pkl.py` →
`hc_date.sh` / `hc_member.sh` → `hc_bc.py`). Each member takes 3.6 min end to end
(rollout + 3a regrid); temporary stores peak at 265 MB and are deleted, leaving ~40 MB.
Building each state takes ~7 min from ARCO plus MARS waves; tape waits of up to several
hours are why `prefetch_waves.py` exists.

**Gate (study plan §2.4, S2S form):** an ERA5-started 20261029 member against that day's
50 IFS members, weeks 3–4: bias ≤ 0.05 ensemble spread, distance from the ensemble mean
0.9–1.2× a typical IFS member, flat rank histogram. Starting from ERA5 makes no
measurable difference at these leads.

**Leave-one-year-out MSE**, significance-masked correction (`sig_a0.5`: only cells whose
20-year bias is significant at p < 0.05, half strength), 40 cases (20 years × 2 weeks):

| Var | Global (land for tas/pr) | Greater Horn of Africa land | Cases improved, global / GHA |
|---|---|---|---|
| tas | −2.7 % | −5.7 % | 36/40 / 30/40 |
| mslp | +0.0 % | −16.7 % | 26/40 / 31/40 |
| pr | −2.6 % | −10.0 % | 35/40 / 37/40 |

Biases found over the GHA: mslp +0.8–0.9 hPa (significant on ~2/3 of the region), tas
−0.15/−0.37 K, pr −3.9/−1.4 mm per week (weeks 3/4). The largest corrections are over the
Tibetan Plateau (tas up to +13 K, mslp −1.4 hPa ×10), where interpolated forecast vs
point-sampled ERA5 and the sea-level reduction add to model drift. They are systematic
against the scoring target, so they stay.

**The past-cycle tables (§0) barely match the 20-year bias** (pattern r −0.18 to +0.29):
they mostly captured the Sep–Oct 2026 weather and the season, so the §0 numbers are
inflated by persistence. Prefer the hindcast correction where one exists. For 20261105:
`aiwq_bc_hc/ensemble_quintile_probabilities_20261105_bchc.nc` changes the modal quintile
in 3–4 % of cells, against 20–24 % for the past-cycle file.

Not yet: the 10 EDA members (CDS token works, `run-pre50r1-dates/.env`) to reach 220 runs
and model-climate quintile edges; other calendar dates.

The 51st member (control) comes from HRES `oper`/`wave` step 0, because open data has no
ENS control (`s3_grib_pkl_input_aifsens_v2_proto.py --control-member 51`,
checked by `check_control_pkl.py`).

### 0.2 Fifteen variables, 20-year same-date hindcast (in progress, 2026-11-11)

The 20-year hindcast is extended from tas / mslp / pr to 15 variables: the 5 AI-WQ
variables plus 10 indices chosen for Africa and the Indian Ocean. All 15 come from fields
the model already writes, so the GPU cost stays the same. Only the extraction, done before
each member's stores are deleted, gets longer.

#### Variables, definitions, and what each needs

Boxes are lat °N × lon °E; weeks 3 and 4 are lead days 19–25 and 26–32 (AI-WQ's
forecast periods), and "daily" means lead days 1–33.

| # | Variable | Definition (model fields, box) | Grid / steps read | Kept per member | Truth for 2006–2025 | Correction | AI-WQ? |
|---|---|---|---|---|---|---|---|
| 1 | tas | weekly mean `2t` | N320 → 1.5° (3a), 432–792 h, 6-hourly | 1.5° weekly maps | ERA5 (`era5_weekly.py`) | bias (§0.1), then model-climate quintiles | yes |
| 2 | mslp | weekly mean `msl` | as 1 | as 1 | ERA5 | as 1 | yes |
| 3 | pr | weekly sum `tp` | as 1 | as 1 | ERA5 | ratio bias, quintiles | yes |
| 4 | Tropical storm days | tracked storms (`ts-mjo/ts_days.py`: `msl 10u 10v u_850 v_850 t_200 t_300 t_500`), unique (storm, day) pairs per basin per week | N320 sidecar, 432–792 h, 6-hourly | `storm_days(week, basin)` | IBTrACS v04r01 (`/tank/projects/ibtracs`), AI-WQ's `count_nstormdays` | tercile edges per basin and week from the model's own climate | yes |
| 5 | MJO | VPM/RMM bands from `u_200 v_200` (velocity potential) and `u_850`, `u_200` (`ts-mjo/vpm_index.py --dump-bands`) | O96, 0–792 h, daily means of 6-hourly | daily normalised bands, lead days 1–33 | ERA5 VPM per year (`/tank/projects/era5_vpm_clim`, 1991+) | amplitude scaling and phase offset per lead day | yes |
| 6 | IOD (DMI) | `skt` west box (−10..10, 50..70) minus east box (−10..0, 90..110) | O96, 0–792 h, daily | daily W, E box means | ERA5 SST (ARCO) | per lead day: bias and variance; terciles at ±0.4 °C on the anomaly | extra |
| 7 | ITCZ | `tp`-weighted latitude in −20..20, sectors Africa (−20..55), Indian Ocean (55..100), East Africa (30..52) | O96, 0–792 h, daily sums | daily latitude per sector | ERA5 tp (ARCO) | per lead day latitude bias | extra |
| 8 | Indian Ocean westerlies / Somali Jet | `u_850` (−5..5, 70..90); `u_925`, `v_925`, speed (−5..10, 40..55) | O96, daily | daily box means | ERA5 u, v | bias, terciles | extra |
| 9 | Soil moisture | `swvl1`, `swvl2` | O96, daily + weekly maps | GHA box daily; Africa weekly maps | ERA5 swvl | bias per cell, terciles | extra |
| 10 | Mascarene High | `msl` (−35..−20, 40..90) | O96, daily | daily box mean | ERA5 msl | bias, terciles | extra |
| 11 | Tropical Easterly Jet | `u_150`, `u_200` (0..15, 40..80) | O96, daily | daily box means | ERA5 u | bias, terciles | extra |
| 12 | African Easterly Jet | `u_600`, `u_700` (10..15, −20..20) | O96, daily | daily box means | ERA5 u | bias, terciles | extra |
| 13 | Western Indian Ocean waves | `swh`, `mwp` (−12..12, 39..60), sea only | O96, daily + weekly maps | daily box means; weekly maps | ERA5 waves (ARCO) or ecWAM hindcast (MARS) | bias, terciles | extra |
| 14 | Moisture into East Africa | `tcw` (−5..15, 30..52); 850 hPa flux `q·u`, `q·v` (−5..10, 40..52) | O96, 6-hourly → daily | daily box means; weekly `tcw` maps | ERA5 tcw, q, u, v | bias, terciles | extra |
| 15 | Heat stress | daily max of `2t` (max of 4 six-hourly values) and daily max wet-bulb temperature (Stull 2011 from `2t`, `2d`), Africa land | O96, 6-hourly | GHA box daily; Africa weekly maps | ERA5 2t, 2d (6-hourly, same daily-max rule) | bias per cell, terciles | extra |

Weekly maps cover the Africa–Indian Ocean domain (−40..40, −25..120) on the O96 cells. The
daily-max rule for 15 uses the 6-hourly values both for the model and for ERA5, so the
comparison is like for like (a true daily maximum needs hourly data).

#### Time steps each run must store

| Store | Grid | Steps | Fields | Serves |
|---|---|---|---|---|
| main | O96 (40 320 cells) | **all 132 steps, 6–792 h** | all model outputs | 5–15 (daily series need every day from day 1; MJO uses the whole lead) |
| sidecar | N320 (542 080 points) | **61 steps, 432–792 h** | `msl tp 2t 10u 10v t_200 t_300 t_500 u_850 v_850` | 1–4 (3a regrid; storm tracking at full resolution) |

These are the real-time chain's own settings (tier B), so the hindcast and the real-time
forecast are processed identically. The extractor (`hc_extract.py`) reads any store, so the
same code makes the real-time products that get corrected.

#### Cost per member and per date

| | Per member | 20 control members | 220 members |
|---|---|---|---|
| GPU, rollout (all steps written to O96) | 4 min 50 s (measured) | 1.6 h | 17.7 h |
| Extraction (3a, storm tracking, MJO, indices) | 30 s, CPU (measured) | 0.2 h | 1.8 h |
| Temporary stores (deleted per member) | 3.1 GB (O96 2.1 GB + N320 1.0 GB, measured) | 3.1 GB | 3.1 GB |
| Kept | ~42 MB (1.5° file 41 MB + products, measured) | 0.85 GB | 9.2 GB |
| Starting states | 0.97 GB pkl | 1 GB at a time | 11 GB per year at a time (13–26 GB peak) |

#### Year-by-year run before the Thursday start

The hindcast for Thursday date D uses D in 2006–2025. It is built Monday to Wednesday, so
the bias tables exist when the real-time run starts on Thursday.

| Day | Control-only (20 runs) | 220 members |
|---|---|---|
| Mon | prefetch MARS waves (40 files, absorbs tape waits); ERA5 verification for D+18..D+32 in 2006–2025 (~17 GB transfer, 24 MB kept); IBTrACS and ERA5 VPM already on disk | as control, plus queue the CDS EDA requests (~20 GB) |
| Tue | states and members, years 2025 → 2006, streamed: ~16 min state build per year (network) overlapping ~5 min GPU per year. **Done in ~5.5 h** | years 2025 → ~2014, 11 members per year (~55 min GPU per year), next year built while the GPU runs |
| Wed | `hc_bc.py evaluate` + bias tables for all 15 variables | years ~2013 → 2006, then evaluate + bias tables |
| Thu | real-time cycle (Step 1 → Step 2 → 3a/3b, TS, MJO, `hc_extract.py`), apply the tables, submit raw and corrected | same |

Per year, the order is always: control state (ARCO + MARS) → (EDA perturbed states) →
rollouts → extraction → delete stores and pkls. After a date is built once, later years
add only that year's members (1 control, or 11), about 5 min or 1 h of GPU.

### 0.3 Fifteen-variable results, 5 Nov, control only (2026-11-12)

20 ERA5-started members (one per year, 2006–2025) with all 15 products (`hc_member.sh`),
against ERA5 (tas/mslp/pr; indices through `era5_truth_indices.py`, the same
`hc_extract.compute()` as the model), IBTrACS (storm days) and AI-WQ RMM (MJO). Full tables:
`/tank/projects/bc-aifs/hc15_eval_1105.md`. Leave-one-year-out throughout; with one member
per year the scores are deterministic (error, hit rate), so they test whether a bias is
stable, not whether probabilities are reliable (COLAB_PLAN.md §5.1).

| Variable | Result | Use for 20261105 |
|---|---|---|
| tas | −4 % error, 16/20 years improved in both weeks | corrected at half strength |
| pr | −3.3 % error, 19/20 and 16/20 years improved | corrected at half strength |
| mslp | no gain (9–10/20 years) | **raw** (best on the AI-WQ leaderboard already) |
| Tropical storm days | model-climate terciles no better than raw (NWP week 3 worse: hit 0.60 vs 0.70); most late-Nov basin-weeks are not scoreable by AI-WQ's rule (observed counts mostly 0). Count biases: NWP under-counted (1.4 vs 2.5 and 0.6 vs 1.8 days), ATL over-counted (1.1 vs 0.3) | **raw**; needs more members or pooled dates |
| MJO | per lead-bin map onto RMM: category hit 0.34→0.40 (days 19–25), 0.32→0.40 (26–33); RMM error 1.51→1.30, 1.45→1.33 | file written, **not validated for real time**: real-time members' amplitude (~3.1 at days 10–19) is about twice the hindcast members' (~1.5). Test ERA5 vs IFS starts on the same date before using it |
| Indices (24 weekly series) | bias removal helps where the bias is significant (|t| > 2): soil moisture, Somali Jet v925 and speed (week 3), TEJ, Africa ITCZ (week 3), moisture flux, wet-bulb; non-significant ones get slightly worse, so they are left raw | corrected only where it lowered the leave-one-out error; tercile probabilities against the model's own climate |
| Weekly maps (Africa–Indian Ocean) | MSE change domain / Greater Horn land: swvl1 −31…−34 %, swvl2 −40…−44 %, skt −27…−32 %, tmax −16…−23 %, wet-bulb max −18…−30 %, tp −6…−8 % / −20 %, swh −1…−3 % / −12…−21 %, mwp −10…−17 %, tcw −2…−4 % / −6…−11 % | corrected (significance-masked, half strength) |

Files for 20261105 in `aifs-run/20261105_0000/aiwq_bc_hc/`: `ensemble_quintile_probabilities_20261105_bchc.nc`
(tas and pr corrected, mslp raw), `mjo_probs_20261105_bchc.nc` (not validated),
`ts_days_probs_20261105_bchc.nc` (flagged not for submission), `indices_20261105_bchc.nc`,
`weekly_maps_20261105_bchc.nc`. None has been submitted.

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
