# MJO and tropical-storm-days diagnostics from the AIFS-ENS Icechunk store

Builds the two AI Weather Quest **non-gridded** targets out of a per-cycle
`icechunk_v2` store (50 members, 120 variables, native N320, hours 432-792):

| Target | AI-WQ definition | Evaluation | This folder |
|---|---|---|---|
| MJO phase | [training_data#mjo-data-processing](https://ecmwf-ai-weather-quest.readthedocs.io/en/latest/training_data.html#mjo-data-processing) | [BSS over 9 categories](https://ecmwf-ai-weather-quest.readthedocs.io/en/latest/forecast_evaluation.html#mjo-phase-probability-forecasts) | `mjo_index.py` |
| Tropical storm days | [training_data#tropical-storm-days-data-processing](https://ecmwf-ai-weather-quest.readthedocs.io/en/latest/training_data.html#tropical-storm-days-data-processing) | [RPSS on terciles per basin](https://ecmwf-ai-weather-quest.readthedocs.io/en/latest/forecast_evaluation.html#tropical-storm-days-ts) | `ts_days.py` |

`store_io.py` holds the shared store readers (written-step detection, valid
times, basin masks, day/7-day-window grouping).

```bash
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
STORE=/tank/projects/aifs-run/20260730_0000/icechunk_v2
TAG=cycle-20260730_0000-final

$PY ts_days.py   --store $STORE --tag $TAG --init 20260730 --out ts_days_probs.nc
$PY mjo_index.py --store $STORE --tag $TAG --init 20260730 --dump-bands mjo_bands.npz
```

A cycle covering days 18-33 yields **16 days -> 2 full 7-day windows**, which is
what AI-WQ scores (its week-3 / week-4 targets).

---

## Two blockers, stated up front

### 1. MJO: the model does not forecast OLR

The canonical RMM combines **OLR + U850 + U200**. `U850`/`U200` are present and
verified; OLR is **not obtainable**, and this is not a writer setting we can flip.

AIFS-ENS-2.0's complete single-level output is 35 variables, **all surface**:

```
100u 100v 10u 10v 2d 2t cdww cp h1012 h1214 h1417 h1721 h2125 h2530 hcc lcc
mcc msl mwp ro sd sf skt snowc sp ssrd stl1 stl2 strd swh swvl1 swvl2 tcc tcw tp
```

`ssrd`/`strd` are *downward surface* fluxes; there is no top-of-atmosphere term
of any kind. `ttr` is outside the checkpoint's output space, so it cannot be
added by re-running inference. ERA5 cannot substitute either: the forecast days
(18-33) are in the future, and ERA5 only covers dates already past.

**Wind-only is not a fallback - it is not a valid projection.** The RMM projects
onto Wheeler & Hendon's *combined* EOFs: a single 3 x 144 = 432-element state
vector ordered `[OLR, U850, U200]`, with `WH04_RMM_stddevs.nc` normalising the
PCs *that basis* produces. Dropping the OLR block and projecting the 288-element
wind vector onto what remains gives something that is not an EOF of the
wind-only space - not orthonormal there, not variance-maximising - so its PCs are
not RMM1/RMM2, the published standard deviations do not apply, and the
`amplitude < 1` inactive test loses its meaning.

`mjo_index.py` therefore **refuses** to project without OLR:

- `--olr-source FILE` - the full, valid 3-field projection
  (`index_kind = "rmm3"`, `submittable = yes`). For *past* dates that file can be
  ERA5 (`olr = -ttr / accumulation_seconds`); for forecast dates only an emulator
  can supply it.
- `--olr-source none` (default) - stops after the normalised band anomalies
  (steps 1-5) and says why. Adding `--allow-wind-only` emits the truncated-EOF
  index anyway, but every variable is renamed `windproxy_*` and the file carries
  `submittable = NO`, so it cannot be mistaken for an MJO product.

**Way forward - only two honest options:**

1. **Statistical OLR emulator.** Tropical OLR is dominated by deep convection,
   and the model does output `hcc`, `tcc`, `cp`, `tp`, `tcw`. Fit `ttr` on those
   predictors in ERA5 (where both sides exist), apply to the forecast fields,
   then run the standard 3-field projection. This keeps the official EOF basis
   intact - the only route that can yield a real RMM - but it is an emulator and
   must be validated against ERA5-derived RMM before any submission.
2. **Do not submit the MJO target**, and submit only what this model supports.

The wind-only mode is *not* a third option: it is a diagnostic aid, kept for
inspecting zonal wind structure, and is barred from producing an MJO file.

### 2. Both targets need external reference data

Neither diagnostic is self-contained in a single forecast:

| Needed | Used by | Source | Why a forecast alone cannot supply it |
|---|---|---|---|
| Observed combined EOFs + RMM sds (`--eofs`) | MJO step 6-7 | **`AI_WQ_package.retrieve_MJO_projection_data(password)`** -> `WH04_combinedEOFs.nc`, `WH04_RMM_stddevs.nc` | the projection basis is Wheeler & Hendon's, from observations |
| ERA5 day-of-year climatology 1979-2025 (`--clim`) | MJO step 3 | build from ERA5 (**not distributed**) | anomalies are defined against observed climatology |
| Preceding 120-day mean (`--lowfreq`) | MJO step 4 | ERA5 up to init + forecast days 1-17 | the filter reaches 120 days *before* each day; a 16-day window cannot contain it |
| Tercile boundaries from a matching climatology (`--tercile-clim`) | TS days | **`AI_WQ_package.TS_processing.download_IBTRACS_compute_TStercile_climatology(date, ibtracs_file, savedir)`** — or build from IBTrACS with *this* detector | the package's version is built from **observed** IBTrACS counts; ours are proxy-detector counts, and the two only interchange if the detector is bias-corrected (see §"Can TS be submitted?") |

The EOF blocker is therefore **already solved by the AI-WQ package** - use the
distributed `WH04_combinedEOFs.nc` rather than computing EOFs from your own data
(self-computed EOFs differ in sign and mode order, which silently rotates every
phase; this is the most common cause of an RMM that will not reproduce the
official `MJO_DAILY` files).

Note the `--lowfreq` row is a second, separate consequence of `--write-hours
432-792`: the store holds no forecast days 1-17, so the trailing 120-day window
has a gap between init and day 18. Storing `u_200`/`u_850` over the +-15 deg band
for *all* lead hours would close it at negligible cost (the tropical band is
135 680 of 542 080 points, two variables).

Without `--eofs`, `mjo_index.py` deliberately **stops after step 5** and writes
the normalised band anomalies (`--dump-bands`) rather than inventing a basis.
Without `--tercile-clim`, `ts_days.py` falls back to the ensemble's own terciles
and prints a loud warning - those probabilities are self-referential and not
submission-grade.

---

## MJO processing (`mjo_index.py`)

Implements the documented chain on the store's native N320 points, with no
regridding step: points within +-15 deg are binned straight into **144 x 2.5 deg
longitude bins** with cosine-latitude weighting (steps 1-2), then daily-averaged
from the 6-hourly steps.

Steps 3-4 are applied only if the corresponding reference file is given; step 5
uses the Wheeler & Hendon (2004) factors (OLR 15.1 W m-2, U850 1.81 m s-1,
U200 4.81 m s-1). Phases follow AI-WQ's 9 categories: octant 1-8 from
`arctan2(RMM2, RMM1)`, **phase 0 when amplitude < 1**. A member's weekly phase is
its modal daily phase; the 50 members are then counted into the 9 categories.

Verified on `cycle-20260730_0000`: 16 days x 144 bins, all bins populated
(135 680 N320 points fall inside +-15 deg).

## Tropical storm days (`ts_days.py`)

AI-WQ's observed diagnostic comes from **IBTrACS**, which a forecast obviously
cannot use, so the storms must be detected in the model fields. The day/week
logic is kept identical to the spec (a day counts once if any sub-daily record
qualifies; daily flags summed over 7 days), with detection:

```
sqrt(10u^2 + 10v^2) >= 17 m/s                      (--wind-threshold)
  AND  msl <= basin-median msl - 5 hPa             (--msl-drop, a closed low)
  AND  t300 >= same-latitude-band median + 0.5 K   (--warm-core, 0 disables)
  AND  cyclonic zeta850 >= 1e-4 1/s                (--vorticity, 0 disables)
```

**Vorticity is derived, not read.** `vo` is not an AIFS-ENS-2.0 output, so
`grid_ops.py` computes `zeta = 1/(a cos phi) [dv/dlambda - d(u cos phi)/dphi]`
directly on the N320 *reduced* Gaussian grid: exact circular differences along
each latitude row, and a precomputed nearest-longitude index map between
adjacent rows. That costs ~1.5 s per member, against ~26 min if every field were
regridded to a regular grid purely to differentiate it. Validated against
physics: zonal means are O(1e-6) 1/s, and the deepest 1 % of `msl` carries
+5.3e-5 1/s cyclonic vorticity while the highest 1 % carries -2.0e-5 1/s.
The sign is hemisphere-corrected so one threshold means "cyclonic" in both.

The warm-core reference is taken **within 2 deg latitude bands**, not over the
whole basin: the boxes span 0-40 deg, where the upper-level temperature gradient
would otherwise label every midlatitude low "cold" and every tropical system
"warm" regardless of structure.

The store's 6-hourly steps are coarser than IBTrACS' 3-hourly records; that can
only miss short qualifying periods, never add them.

### Detector caveat - measured, not hypothetical

It is a wind/pressure/warm-core proxy, **not a tracker**: no track continuity, no
`vo` (absent from the store), and N320 under-resolves TC peak winds. Running it
on `cycle-20260730_0000` (valid mid-late August) shows exactly the expected
failure and the expected sensitivity:

| filter | ATL | NWP | SWIO | SEIO |
|---|---|---|---|---|
| wind + msl only | 3.2 | 6.0 | 6.0 | 6.3 |
| + warm core 0.5 K | 2.7 | 6.0 | 4.8 | 5.5 |
| + warm core 2.0 K | 2.3 | 5.7 | 3.3 | 5.3 |
| + warm core 4.0 K | 2.0 | 5.3 | 2.3 | 3.7 |
| + **vorticity 1e-4** (warm core 0.5 K) | 2.4 | **5.8** | **3.0** | 4.2 |

(mean storm days, first week, first members.)

**SWIO/SEIO are in austral winter and should be near zero**, so the un-tuned
detector over-triggers on midlatitude systems reaching into the 0-40 deg S part
of the boxes. Every filter behaves the way physics says it should: adding
vorticity cuts SWIO 5.0 -> 3.0 and SEIO 5.8 -> 4.2 while leaving **NWP unchanged
at 5.8** - the North-West Pacific is genuinely active in August, so a correct
filter must *not* suppress it. Vorticity buys that separation more cheaply than
the warm-core threshold, which only reaches it by being tightened to a level
(+4 K) that starts eroding NWP too.

But SWIO at 3.0 is still well above the ~0 August climatology implies, so
**the thresholds remain uncalibrated guesses**. They must be fitted against
IBTrACS over many cycles - and the remaining error is most likely the missing
piece rather than the threshold values: this is still a point-wise detector with
no track continuity, so a single transient wind maximum satisfying all four
tests on one 6-hourly step marks the whole day. Requiring a coherent centre to
persist across consecutive steps is the next structural fix, not more tuning.

---

## Outputs

`ts_days_probs.nc`
- `TS_tercile_probability` (week, basin, category) - below/near/above
- `storm_days` (member, week, basin) - the raw per-member counts, always kept
- `tercile_bounds` (basin, bound), plus detector settings and `tercile_source`
  in the attributes

`mjo_probs.nc` (only with `--eofs`)
- `MJO_phase_probability` (week, MJO_phase=0..8)
- `rmm1`, `rmm2`, `amplitude`, `phase` (member, day)
- attributes record `index_kind`, whether OLR was available, and which of the
  climatology / low-frequency steps were actually applied

Both files record their source store and cycle, so a product can never be
mistaken for one built with a different detector or a different index basis.

## Do we need the AIFS ENS v2 BUFR tropical-cyclone tracks? **No**

AIFS ENS v2 now disseminates TC tracks as BUFR
([ECMWF implementation notes](https://confluence.ecmwf.int/display/FCST/Implementation+of+AIFS+ENS+v2#ImplementationofAIFSENSv2-Newparameters)),
which looks like the obvious shortcut - a real tracker, storm identity included.
It does not help this target, for three independent reasons:

1. **Lead time is short by a week.** The tracks run to **step 360 = day 15**.
   AI-WQ's TS target is valid at **day 19 or day 26**, and our store covers days
   18-33. The BUFR tracks stop before the forecast window even begins.
2. **They are ECMWF's forecast, not ours.** They come from ECMWF's operational
   AIFS ENS run (52 members: 50 perturbed + control + AIFS Single). Submitting a
   diagnostic derived from them would be submitting somebody else's forecast.
3. **Access.** Dissemination is described via operational ECPDS for authorised
   users; open-data availability is not stated.

So storms must be detected in our own output - which is what `ts_days.py` does,
and why deriving vorticity was worth doing. The BUFR tracks remain useful as an
**independent check** on the detector for days 1-15 in a hindcast setting, if
they can be obtained; they are not a submission path.

## Can TS be submitted alongside `tas` / `mslp` / `pr`?

**Mechanically yes, today. Scientifically not yet — and the gap is narrower than this
file previously implied.** Evaluated 2026-08-23 against AI-WQ 3.29 and a live run on the
`20260820` N320 store.

### 1. The submission path exists and is small

`TS` is a first-class Edition-2 variable, not something to be bolted on:

```python
check_fc_submission.check_filename_characteristics  ->  ['tas','mslp','pr','MJO','TS']
AI_WQ_forecast_submission(data, variable='TS', fc_start_date, fc_period, ...)
```

The expected object is tiny — `AI_WQ_create_empty_dataarray` (`forecast_submission.py:220`):

| | |
|---|---|
| dims | `('tercile', 'basin')`, shape **(3, 4)** |
| `tercile` | `[1/3, 2/3, 1.0]` — checked by `check_tercile_range` |
| `basin` | `['ATL', 'NWP', 'SWIO', 'SEIO']` |
| `fc_period` `'1'` | days **18–24** |
| `fc_period` `'2'` | days **25–31** |
| per cycle | **2 files**, 12 numbers each |

Our `ts_days_probs.nc` already carries exactly this: `TS_tercile_probability
(week, basin, category)` → transpose per week. The plumbing is a dozen lines; **it is not
the reason to wait.**

### 2. Only two basins are ever scored — and for boreal-season inits they are the two
the detector handles best

`check_fc_submission.py:246-258` builds an `active_mask` from the init month (shifted back
7 days for `fc_period` 2) and requires the probabilities to sum to 1 **only there**:

| init month | basins scored |
|---|---|
| **Jun–Nov** | **ATL, NWP** |
| **Dec–Feb** | SWIO, SEIO |
| Mar–May | *none* — the check passes vacuously |

This matters more than anything else in this file. The headline measured failure above is
that **SWIO/SEIO over-trigger in austral winter** — and for a Jun–Nov init those basins are
not scored at all. The sensitivity table's other half is the reassuring one: every filter
leaves **NWP unchanged at 5.8** while cutting SWIO 5.0 → 3.0, which is the correct physical
signature. The detector's worst known bias lands outside the scored set for exactly the
half of the year we have been running.

That narrows the problem from four basins to two. It does not calibrate them.

### 3. The tercile climatology is obtainable — this file used to say otherwise

`AI_WQ_package.TS_processing` ships the builder:

```python
download_IBTRACS_compute_TStercile_climatology(date_str, ibtracs_filename, savedir)
#  -> TS_20yrCLIM_WEEKLYTSDAYS_terciles_<date>.nc   (tercile, basin)
```

20 years × 5 day-offsets (−4…+4) = 100 weekly samples per basin, storms filtered at
`usa_wind >= 33.0454` kt (17 m/s), 33rd/67th percentiles. Note `download_IBTRACS` does
**not** download — it opens a local IBTrACS netCDF you supply (free from NOAA/NCEI).

So `--tercile-clim` can be fed. The question is whether it *should* be — §5.

### 4. Measured on 20260820 (N320 store, 50 members, verification 2026-09-07 / 09-14)

```
$PY ts_days.py --store /tank/projects/aifs-run/20260820_0000/icechunk_v2 \
    --tag cycle-20260820_0000 --init 20260820 --out ts_days_probs_20260820.nc
```

~3.5 min for 50 members. `tercile_source = ENSEMBLE_SELF`, so the probabilities below are
self-referential — read the **counts**, not the probabilities:

| week | basin | mean storm days | range | P(below/near/above) |
|---|---|---|---|---|
| 2026-09-07 | **ATL** | 2.3 | 0–7 | 0.42 / 0.34 / 0.24 |
| 2026-09-07 | **NWP** | **6.5** | **2–7** | 0.30 / 0.70 / **0.00** |
| 2026-09-07 | SWIO | 3.3 | 1–7 | 0.38 / 0.34 / 0.28 |
| 2026-09-07 | SEIO | 3.3 | 0–7 | 0.34 / 0.22 / 0.44 |
| 2026-09-14 | **ATL** | 2.6 | 0–7 | 0.44 / 0.22 / 0.34 |
| 2026-09-14 | **NWP** | **6.0** | 1–7 | 0.44 / 0.56 / **0.00** |
| 2026-09-14 | SWIO | 2.8 | 0–7 | 0.48 / 0.34 / 0.18 |
| 2026-09-14 | SEIO | 1.9 | 0–5 | 0.68 / 0.14 / 0.18 |

SWIO at 2.8–3.3 in September repeats the known over-trigger — now harmless, because
September inits score ATL/NWP only.

### 5. The two blockers that remain, both in the scored basins

**(a) NWP saturates against the 7-day ceiling.** A week has 7 days, and the detector marks
6.0–6.5 of them. `P(above) = 0.00` in both weeks is not a forecast — it is the arithmetic
of a variable pinned at its maximum. A saturated predictor cannot discriminate, so RPSS in
NWP would be near zero at best regardless of how good the underlying rollout is. This is
the point-wise-detector problem the caveat above already names: any single 6-hourly wind
maximum anywhere in a 38,000-point basin marks the whole day, and over a basin that size
in peak season something always qualifies. **Track continuity — requiring a coherent centre
to persist across consecutive steps — is the fix, not threshold tuning.**

**(b) Detector counts and IBTrACS counts are not the same quantity.** AI-WQ's terciles come
from observed IBTrACS storm days; ours come from a proxy detector with a different bias. If
the detector's climatological mean sits above the observed one, *every* forecast lands in
"above normal" and RPSS is negative **by construction** — a systematic error that no amount
of ensemble skill recovers. Feeding `--tercile-clim` with the package's IBTrACS file
without first measuring that offset would produce a confidently wrong submission.

### 6. The bounded next step — **done 2026-08-23. Verdict: do not submit.**

Run, with the AI-WQ package's own builder against IBTrACS `v04r01` (23 MB, NOAA/NCEI,
downloaded to `/tank/projects/ibtracs/`):

```bash
$PY -c "
from AI_WQ_package import TS_processing as T
ib = T.download_IBTRACS('/tank/projects/ibtracs/IBTrACS.ALL.v04r01.nc')   # opens a LOCAL file
for d in ('20260907','20260914'):
    T.compute_20yr_TStercile_climatology(d, ib, '/tank/projects/ibtracs/clim')"
```

**Observed weekly storm days** (100 samples = 20 yr × 5 day-offsets):

| week | basin | mean | median | p33 | p67 | max | zero-weeks |
|---|---|---|---|---|---|---|---|
| 09-07 | ATL | 8.72 | 8 | 6.00 | 11.00 | 24 | 3 % |
| 09-07 | NWP | 5.81 | 6 | 4.00 | 7.00 | 14 | 5 % |
| 09-07 | SWIO | 0.17 | 0 | 0.00 | 0.00 | 4 | 93 % |
| 09-07 | SEIO | 0.00 | 0 | 0.00 | 0.00 | 0 | 100 % |
| 09-14 | ATL | 8.88 | 8 | 5.00 | 12.00 | 28 | 7 % |
| 09-14 | NWP | 6.49 | 6.5 | 5.00 | 8.00 | 18 | 6 % |
| 09-14 | SWIO | 0.15 | 0 | 0.00 | 0.00 | 4 | 95 % |
| 09-14 | SEIO | 0.00 | 0 | 0.00 | 0.00 | 0 | 100 % |

#### The blocker is not bias — it is a units mismatch

**A week cannot contain 28 storm days.** AI-WQ's observed diagnostic counts unique
**(storm, day) pairs** (`TS_processing.count_nstormdays`: `np.unique(stack([storm_ids, day]))`),
so two concurrent hurricanes on the same day contribute **two** storm days. The quantity is
*storm*-days summed across storms, and it is **unbounded**.

`ts_days.py` counts **basin-days** — `by_day[day] = step_hit.any()`, a boolean per day,
summed over 7. It is **capped at 7 by construction**.

These are different quantities. No threshold tuning, and no bias correction, makes a
0–7 counter comparable to a 0–28 one. The earlier framing in this file — "the detector is
uncalibrated" — understated it.

#### What that does to a submission, measured

Applying the real IBTrACS terciles to the 50 member counts from `20260820`:

| week | basin | fc mean | fc max | obs p33 | obs p67 | P(below) | P(near) | P(above) |
|---|---|---|---|---|---|---|---|---|
| 09-07 | **ATL** | 2.34 | 7 | 6.00 | 11.00 | **0.96** | 0.04 | **0.00** |
| 09-07 | **NWP** | 6.46 | 7 | 4.00 | 7.00 | 0.02 | 0.98 | **0.00** |
| 09-14 | **ATL** | 2.58 | 7 | 5.00 | 12.00 | **0.86** | 0.14 | **0.00** |
| 09-14 | **NWP** | 6.00 | 7 | 5.00 | 8.00 | 0.24 | 0.76 | **0.00** |
| 09-07 | SWIO | 3.34 | 7 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| 09-14 | SEIO | 1.94 | 5 | 0.00 | 0.00 | 0.24 | 0.00 | 0.76 |

(bold = the basins actually scored for an August init.)

- **`P(above) = 0.00` in NWP is structural, not a forecast.** The observed upper tercile is
  7.0 and 8.0 storm days; our counter's maximum *is* 7, so `c > hi` is unsatisfiable. The
  ensemble is incapable of predicting an above-normal North-West Pacific week.
- **ATL is a near-deterministic "below normal"** (0.96 / 0.86) for the same reason: the
  ceiling of 7 sits barely above the observed p33 and far below p67.
- SWIO/SEIO invert to `P(above) ≈ 1` against an observed climatology that is **zero in
  93–100 % of weeks** — the over-trigger, now quantified. Unscored for these inits, but it
  is the same defect.

Submitting this would score a strongly negative RPSS in every basin, and would do so
whatever the underlying rollout looked like. **Do not wire it up.**

#### What would actually unblock it

Counting (storm, day) pairs requires **storm identity** — which storm, persisting across
steps. That is a tracker, not a threshold. So track continuity moves from "the next
structural fix" (as this file had it) to **a prerequisite for submitting at all**:

1. group qualifying points into connected centres per step;
2. link centres across consecutive 6-hourly steps into tracks;
3. count `(track_id, day)` pairs — the same quantity AI-WQ counts;
4. *then* compare against the IBTrACS terciles, which are already built and on disk at
   `/tank/projects/ibtracs/clim/`.

Only after step 4 shows comparable distributions does calibration become the question.

A smaller gap to fix on the way: `--tercile-clim` expects an `.npz` with a `bounds`
`(basin, 2)` array applied to **all** weeks, while AI-WQ writes one `.nc` per valid week
with dims `(tercile, basin)`. Per-week bounds and a reader for the official file.

### 7. A constraint worth knowing before planning this

The detector needs `10u`, `10v`, `msl`, `t_300`, `u_850`, `v_850` over days 18–33, and
`grid_ops.py` differentiates on the **N320 reduced Gaussian** rows. Only **two N320 stores
now exist** — `20260806` and `20260820`; the rest of the archive was released or is O96-only.
Two cycles is not enough to characterise the detector's own distribution, so a
detector-native climatology is not currently reachable from what is on disk. The O96 corpora
hold all six variables over the *full* 0–792 h, which is tempting, but O96 is ~112 km and
under-resolves TC winds further than N320 already does — it would most likely move the bias,
not remove it. **Untested; do not assume it substitutes.**

---

## Not done

- No `ttr` in the store, so no true RMM (see blocker 1).
- No EOF / climatology / tercile reference files are bundled here - they are
  observational products and belong with the AI-WQ package data, not in this repo.
- **The IBTrACS comparison has now been run (§6): the detector counts a different quantity
  from AI-WQ, and a submission built on it would be systematically wrong.** A tracker with
  storm identity is a prerequisite, not an improvement.
- No track continuity, so the counter is capped at 7 while the target is unbounded.
- Tested against `20260730` and `20260820`.
