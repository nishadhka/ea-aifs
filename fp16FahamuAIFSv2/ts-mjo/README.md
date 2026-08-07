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

So `mjo_index.py` runs in one of two modes:

- `--olr-source none` (default) - projects **wind-only**, and labels the output
  `index_kind = "wind_only_proxy"`. This is **not** the official RMM and must not
  be submitted as one.
- `--olr-source FILE` - supply daily OLR already on the (days, 144) band grid,
  giving the full 3-field projection (`index_kind = "rmm3"`). For *past* dates
  that file can come from ERA5 (`olr = -ttr / accumulation_seconds`); for
  forecast dates it can only come from a proxy - see option 2 below.

**Way forward - three honest options:**

1. **Wind-only index** (implemented). Cheapest; never label it RMM.
2. **Statistical OLR proxy.** Tropical OLR is dominated by deep convection, and
   the model does output `hcc`, `tcc`, `cp`, `tp`, `tcw`. Fit `ttr` on those
   predictors in ERA5 (where both sides exist), apply to the forecast fields,
   then run the standard 3-field projection. This keeps the official EOF basis
   and is the route most likely to score - but it is an emulator and must be
   validated against ERA5-derived RMM before any submission.
3. **Drop MJO** and submit only the targets this model can actually support.

### 2. Both targets need external reference data

Neither diagnostic is self-contained in a single forecast:

| Needed | Used by | Source | Why a forecast alone cannot supply it |
|---|---|---|---|
| Observed combined EOFs + RMM sds (`--eofs`) | MJO step 6-7 | **`AI_WQ_package.retrieve_MJO_projection_data(password)`** -> `WH04_combinedEOFs.nc`, `WH04_RMM_stddevs.nc` | the projection basis is Wheeler & Hendon's, from observations |
| ERA5 day-of-year climatology 1979-2025 (`--clim`) | MJO step 3 | build from ERA5 (**not distributed**) | anomalies are defined against observed climatology |
| Preceding 120-day mean (`--lowfreq`) | MJO step 4 | ERA5 up to init + forecast days 1-17 | the filter reaches 120 days *before* each day; a 16-day window cannot contain it |
| Tercile boundaries from a matching climatology (`--tercile-clim`) | TS days | build from IBTrACS with **this** detector (**not distributed**) | terciles must come from a climatology built with the same detector |

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

## Not done

- No `ttr` in the store, so no true RMM (see blocker 1).
- No EOF / climatology / tercile reference files are bundled here - they are
  observational products and belong with the AI-WQ package data, not in this repo.
- The detector is unfitted; no IBTrACS comparison has been run.
- Only tested against the `20260730` cycle.
