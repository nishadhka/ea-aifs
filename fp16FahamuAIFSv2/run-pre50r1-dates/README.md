# Running AIFS-ENS v2 for dates before IFS Cy50r1

**Question.** We wanted a v2 cycle for **12 February 2026** — a MAM 2026 forecast window
(days 18–33 → 2 Mar – 17 Mar). Step 1 failed immediately:

```
LookupError: 0 msgs for {'param': 'wmb', 'levtype': 'sfc', 'number': 1} (expected 1)
```

**Answer: it cannot be done, from any source.** Thirteen of the model's 112 input fields
are outputs of IFS Cycle 50r1, which went operational with the 06 UTC run on **12 May
2026 — jointly with AIFS v2 itself**. Before that date those fields were not withheld,
they *did not exist*. The earliest initialisable v2 cycle is **2026-05-13 00z**.

This is not a data-access problem that a licence or a different mirror would solve, and
that is the point of this folder: to record what was checked, so nobody re-opens it.

---

## 1. What is missing, and why

| group | fields | first produced |
|---|---|---|
| Period-band significant wave height | `h1012` `h1214` `h1417` `h1721` `h2125` `h2530` | Cy50r1 |
| Wave drag / bathymetry | `cdww` `wmb` | Cy50r1 |
| **New 10 hPa level** | `t_10` `u_10` `v_10` `w_10` `z_10` | Cy50r1 |

The wave bands are significant wave heights within fixed period bands — 10–12 s, 12–14 s,
14–17 s, 17–21 s, 21–25 s, 25–30 s. The AIFS ocean paper describes them as "the
decomposition of significant wave height into six distinct period bands for all waves
with periods larger than 10 seconds."

The 10 hPa level is easy to miss and matters more: the model card notes v2 has "14
pressure levels, now including a new 10 hPa level for improved stratospheric
representation." Stratospheric initial state is a recognised S2S predictability source,
so this is a more serious gap than the ocean-wave fields.

Measured cutover, by bisection over the open-data indexes — exactly consistent with a
12 May 06z implementation making 13 May the first complete 00z cycle:

| date | waef params | 10 hPa level |
|---|---|---|
| 2026-02-12 … 2026-05-12 | 5 | absent |
| **2026-05-13 →** | **13** | **present** |

## 2. Every source checked

| source | wave bands | `cdww` | `wmb` | verdict |
|---|---|---|---|---|
| ECMWF open data `waef` (ensemble), pre-13 May | ✗ | ✗ | ✗ | only `mwd mwp swh pp1d mp2` |
| ECMWF open data `wave` (deterministic), pre-13 May | ✗ | ✗ | ✗ | identical 5 params — not an ensemble-only gap |
| ECMWF open data `enfo` / `oper` | ✗ | ✗ | ✗ | no wave fields at all |
| MARS operational archive | ✗ | ✗ | ✗ | the IFS did not compute them before 50r1 |
| ERA5 (CDS) | ✗ | ✓ | ✓ | ERA5 is Cy41r2 (2016); has swell *partitions*, not period bands |
| ERA5 on Google Earth Engine | ✗ | ✗ | ✗ | atmospheric-only subset, no wave variables |
| ERA5 on AWS (`s3://nsf-ncar-era5`) | ✗ | ✗ | ✗ | 8 streams, 754 keys checked — **zero** GRIB table-140 files |
| NOAA GEFS-Wave / GFS-Wave (WW3) | ✗ | ✗ | ✗ | `SWELL 1/2/3` partitions; also no `cdww`/`wmb` |

Reproduce the open-data half of this table for any date:

```bash
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
$PY check_open_data_inputs.py --date 20260212      # -> v2 NOT runnable, v1 RUNNABLE
$PY check_open_data_inputs.py --scan 20260510 20260515   # -> finds the cutover
```

### "But v2 was trained on ERA5 — surely the fields are in ERA5?"

A reasonable assumption, and the single most useful thing we learned: **no**. ECMWF built
a purpose-made dataset rather than using ERA5, precisely because ERA5 lacks these fields.
From the AIFS ocean paper:

> "To train the additional wave-related fields, a **dedicated hindcast dataset covering
> the period 1979–2025** was produced using ECMWF's most recent wave model (ecWAM) and
> altimeter wave height Data Assimilation (DA) system at a resolution of approximately
> 9 km."

That hindcast is not published. So the fields exist in exactly two places: ECMWF's
internal training hindcast (1979–2025) and operational 50r1 (12 May 2026 →).

**Which leaves early 2026 in a gap between the two.** February 2026 is after the hindcast
ends and before 50r1 went live — plausibly a period for which nobody, ECMWF included, has
these fields.

### The one route not fully closed

The band edges (10, 12, 14, 17, 21, 25, 30 s) fall exactly on ERA5's 1.1-ratio spectral
frequency grid, two bins per band, so the six fields are in principle *derivable* from
ERA5 2D wave spectra:

```
Hs_band = 4 · √( Σ_bins Σ_directions E(f,θ)·Δf·Δθ )
```

Viable but expensive (spectra are tape-backed, not on CDS disks, on their own grid), and
it would reconstruct a 9 km ecWAM product from a coarse 2016-cycle reanalysis. It also
does nothing about the missing 10 hPa level. Not recommended without evidence that the
fields matter — which is what §3 measures.

## 3. Does filling the gap actually work? (`wave_sensitivity.py`)

Rather than argue about it, measure it: take **20260813** (a date where all 13 fields are
real), and re-run member 1 with them replaced by what a pre-50r1 run would have to invent.

**Design note — why this is a paired test.** aifs-ens-2.0 is stochastic: two runs from the
same input differ by 2t rms ≈ 2.8 K. That would drown any signal. But the noise comes
from torch's global RNG, so seeding pairs the runs — at 24 h, same seed gives
max|d| = 0.16 K against 12.8 K for different seeds, an 80× separation. Every
configuration is therefore run under identical seeds, and two baselines make the numbers
readable:

* **floor** — `controlB` vs `control`, same seed: irreducible FP16/cuDNN
  non-determinism. Chaos amplifies it over a 33-day rollout, so at day 18–33 this, not
  zero, is the honest baseline.
* **ensemble scale** — `control` seed 0 vs seed 1: as different as drawing another member.

A fill is usable only if it sits near the floor *and* well below the ensemble scale.
Landing between them means the fill changes the forecast in a way one member cannot
separate from noise — which is not the same as harmless.

| mode | what it does |
|---|---|
| `control` / `controlB` | real fields (identical runs; their difference is the floor) |
| `zero` / `nan` | the 8 wave fields set to 0 / NaN |
| `wmb` | only `wmb` zeroed — the runner reports it as an unsupported coupled forcing |
| `l10copy` | 10 hPa level carried up from 50 hPa (the realistic reconstruction) |
| `l10zero` | 10 hPa level zeroed (pessimistic bound) |
| `pre50r1` | wave NaN + 10 hPa from 50 hPa — the honest best-effort pre-50r1 input |

```bash
$PY wave_sensitivity.py --out RESULTS --seeds 0,1 \
      --modes control,controlB,zero,nan,wmb,l10copy,l10zero,pre50r1
$PY analyse_sensitivity.py --dir RESULTS
```

### Result 1 (day 18–33, the product window): the 10 hPa level is fatal; the waves are unmeasurable

14 rollouts — 7 configurations × 2 seeds. 2t global rms against the control at the same
seed, as a multiple of the non-determinism floor and as a percentage of the ensemble
scale:

| treatment | 2t rms (K) | × floor | % of ensemble | reading |
|---|---|---|---|---|
| `controlB` — identical run | 2.86 / 3.06 | 1.0 | 88 % / 95 % | **the floor** |
| `wmb` alone | 2.96 / 3.16 | 1.0 / 1.1 | 92 % / 98 % | at floor |
| `nan` — 8 wave fields → NaN | 2.95 / 3.33 | 1.0 / 1.2 | 91 % / 103 % | at floor |
| `zero` — 8 wave fields → 0 | 3.31 / 3.15 | 1.2 / 1.1 | 102 % / 98 % | at floor |
| **`l10copy` — 10 hPa from 50 hPa** | **9.05 / 8.69** | **3.2 / 3.0** | **280 % / 269 %** | **far above** |
| **`pre50r1` — waves + 10 hPa** | **8.75 / 8.47** | **3.1 / 3.0** | **271 % / 262 %** | **far above** |

**This test is one-sided.** Because the floor sits at 85–97 % of the ensemble scale, it
cannot prove a perturbation is *safe* — anything near 100 % is simply below the
resolution. But it can prove one is *unacceptable*, and it does.

**The wave fields are at the floor.** All three wave treatments are indistinguishable
from re-running the identical input. Note this does not prove they are harmless; it
proves the day-18–33 single-member test cannot see them (Result 2 looks where it can).

**The 10 hPa substitution is not survivable.** Carrying 50 hPa up to 10 hPa — the
realistic reconstruction, and the only one available without a 50r1 analysis — puts the
forecast **~2.7× outside the model's own ensemble spread**. A merely decorrelated
forecast saturates at ~100 %; 270 % means a *systematic* shift, not chaotic divergence.
That is physically unsurprising: 50 hPa temperatures (~210–220 K) and winds are simply
wrong at 10 hPa, and the stratospheric bias propagates down over 33 days.

`pre50r1` ≈ `l10copy` (271 % vs 280 %) confirms the decomposition: **the stratospheric
level accounts for essentially the whole effect, and the wave fields add nothing
measurable.** The irony is that the run originally failed on `wmb`, a wave field — and
the wave fields turn out to be the part that does not matter.

A perturbation the model ignores producing the same signal as one that should matter is
the signature of **chaotic saturation**. The duplicate-control run proves it outright —
two **byte-identical** rollouts under the **same seed**, differing only by FP16/cuDNN
non-determinism, diverge to almost the full ensemble spread by day 18:

| var | floor (`controlB` vs `control`, same seed) | ensemble scale | floor / ensemble |
|---|---|---|---|
| 2t | 2.86 K | 3.23 K | **88 %** |
| tp | 0.00315 | 0.00324 | **97 %** |
| msl | 766 Pa | 904 Pa | **85 %** |

The floor is at the ceiling: every treatment (91–107 %) is indistinguishable from
changing nothing at all. The `wmb` run was included as an internal control and it did its
job — without it and without `controlB`, these numbers read as "the wave fields matter
enormously," which is the opposite of what they show.

Two consequences worth carrying beyond this experiment:

1. **A single-member day-18–33 difference is not a sensitivity measurement.** It measures
   chaos, whatever it is nominally comparing.
2. **Single-member AIFS-ENS forecasts at S2S range are not reproducible on the same
   hardware.** Bit-identical inputs and a fixed seed still diverge to ensemble spread. The
   ensemble is the product; no individual member is.

**Do not read a single-member day-18–33 comparison as a sensitivity measurement.** It
measures chaos. Two ways out: measure at short lead where the signal is still traceable
(Result 2), or compare *distributions* with many members per configuration (not done —
would take ~40 rollouts).

### Result 2 (6–120 h): everything separates, and the ranking is unambiguous

At 5-day lead the non-determinism floor collapses to **3 % of the ensemble scale** (2t
0.037 K vs 1.11 K), so every treatment is resolvable. 16 rollouts, 8 configurations ×
2 seeds:

| treatment | 2t × floor | 2t % of ensemble | msl × floor | verdict |
|---|---|---|---|---|
| `controlB` — identical run | 1.0 | 3 % | 1.0 | the floor |
| `wmb` alone | 8.6 / 8.8 | 29 % | 9.0 | **used after all** |
| `nan` — 8 wave fields → NaN | 12.5 / 11.6 | 42 % / 39 % | 26 / 22 | real, sub-ensemble |
| `zero` — 8 wave fields → 0 | 14.1 / 14.2 | 47 % | 23 / 22 | real, sub-ensemble |
| **`l10copy` — 10 hPa from 50 hPa** | **164 / 162** | **547 % / 541 %** | **371 / 377** | **catastrophic** |
| `l10zero` — 10 hPa → 0 | 237 / 236 | 792 % / 789 % | 306 | catastrophic |
| **`pre50r1` — waves + 10 hPa** | **164 / 163** | **548 % / 544 %** | **370 / 373** | **catastrophic** |

Three things this settles that Result 1 could not:

1. **The wave fields *are* used** — 12–14× the floor, ~40–47 % of the ensemble scale.
   They were not "unimportant" at day 18–33, merely saturated. Faking them costs
   something real, roughly half an ensemble member's worth of perturbation.
2. **`wmb` is used too** — 8.6–8.8× the floor on its own. The runner's
   `Coupled forcings are not supported by this runner: [... 'wmb' ...]` message refers to
   *updating* the forcing through the rollout, **not** to ignoring the field at input.
   Do not read that message as "this field is free."
3. **The 10 hPa level is ~12× more damaging than all eight wave fields combined**, and
   its best available substitute already produces **5.5× the ensemble spread at day 5**.
   `l10copy` (547 %) beats `l10zero` (792 %) as expected — carrying 50 hPa up is better
   than zeros — but both are far outside anything usable. `tp` is the least affected
   field (101 % of ensemble), consistent with a stratospheric temperature/height bias
   hitting `2t` and `msl` hardest.

`pre50r1` ≈ `l10copy` at both leads: the stratospheric level dominates completely.

## 4. Conclusion

**AIFS-ENS v2 cannot be run for dates before 2026-05-13.** The blocker is not the ocean
wave fields that made Step 1 fail — those are fillable at a real but bounded cost
(~40–47 % of an ensemble spread). It is the **10 hPa level**, which has no acceptable
substitute: the best reconstruction available without a 50r1 analysis puts the forecast
5.5× outside the model's own ensemble spread by day 5, and 2.7× outside it at day 18–33.

Had the investigation stopped at the wave fields — the error Step 1 actually raised — the
answer would have been "looks fillable, worth trying." `check_open_data_inputs.py`
catching the missing 10 hPa level is what turned that around.

## 5. What to do for pre-50r1 dates

**Use AIFS v1.** It predates the coupled wave inputs and the 10 hPa level, so it needs
none of the missing fields. Verified against the 12 Feb 2026 open-data index:

```
AIFS v1: RUNNABLE
    sfc     8/8      pl      6/6
    sol     1/1      levels 13/13
```

`s3_grib_pkl/s3_grib_pkl_input_aifsens.py` builds v1 inputs from open data (50 members,
IFS ENS), and `era5tFp16FahamuAIFSv1/` holds the downstream chain. Note the v1 runner is
separate from `run_local_icechunk_v2.py`, so O96 output would need wiring in.

## 6. Files

| file | what it is |
|---|---|
| `check_open_data_inputs.py` | is a date runnable for v1 / v2? also `--scan` a range to locate a cutover |
| `wave_sensitivity.py` | the paired sensitivity experiment |
| `analyse_sensitivity.py` | scores treatments against the floor and the ensemble scale |

Raw rollouts (30 × 61 or 20 captured steps of `2t`/`tp`/`msl` on N320) are kept outside
the repo — 3.3 GB in `/tank/projects/aifs-run/wave_sensitivity_20260813/` and 1.2 GB in
`…_short/`. Delete them once the numbers above are trusted; `wave_sensitivity.py` skips
configurations whose `.npz` already exists, so a partial re-run is cheap.

## 7. If ECMWF changes the inputs again

The same trap will recur at the next cycle upgrade, and the failure mode is misleading:
Step 1 dies on whichever missing field it happens to look up first, which need not be the
one that matters. Before assuming a substitute is viable:

1. `check_open_data_inputs.py --date <D>` — enumerate **every** missing field, not just
   the one in the traceback.
2. Measure at **short lead** (5 days). At S2S range the non-determinism floor reaches
   ~90 % of the ensemble spread and hides everything below it.
3. Always include a **duplicate control** and a **should-be-inert treatment**. Both
   internal controls earned their place here — `controlB` exposed the saturation, and
   `wmb` disproved the assumption that the runner's "coupled forcings are not supported"
   message meant the field was unused.
