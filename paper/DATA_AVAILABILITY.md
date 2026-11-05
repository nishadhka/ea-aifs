# Data availability — ERA5 starting states for AIFS-ENS v2, including the 6 wave bands

**Checked:** 2026-11-05 · **Status:** exploration only. Nothing is built, no scripts are committed,
and no bulk download has started. Companion to `AIFS_ENS_V2_S2S_EXTREMES_STUDY_PLAN.md` §2.

**Bottom line:** every one of the 112 AIFS-ENS v2 input fields is available for any date
1979–2026: 106 from ERA5 (ARCO) and the 6 wave period bands from ECMWF's published AIFS-training
wave hindcast. The ERA5 starting-state route is intact. **Scope: download only the starting dates
the 22 (+1) events and their verification periods need, not a full 20-year archive.**

---

## 1. Field-by-field sources

AIFS-ENS v2 needs 112 input fields: 9 surface + 4 constants + 4 soil + 12 wave + 83 pressure-level.

| Field group | Count | Source | Period | Grid |
|---|---|---|---|---|
| Surface `10u 10v 2d 2t msl skt sp tcw sd` | 9 | ARCO-ERA5 `ar/full_37-1h-0p25deg-chunk-1.zarr-v3` | 1940 → 2026-06-30 final, → 2026-10-30 ERA5T | 0.25°, hourly → regrid to N320 |
| Constants `lsm z slor sdor` | 4 | ARCO-ERA5 (static) | — | 0.25° |
| Soil `stl1 stl2 swvl1 swvl2` | 4 | ARCO-ERA5 | as above | 0.25° |
| Pressure levels `t u v w z` × 14, `q` × 13 (incl. 10 hPa) | 83 | ARCO-ERA5 (37 levels) | as above | 0.25° |
| Waves `swh mwp cos_mwd sin_mwd cdww wmb` | 6 | ARCO-ERA5 (`significant_height_of_combined_wind_waves_and_swell`, `mean_wave_period`, `mean_wave_direction`, `coefficient_of_drag_with_waves`, `model_bathymetry`), or the wave hindcast below | as above | 0.25° |
| **6 wave period bands `h1012 h1214 h1417 h1721 h2125 h2530`** | **6** | **ECMWF AIFS-training wave hindcast (§2)**. Not in any ERA5 store | 1979–2026 | **N320 directly from MARS** |
| Ensemble perturbations | — | ERA5 EDA, 10 members, CDS `product_type: ensemble_members` | 1940 → | ~0.5°, 3-hourly |

ARCO holds no ensemble members, so the 10 EDA perturbations come from CDS. Whether every needed
variable (including `w` and 10 hPa) is published for the EDA members is **not yet checked**.

## 2. The wave hindcast series (ECMWF experiment catalogue)

Source: <https://apps.ecmwf.int/ifs-experiments/>. Each is standalone ecWAM CY50R1, Tco1279
(~9 km), 6-hourly, carrying the bands (140114–140119), `cdww` (140233) and `wmb` (140219). Same
model and parameters as `j1r2`, already used in `../fp16FahamuAIFSv2/run-pre50r1-dates/`.
Reference: Bidlot et al. 2026, *Wave Hindcast for ERA6 Preparation and Training Data Driven
Models*, ECMWF Tech Memo (in preparation).

| expver | Period | Wind forcing | Covers events |
|---|---|---|---|
| `ikxa` | 1979–1989 | corrected ERA5 (no altimeter assimilation) | — |
| `ikp7` | 1990–1999 | corrected ERA5 + altimeter assimilation | — |
| **`ikp6`** | **2000–2015** | corrected ERA5 + altimeter assimilation | climatology years only |
| **`ikg0`** | **2016–2023** | corrected ERA5 + altimeter assimilation | Rwanda 2016; all 2019, 2020–23, 2021, 2023 events |
| **`itul`** | **2024–2025** | corrected ERA5 + altimeter assimilation | Burundi, Nairobi, Tanzania 2024 |
| **`j5rf`** | **2026** | corrected ERA5 + altimeter assimilation | Nairobi Mar 2026; Sep–Oct 2026 cycles |
| `ikfz`, `ikdi` | 2016–2018, 2019–2025 | operational winds | alternative |
| `j6bd`, `j6bf` | 2016–2020, 2021–2026-02 | operational winds | alternative |
| `j1j2`, `j1j3`, `j1r2` | 2016–2026 | operational winds + ORAS6 currents | `j1r2` = existing pre-50r1 donor |

Use the **ERA5-forced** series (`ikp6`, `ikg0`, `itul`, `j5rf`) with ERA5 starting states, so that
the waves come from the same winds as the atmosphere.

### 2.1 Retrieval test (one date, then stopped)

| expver | Date | Result | Time |
|---|---|---|---|
| `ikp6` | 2011-01-06 00z | **OK**: 4 fields, 542 080 points each (N320, the AIFS grid). `h1012` 0.001–5.09 m, `h2530` 0.001–0.090 m, `cdww` 9.4e-4–3.7e-3, `wmb` 2–999 | **615 s** (tape) |
| `ikg0`, `itul` | 2019-05-02, 2024-04-04 | not tested (stopped on request) | — |

The values are physical: band height falls with period, `wmb` ≤ 999 m, and `cdww` is in the
range `j1r2` showed. **~10 minutes per request, mostly tape staging**, so batch all fields and
both times (t−6h, t0) for one date into one request.

Request used (same as `fetch_j1r2.py`, with `expver` changed):

```python
{"class": "rd", "dataset": "research", "expver": "ikp6", "stream": "wave", "type": "an",
 "levtype": "sfc", "date": "20110106", "time": "00", "step": "0",
 "param": "140114/140119/140233/140219", "grid": "N320", "target": "ikp6_20110106.grib"}
```

Credentials: `../fp16FahamuAIFSv2/run-pre50r1-dates/.ecmwfapirc` (ECMWF Web API key, `class=rd`
access). This is not the CDS token. See that folder's README §2.7.

## 3. Alternative ERA5 store, not used

Earthmover `s3://earthmover-icechunk-era5/icechunkV2` (Icechunk, anonymous): 1940 → 2026-06-30,
updated quarterly, 0.25° hourly, **13 levels (no 10 hPa), no wave fields**. That is 95 of 112
fields. Needs `numcodecs[pcodec]`, which the `aifs-gpu` env lacks. Keep it as a cross-check only.

## 4. What to download: only the event dates

The volume follows from the starting dates, not from the archive length. For each event:

- **Starting dates:** the Thursdays 1–4 weeks before the event window opens (leads that put
  the window inside days 18–33 and days 5–17).
- **Same calendar dates in the 20 climatology years** (2005–2024, plus 2025–2026 out of sample),
  needed for calibration and for the all-years detection scores (study plan §4, §5).
- **Each starting date needs two times** (t−6h, t0): ARCO for 106 fields, the wave hindcast for 6
  bands, CDS EDA for the 10 perturbations.

Events sharing a window share starting dates: Eritrea and Khartoum both use Aug 2019; the
2020–23 droughts share their failed seasons; Burundi, Nairobi and Tanzania share Apr 2024. Merge
these before counting.

Volume, roughly: 12–24 calendar dates × 22 years = **264–528 starting dates**. That means about
264–528 wave requests at ~10 minutes each (~2–4 days of queue if run serially; MARS allows a few
in parallel), plus the same number of ARCO reads (fast) and CDS EDA requests (queue unknown).

## 5. Not done yet

- [ ] Final list of starting dates from the event windows (`~/crma/scenario-sim/scenarios/*.json`)
- [ ] Retrieval test for `ikg0`, `itul`, `j5rf`
- [ ] CDS EDA variable check (`w`, 10 hPa, soil, `sd` for the ensemble members)
- [ ] Input-space check of the hindcast bands against operational 50r1 on a 2026 overlap date
      (as `run-pre50r1-dates` §2.4 did for `j1r2`)
- [ ] `--expver` option in `fetch_j1r2.py`, and an ERA5 (ARCO) starting-state builder. Neither
      exists yet.
