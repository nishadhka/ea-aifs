# CRMA layer 3 from each AIFS-ENS cycle — extract, then purge

`crma_layer3_extract.py` reduces one cycle's O96 corpus (~158 GB, 256 GB with
the sidecar) to **CRMA layer 3**, about 50–100 MB. Afterwards the store is no
longer the only source of what CRMA needs, and can go through the normal purge.

**Status: DRAFT (2026-10-10).** It was written on the CRMA host, where `/tank`
is not mounted, and is syntax-checked only. Run it on one complete cycle and
read the manifest before relying on it or wiring it into the purge guard.

The full plan, including ERA5's role, is in the CRMA repo:
`crma/hazards/dryspell/AIFS_ERA5_LAYER3_PLAN.md`.

## Run

```bash
python crma_layer3_extract.py \
  --store /tank/projects/aifs-run/20261001_0000/icechunk_o96 --cycle 20261001 \
  --crma-repo /path/to/crma --out /tank/projects/crma_layer3
```

`--crma-repo` is required. Region definitions (`bn-evidence/regions.py`) and
the regional drivers and regime inputs (`hazards/dryspell/regional_situation.py`)
are imported from CRMA, not copied, so an AIFS row and an S2S row are the same
quantity.

## Output (`/tank/projects/crma_layer3/<cycle>/`, outside `aifs-run`)

| file | content |
|---|---|
| `evidence.parquet` | CRMA long layout (`init_time, window, region_id, region_type, feature, node, member, value`), windows D1-10, D11-20, D21-30, D31-33, 50 members. The 55 lev04 basins are read through a 1.5° buffer; also the synoptic, global and ocean loci. Features: the 16 S2S node features under the same names, plus `w700`, `cp_frac`, `swvl1`, `ro`. |
| `tp_basins.npz` | daily basin rainfall per member, days 0–32, mm/day |
| `regional_compact.npz` | regime box and the ten gridded drivers per member and window, on the S2S 1.5° regional grid |
| `manifest.json` | rows, members, windows, variables used and missing, sha256 per file |

**Known gaps:**

- **CAPE is absent.** The convective node uses the lapse rate only.
- **There is no OLR**, so the MJO comes from the existing `ts-mjo` product.
- **Ocean loci** are written only if the store carries `sst`. The O96 corpus
  does not, so in practice AIFS layer 3 has no `ocean` rows.

## Purge guard (add once validated)

Purge the O96 store only if `manifest.json` exists, `members == 50`, and every
listed file's sha256 matches. This is the same pattern as the TS/MJO rule in
`cleanup_aifs_run.py`.

## Checks to run on the first cycle

1. Row counts per window × region type × feature in the manifest. NaN shares
   should be zero except for `cape`, and for `ocean` if there is no `sst`.
2. `tp` units: the script converts metres per 6 h to mm when the maximum is
   below 1. Confirm on one member against the N320 sidecar.
3. The lead-hour coordinate: `lead_hours()` guesses `step`, `lead_time` or
   `time`. Confirm that 132 steps map to 6–792 h.
4. Compare D11-20 basin features for 20261001 with the IFS 46-day rows for the
   same init (`crma/layer3/ifs46/evidence_d1/op_2026_10.parquet`).

## First run: 20260903 (2026-10-10)

The run took about 5 min on the AIFS host. It needs `geopandas`, `pyogrio` and
`pyarrow`. At the time `aifs-gpu` lacked them and they were supplied on
`PYTHONPATH`. They have since been pip-installed into `aifs-gpu`
(geopandas 1.2.0, pyogrio 0.13.0, pyarrow 26.0.0, nothing else upgraded).
The output is **3.2 MB**, not 50–100 MB: `evidence.parquet` is 2.0 MB and
`regional_compact.npz` is 0.8 MB. The manifest reports 50 members, 132 steps,
71 regions (55 basins, 11 synoptic, 5 global, no ocean) and 284,000 rows, which
is 50 × 4 windows × 71 × 20 features. All three sha256 values match.

Fixed before this run: region means were a plain `x @ W.T`, so a single NaN
point made the whole region NaN. `swvl1` and `ro` are NaN over sea and lakes
(70% of points), so every coastal or lake basin, and most loci, was lost. They
now go through `region_mean`, which skips NaN points and renormalises the weights.

1. Every window × region type × feature cell has the full count (50 × regions).
   NaN shares are zero apart from four expected cases. `cape` is 100% NaN.
   `swvl1` and `ro` are 60% NaN in `global`, from the 3 all-ocean loci. `cp_frac`
   is 1.9% NaN in basins, where a window has no rain. The `ea_olr` driver is all
   NaN because the store has no OLR.
2. `tp` is a per-6-h interval in metres, not an accumulation. Over the East
   Africa box (12°S–15°N, 22–52°E) at h432–792, O96 and the N320 sidecar agree
   to 1.4–2.4% for members 1, 26 and 50.
3. `time` is datetime, and maps to 6–792 h every 6 h. The windows hold 40, 40,
   40 and 12 steps.
4. Run later on the same day; see "Check 4 and the regional grid" below.

## All five cycles (2026-10-10)

20260903, 20260910, 20260917, 20260924 and 20261001 were extracted with the
fixed script. Each has 50 members, 132 steps (6–792 h), 71 regions,
284,000 rows and 3.2 MB, and its sha256 values match. Checks 1–3 pass on every
cycle. On check 2, the O96/N320 rainfall ratio is 1.011–1.043 across the 15
cycle-member pairs. O96 is always slightly wetter over the East Africa box,
which is consistent with the coarser grid, not a units error.

Checks 1–3 are `crma_layer3_check.py <cycle> ...`.

## Check 4 and the regional grid (2026-10-10)

Both turned out to be runnable from the AIFS host. The IFS 46-day stores that
`s2s_features.py` reads are public and anonymous on S3
(`dynamical-ecmwf-ifs-ens`, `planette-ifs-46-day`).

**Regional grid: latitude order matches, but longitude was off by half a
cell.** `s2s_features.reader()` cuts REGION from the IFS 1.5° grids. Both the
reforecast and the operational store give lat 30 → -39 descending (47 rows)
and lon **10.5** → 90 (54 columns). The extractor used `arange(10.0, 90.0, 1.5)`,
which also gives 54 columns and a 918-cell regime box, so nothing failed. Every
AIFS cell was simply 0.5° west of the matching IFS cell. Fixed: the grid is now
built exactly as `reader()` builds it, with an assert on its ends, and all five
cycles have been re-extracted. The drivers moved by 0.44 on average (mean
absolute change). Checks 1–3 still pass on all five.

**Check 4: AIFS 20261001 vs the IFS 46-day operational run of 2026-10-01.**
The IFS rows were regenerated with CRMA's own `s2s_features.py`
(`--source op --years 2026 --limit 1 --dilate 1`, with REPO redirected to
scratch). The `layer3/observations` basin list was replaced by a stub listing
the same 55 ICPAC basins. Then `crma_layer3_check4.py` was run. All 55 basins
match. All 15 shared features agree in units and sign, and the across-basin
correlation of the ensemble means is 0.92–0.99 for every moisture, rain,
pressure and transport feature, in every window. z500 (0.59–0.79) and
divergence (0.65–0.79) are lower, as expected for fields that vary little
across basins. There are consistent model biases, which per-model
climatologies are meant to absorb; none of them is an extraction error:

| feature | IFS D11-20 | AIFS D11-20 | |
|---|---|---|---|
| tp, mm/day | 3.43 | 2.68 | AIFS 22–32% drier in every window |
| slp_mean, hPa | 1012.6 | 1014.4 | +1.2 to +1.8 hPa |
| z500, m | 5884 | 5902 | +12 to +17 m |
| vq850 | 11.2 | 7.9 | weaker southerly moisture flux, D1-10 and D11-20 |

AIFS ensemble spread is narrower than IFS, at 0.6–1.0 of IFS's spread (median
across basins). Part of that is 50 members against 101.

## Before the purge: loci schema, not yet fixed

Basins match the S2S layout exactly. **The loci do not**, and unlike a name,
a missing variable cannot be recovered once the O96 store is gone:

- **synoptic**: S2S writes `<field>_mean` (e.g. `q850_mean`, `tp_mean`) plus
  `vo850_max`, `mfx850_max`, `slp_boxmax` and `slp_boxmin`. AIFS writes the
  basin names (`q850`, `tp`) and no box maxima or minima.
- **global**: S2S writes `u200_mean`, `u850_mean`, `u50_mean`, `u10_mean` and
  `slp_mean`. AIFS writes the 20 basin features instead, and has no u50 or u10
  although the store carries `u_50` and `u_10` (the QBO locus needs them).
- **ocean**: S2S writes `sst_mean` for INDIAN_OCEAN_LOCI and ENSO_LOCI. AIFS
  writes none because there is no `sst`, but the store has `skt`, which over the
  sea is the SST.
- **member**: S2S numbers members from 0, AIFS from 1.

**No store has been purged.** Checks 1–4 and the grid now pass, but the loci
should match the S2S schema before any store is purged.
