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
`pyarrow`, which the `aifs-gpu` env lacks; they were supplied on `PYTHONPATH`.
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
4. **Not run.** `layer3/ifs46/evidence_d1/` is not in the CRMA checkout on the
   AIFS host.

Still open: the S2S regional grids (`s2s_features --regional` npz) are not on
this host either. So it is unconfirmed that `regional_compact.npz`, with
latitude running 30 → -40 (descending), has the same orientation as the IFS
regime catalogues. `regime_inputs` flattens a box mask, so a flipped latitude
axis would silently permute the regime vector.


## All five cycles (2026-10-10)

20260903, 20260910, 20260917, 20260924 and 20261001 were extracted with the
fixed script. Each has 50 members, 132 steps (6–792 h), 71 regions,
284,000 rows and 3.2 MB, and its sha256 values match. Checks 1–3 pass on every
cycle. On check 2, the O96/N320 rainfall ratio is 1.011–1.043 across the 15
cycle-member pairs. O96 is always slightly wetter over the East Africa box,
which is consistent with the coarser grid, not a units error.

**No store has been purged.** The purge guard waits on check 4 and on the
regional-grid latitude order. Both need inputs that are only on the CRMA host.
