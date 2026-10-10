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
- **Ocean loci** are written only if the store carries `sst`.

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
