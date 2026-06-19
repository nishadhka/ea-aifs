# fp16FahamuAIFSv2 (AIFS ENS v2.0 / FP16 / ECMWF Open Data)

New model version targeting **AIFS-ENS-2.0** (checkpoint `ecmwf/aifs-ens-2.0`), run
at FP16. This folder currently implements the **v2.0 input data preparation**; the GPU
inference runner, post-processing and submission for v2 are not done yet (the v1/era5t
Steps 3–5 in `shared/` are reusable once v2 GRIB output exists).

Reference notebook: `run_AIFS_ENS_v2.0.ipynb`
(https://huggingface.co/ecmwf/aifs-ens-2.0/blob/main/run_AIFS_ENS_v2.0.ipynb).

## Scoping — what changed vs `*AIFSv1`

Compared against `shared/ecmwf_opendata_pkl_input_aifsens.py`:

| Group | v1 | **v2.0** |
|-------|----|----------|
| Surface | `10u,10v,2d,2t,msl,skt,sp,tcw` | **+ `sd`** (snow depth) |
| Constants | `lsm,z,slor,sdor` | same |
| Soil | `sot` → `stl1,stl2` | **`vsw,sot`** → `stl1,stl2,**swvl1,swvl2**` |
| Wave | *(none)* | **11 new** `wmb,h1012,h1214,h1417,h1721,h2125,h2530,mwd,cdww,mwp,swh` (stream `waef`/`wave`) |
| Pressure levels | 13 (`1000…50`) | **14 (`1000…10`)** |
| PL params | `gh,t,u,v,w,q` | same, **drop `q_10`** |
| Transforms | `gh→z` | `gh→z` **+ `mwd→cos_mwd/sin_mwd`** **+ LSM-mask `sd,swvl1,swvl2`→NaN over sea** |
| **Field count** | ~92 | **112** (verified) |
| Inference env | regrid 0.4.0 / anemoi (v1) | regrid **0.5.1**, anemoi-models **0.11.2**, anemoi-inference **0.8.3**, torch **2.7**, flash-attn **2.7.4** |
| Checkpoint | `ecmwf/aifs-ens-1.0` | **`ecmwf/aifs-ens-2.0`** |

Retrieval mechanics are unchanged: download 0.25° open data over `[date-6h, date]`,
roll −180..180 → 0..360, interpolate to **N320** with `earthkit-regrid`, stack the two
timesteps. Constants (`lsm,z,slor,sdor`) are still fetched once and replicated.

## Scripts

| Script | Role |
|--------|------|
| `fp16FahamuAIFSv2/ecmwf_opendata_pkl_input_aifsens_v2.py` | **Step 1 (v2):** ECMWF Open Data → 112-field input-state pkl per member, upload to GCS |

The 112-field set: 9 surface + 4 constants + 4 soil (`stl1/2`,`swvl1/2`) + 12 wave
(`mwd`→`cos_mwd/sin_mwd`) + 83 pressure-level (6 params × 14 levels − `q_10`).

## Run

```bash
# Latest open-data date, all 50 members, upload to gs://…/<date>/input_v2/
python fp16FahamuAIFSv2/ecmwf_opendata_pkl_input_aifsens_v2.py --members 1-50

# Pin a date / subset / skip upload while testing
python fp16FahamuAIFSv2/ecmwf_opendata_pkl_input_aifsens_v2.py \
    --date 20260611 --members 1 --no-upload --keep-local
```

- **Output:** `gs://aifs-aiquest-us-20251127/<date>_0000/input_v2/input_state_member_NNN.pkl`
  (kept separate from v1's `input/` so both versions can coexist).
- **Requires:** `coiled-data.json` (GCS key) unless `--no-upload`. CPU only.
- `--source` chooses the open-data mirror (`ecmwf`/`azure`/`aws`/`google`).

## Caveats / TODO

- **earthkit-regrid:** the v2 notebook pins **0.5.1**; the repo `aifs-etl` env has
  **0.4.0**. The N320 interpolation call is unchanged, but bump the env to 0.5.1 for an
  exact match before a production run.
- **Wave stream availability:** wave fields come from `waef` (ensemble) / `wave`
  (control). If a member 404s on the wave group, check open-data wave availability for
  that cycle.
- **LSM mask:** derived here from the already-N320 `lsm` field (`lsm==0`), instead of the
  notebook's separate `lsm.grib` file — equivalent and avoids a side artifact.
- **Not yet implemented:** v2 GPU runner (`aifs-ens-2.0` checkpoint, anemoi-models
  0.11.2, flash-attn 2.7.4) and wiring Steps 3–5. The 112-field pkl is the contract the
  runner will consume.
