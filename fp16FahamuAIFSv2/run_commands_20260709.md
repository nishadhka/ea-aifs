# AIFS ENS v2.0 PKL Creation Commands - 20260709

**Date:** 20260709  
**Members:** 1-50 (all 50 ensemble members)  
**Step:** 1 (CPU) - Input preparation

## Environment Setup

```bash
# Activate the aifs-etl micromamba environment
micromamba activate aifs-etl
```

## PKL Creation Command

```bash
# Create input-state pickle files for 50 members
# Date: 20260709 | Source: AWS S3 mirror (hardcoded) | Upload to GCS: Yes
micromamba run -n aifs-etl python ecmwf_opendata_pkl_input_aifsens_v2.py \
  --date 20260709 \
  --members 1-50
```

> **No `--source` flag.** The open-data mirror is hardcoded to the **AWS S3** replica
> (the direct `ecmwf` portal is throttled to 500 simultaneous connections). Passing
> `--source` fails with `error: unrecognized arguments: --source`.

## Command Breakdown

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--date` | `20260709` | Forecast initialization date (YYYYMMDD) |
| `--members` | `1-50` | All 50 ensemble members |
| (fixed) | mirror | AWS S3 — hardcoded, not selectable |
| (default) | `--gcs-subpath` | `input_v2` - GCS storage path |
| (default) | `--out-dir` | `input_states_v2/` - local output directory |

## Alternative Command Examples

```bash
# Subset members only (e.g., members 1-10 for testing)
micromamba run -n aifs-etl python ecmwf_opendata_pkl_input_aifsens_v2.py \
  --date 20260709 --members 1-10

# Specific members (e.g., 1, 5, 10, 25)
micromamba run -n aifs-etl python ecmwf_opendata_pkl_input_aifsens_v2.py \
  --date 20260709 --members 1,5,10,25

# Skip GCS upload (local testing only), keep the pkls on disk
micromamba run -n aifs-etl python ecmwf_opendata_pkl_input_aifsens_v2.py \
  --date 20260709 --members 1-50 --no-upload --keep-local
```

## Expected Output

The script will:
1. Download ECMWF Open Data for each member (date-6h and date)
2. Regrid from 0.25° to N320 resolution
3. Apply v2.0 transformations (112 fields including new wave params)
4. Save to `input_states_v2/input_state_member_NNN.pkl`
5. Upload to GCS: `gs://aifs-aiquest-us-20251127/20260709_0000/input_v2/input_state_member_NNN.pkl`

**Per-member processing time:** ~1-2 minutes each  
**Total time for 50 members:** ~50-100 minutes (depends on network speed)

## Output Files

```
input_states_v2/
├── input_state_member_001.pkl  (~0.97 GB)
├── input_state_member_002.pkl  (~0.97 GB)
│   ...
└── input_state_member_050.pkl  (~0.97 GB)
```

**Total local size:** ~48 GB (measured: 112 fields × 2 timesteps × 542,080 N320 cells, float64)  
**GCS location:** `gs://aifs-aiquest-us-20251127/20260709_0000/input_v2/`

## Next Steps (Step 2: GPU Inference)

Once pkl files are ready, run the GPU inference pipeline:

```bash
micromamba run -n gpu-env python fp16FahamuAIFSv2/fp16_automate_aifs_gpu_pipeline_v2.py \
  --date 20260709_0000 --members 1-50 --lead-time 960
```

This reads the input_v2/ pkl files and produces GRIB forecasts.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "earthkit-data serialises open-data downloads with a file lock" (hangs) | Use Ctrl-C (not Ctrl-Z) to stop; remove lock: `rm -f /tmp/earthkit-data-cloudenv/*.cache.lock` |
| `error: unrecognized arguments: --source` | The `--source` flag was removed; the AWS S3 mirror is hardcoded. Drop the flag. |
| Network timeout / slow downloads | Simply re-run; the mirror is already the faster AWS S3 replica |
| "coiled-data.json not found" warning | GCS upload skipped (local-only mode) — expected if running without GCS credentials |
| Member fails verification | Check network connectivity; member will be skipped and flagged in summary |

## Links

- **Reference:** [AIFS ENS v2.0 Notebook](https://huggingface.co/ecmwf/aifs-ens-2.0/blob/main/run_AIFS_ENS_v2.0.ipynb)
- **Input prep script:** `fp16FahamuAIFSv2/ecmwf_opendata_pkl_input_aifsens_v2.py`
- **v2 README:** `fp16FahamuAIFSv2/README.md`

---

**Status:** Job started - see progress below

[Processing results will be appended here once complete]
