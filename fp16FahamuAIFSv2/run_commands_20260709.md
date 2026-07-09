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
# Date: 20260709 | Source: AWS S3 mirror | Upload to GCS: Yes
micromamba run -n aifs-etl python ecmwf_opendata_pkl_input_aifsens_v2.py \
  --date 20260709 \
  --members 1-50 \
  --source aws
```

## Command Breakdown

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--date` | `20260709` | Forecast initialization date (YYYYMMDD) |
| `--members` | `1-50` | All 50 ensemble members |
| `--source` | `aws` | Use AWS S3 mirror (faster than direct ECMWF, recommended) |
| (default) | `--gcs-subpath` | `input_v2` - GCS storage path |
| (default) | `--out-dir` | `input_states_v2/` - local output directory |

## Alternative Command Examples

```bash
# Subset members only (e.g., members 1-10 for testing)
micromamba run -n aifs-etl python ecmwf_opendata_pkl_input_aifsens_v2.py \
  --date 20260709 --members 1-10 --source aws

# Specific members (e.g., 1, 5, 10, 25)
micromamba run -n aifs-etl python ecmwf_opendata_pkl_input_aifsens_v2.py \
  --date 20260709 --members 1,5,10,25 --source aws

# Skip GCS upload (local testing only)
micromamba run -n aifs-etl python ecmwf_opendata_pkl_input_aifsens_v2.py \
  --date 20260709 --members 1-50 --source aws --no-upload

# Use ECMWF direct portal (slower, may hit rate limits)
micromamba run -n aifs-etl python ecmwf_opendata_pkl_input_aifsens_v2.py \
  --date 20260709 --members 1-50 --source ecmwf
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
├── input_state_member_001.pkl  (~100 MB)
├── input_state_member_002.pkl  (~100 MB)
│   ...
└── input_state_member_050.pkl  (~100 MB)
```

**Total local size:** ~5 GB  
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
| Network timeout / slow downloads | Retry with `--source aws` (default) instead of ecmwf |
| "coiled-data.json not found" warning | GCS upload skipped (local-only mode) — expected if running without GCS credentials |
| Member fails verification | Check network connectivity; member will be skipped and flagged in summary |

## Links

- **Reference:** [AIFS ENS v2.0 Notebook](https://huggingface.co/ecmwf/aifs-ens-2.0/blob/main/run_AIFS_ENS_v2.0.ipynb)
- **Input prep script:** `fp16FahamuAIFSv2/ecmwf_opendata_pkl_input_aifsens_v2.py`
- **v2 README:** `fp16FahamuAIFSv2/README.md`

---

**Status:** Job started - see progress below

[Processing results will be appended here once complete]
