#!/usr/bin/env bash
# Bootstrap a Colab GPU runtime for the hindcast rollout (COLAB_PLAN.md Phase B).
# Same stack as the GPU box (fp16FahamuAIFSv2/LOCAL_GPU_RUN.md section 1): Python 3.12,
# torch 2.7.0+cu126, anemoi-inference 0.8.3, flash-attn 2.7.4.post1 (cxx11abi TRUE wheel),
# icechunk 2.1.0, zarr 3.2.1, and the ecmwf/aifs-ens-2.0 checkpoint in HF_HOME.
# Colab's own Python is 3.13 with a newer torch, so a separate uv venv keeps the versions
# identical to the box that produced the 20 control members.
#
#   bash hc_gpu_setup.sh           -> /content/venv, prints "SETUP OK" and timings
set -eu
T0=$(date +%s)
V=/content/venv
export HF_HOME=${HF_HOME:-/content/hf_cache}
uv venv -q --python 3.12 $V
uv pip install -q --system boto3          # the worker itself runs on Colab's python
PY=$V/bin/python
uv pip install -q --python $PY \
  torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple
uv pip install -q --python $PY \
  torch==2.7.0 torch-geometric==2.6.1 \
  anemoi-inference==0.8.3 anemoi-models==0.11.2 anemoi-utils==0.4.35.post3 \
  earthkit-regrid==0.5.1 "earthkit-data<1.0.0" ecmwf-opendata==0.3.29 \
  icechunk==2.1.0 zarr==3.2.1 google-cloud-storage python-dotenv netcdf4 xarray \
  huggingface_hub boto3 scipy pandas eccodes gribberish obstore \
  --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
T1=$(date +%s)
ABI=$($PY -c "import torch; print(str(torch._C._GLIBCXX_USE_CXX11_ABI).upper())")
WHL="flash_attn-2.7.4.post1+cu12torch2.7cxx11abi${ABI}-cp312-cp312-linux_x86_64.whl"
curl -sSL -o /content/$WHL \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/${WHL//+/%2B}"
uv pip install -q --python $PY --no-deps /content/$WHL
rm -f /content/$WHL
T2=$(date +%s)
$PY - <<'EOF'
import torch, flash_attn
from flash_attn import flash_attn_func
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0))
q = torch.randn(1, 64, 8, 64, device="cuda", dtype=torch.float16)
print("flash_attn OK", tuple(flash_attn_func(q, q, q).shape))
import anemoi.inference, icechunk, zarr
print("anemoi-inference", anemoi.inference.__version__, "| icechunk", icechunk.__version__, "| zarr", zarr.__version__)
from huggingface_hub import HfApi, hf_hub_download
files = [f for f in HfApi().list_repo_files("ecmwf/aifs-ens-2.0") if f.endswith(".ckpt")]
for f in files:      # only the checkpoint(s), as anemoi's huggingface loader fetches them
    print("checkpoint", hf_hub_download("ecmwf/aifs-ens-2.0", f))
EOF
T3=$(date +%s)
echo "SETUP OK: pip $((T1-T0)) s, flash-attn $((T2-T1)) s, checkpoint+verify $((T3-T2)) s, total $((T3-T0)) s"
