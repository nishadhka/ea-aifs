#!/usr/bin/env bash
#
# setup_gpu_env_v2.sh — build the AIFS-ENS-2.0 (Step 2) GPU inference env.
#
# Creates a self-contained venv with the pins from the v2 notebook, plus the
# three things the README's pin list leaves out (see LOAD_TEST_RESULTS.md §1):
#   * huggingface_hub   — anemoi resolves {"huggingface": "ecmwf/aifs-ens-2.0"} through it
#   * a host C compiler — aifs-ens-2.0's GraphTransformer JIT-builds a Triton CUDA shim
#   * a prebuilt flash-attn wheel — a source build needs nvcc, which GPU boxes rarely have
#
# The flash-attn wheel is chosen at runtime from the interpreter's Python tag and
# torch's C++11 ABI flag, because a mismatch on either installs silently and then
# fails at import. torch 2.7 linux wheels are cxx11abi=TRUE; older ones were FALSE.
#
# Usage:
#   ./setup_gpu_env_v2.sh                                   # ./venv-aifs-v2, Python 3.10
#   ./setup_gpu_env_v2.sh --prefix /scratch/venv-aifs-v2    # pick the location
#   ./setup_gpu_env_v2.sh --python 3.12 --with-icechunk     # + local Icechunk deps
#   ./setup_gpu_env_v2.sh --verify-only --prefix DIR        # re-run the checks on an env
#
# Verified: L4 22 GB / driver 570 / CUDA 12.8, Python 3.10 (uv) — 10.77 GB allocated,
# 11.73 GB reserved at FP16 + 16 chunks, matching LOAD_TEST_RESULTS.md.

set -euo pipefail

PREFIX="${AIFS_V2_VENV:-$PWD/venv-aifs-v2}"
PYVER="3.10"
WITH_ICECHUNK=0
VERIFY_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)        PREFIX="$2"; shift 2 ;;
        --python)        PYVER="$2"; shift 2 ;;
        --with-icechunk) WITH_ICECHUNK=1; shift ;;
        --verify-only)   VERIFY_ONLY=1; shift ;;
        -h|--help)       sed -n '2,26p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

PY="$PREFIX/bin/python"

# --- 0. host prerequisites -------------------------------------------------
# gcc is not a build-time nicety here: anemoi.models.triton.gt compiles a CUDA
# shim on the first forward pass, so a missing compiler surfaces mid-inference
# as "RuntimeError: Failed to find C compiler". (ptxas ships inside the triton wheel.)
if ! command -v gcc >/dev/null 2>&1; then
    echo "ERROR: no gcc on PATH — the Triton JIT kernel will fail at the first step." >&2
    echo "       install it first, e.g.: sudo apt-get install -y gcc" >&2
    exit 1
fi
export CC="${CC:-$(command -v gcc)}"

command -v nvidia-smi >/dev/null 2>&1 \
    && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
    || echo "WARNING: nvidia-smi not found — this env only runs on an Ampere+ GPU box." >&2

if [[ "$VERIFY_ONLY" -eq 0 ]]; then
    # --- 1. create the venv ------------------------------------------------
    if command -v uv >/dev/null 2>&1; then
        uv venv --python "$PYVER" "$PREFIX"
        PIP=(uv pip install)
        export VIRTUAL_ENV="$PREFIX"
    else
        "python$PYVER" -m venv "$PREFIX"
        PIP=("$PY" -m pip install)
        "$PY" -m pip install -q --upgrade pip
    fi

    # --- 2. torch first ----------------------------------------------------
    # PyPI's torch 2.7.0 is the cu126 build; it runs fine on CUDA 12.8/13.x drivers.
    # It must land before flash-attn (whose wheel is picked from torch's ABI) and
    # before anemoi (so nothing pulls a different torch in as a dependency).
    "${PIP[@]}" torch==2.7.0 torch-geometric==2.6.1

    # --- 3. the pinned anemoi / earthkit stack -----------------------------
    "${PIP[@]}" \
        anemoi-inference==0.8.3 \
        anemoi-models==0.11.2 \
        anemoi-utils==0.4.35.post3 \
        earthkit-regrid==0.5.1 \
        ecmwf-opendata==0.3.29 \
        'earthkit-data<1.0.0'

    # --- 4. what the pin list omits ----------------------------------------
    "${PIP[@]}" huggingface_hub google-cloud-storage matplotlib

    [[ "$WITH_ICECHUNK" -eq 1 ]] && "${PIP[@]}" icechunk zarr xarray netcdf4 python-dotenv

    # --- 5. flash-attn, matched to (python tag, torch C++11 ABI) -----------
    PYTAG=$("$PY" -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
    ABI=$("$PY" -c 'import torch; print("TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE")')
    WHL="flash_attn-2.7.4.post1+cu12torch2.7cxx11abi${ABI}-${PYTAG}-${PYTAG}-linux_x86_64.whl"
    echo "flash-attn wheel: $WHL"
    # --no-deps: torch is already installed and the wheel would otherwise re-resolve it.
    "${PIP[@]}" --no-deps \
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/${WHL}"
fi

# --- 6. verify -------------------------------------------------------------
"$PY" - <<'EOF'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")

from flash_attn import flash_attn_func
q = torch.randn(1, 64, 8, 64, device="cuda", dtype=torch.float16)
print("flash_attn OK", tuple(flash_attn_func(q, q, q).shape))

import anemoi.inference, anemoi.models, earthkit.regrid, ecmwf.opendata
from anemoi.inference.runners.simple import SimpleRunner
from anemoi.inference.outputs.gribfile import GribFileOutput   # noqa: F401
from anemoi.inference.outputs.printer import print_state       # noqa: F401
print("anemoi-inference", anemoi.inference.__version__,
      "| anemoi-models", anemoi.models.__version__,
      "| earthkit-regrid", earthkit.regrid.__version__)

# Resolve the checkpoint through huggingface_hub and confirm the field contract
# the v2 scripts assume: 112 inputs in, 120 outputs out, tp/msl/2t retained.
runner = SimpleRunner({"huggingface": "ecmwf/aifs-ens-2.0"}, device="cuda", precision="16")
ck = runner.checkpoint
computed = set(ck.model_computed_variables)
n_in = len([v for v in ck.input_tensor_index_to_variable.values() if v not in computed])
out = set(ck.output_tensor_index_to_variable.values())
print(f"checkpoint OK | inputs {n_in} | outputs {len(out)} | grid {ck.number_of_grid_points}"
      f" | timestep {ck.timestep}")
assert n_in == 112, f"expected 112 input fields, got {n_in}"
assert {"tp", "msl", "2t"} <= out, "Step 3 params missing from the checkpoint outputs"
print("ENV OK")
EOF

echo
echo "Env ready: $PREFIX"
echo "Use it with:  export HF_HOME=<cache dir> CC=/usr/bin/gcc"
echo "              $PY fp16FahamuAIFSv2/fp16_automate_aifs_gpu_pipeline_v2.py --help"
