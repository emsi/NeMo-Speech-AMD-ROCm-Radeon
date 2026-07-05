#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

VENV="${VENV:-.venv-rocm}"
PY="${VENV}/bin/python"

if [[ ! -x "${PY}" ]]; then
  echo "Missing ${PY}. Run: bash scripts/rocm/setup_rocm_uv_env.sh" >&2
  exit 1
fi

echo "== AMD ROCm Radeon devices =="
rocminfo | grep -E "Name: +gfx|Marketing Name: +Radeon" || {
  echo "ROCm did not report an AMD GPU." >&2
  exit 1
}

echo
echo "== AMD ROCm PyTorch + NeMo Speech =="
"${PY}" - <<'PY'
import torch
import nemo
from nemo.collections.asr.models import ASRModel
from nemo.collections.tts.models import FastPitchModel
import nemo.collections.audio as audio
import nemo.collections.speechlm2 as speechlm2

print("nemo", nemo.__version__)
print("torch", torch.__version__)
print("hip", torch.version.hip)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("ROCm PyTorch did not expose a CUDA-compatible device")

print("gpu", torch.cuda.get_device_name(0))
a = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
b = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
c = a @ b
torch.cuda.synchronize()
if not torch.isfinite(c).all().item():
    raise SystemExit("FP16 matmul produced non-finite values")

print("fp16_matmul", c.dtype, "ok")
print("imports", ASRModel.__name__, FastPitchModel.__name__, audio.__name__, speechlm2.__name__)
PY

echo
echo "== Python package consistency =="
"${PY}" -m pip check

echo
echo "== NVIDIA/CUDA extras absence check =="
bad_packages="$("${PY}" -m pip list --format=freeze | grep -Ei '^(cuda|numba-cuda|nvidia|transformer-engine|flash-attn|flashoptim|deep-ep|mamba|causal-conv1d|nv-grouped-gemm)' || true)"
if [[ -n "${bad_packages}" ]]; then
  echo "Unexpected CUDA/NVIDIA packages found:" >&2
  echo "${bad_packages}" >&2
  exit 1
fi
echo "ok"
