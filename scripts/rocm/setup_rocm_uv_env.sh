#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it first: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

if ! command -v python3.12 >/dev/null 2>&1; then
  echo "Python 3.12 is required for the AMD Radeon ROCm wheels used here." >&2
  exit 1
fi

VENV="${VENV:-.venv-rocm}"
PY="${VENV}/bin/python"

uv venv --python 3.12 "${VENV}"

uv pip install --python "${PY}" --link-mode=copy --upgrade \
  pip wheel setuptools 'numpy==1.26.4'

uv pip install --python "${PY}" --link-mode=copy \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl' \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchvision-0.24.0%2Brocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl' \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchaudio-2.9.0%2Brocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl' \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/triton-3.5.1%2Brocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl'

# Install NeMo Speech dependencies manually. Do not use cu12, cu13, compiled,
# compiled-a100, all, or uv sync on AMD hosts.
uv pip install --python "${PY}" --link-mode=copy \
  'numpy==1.26.4' \
  aistore 'fsspec>=2024.12.0' 'huggingface_hub>=0.24' 'onnx>=1.7.0' \
  scikit-learn smart-open tensorboard text-unidecode 'tqdm>=4.41.0' wrapt \
  'hydra-core>1.3,<=1.3.2' 'lightning>2.2.1,<=2.4.0' 'omegaconf<=2.3' \
  'torchmetrics>=0.11.0' transformers wandb 'webdataset>=0.2.86' \
  'nv_one_logger_core>=2.3.1' \
  'nv_one_logger_training_telemetry>=2.3.1' \
  'nv_one_logger_pytorch_lightning_integration>=2.3.1' \
  'datasets>=3.2.0' einops pandas 'sentencepiece<1.0.0' \
  braceexpand kaldialign 'lhotse>=1.33.0' 'librosa>=0.10.1' \
  packaging sacrebleu 'scipy>=0.14' soundfile whisper_normalizer \
  matplotlib 'librosa>=0.10.0' \
  'pesq; (platform_machine != "x86_64" or platform_system != "Darwin")' \
  pystoi \
  janome jieba nemo_text_processing nltk pypinyin pypinyin-dict pyopenjtalk \
  'peft<=0.18.0'

uv pip install --python "${PY}" --link-mode=copy --no-deps --reinstall -e .

echo
echo "ROCm NeMo environment ready: ${VENV}"
echo "Activate with: source ${VENV}/bin/activate"
