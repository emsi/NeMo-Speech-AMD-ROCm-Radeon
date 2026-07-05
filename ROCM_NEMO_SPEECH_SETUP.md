# ROCm NeMo Speech Setup

Date: 2026-07-05

This checkout is configured for NeMo Speech on an AMD Radeon RX 7900 XT using AMD Radeon ROCm PyTorch wheels. The environment intentionally avoids NeMo's CUDA extras, CUDA PyTorch wheels, `cuda-bindings`, `cuda-python`, `numba-cuda`, Transformer Engine, FlashAttention, DeepEP, Mamba, and other NVIDIA compiled extras.

## Sources Used

- AMD Radeon PyTorch install guide: https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-pytorch.html
- AMD Radeon Linux compatibility matrix: https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html
- AMD Radeon software install guide: https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-radeon.html

AMD's Radeon matrix lists PyTorch 2.9.1 + ROCm 7.2.1 as official production support for Radeon 7000-series GPUs, and lists FP32, FP16, and mixed FP32/FP16 as supported data types. The install guide recommends `repo.radeon.com` wheels over PyTorch.org nightlies for Radeon.

## Host Changes

The host is Ubuntu 24.04.3 with kernel `6.8.0-124-generic`.

Installed AMD's ROCm 7.2.1 apt repository package:

```bash
wget -O /tmp/amdgpu-install_7.2.1.70201-1_all.deb \
  https://repo.radeon.com/amdgpu-install/7.2.1/ubuntu/noble/amdgpu-install_7.2.1.70201-1_all.deb
apt-get install -y /tmp/amdgpu-install_7.2.1.70201-1_all.deb
```

Installed ROCm user space without DKMS:

```bash
amdgpu-install -y --usecase=rocm --no-dkms
```

Important installed ROCm packages include:

```text
rocm 7.2.1.70201-81~24.04
rocm-core 7.2.1.70201-81~24.04
hip-runtime-amd 7.2.53211.70201-81~24.04
miopen-hip 3.5.1.70201-81~24.04
rocblas 5.2.0.70201-81~24.04
rccl 2.27.7.70201-81~24.04
roctracer 4.1.70201.70201-81~24.04
rocminfo 1.0.0.70201-81~24.04
```

This install also created `/opt/rocm -> /opt/rocm-7.2.1` via alternatives.

## Python Environment

The environment is managed by `uv` at:

```bash
/workspace/Speech/.venv-rocm
```

Created with:

```bash
uv venv --python 3.12 .venv-rocm
uv pip install --python .venv-rocm/bin/python --upgrade pip wheel setuptools 'numpy==1.26.4'
```

Installed AMD Radeon ROCm PyTorch wheels by direct URL:

```bash
uv pip install --python .venv-rocm/bin/python \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl' \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchvision-0.24.0%2Brocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl' \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchaudio-2.9.0%2Brocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl' \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/triton-3.5.1%2Brocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl'
```

Installed NeMo common, ASR, audio, TTS, and the non-CUDA `speechlm2` import dependency `peft<=0.18.0` manually, then installed the local package editable with no dependency resolution:

```bash
uv pip install --python .venv-rocm/bin/python --no-deps -e .
```

The local `pyproject.toml` was changed to remove the Linux base dependency on `cuda-bindings`. This keeps package metadata consistent with the ROCm install path.

## Verification

ROCm sees the GPU:

```text
Name: gfx1100
Marketing Name: Radeon RX 7900 XT
```

PyTorch sees and uses the GPU:

```text
torch 2.9.1+rocm7.2.1.gitff65f5bc
torch.version.hip 7.2.53211-e1a6bc5663
cuda_available True
device_count 1
device_name Radeon RX 7900 XT
fp16_matmul torch.float16 (1024, 1024) True
```

NeMo imports pass:

```text
ASRModel import ok
audio import ok
tts ok FastPitchModel HifiGanModel
speechlm2 import ok
```

No CUDA/NVIDIA compiled extras are installed in the venv:

```bash
.venv-rocm/bin/python -m pip list --format=freeze | rg -i '^(cuda|numba-cuda|nvidia|transformer-engine|flash-attn|flashoptim|deep-ep|mamba|causal-conv1d|nv-grouped-gemm)'
```

This returns no packages.

Dependency metadata is clean:

```text
No broken requirements found.
```

End-to-end NeMo ASR smoke test passed:

```text
Model: stt_en_jasper10x5dr
Input: generated 1 second 16 kHz WAV
Execution: model on cuda:0 under torch.amp.autocast('cuda', dtype=torch.float16)
GPU: Radeon RX 7900 XT
HIP: 7.2.53211-e1a6bc5663
Result: transcription completed successfully
```

## Usage

Activate the environment:

```bash
source /workspace/Speech/.venv-rocm/bin/activate
```

Prefer FP16 or mixed FP32/FP16 first:

```python
with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
    ...
```

Do not enable NeMo's `cu12`, `cu13`, `compiled`, or `compiled-a100` extras on this host.
