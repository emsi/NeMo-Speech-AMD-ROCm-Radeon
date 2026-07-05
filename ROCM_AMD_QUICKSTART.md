# AMD ROCm Radeon NeMo Speech Quickstart

This is the AMD-compatible NVIDIA NeMo Speech variant. It is the direct path for running NeMo Speech on an AMD Radeon GPU
such as the Radeon RX 7900 XT (`gfx1100`) with AMD ROCm PyTorch.

Use the ROCm setup scripts in this guide for AMD Radeon hosts.

The Python package name remains `nemo-toolkit` and imports remain under `nemo`; the forked runtime path is AMD ROCm
Radeon first.

## Tested AMD Radeon Target

- OS: Ubuntu 24.04
- GPU: Radeon RX 7900 XT, `gfx1100`
- ROCm runtime: 7.2.1
- PyTorch: AMD Radeon wheel, `torch==2.9.1+rocm7.2.1`
- Python: 3.12
- Precision: FP16 / mixed precision first

## 1. Install ROCm Runtime

Run as root on Ubuntu 24.04:

```bash
cd /path/to/NeMo-Speech-AMD-ROCm-Radeon
bash scripts/rocm/install_rocm_runtime_ubuntu24.sh
```

This installs AMD ROCm 7.2.1 user-space packages without DKMS:

```bash
amdgpu-install -y --usecase=rocm --no-dkms
```

## 2. Create the ROCm Python Environment

Run from the repo root:

```bash
cd /path/to/NeMo-Speech-AMD-ROCm-Radeon
bash scripts/rocm/setup_rocm_uv_env.sh
```

This creates `.venv-rocm`, installs AMD's Radeon ROCm PyTorch wheels, installs NeMo Speech dependencies, and installs
this AMD ROCm Radeon fork editable for the ROCm runtime.

Activate it:

```bash
source .venv-rocm/bin/activate
```

## 3. Verify AMD GPU + NeMo

```bash
cd /path/to/NeMo-Speech-AMD-ROCm-Radeon
bash scripts/rocm/verify_rocm_nemo.sh
```

Expected result:

```text
GPU: Radeon RX 7900 XT
FP16 matmul: ok
NeMo imports: ok
No broken requirements found.
```

## 4. Test Mic Dictation

List capture devices:

```bash
arecord -l
```

Record and transcribe 7 seconds from a specific ALSA input:

```bash
cd /path/to/NeMo-Speech-AMD-ROCm-Radeon
bash scripts/rocm/transcribe_mic_rocm.sh default:CARD=MV7 7
```

For another mic, replace `default:CARD=MV7` with the device shown by `arecord -L`, for example:

```bash
bash scripts/rocm/transcribe_mic_rocm.sh plughw:CARD=Generic,DEV=0 7
```

The script records `/tmp/nemo_rocm_mic.wav`, checks that it is not silent, and transcribes it with NeMo on `cuda:0`
backed by ROCm/HIP. PyTorch intentionally exposes ROCm GPUs through the CUDA-compatible API surface.

## AMD ROCm Container Device Pass-Through

The container must receive these devices and groups:

```yaml
devices:
  - /dev/kfd:/dev/kfd
  - /dev/dri:/dev/dri
  - /dev/snd:/dev/snd
group_add:
  - video
  - render
  - audio
ipc: host
shm_size: 8g
security_opt:
  - seccomp=unconfined
environment:
  XDG_RUNTIME_DIR: /run/user/1000
  PULSE_SERVER: unix:/run/user/1000/pulse/native
volumes:
  - /run/user/1000:/run/user/1000
```

After recreating the container, verify:

```bash
rocminfo | grep -E 'gfx1100|Radeon'
arecord -l
```

## ROCm Environment Sanity Check

These package names indicate CUDA-only extras in the ROCm environment:

```text
cuda-bindings
cuda-python
numba-cuda
transformer-engine
flash-attn
flashoptim
deep_ep
mamba-ssm
causal-conv1d
nv-grouped-gemm
```

Check the environment:

```bash
.venv-rocm/bin/python -m pip list --format=freeze \
  | rg -i '^(cuda|numba-cuda|nvidia|transformer-engine|flash-attn|flashoptim|deep-ep|mamba|causal-conv1d|nv-grouped-gemm)'
```

A clean ROCm environment has no matches from that command.
