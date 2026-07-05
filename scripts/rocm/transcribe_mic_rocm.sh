#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

VENV="${VENV:-.venv-rocm}"
PY="${VENV}/bin/python"
DEVICE="${1:-default:CARD=MV7}"
DURATION="${2:-7}"
WAV="${WAV:-/tmp/nemo_rocm_mic.wav}"
MODEL="${MODEL:-stt_en_jasper10x5dr}"

if [[ ! -x "${PY}" ]]; then
  echo "Missing ${PY}. Run: bash scripts/rocm/setup_rocm_uv_env.sh" >&2
  exit 1
fi

if ! command -v arecord >/dev/null 2>&1; then
  echo "arecord is missing. Install alsa-utils or run scripts/rocm/install_rocm_runtime_ubuntu24.sh." >&2
  exit 1
fi

echo "Recording ${DURATION}s from ALSA device for AMD ROCm Radeon NeMo Speech: ${DEVICE}"
rm -f "${WAV}"
arecord -D "${DEVICE}" -f S16_LE -r 16000 -c 1 -d "${DURATION}" "${WAV}"

"${PY}" - "${WAV}" <<'PY'
import audioop
import sys
import wave

path = sys.argv[1]
with wave.open(path, "rb") as wav:
    data = wav.readframes(wav.getnframes())
    rms = audioop.rms(data, wav.getsampwidth())
    peak = audioop.max(data, wav.getsampwidth())
    duration = wav.getnframes() / wav.getframerate()

print(f"clip={path}")
print(f"duration_sec={duration:.3f}")
print(f"rms={rms}")
print(f"peak={peak}")

if rms <= 50 or peak <= 500:
    raise SystemExit("Recorded audio looks silent. Check mic mute/gain or choose another arecord device.")
PY

"${PY}" - "${WAV}" "${MODEL}" <<'PY'
import sys
import torch
from nemo.collections.asr.models import EncDecCTCModel

wav_path, model_name = sys.argv[1], sys.argv[2]
model = EncDecCTCModel.from_pretrained(model_name, map_location="cpu")
model.eval().to("cuda")

with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
    result = model.transcribe([wav_path], batch_size=1, return_hypotheses=False)

text = getattr(result[0], "text", result[0])
print(f"device={next(model.parameters()).device}")
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"transcript={text!r}")
PY
