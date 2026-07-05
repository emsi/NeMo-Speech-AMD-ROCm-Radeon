#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script as root: sudo bash scripts/rocm/install_rocm_runtime_ubuntu24.sh" >&2
  exit 1
fi

if [[ ! -r /etc/os-release ]]; then
  echo "Cannot read /etc/os-release" >&2
  exit 1
fi

# shellcheck disable=SC1091
source /etc/os-release

if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "24.04" ]]; then
  echo "This script is for Ubuntu 24.04. Detected: ${PRETTY_NAME:-unknown}" >&2
  exit 1
fi

AMDGPU_DEB="/tmp/amdgpu-install_7.2.1.70201-1_all.deb"
AMDGPU_URL="https://repo.radeon.com/amdgpu-install/7.2.1/ubuntu/noble/amdgpu-install_7.2.1.70201-1_all.deb"

apt-get update
apt-get install -y wget ca-certificates python3-setuptools python3-wheel alsa-utils pulseaudio-utils

wget -O "${AMDGPU_DEB}" "${AMDGPU_URL}"
apt-get install -y "${AMDGPU_DEB}"

# --no-dkms keeps the existing kernel driver and installs ROCm user space.
amdgpu-install -y --usecase=rocm --no-dkms

echo
echo "AMD ROCm Radeon runtime installed for NeMo Speech. Quick checks:"
rocminfo 2>/dev/null | grep -E "Name: +gfx|Marketing Name: +Radeon" || true
rocm-smi --showproductname --showdriverversion || true
