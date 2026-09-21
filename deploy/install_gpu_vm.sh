#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -eq 0 ]]; then
    SUDO=
else
    SUDO=sudo
fi

echo "Installing Docker and NVIDIA Container Toolkit..."
$SUDO apt-get update
$SUDO apt-get install -y ca-certificates curl gnupg rsync ubuntu-drivers-common

if ! nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA driver not detected; installing the Ubuntu-recommended driver..."
    $SUDO ubuntu-drivers install
fi

if ! command -v docker >/dev/null 2>&1; then
    curl -fsSL https://get.docker.com | $SUDO sh
fi

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | $SUDO gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null

$SUDO apt-get update
$SUDO apt-get install -y nvidia-container-toolkit
$SUDO nvidia-ctk runtime configure --runtime=docker
$SUDO systemctl restart docker

if ! nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA driver installation completed. Reboot the VM, then rerun this script." >&2
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$SUDO docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p roms checkpoints
echo
echo "GPU Docker is ready."
echo "Copy the ROM to: $(pwd)/roms/Super Mario Bros. Deluxe.gbc"
echo "Build with:      docker compose build"
echo "Start training:  docker compose up trainer"
