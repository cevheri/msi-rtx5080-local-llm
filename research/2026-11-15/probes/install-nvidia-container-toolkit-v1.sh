#!/usr/bin/env bash
# Install NVIDIA Container Toolkit 1.20.1-1 following NVIDIA's install guide, then configure Docker.
# Source of the commands: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
# Run as: sudo bash install-nvidia-container-toolkit.sh
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "Run this script with sudo." >&2
  exit 1
fi

V=1.20.1-1
KEYRING=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
LIST=/etc/apt/sources.list.d/nvidia-container-toolkit.list

echo "== 1/6 Prerequisites"
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl gnupg2

echo "== 2/6 NVIDIA repository key and list"
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --batch --yes --dearmor -o "$KEYRING"
gpg --show-keys --with-fingerprint "$KEYRING"
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed "s#deb https://#deb [signed-by=$KEYRING] https://#g" > "$LIST"
cat "$LIST"

echo "== 3/6 Install toolkit $V"
apt-get update
apt-get install -y \
  nvidia-container-toolkit="$V" \
  nvidia-container-toolkit-base="$V" \
  libnvidia-container-tools="$V" \
  libnvidia-container1="$V"

echo "== 4/6 CDI specification"
if ! nvidia-ctk cdi list 2>/dev/null | grep -q 'nvidia.com/gpu=all'; then
  nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml
fi
nvidia-ctk cdi list

echo "== 5/6 Docker runtime configuration"
nvidia-ctk runtime configure --runtime=docker
cat /etc/docker/daemon.json

echo "== 6/6 Restart Docker (only when no container is running)"
running=$(docker ps -q | wc -l)
if [ "$running" -gt 0 ]; then
  echo "$running container(s) are running, so Docker was not restarted." >&2
  echo "Stop them, then run: sudo systemctl restart docker" >&2
  exit 2
fi
systemctl restart docker
docker info | grep -E 'Runtimes|Default Runtime|CDI spec' -A2
echo "Done."
