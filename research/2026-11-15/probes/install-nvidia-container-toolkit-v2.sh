#!/usr/bin/env bash
# Install NVIDIA Container Toolkit 1.20.1-1 following NVIDIA's install guide, then configure Docker.
# Non-interactive version: apt never touches the terminal, so it cannot be stopped by job control.
# Source of the commands: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
# Run as: pkexec bash install-nvidia-container-toolkit-v2.sh   (or: sudo bash install-nvidia-container-toolkit-v2.sh)
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "Run this script as root (pkexec or sudo)." >&2
  exit 1
fi

SCRATCH=$SCRATCH
LOG="$SCRATCH/toolkit-install-v2.log"
exec > >(tee -a "$LOG") 2>&1

export DEBIAN_FRONTEND=noninteractive
APT=(apt-get -y -q -o Dpkg::Use-Pty=0 -o APT::Color=0 -o DPkg::Lock::Timeout=300)
V=1.20.1-1
KEYRING=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
LIST=/etc/apt/sources.list.d/nvidia-container-toolkit.list
OLD="^(sudo )?bash $SCRATCH/install-nvidia-container-toolkit\.sh\$"

echo "== 0/6 Stop the earlier run that got stuck"
for pid in $(pgrep -f "$OLD" || true); do
  echo "ending earlier script process $pid"
  kill -TERM "$pid" 2>/dev/null || true
done
for pid in $(pgrep -x apt-get || true); do
  case "$(ps -o stat= -p "$pid" 2>/dev/null || true)" in
    T*) echo "resuming and ending stopped apt-get $pid"
        kill -CONT "$pid" 2>/dev/null || true
        sleep 1
        kill -TERM "$pid" 2>/dev/null || true ;;
  esac
done
sleep 2
dpkg --configure -a

echo "== 1/6 Prerequisites"
"${APT[@]}" update < /dev/null
"${APT[@]}" install --no-install-recommends ca-certificates curl gnupg2 < /dev/null

echo "== 2/6 NVIDIA repository key and list"
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --batch --yes --dearmor -o "$KEYRING"
gpg --batch --show-keys --with-fingerprint "$KEYRING"
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed "s#deb https://#deb [signed-by=$KEYRING] https://#g" > "$LIST"
cat "$LIST"

echo "== 3/6 Install toolkit $V"
"${APT[@]}" update < /dev/null
"${APT[@]}" install \
  nvidia-container-toolkit="$V" \
  nvidia-container-toolkit-base="$V" \
  libnvidia-container-tools="$V" \
  libnvidia-container1="$V" < /dev/null

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
docker info | grep -E -A2 'Runtimes|Default Runtime|CDI spec'
echo "Done."
