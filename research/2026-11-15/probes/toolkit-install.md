# NVIDIA Container Toolkit install and verification

Installed on 2026-11-15 on this laptop (RTX 5080 Laptop GPU, driver 595.99.02, Ubuntu 26.04.1, Docker 29.9.3).

## What was run

The user ran [install-nvidia-container-toolkit-v2.sh](install-nvidia-container-toolkit-v2.sh) with `sudo` in a terminal.
It follows NVIDIA's [install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html): NVIDIA's apt key and list, the four packages pinned to 1.20.1-1, a CDI spec check, `nvidia-ctk runtime configure --runtime=docker`, and a Docker restart that runs only when no container is running.
The full log is [toolkit-install-v2.log](toolkit-install-v2.log).

The first attempt, [install-nvidia-container-toolkit-v1.sh](install-nvidia-container-toolkit-v1.sh), was run as `sudo bash ... | tee ...` and stalled after step 1 ([log](toolkit-install-v1-stalled.log)).
Its `apt-get` process sat in state `T` (stopped) in the foreground group of the pseudo-terminal that `sudo` created, with nothing from NVIDIA installed yet (observed with `ps` and `/proc/<pid>/status`).
The cause is likely terminal job control around apt's own terminal handling when its output is piped (inferred, not proven).
Version 2 avoids the terminal entirely (`DEBIAN_FRONTEND=noninteractive`, `-o Dpkg::Use-Pty=0`, input from `/dev/null`, `-o DPkg::Lock::Timeout=300`), writes its own log, and ended the stalled run before starting.
Running `sudo` through Claude Code's `!` prefix fails with `sudo: A terminal is required to authenticate`, so privileged steps run in a separate terminal or through `pkexec`.

## Result (observed)

| Check | Result |
|---|---|
| Packages | `nvidia-container-toolkit`, `nvidia-container-toolkit-base`, `libnvidia-container-tools`, `libnvidia-container1`, all 1.20.1-1; 37.7 MB on disk |
| `nvidia-ctk --version` | 1.20.1, commit dffc40b4f820ce5c512633bac9e0418d0e05a2ee |
| Repository key | NVIDIA CORPORATION (Open Source Projects), fingerprint `C95B 321B 61E8 8C18 09C4 F759 DDCA E044 F796 ECB0` |
| CDI refresh units | `nvidia-cdi-refresh.path` and `.service` enabled; the path unit is active |
| CDI spec | `/var/run/cdi/nvidia.yaml`, devices `nvidia.com/gpu=0`, `nvidia.com/gpu=<UUID>`, `nvidia.com/gpu=all` |
| `/etc/docker/daemon.json` | Only a `runtimes.nvidia` entry with path `nvidia-container-runtime`; default runtime stays `runc` |
| Docker | Runtimes `io.containerd.runc.v2 nvidia runc`, default `runc`, live restore off, CDI spec directories `/etc/cdi` and `/run/cdi` |

## GPU tests in containers (measured on this laptop)

| Test | Command (all with `--memory` and an equal `--memory-swap`) | Result |
|---|---|---|
| CDI | `docker run --rm --device nvidia.com/gpu=all ubuntu:24.04 nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv` | `NVIDIA GeForce RTX 5080 Laptop GPU, 595.99.02, 16303 MiB, 12.0` |
| NVIDIA runtime | `docker run --rm --runtime=nvidia --gpus all ubuntu:24.04 nvidia-smi -L` | GPU 0 listed |
| Plain `--gpus` | `docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L` | GPU 0 listed |
| CUDA kernel | `docker run --rm --device nvidia.com/gpu=all nvcr.io/nvidia/k8s/cuda-sample:vectoradd-cuda12.5.0-ubuntu22.04` | `Test PASSED` |

The vectorAdd sample is a CUDA 12.5 build, and CUDA 12.5 cannot emit sm_120 machine code, so its kernel most likely ran through the driver's PTX JIT compiler, which the toolkit mounts into the container (inferred).
The images `ubuntu:24.04` (119 MB) and `nvcr.io/nvidia/k8s/cuda-sample:vectoradd-cuda12.5.0-ubuntu22.04` (343 MB) were pulled for these tests.

## How to undo

`sudo apt-get purge nvidia-container-toolkit nvidia-container-toolkit-base libnvidia-container-tools libnvidia-container1`, delete `/etc/apt/sources.list.d/nvidia-container-toolkit.list` and `/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg`, delete `/etc/docker/daemon.json` (it held only the `nvidia` runtime entry), and run `sudo systemctl restart docker`.
