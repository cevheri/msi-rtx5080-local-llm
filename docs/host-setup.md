# Host setup for GPU containers

Last verified: 2026-11-18.

Before any model server can run, install the NVIDIA Container Toolkit from NVIDIA's apt repo and give containers the GPU through CDI (`--device nvidia.com/gpu=all`).
The `nvidia-container-toolkit-base` package alone is enough for that CDI path, and a toolkit-free manual setup also works but is fragile ([Smaller options](#smaller-options-base-package-only-or-no-toolkit)).
Then publish every server port on `127.0.0.1` only, because Docker-published ports bypass ufw.
Give every container its own memory cap, because the `heavy` wrapper does not reach containers.
Start servers on demand with no restart policy, one at a time, and keep one Hugging Face cache under `/home` that vLLM and llama.cpp share.
The toolkit was installed and verified on this laptop on 2026-11-15, including a CUDA kernel run inside a container ([install record](../research/2026-11-15/probes/toolkit-install.md)); no model server has been run yet.

## Current host state (observed)

These values were observed on 2026-11-15 with `nvidia-smi`, `lscpu`, `free`, `lsblk` and `docker info`.

| Item | Observed value |
|---|---|
| GPU | NVIDIA GeForce RTX 5080 Laptop GPU, Blackwell GB203M, compute capability 12.0 (sm_120), 16303 MiB GDDR7, 256-bit |
| GPU power | Power limit 140 W (default 80 W, max 175 W); NVIDIA lists 80-150 W subsystem power and 896 GB/s for this GPU ([NVIDIA laptop compare](https://www.nvidia.com/en-us/geforce/laptops/compare/)) |
| Graphics mode | PRIME on-demand, the Intel iGPU drives the desktop, about 112 MiB VRAM in use at idle |
| Driver | 595.99.02, `nvidia-driver-595-open`, CUDA driver API 13.2, no CUDA toolkit on the host |
| CPU and RAM | Intel Core Ultra 9 290HX Plus, 24 cores; 61 GiB usable RAM, about 20 GiB used by the desktop, so about 40 GiB free |
| Disk | 2 TB NVMe; 562 GB free on `/home`, 405 GB free on `/`; Docker root is `/var/lib/docker` on `/` |
| OS | Ubuntu 26.04.1 LTS, kernel 7.0.0-43 |
| Docker | Engine 29.9.3 (rootful, the user is in the `docker` group), Compose v5.7.1, cgroup v2 with the systemd cgroup driver |
| Docker GPU support | Before the install: runtimes `runc` only and no `/etc/docker/daemon.json`. After the install on 2026-11-15: runtimes `io.containerd.runc.v2 nvidia runc`, default `runc`, `daemon.json` holding only the `nvidia` runtime entry, CDI spec `/var/run/cdi/nvidia.yaml` |
| NVIDIA Container Toolkit | 1.20.1-1 installed on 2026-11-15 (all four packages); CDI devices `nvidia.com/gpu=0` and `nvidia.com/gpu=all`; CDI, `--runtime=nvidia` and plain `--gpus all` all verified ([install record](../research/2026-11-15/probes/toolkit-install.md)) |
| Firewall | ufw service active and enabled |
| Model cache | `~/.cache/huggingface` does not exist yet |

The user is in the `docker` group, so every `docker` command below runs without `sudo`; that group is root-equivalent, so treat it with the same care as `sudo`, and the commands that need `sudo` are listed in [Commands that need sudo](#commands-that-need-sudo).

## Install the NVIDIA Container Toolkit

Ubuntu 26.04 is listed for amd64 in the toolkit's [supported platforms table](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/supported-platforms.html), and Docker Engine lists "Ubuntu Resolute 26.04 (LTS)" as supported ([Docker install docs](https://docs.docker.com/engine/install/ubuntu/)).
The [install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) pins `NVIDIA_CONTAINER_TOOLKIT_VERSION=1.20.1-1`, which is the latest [GitHub release](https://github.com/NVIDIA/nvidia-container-toolkit/releases) (v1.20.1, 2026-09-19).
NVIDIA's stable apt repo also serves `1.20.2-1` ([Packages index](https://nvidia.github.io/libnvidia-container/stable/deb/amd64/Packages)), from a signed tag v1.20.2 dated 2026-11-06 whose [CHANGELOG](https://github.com/NVIDIA/nvidia-container-toolkit/blob/v1.20.2/CHANGELOG.md) lists only bug fixes and dependency bumps, but it has no GitHub release object and no docs release note.
Either version works on a single-GPU laptop; this page uses the documented pin.
Use NVIDIA's repo, not Ubuntu's archive: Ubuntu only has `nvidia-container-toolkit` 1.19.0 in `resolute-proposed` ([Launchpad](https://launchpad.net/ubuntu/+source/nvidia-container-toolkit)), and mixing the two sources invites version conflicts.
The repo list file is distribution-agnostic (`deb https://nvidia.github.io/libnvidia-container/stable/deb/$(ARCH) /`), so no Ubuntu 26.04 specific path is needed ([list file](https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list)).

These are the official commands from the [install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), unchanged except for the comment:

```bash
sudo apt-get update && sudo apt-get install -y --no-install-recommends ca-certificates curl gnupg2
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.20.1-1   # docs pin; 1.20.2-1 is also in the stable repo
sudo apt-get install -y \
    nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
```

Version 1.20.1 orders `nvidia-cdi-refresh` before Docker at boot and keeps existing Docker feature flags when `nvidia-ctk runtime configure` enables CDI ([release notes](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/release-notes.html)).
The generated CDI spec uses schema v0.7.0, which needs Docker 26.1.0 or newer; Docker 29.9.3 qualifies ([release notes](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/release-notes.html)).
Toolkit 1.20.1 features such as `NVIDIA_GPU_MEMORY_LIMIT` need an R615 or newer driver, so they do not work on driver 595 ([release notes](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/release-notes.html)).

### Install notes from 2026-11-15

These commands were run on this laptop through a script, and the result is recorded in the [install record](../research/2026-11-15/probes/toolkit-install.md).
Running `sudo` from Claude Code's `!` prefix fails with `sudo: A terminal is required to authenticate`, so run privileged steps in a separate terminal or with `pkexec`.
Do not pipe `sudo apt-get` through `tee`: the first attempt stopped with `apt-get` in state `T` after step 1 (observed; the cause is likely terminal job control, inferred).
The working script ran apt non-interactively with `DEBIAN_FRONTEND=noninteractive`, `-o Dpkg::Use-Pty=0`, input from `/dev/null` and `-o DPkg::Lock::Timeout=300`, and wrote its own log ([script](../research/2026-11-15/probes/install-nvidia-container-toolkit-v2.sh)).

## Smaller options: base package only, or no toolkit

This section answers whether Docker can use the GPU without the NVIDIA Container Toolkit; it was tested and source-checked on 2026-11-15.

### What plain Docker does without the toolkit (observed)

Docker does not add GPU device nodes or driver libraries to a container by itself.
On this host, `docker run --rm --gpus all hello-world` fails with `failed to discover GPU vendor from CDI: no known GPU vendor found`, and `docker run --rm --device nvidia.com/gpu=all hello-world` fails with `CDI device injection failed: unresolvable CDI devices nvidia.com/gpu=all` (observed, [probe log](../research/2026-11-15/probes/no-toolkit-gpu-test.md)).
Docker registers its NVIDIA handler for `--gpus` at daemon start, and only when `nvidia-cdi-hook` or `nvidia-container-runtime-hook` is on the PATH ([devices_nvidia_linux.go](https://github.com/moby/moby/blob/master/daemon/devices_nvidia_linux.go), [devices_linux.go](https://github.com/moby/moby/blob/master/daemon/devices_linux.go)).

### Base package only (the smallest supported install)

For CDI-only use, NVIDIA accepts either the full toolkit or just the `nvidia-container-toolkit-base` package, and says the runtime hook and its dependencies are not needed on machines that use CDI exclusively ([CDI support](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html)).
At v1.20.1 the base package installs `nvidia-ctk`, `nvidia-cdi-hook`, `nvidia-container-runtime` and the `nvidia-cdi-refresh` units with their udev rule ([base package file list](https://github.com/NVIDIA/nvidia-container-toolkit/blob/v1.20.1/packaging/debian/nvidia-container-toolkit-base.install)).
The full `nvidia-container-toolkit` package adds only `nvidia-container-runtime-hook`, which brings in `libnvidia-container` ([full package file list](https://github.com/NVIDIA/nvidia-container-toolkit/blob/v1.20.1/packaging/debian/nvidia-container-toolkit.install)).
With the base package alone, skip `nvidia-ctk runtime configure`, so `/etc/docker/daemon.json` stays untouched.
`--device nvidia.com/gpu=all` should then work without restarting Docker, because Docker creates its CDI cache with the library default of automatic refresh ([daemon/cdi.go](https://github.com/moby/moby/blob/master/daemon/cdi.go), [CDI cache.go](https://github.com/cncf-tags/container-device-interface/blob/main/pkg/cdi/cache.go)) (inferred from source, not tested).
`--gpus all` needs one Docker restart after the install, because its handler is registered when the daemon starts (inferred from source, not tested).
Ubuntu 26.04's enabled archive pockets have no candidate for either package (`apt-cache policy` shows none, observed), so install from NVIDIA's apt repo as above while listing only `nvidia-container-toolkit-base`, or install the base `.deb` from the [v1.20.1 release](https://github.com/NVIDIA/nvidia-container-toolkit/releases/tag/v1.20.1) tarball (`nvidia-container-toolkit_1.20.1_deb_amd64.tar.gz`) with `dpkg -i`.

```bash
# after adding NVIDIA's apt repo with the commands above
sudo apt-get install -y nvidia-container-toolkit-base=1.20.1-1
nvidia-ctk cdi list        # expect nvidia.com/gpu=0 and nvidia.com/gpu=all
# if the list is empty: sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml
docker run --rm --memory=1g --memory-swap=1g --device nvidia.com/gpu=all nvidia/cuda:13.0.3-base-ubuntu24.04 nvidia-smi
```

The base package's post-install script writes the runtime config file and reloads the udev rules; it does not run `nvidia-ctk runtime configure` ([postinst](https://github.com/NVIDIA/nvidia-container-toolkit/blob/v1.20.1/packaging/debian/nvidia-container-toolkit-base.postinst)).
To undo it, remove the package and the repo list file; the generated spec lives in `/var/run/cdi`, which does not survive a reboot.

### No toolkit at all (tested, not recommended for daily use)

Plain Docker can use the GPU if the device nodes and the host driver's user-space libraries are passed in by hand, which is the work the toolkit automates.
In a test with `python:3.12-slim`, `nvidia-smi` listed the RTX 5080 Laptop GPU, and a CUDA driver API probe reported compute capability 12.0, 15841 MiB, and a successful 1 GiB allocation (measured on this laptop, [probe log](../research/2026-11-15/probes/no-toolkit-gpu-test.md)).
The tested command, with a memory cap added here:

```bash
L=/usr/lib/x86_64-linux-gnu
docker run --rm --memory=4g --memory-swap=4g \
  --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
  -v $L/libcuda.so.1:$L/libcuda.so.1:ro \
  -v $L/libnvidia-ml.so.1:$L/libnvidia-ml.so.1:ro \
  -v $L/libnvidia-ptxjitcompiler.so.1:$L/libnvidia-ptxjitcompiler.so.1:ro \
  -v $L/libnvidia-nvvm.so.4:$L/libnvidia-nvvm.so.4:ro \
  -v $L/libnvidia-gpucomp.so.595.99.02:$L/libnvidia-gpucomp.so.595.99.02:ro \
  -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro \
  python:3.12-slim nvidia-smi -L
```

It is not the recommended path, for these reasons:

- Neither NVIDIA nor vLLM documents it, so problems have no upstream answer.
- The library list is maintained by hand: a missing library fails only on the code path that needs it, and `libnvidia-gpucomp.so.595.99.02` carries the driver version, so that mount breaks at the next driver update.
- vLLM needs more than the test above: Triton finds libcuda through `TRITON_LIBCUDA_PATH`, `ldconfig -p` or `LD_LIBRARY_PATH` and links against the unversioned `libcuda.so` ([Triton driver.py](https://github.com/triton-lang/triton/blob/main/third_party/nvidia/backend/driver.py)), so a vLLM container would also need `-v $L/libcuda.so:$L/libcuda.so:ro` and `-e TRITON_LIBCUDA_PATH=$L` (inferred; vLLM was not run this way).
- It skips the `NVIDIA_REQUIRE_CUDA` check that refuses an image built for a newer CUDA than the driver, so such a mismatch fails later and less clearly.
- The device nodes must already exist on the host; they did during the test (observed), and `nvidia-modprobe` is not installed here to create them.

Use it for quick experiments; for daily use, install the base package.

## Risks and side effects of the toolkit

The toolkit does not install a driver or any CUDA library, so it cannot conflict with PyTorch or CUDA versions used by other projects.
NVIDIA's apt repo holds only container packages (`libnvidia-container*`, `nvidia-container-runtime`, `nvidia-container-toolkit-base`, `nvidia-container-toolkit`, `nvidia-docker2`), so an `apt upgrade` from it cannot replace the driver or CUDA ([Packages index](https://nvidia.github.io/libnvidia-container/stable/deb/amd64/Packages)).
At 1.20.1-1, `nvidia-container-toolkit-base` has no package dependencies and an installed size of 27752 KiB, and the full stack adds only `libc6`, `libcap2` and `libseccomp2` as dependencies ([Packages index](https://nvidia.github.io/libnvidia-container/stable/deb/amd64/Packages)).
Host processes never pass through the toolkit: it acts only when a container requests a GPU.
Each container and each Python environment brings its own CUDA user-space libraries; for example the PyPI `torch` 2.15.0 wheel depends on `cuda-toolkit==13.2.2` and `nvidia-cudnn-cu13` pip packages ([PyPI metadata](https://pypi.org/pypi/torch/json)), while the vLLM image carries CUDA 13.0.3 and the llama.cpp image CUDA 12.8.1.
The only shared piece is the driver, and that limit exists with or without the toolkit: driver 595 runs CUDA 13.2 and older natively ([CUDA release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)).

What does need care:

- VRAM is shared: a vLLM container reserves 92% of the 16 GB at start, so a PyTorch job from another project and a model server cannot both use the GPU fully; stop the model container first.
- `nvidia-ctk runtime configure` edits `/etc/docker/daemon.json` and needs a Docker restart, and with `Live Restore Enabled: false` (observed) that stops every running container; the base-only CDI install skips this step.
- The toolkit's hooks run with root privileges at container start, and container-escape class vulnerabilities have been fixed several times, most recently CVE-2026-24260 (CVSS 8.5, affected up to 1.19.0, fixed in 1.19.1) ([June 2026 bulletin](https://nvidia.custhelp.com/app/answers/detail/a_id/5850)); stay on a current release such as 1.20.1 and give the GPU only to trusted images.
- Some past issues spared CDI: NVIDIA said CVE-2025-23359 "does not impact use cases where CDI is used" ([February 2025 bulletin](https://nvidia.custhelp.com/app/answers/detail/a_id/5616)).
- Installing from NVIDIA's repo brings future fixes through `apt upgrade`; installing a `.deb` without the repo means tracking security releases by hand.
- To remove it: `sudo apt-get purge nvidia-container-toolkit-base`, delete the repo list and keyring files, and, if `nvidia-ctk runtime configure` was run, remove the `nvidia` runtime entry from `/etc/docker/daemon.json` and restart Docker; the spec in `/var/run/cdi` disappears at reboot.

## Configure Docker for the GPU

There are two ways to hand the GPU to a container.
The legacy path is the NVIDIA runtime with `--gpus all`, which is what the [vLLM Docker docs](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/deployment/docker.md) and NVIDIA's [sample workload](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html) show.
The other path is CDI, where Docker injects the devices named in a CDI spec with `--device nvidia.com/gpu=all`.
Docker Engine has CDI on by default since 28.3.0 and reads specs from `/etc/cdi` and `/var/run/cdi` ([dockerd reference](https://github.com/moby/moby/blob/master/docs/reference/dockerd.md#configure-cdi-devices)).
Since toolkit 1.18.0, the `nvidia-cdi-refresh` systemd units write `/var/run/cdi/nvidia.yaml` automatically on toolkit install, driver install and reboot ([CDI support](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html)).

### Why CDI is preferred on this host

NVIDIA's [troubleshooting page](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html) says that when the GPU is injected by the NVIDIA Container Runtime Hook (the `--gpus` flag, or the NVIDIA runtime in legacy mode) and systemd manages container cgroups, `systemctl daemon-reload` is enough to make running containers lose the GPU with `Failed to initialize NVML: Unknown Error`.
This host uses the systemd cgroup driver on cgroup v2, and package upgrades commonly run `daemon-reload` (inferred), so a long-running vLLM or llama.cpp container could lose the GPU in the middle of a session.
The documented mitigations are the cgroupfs driver, explicit `--device` nodes, or CDI ([troubleshooting](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html)).
CDI is the least invasive of the three, so every command in this knowledge base uses `--device nvidia.com/gpu=all`.
If a container that was started with `--gpus` loses the GPU, recreate it.

### Configure and verify

Configure the NVIDIA runtime as well, so the official sample and upstream `--gpus` commands also work ([install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)).
With the base-only install from [Smaller options](#smaller-options-base-package-only-or-no-toolkit), skip this step and the legacy check below; `--device nvidia.com/gpu=all` is all this knowledge base uses.
`nvidia-ctk runtime configure` edits `/etc/docker/daemon.json`, and restarting Docker stops every running container.

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

nvidia-ctk --version
sudo systemctl status nvidia-cdi-refresh.path nvidia-cdi-refresh.service
nvidia-ctk cdi list                       # expect nvidia.com/gpu=0, nvidia.com/gpu=all, ...
docker info | grep -i -A3 cdi             # spec directories /etc/cdi and /var/run/cdi
docker info | grep -i runtimes            # nvidia should now be listed next to runc

# CDI path (used by this knowledge base)
docker run --rm --memory=1g --memory-swap=1g --device nvidia.com/gpu=all nvidia/cuda:13.0.3-base-ubuntu24.04 nvidia-smi

# Legacy path (NVIDIA's own sample, plus a memory cap)
docker run --rm --memory=1g --memory-swap=1g --runtime=nvidia --gpus all ubuntu nvidia-smi
```

If `nvidia-ctk cdi list` is empty, regenerate the spec with `sudo systemctl restart nvidia-cdi-refresh.service` or `sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml` ([CDI support](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html)).
The refresh units do not handle driver removal, so regenerate after any driver change that was not a plain upgrade ([CDI support](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html)).

### Which CUDA images the 595 driver runs

Driver branch R595 corresponds to CUDA 13.2; 13.0 is R580, 13.1 is R590, 13.3 is R610, 13.4 is R615 and 13.5 is R620 ([CUDA release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)).
CUDA 12.x and 13.0 to 13.2 images therefore run natively on this driver.
CUDA 13.3 to 13.5 images run only under [minor version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html), where features that need a newer driver fail with `cudaErrorCallRequiresNewerDriver` and applications that JIT-compile PTX do not work.
[Forward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html) packages cover data center GPUs and select RTX server SKUs, not GeForce laptop GPUs, so there is no way to lift that limit here.
Use `nvidia/cuda:13.0.3-base-ubuntu24.04` for verification: it matches the CUDA 13.0.3 base of the vLLM v0.33.0 image ([Dockerfile](https://github.com/vllm-project/vllm/blob/v0.33.0/docker/Dockerfile)); `13.2.1-base-ubuntu24.04` also fits ([Docker Hub tags](https://hub.docker.com/r/nvidia/cuda/tags)).
The `ubuntu26.04` CUDA base tags start at 13.3.0, which is newer than R595, so avoid them for the smoke test ([Docker Hub tags](https://hub.docker.com/r/nvidia/cuda/tags)).
The same rule is why the llama.cpp guide pins `server-cuda-b12407` (CUDA 12.8.1, [cuda.Dockerfile](https://github.com/ggml-org/llama.cpp/blob/b12418/.devops/cuda.Dockerfile)) rather than `server-cuda13`, and why vLLM `cu135-nightly` tags are avoided; see [llama.cpp](../llamacpp/README.md) and [vLLM](../vllm/README.md).

`nvidia-smi` inside a container needs the `utility` driver capability; when `NVIDIA_DRIVER_CAPABILITIES` is unset the default is `utility,compute`, and the CUDA base images set it ([specialized configurations](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html)).

## Compose GPU syntax

Compose accepts three forms, and all three were checked with `docker compose config` on this host's Compose v5.7.1 (syntax check only, no container started).
The CDI form uses the service-level `devices` list, which accepts CDI names ([Compose services reference](https://docs.docker.com/reference/compose-file/services/)).
The `gpus: all` short form is a device request with an implicit `gpu` capability and needs Compose 2.30.0 or newer ([Compose services reference](https://docs.docker.com/reference/compose-file/services/)).
The `deploy.resources.reservations.devices` form needs `capabilities`, and omitting it is an error at deployment; `count` and `device_ids` are mutually exclusive ([Compose GPU support](https://docs.docker.com/compose/how-tos/gpu-support/)).

```yaml
services:
  cdi-form:                     # preferred here: survives systemctl daemon-reload
    image: nvidia/cuda:13.0.3-base-ubuntu24.04
    command: ["nvidia-smi"]
    mem_limit: 1g
    memswap_limit: 1g
    devices:
      - nvidia.com/gpu=all
  short-form:                   # Compose 2.30.0 or newer
    image: nvidia/cuda:13.0.3-base-ubuntu24.04
    command: ["nvidia-smi"]
    mem_limit: 1g
    memswap_limit: 1g
    gpus: all
  deploy-form:
    image: nvidia/cuda:13.0.3-base-ubuntu24.04
    command: ["nvidia-smi"]
    mem_limit: 1g
    memswap_limit: 1g
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all        # or device_ids: ["0"], not both
              capabilities: [gpu]
```

A full vLLM service with the port, memory cap and cache mounts from the sections below looks like this; the model name and engine arguments come from [vLLM models](../vllm/models.md).
It has no `restart:` key on purpose (see [Laptop power](#laptop-power-and-restart-policy)).

```yaml
services:
  vllm:
    image: vllm/vllm-openai:v0.33.0
    devices:
      - nvidia.com/gpu=all
    ipc: host
    ports:
      - "127.0.0.1:8000:8000"
    environment:
      HF_TOKEN: ${HF_TOKEN:-}
    volumes:
      - ${HOME}/.cache/huggingface:/root/.cache/huggingface
      - vllm-cache:/root/.cache/vllm
    mem_limit: 32g
    memswap_limit: 32g
    command: ["<org>/<model>"]
volumes:
  vllm-cache: {}
```

The `ipc: host`, HF cache path and `/root/.cache/vllm` compile-cache volume follow the [vLLM Docker docs at v0.33.0](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/deployment/docker.md); [vLLM](../vllm/README.md) explains them.

## Network exposure

Docker's own docs say that on Ubuntu with ufw, ports published by Docker bypass the firewall rules ([Docker install docs](https://docs.docker.com/engine/install/ubuntu/)).
Docker routes container traffic in the `nat` table, so packets are diverted before they reach the `INPUT` and `OUTPUT` chains that ufw uses ([Docker and ufw](https://docs.docker.com/engine/network/packet-filtering-firewalls/#docker-and-ufw)).
ufw is active on this laptop, so a plain `-p 8000:8000` would expose an unauthenticated model server to every network the laptop joins, and `ufw status` would not show it.

Publish every port on the loopback address only: `-p 127.0.0.1:8000:8000` for vLLM and `-p 127.0.0.1:8080:8080` for llama.cpp.
With a localhost address in the publish flag, only the Docker host can reach the port ([port publishing](https://docs.docker.com/engine/network/port-publishing/)).
Before Docker 28.0.0, hosts on the same L2 segment could still reach ports published to localhost ([moby/moby#45610](https://github.com/moby/moby/issues/45610)); Docker 29.9.3 is not affected ([port publishing](https://docs.docker.com/engine/network/port-publishing/)).
Inside the container the server must still listen on all interfaces: the llama.cpp CUDA image sets `LLAMA_ARG_HOST=0.0.0.0` and `LLAMA_ARG_PORT=8080` ([cuda.Dockerfile at b12418](https://github.com/ggml-org/llama.cpp/blob/b12418/.devops/cuda.Dockerfile)), and vLLM v0.33.0 defaults to `--host` unset and `--port 8000` ([cli_args.py at v0.33.0](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/entrypoints/common/cli_args.py)), which its official container example reaches through a published port ([vLLM Docker docs](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/deployment/docker.md)).
Do not pass `--host 127.0.0.1` to either server inside the container, or the published port stops working (inferred from Docker's bridge networking).
The [OpenCode configuration](opencode.md) points at `http://127.0.0.1:8000/v1` and `http://127.0.0.1:8080/v1` to match.

Check what is actually listening after a server starts:

```bash
ss -ltn | grep -E ':(8000|8080) '         # expect 127.0.0.1:8000 or 127.0.0.1:8080, never 0.0.0.0 or [::]
docker port <container>                   # expect 127.0.0.1 in the mapping
```

As a safety net, Docker can default unqualified port bindings to loopback: `"ip": "127.0.0.1"` in `/etc/docker/daemon.json` covers the default bridge, and `default-network-opts` with `com.docker.network.bridge.host_binding_ipv4` covers user-defined bridges such as Compose networks ([port publishing](https://docs.docker.com/engine/network/port-publishing/)).
Both need a Docker restart, and `daemon.json` is also edited by `nvidia-ctk`, so merge the keys rather than overwrite the file.
This is optional; the explicit `127.0.0.1:` prefix in every command is the primary control.

An API key is optional on a loopback-only server, but it limits what another local process or browser page can do.
vLLM v0.33.0 accepts `--api-key` or the `VLLM_API_KEY` environment variable, and the CLI flag wins ([register.py at v0.33.0](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/entrypoints/serve/middleware/register.py)).
llama.cpp accepts `--api-key` (environment `LLAMA_API_KEY`) or `--api-key-file`, and `/health` stays public ([llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/b12418/tools/server/README.md)).
Pass the key from the environment (`-e VLLM_API_KEY`) rather than on the command line, and set the same value as `apiKey` in [OpenCode](opencode.md).

## Container memory caps

The `heavy` wrapper (`~/.local/bin/heavy`) runs a command with `systemd-run --user --scope --slice=heavy.slice`, and `heavy.slice` caps all such runs together at `MemoryMax=32G`, `MemorySwapMax=0` and `CPUQuota=1200%` (read from the files on this host).
It does not cap containers.
With the systemd cgroup driver the default cgroup parent is `system.slice` ([dockerd reference](https://github.com/moby/moby/blob/master/docs/reference/dockerd.md#default-cgroup-parent)), and a container's cgroup is `/sys/fs/cgroup/system.slice/docker-<id>.scope` ([runtime metrics](https://docs.docker.com/engine/containers/runmetrics/)).
The root `dockerd` creates that cgroup, so `heavy docker run ...` caps only the Docker CLI process, not the model server.

Give every model container its own hard cap with `--memory` and an equal `--memory-swap`.
Equal values mean no swap; if `--memory-swap` is unset the container may use as much swap again as `--memory`, and `-1` means unlimited swap ([resource constraints](https://docs.docker.com/engine/containers/resource_constraints/)).
Without `--memory`, a container that runs away can trigger the host OOM killer against desktop processes ([resource constraints](https://docs.docker.com/engine/containers/resource_constraints/)).
In Compose, use `mem_limit` and `memswap_limit` with equal values ([Compose services reference](https://docs.docker.com/reference/compose-file/services/)).

Sizing guidance (estimates, except where a row says measured):

| Workload | Suggested cap | Why |
|---|---|---|
| vLLM, model fully on the GPU | `--memory=32g --memory-swap=32g` | Room for loading, compilation and the Python process; the [vLLM guide](../vllm/README.md) gives the per-model numbers |
| vLLM, first start of an NVFP4 W4A4 model | Same cap plus `-e MAX_JOBS=2` | The FlashInfer JIT build on first start can use about 52 GiB of host RAM ([vllm#46268](https://github.com/vllm-project/vllm/issues/46268), open), which a 32g cap would kill with exit 137; the open fix [vllm#66864](https://github.com/vllm-project/vllm/pull/66864) budgets about 8 GiB per compiler job, so 2 jobs fit in 32g (estimate, not tested here) |
| llama.cpp, model fully on the GPU | `--memory=16g --memory-swap=16g` | Server process plus the host prompt cache (`--cache-ram`, default 8192 MiB), as in the [llama.cpp models](../llamacpp/models.md) recipes (estimate) |
| llama.cpp with MoE experts in RAM | RAM-resident expert size plus the host prompt cache plus the context checkpoints plus about 2-3 GiB | 20g for Qwen3.6-35B-A3B (10.5 GiB of pinned experts plus the default 8 GiB prompt cache; the container peaked at 14.9 GiB) and 35g for Qwen3-Coder-Next (26.4 GiB of pinned experts with `--cache-ram 4096` and `-ctxcp 8`; it peaked at 31.9 GiB), both measured on 2026-11-16; see [llama.cpp models](../llamacpp/models.md) |
| `hf download` helper container | `--memory=8g --memory-swap=8g` | Example value, not measured |

Run one model server at a time.
The GPU has 16 GiB, and about 40 GiB of RAM is free once the desktop is up, so a 35g offload container and a full `heavy` gate (up to 32G) cannot both fit (estimate).
Do not run a full test or benchmark gate through `heavy` while a large offload model is loaded.
Check the cap that a running container actually got:

```bash
docker stats --no-stream <container>
cat /sys/fs/cgroup/system.slice/docker-$(docker inspect -f '{{.Id}}' <container>).scope/memory.max
```

A shared cap for all LLM containers is possible with a system slice and `--cgroup-parent`, because under the systemd driver the parent must be a slice name ([dockerd reference](https://github.com/moby/moby/blob/master/docs/reference/dockerd.md#default-cgroup-parent)).
For example, `/etc/systemd/system/llm.slice` with `MemoryMax=32G` and `docker run --cgroup-parent=llm.slice ...` (inferred from the dockerd and systemd slice semantics, not tested).
Rootful Docker cannot place containers in the user's `heavy.slice`, which lives under the user manager (inferred).
Creating the slice needs `systemctl daemon-reload`, which is another reason to use CDI.

## Laptop power and restart policy

On Ampere or newer notebooks, runtime D3 (RTD3) power management is on by default (`NVreg_DynamicPowerManagement=0x03`, fine-grained), so the dGPU powers down when idle ([driver 595.99.02 README, RTD3](https://download.nvidia.com/XFree86/Linux-x86_64/595.99.02/README/dynamicpowermanagement.html)).
The same chapter says the GPU "will remain in an active state if a CUDA application is running", and a loaded vLLM or llama.cpp server is such an application.
A container with `--restart always` (or `unless-stopped`, if it was not stopped by hand) starts again with the Docker daemon after every reboot and keeps the dGPU awake, drawing power and producing heat while nothing uses it ([restart policies](https://docs.docker.com/engine/containers/start-containers-automatically/)).
So no command in this knowledge base sets a restart policy: Docker's default is `no` ([restart policies](https://docs.docker.com/engine/containers/start-containers-automatically/)), and servers are started when needed and stopped afterwards with `docker stop <container>`.
Keep `nvidia-persistenced` out of UVM persistence mode, because that mode disables RTD3 ([RTD3 README](https://download.nvidia.com/XFree86/Linux-x86_64/595.99.02/README/dynamicpowermanagement.html)).
The current RTD3 state is in `/proc/driver/nvidia/gpus/<PCI-BUS-ID>/power` ([RTD3 README](https://download.nvidia.com/XFree86/Linux-x86_64/595.99.02/README/dynamicpowermanagement.html)).

PRIME render offload variables (`__NV_PRIME_RENDER_OFFLOAD`, `__GLX_VENDOR_LIBRARY_NAME`) apply to GLX, Vulkan and EGL rendering ([PRIME render offload README](https://download.nvidia.com/XFree86/Linux-x86_64/595.99.02/README/primerenderoffload.html)); headless CUDA containers do not need them (inferred, no NVIDIA page states it outright).
Blackwell is supported only by the open kernel modules, which matches the installed `nvidia-driver-595-open`; NVIDIA notes slower GPU init and longer RTD3 entry and exit with the open modules ([open kernel modules README](https://download.nvidia.com/XFree86/Linux-x86_64/595.99.02/README/kernel_open.html)).

The observed power limit is 140 W, with an 80 W default and a 175 W maximum, which is above NVIDIA's 80-150 W subsystem range ([NVIDIA laptop compare](https://www.nvidia.com/en-us/geforce/laptops/compare/)); the extra headroom is consistent with OEM configuration and Dynamic Boost (inferred, [Dynamic Boost README](https://download.nvidia.com/XFree86/Linux-x86_64/595.99.02/README/dynamicboost.html)).
Whether this laptop runs `nvidia-powerd` was not checked.
Desktop RTX 5080 numbers come from a 360 W card with 960 GB/s, so they do not transfer to this GPU; see [community benchmarks](community-benchmarks.md).
Record power state with every benchmark run, because the result depends on it:

```bash
nvidia-smi --query-gpu=power.draw,power.limit,pstate,temperature.gpu,clocks.sm,clocks.mem --format=csv
```

## Model cache

Keep one Hugging Face cache at `~/.cache/huggingface` on `/home`, which has 562 GB free, and bind-mount it into every container at `/root/.cache/huggingface`.
The HF hub default cache is `~/.cache/huggingface/hub`, with content-addressed files in `blobs/` and per-revision symlinks in `snapshots/` ([HF cache guide](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)).
vLLM's official example mounts the cache at that path ([vLLM Docker docs](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/deployment/docker.md)).
llama.cpp's `-hf` download uses the same hub layout (search order `LLAMA_CACHE`, `HF_HUB_CACHE`, `HF_HOME/hub`, then `~/.cache/huggingface/hub`), and its CUDA image runs as root, so the same mount works ([hf-cache.cpp at b12418](https://github.com/ggml-org/llama.cpp/blob/b12418/common/hf-cache.cpp)).
One cache means a model downloaded once is visible to both runtimes; see [vLLM](../vllm/README.md) and [llama.cpp](../llamacpp/README.md) for the run commands.

Create the directory as your user before the first container starts.
If it does not exist, `docker run -v` creates it ([bind mounts](https://docs.docker.com/engine/storage/bind-mounts/)), and the rootful daemon creates it owned by root (inferred).

```bash
mkdir -p ~/.cache/huggingface
```

Both images run as root, so files they download into the cache are owned by root on the host, and deleting them needs `sudo` or a container (standard bind-mount behaviour with rootful Docker).

### Download through a container

The host has no `hf` CLI.
The vLLM v0.33.0 image installs `huggingface_hub>=1.31.0` ([requirements/common.txt at v0.33.0](https://github.com/vllm-project/vllm/blob/v0.33.0/requirements/common.txt)), which provides the `hf` command, so download through that image into the shared cache.
The user's earlier DGX Spark deployment used the same pattern with `hf download` inside a vLLM container; it has not been run on this laptop yet.
A download does not need the GPU.

```bash
docker run --rm \
  --memory=8g --memory-swap=8g \
  -e HF_TOKEN \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  --entrypoint hf \
  vllm/vllm-openai:v0.33.0 \
  download <org>/<repo>

# GGUF repos: fetch only the quant you need
docker run --rm --memory=8g --memory-swap=8g -e HF_TOKEN \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  --entrypoint hf vllm/vllm-openai:v0.33.0 \
  download <org>/<repo>-GGUF --include "*Q4_K_M*"
```

`hf download` takes `--include` and `--exclude` patterns ([HF CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli)).
The exact repo ids and file patterns per model are in [vLLM models](../vllm/models.md) and [llama.cpp models](../llamacpp/models.md).

### HF_TOKEN

A token is needed for gated repos and avoids anonymous rate limits.
Keep it in your shell environment (for example from a file that is not in any repo) and pass it as `-e HF_TOKEN` with no value: Docker then copies the value from the local environment, and if the variable is not exported it stays unset in the container ([docker run --env](https://docs.docker.com/reference/cli/docker/container/run/#env)).
That keeps the token out of shell history and out of Compose files; in Compose use `HF_TOKEN: ${HF_TOKEN:-}`.
Never commit the token or paste it into a command line.

### Incomplete downloads

`huggingface_hub` writes each file to `<blob>.incomplete` while downloading and renames it when done ([file_download.py](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/file_download.py)).
On the user's earlier DGX Spark deployment, interrupted transfers left `*.incomplete` files that were never cleaned up and silently used disk, and a truncated shard did not fail the download but failed later at model load with a tensor-shape or safetensors-header error.
Check after every download and before blaming a runtime for a load error:

```bash
find ~/.cache/huggingface/hub -name '*.incomplete'     # must print nothing
du -sh ~/.cache/huggingface/hub/models--*              # per-model size
df -h /home /
```

A size check on `snapshots/` alone is misleading because it holds symlinks ([HF cache guide](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)).
Re-run the same `hf download` command to resume, then delete leftover `*.incomplete` files with `sudo rm` if they remain.
Current `huggingface_hub` documents `hf cache ls`, `hf cache verify <repo>`, `hf cache rm` and `hf cache prune` ([HF cache guide](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)); whether the version inside the vLLM image has all of them is unknown, so check `hf cache --help` in the image first.

### Disk budget

| Item | Size | Location |
|---|---|---|
| llama.cpp image `ghcr.io/ggml-org/llama.cpp:server-cuda-b12407` | 7.05 GB on disk as `docker image ls` reports it (measured on the MSI Raider 18 Max HX, Docker 29.9.3, 2026-11-18) | `/var/lib/docker` on `/` |
| vLLM image `vllm/vllm-openai:v0.33.0` | 35.4 GB on disk as `docker image ls` reports it (measured on the MSI Raider 18 Max HX, Docker 29.9.3, 2026-11-18); 9.74 GB compressed; `v0.33.0-x86_64-zstd` is 7.70 GB compressed ([Docker Hub](https://hub.docker.com/r/vllm/vllm-openai/tags)) | `/var/lib/docker` on `/` (405 GB free) |
| SGLang image `lmsysorg/sglang:v0.5.23-runtime` | 49.7 GB on disk as `docker image ls` reports it (measured on the MSI Raider 18 Max HX, Docker 29.9.3, 2026-11-18) | `/var/lib/docker` on `/` |
| vLLM compile cache volume | Unknown, grows per model and config | `/var/lib/docker` on `/` |
| All ten ranked picks, one file each, plus gpt-oss-20b in both GGUF and safetensors | About 174 GiB (estimate, sum of the sizes in [model selection](model-selection.md) plus the 12.82 GiB gpt-oss-20b safetensors) | `~/.cache/huggingface` on `/home` (562 GB free) |

Disk is not the constraint here; RAM and VRAM are.

## Commands that need sudo

The user has to run these, because the agents working on this repository do not install software:

- Done on 2026-11-15: `sudo apt-get update` and `sudo apt-get install` for the toolkit packages, plus the `gpg --dearmor` and `tee` steps that write the keyring and the apt source list ([install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)).
- Done on 2026-11-15: `sudo nvidia-ctk runtime configure --runtime=docker`, then `sudo systemctl restart docker`.
- `sudo systemctl status nvidia-cdi-refresh.path nvidia-cdi-refresh.service`, and if the spec is missing, `sudo systemctl restart nvidia-cdi-refresh.service` or `sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml`.
- Optional: edit `/etc/docker/daemon.json` for the loopback default binding, then `sudo systemctl restart docker`.
- Optional: create `/etc/systemd/system/llm.slice` and run `sudo systemctl daemon-reload` for a shared container cap.
- Optional: `sudo dmidecode -t memory` to read the configured DIMM speed, which is still unverified.
- Cleanup: `sudo rm` of root-owned `*.incomplete` files or model directories in `~/.cache/huggingface`.

Everything else on this page, including `docker run`, `docker compose`, `nvidia-ctk cdi list` and the verification commands, runs as the user.

## Sources

- NVIDIA Container Toolkit: [install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), [supported platforms](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/supported-platforms.html), [CDI support](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html), [troubleshooting](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html), [sample workload](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html), [release notes](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/release-notes.html), [specialized configurations](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html), [GitHub releases](https://github.com/NVIDIA/nvidia-container-toolkit/releases), [v1.20.2 CHANGELOG](https://github.com/NVIDIA/nvidia-container-toolkit/blob/v1.20.2/CHANGELOG.md), [apt Packages index](https://nvidia.github.io/libnvidia-container/stable/deb/amd64/Packages).
- CUDA compatibility: [CUDA toolkit release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html), [minor version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html), [forward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html), [nvidia/cuda tags](https://hub.docker.com/r/nvidia/cuda/tags).
- NVIDIA driver 595.99.02 README: [RTD3](https://download.nvidia.com/XFree86/Linux-x86_64/595.99.02/README/dynamicpowermanagement.html), [PRIME render offload](https://download.nvidia.com/XFree86/Linux-x86_64/595.99.02/README/primerenderoffload.html), [open kernel modules](https://download.nvidia.com/XFree86/Linux-x86_64/595.99.02/README/kernel_open.html), [Dynamic Boost](https://download.nvidia.com/XFree86/Linux-x86_64/595.99.02/README/dynamicboost.html); [NVIDIA laptop GPU compare](https://www.nvidia.com/en-us/geforce/laptops/compare/).
- Docker: [install on Ubuntu](https://docs.docker.com/engine/install/ubuntu/), [packet filtering and firewalls](https://docs.docker.com/engine/network/packet-filtering-firewalls/), [port publishing](https://docs.docker.com/engine/network/port-publishing/), [moby/moby#45610](https://github.com/moby/moby/issues/45610), [resource constraints](https://docs.docker.com/engine/containers/resource_constraints/), [restart policies](https://docs.docker.com/engine/containers/start-containers-automatically/), [runtime metrics](https://docs.docker.com/engine/containers/runmetrics/), [dockerd reference](https://github.com/moby/moby/blob/master/docs/reference/dockerd.md), [docker run](https://docs.docker.com/reference/cli/docker/container/run/), [bind mounts](https://docs.docker.com/engine/storage/bind-mounts/), [Compose services reference](https://docs.docker.com/reference/compose-file/services/), [Compose GPU support](https://docs.docker.com/compose/how-tos/gpu-support/).
- vLLM v0.33.0: [Docker deployment docs](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/deployment/docker.md), [Dockerfile](https://github.com/vllm-project/vllm/blob/v0.33.0/docker/Dockerfile), [requirements/common.txt](https://github.com/vllm-project/vllm/blob/v0.33.0/requirements/common.txt), [API key middleware](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/entrypoints/serve/middleware/register.py), [vllm#46268](https://github.com/vllm-project/vllm/issues/46268), [vllm#66864](https://github.com/vllm-project/vllm/pull/66864), [Docker Hub tags](https://hub.docker.com/r/vllm/vllm-openai/tags).
- llama.cpp b12418: [server README](https://github.com/ggml-org/llama.cpp/blob/b12418/tools/server/README.md), [cuda.Dockerfile](https://github.com/ggml-org/llama.cpp/blob/b12418/.devops/cuda.Dockerfile), [hf-cache.cpp](https://github.com/ggml-org/llama.cpp/blob/b12418/common/hf-cache.cpp).
- Hugging Face: [cache guide](https://huggingface.co/docs/huggingface_hub/guides/manage-cache), [CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli), [file_download.py](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/file_download.py).
- Ubuntu archive: [nvidia-container-toolkit on Launchpad](https://launchpad.net/ubuntu/+source/nvidia-container-toolkit).
- Full catalog with evidence classes: [sources](sources.md).

## Maintenance

Before relying on this page, re-check the toolkit version pin in the install guide (1.20.2 may get a release note, or a newer version may appear) and run `apt-cache policy nvidia-container-toolkit`.
After any driver upgrade, confirm the R-branch mapping in the CUDA release notes, confirm `nvidia-ctk cdi list` still shows the GPU, and re-run the CUDA smoke test.
After a Docker upgrade, re-run `ss -ltn` against a running server to confirm the loopback-only binding and re-read the Docker and ufw section.
Re-check that the `hf` CLI and its `cache` subcommands exist in whichever vLLM image tag is pinned in [vLLM](../vllm/README.md).
Replace the memory-cap estimates with measured peak RSS (`docker stats`) once the first models have run on this laptop.
