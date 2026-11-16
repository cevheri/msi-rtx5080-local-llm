# GPU in plain Docker without the NVIDIA Container Toolkit

Run on 2026-11-15 on this laptop (RTX 5080 Laptop GPU, driver 595.99.02, Docker 29.9.3 with the default runc runtime, no NVIDIA Container Toolkit installed).

## Without any GPU plumbing

```text
$ docker run --rm --gpus all hello-world
docker: Error response from daemon: failed to discover GPU vendor from CDI: no known GPU vendor found

$ docker run --rm --device nvidia.com/gpu=all hello-world
docker: Error response from daemon: CDI device injection failed: unresolvable CDI devices nvidia.com/gpu=all
```

## With device nodes and driver libraries passed in by hand

The probe script is [cuda_probe.py](cuda_probe.py); it calls the CUDA driver API through ctypes.

```text
$ L=/usr/lib/x86_64-linux-gnu
$ docker run --rm \
    --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
    -v $L/libcuda.so.1:$L/libcuda.so.1:ro \
    -v $L/libnvidia-ml.so.1:$L/libnvidia-ml.so.1:ro \
    -v $L/libnvidia-ptxjitcompiler.so.1:$L/libnvidia-ptxjitcompiler.so.1:ro \
    -v $L/libnvidia-nvvm.so.4:$L/libnvidia-nvvm.so.4:ro \
    -v $L/libnvidia-gpucomp.so.595.99.02:$L/libnvidia-gpucomp.so.595.99.02:ro \
    -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro \
    -v <scratch>/cuda_probe.py:/probe.py:ro \
    python:3.12-slim sh -c 'nvidia-smi -L; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; python3 /probe.py'
GPU 0: NVIDIA GeForce RTX 5080 Laptop GPU (UUID: <redacted>)
NVIDIA GeForce RTX 5080 Laptop GPU, 595.99.02, 16303 MiB
cuInit: OK
driver CUDA version: 13020
device count: 1
device: NVIDIA GeForce RTX 5080 Laptop GPU
compute capability: 12.0
total memory MiB: 15841
cuDevicePrimaryCtxRetain: OK
cuCtxSetCurrent: OK
cuMemAlloc 1 GiB: OK
cuMemFree: OK
exit=0
```

The `python:3.12-slim` image was removed after the test.
vLLM itself was not run this way; see [docs/host-setup.md](../../../docs/host-setup.md) for the extra steps it would need and why this path is not recommended for daily use.
