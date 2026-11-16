"""GPU memory as nvidia-smi reports it."""

from launcher.errors import PreflightError
from launcher.model import GpuInfo

QUERY = ["nvidia-smi", "--query-gpu=memory.total,memory.reserved,memory.used,memory.free",
         "--format=csv,noheader,nounits"]
APPS = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"]


def gpu_info(host) -> GpuInfo:
    result = host.run(QUERY)
    if result.returncode == 127:
        raise PreflightError("nvidia-smi was not found; install the NVIDIA driver for this GPU first")
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        raise PreflightError(f"nvidia-smi failed: {detail[-1] if detail else 'no output'}; is the NVIDIA driver loaded?")
    try:
        total, reserved, used, free = (int(part.strip()) for part in result.stdout.strip().splitlines()[0].split(","))
    except (ValueError, IndexError):
        raise PreflightError(f"could not read nvidia-smi's memory figures: {result.stdout.strip()!r}") from None
    return GpuInfo(total, reserved, used, free)


def gpu_apps(host) -> list[tuple[int, str, int]]:
    result = host.run(APPS)
    apps = []
    for line in result.stdout.strip().splitlines() if result.returncode == 0 else []:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3 and parts[0].isdigit() and parts[2].isdigit():
            apps.append((int(parts[0]), parts[1].rsplit("/", 1)[-1], int(parts[2])))
    return apps
