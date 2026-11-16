"""The checks before a model starts, in order, each refusing with what was found and what to do."""

import re
from pathlib import Path

from launcher.docker_state import Container, by_name, docker_ok, image_present, managed
from launcher.errors import PreflightError
from launcher.gpu import gpu_apps, gpu_info
from launcher.model import Recipe, RuntimeConfig


def pull_hint(recipe: Recipe) -> str:
    runtime = "" if recipe.runtime == "llamacpp" else f" --runtime {recipe.runtime}"
    return f"./llmctl pull {recipe.id}{runtime}"


def check_static(host, recipe: Recipe, runtime: RuntimeConfig, adapter, hf_home: Path) -> None:
    docker_ok(host)
    if not any("nvidia.com/gpu" in spec for spec in host.cdi_specs()):
        raise PreflightError("no CDI spec for nvidia.com/gpu in /etc/cdi or /var/run/cdi; install the NVIDIA "
                             "Container Toolkit and generate the spec (docs/host-setup.md)")
    if not image_present(host, runtime.image):
        raise PreflightError(f"the image {runtime.image} is not on this machine; run {pull_hint(recipe)}")
    problems = adapter.file_problems(recipe, hf_home)
    if problems:
        raise PreflightError(f"{recipe.id}: the model files are not ready in {hf_home}: {'; '.join(problems)}; "
                             f"run {pull_hint(recipe)}")


def check_containers(host, runtime: RuntimeConfig) -> list[Container]:
    """Refuse a foreign container that holds the name; return the managed servers that are running."""
    existing = by_name(host, runtime.container_name)
    if existing is not None and not existing.is_managed:
        note = ", and it is running, so removing it stops that server" if existing.state == "running" else ""
        raise PreflightError(f"a container named '{runtime.container_name}' exists ({existing.state}) and was not "
                             f"started by llmctl{note}; remove it yourself with `docker rm -f {runtime.container_name}`")
    return [container for container in managed(host) if container.state == "running"]


def check_port(host, runtime: RuntimeConfig) -> None:
    if host.port_open(runtime.host_port):
        raise PreflightError(f"port 127.0.0.1:{runtime.host_port} is already in use by something llmctl did not "
                             "start; stop it first")


def check_resources(host, recipe: Recipe, runtime: RuntimeConfig, adapter) -> None:
    gpu = gpu_info(host)
    need = adapter.gpu_need_mib(recipe, runtime, gpu)
    if gpu.free_mib < need:
        apps = ", ".join(f"{name} (pid {pid}, {mib} MiB)" for pid, name, mib in gpu_apps(host))
        users = apps or "no compute processes; the desktop and other graphics programs hold the rest (see nvidia-smi)"
        raise PreflightError(f"{recipe.id} needs {need} MiB of free GPU memory and {gpu.free_mib} MiB is free; "
                             f"GPU memory is used by: {users}; close them, or {adapter.gpu_advice(recipe)} "
                             "(docs/llmctl.md)")
    match = re.search(r"^MemAvailable:\s+(\d+)\s+kB", host.meminfo(), re.M)
    if match is None:
        raise PreflightError("could not read MemAvailable from /proc/meminfo")
    available = int(match.group(1)) / 2**20
    if available < recipe.memory_gib:
        raise PreflightError(f"{recipe.id} may use up to {recipe.memory_gib} GiB of RAM (its container cap) and "
                             f"{available:.1f} GiB is available; close programs first")
