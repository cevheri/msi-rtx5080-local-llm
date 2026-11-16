"""What Docker has: whether it answers, the images present, and the containers by name or by llmctl's label."""

import json
from dataclasses import dataclass

from launcher.command import LABEL_PREFIX
from launcher.errors import PreflightError

MANAGED = LABEL_PREFIX + "managed"


@dataclass(frozen=True)
class Container:
    name: str
    id: str
    state: str
    labels: dict

    @property
    def is_managed(self) -> bool:
        return self.labels.get(MANAGED) == "true"


def _last_line(result) -> str:
    lines = (result.stderr or result.stdout).strip().splitlines()
    return lines[-1] if lines else "no output"


def docker_ok(host) -> None:
    result = host.run(["docker", "info", "--format", "{{.ServerVersion}}"])
    if result.returncode == 0:
        return
    if result.returncode == 127:
        raise PreflightError("docker was not found; install Docker Engine (docs/host-setup.md)")
    if "permission denied" in result.stderr.lower():
        raise PreflightError("Docker refused the connection (permission denied on its socket); add your user to "
                             "the docker group with `sudo usermod -aG docker $USER` and log in again")
    raise PreflightError(f"Docker is not answering: {_last_line(result)}; start it with `sudo systemctl start docker`")


def image_present(host, image: str) -> bool:
    return host.run(["docker", "image", "inspect", "--format", "{{.Id}}", image]).returncode == 0


def by_name(host, name: str) -> Container | None:
    result = host.run(["docker", "container", "inspect", "--format",
                       "{{.Id}}|{{.State.Status}}|{{json .Config.Labels}}", name])
    if result.returncode != 0:
        return None
    container_id, state, labels = result.stdout.strip().split("|", 2)
    return Container(name, container_id, state, json.loads(labels) or {})


def managed(host) -> list[Container]:
    result = host.run(["docker", "ps", "-a", "--filter", f"label={MANAGED}=true",
                       "--format", "{{.Names}}|{{.ID}}|{{.State}}|{{.Labels}}"])
    if result.returncode != 0:
        docker_ok(host)  # raises the specific error: missing, permission denied, not answering
        raise PreflightError(f"docker ps failed: {_last_line(result)}")
    containers = []
    for line in result.stdout.strip().splitlines():
        name, container_id, state, labels = line.split("|", 3)
        pairs = dict(item.split("=", 1) for item in labels.split(",") if "=" in item)
        containers.append(Container(name, container_id, state, pairs))
    return containers


def state(host, name: str) -> str:
    result = host.run(["docker", "inspect", "--format", "{{.State.Status}}", name])
    return result.stdout.strip() if result.returncode == 0 else "missing"


def logs_tail(host, name: str, lines: int) -> str:
    result = host.run(["docker", "logs", "--tail", str(lines), name])
    return (result.stdout + result.stderr).rstrip()


def stop(host, name: str) -> None:
    result = host.run(["docker", "stop", "-t", "60", name])
    if result.returncode != 0:
        raise PreflightError(f"could not stop {name}: {_last_line(result)}")


def remove(host, name: str) -> None:
    result = host.run(["docker", "rm", name])
    if result.returncode != 0:
        raise PreflightError(f"could not remove {name}: {_last_line(result)}")
