"""Data shared by every module. Runtime-specific model details live in the adapters."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Source:
    kind: str  # "shipped" or "override"
    path: Path
    sha256: str


@dataclass(frozen=True)
class Evidence:
    kind: str  # "measured" or "estimate"
    summary: str
    date: str | None
    hardware: str | None
    record: str | None
    doc: str | None
    decode_tok_s: float | None = None  # measured decode speed, for list


@dataclass(frozen=True)
class RuntimeConfig:
    name: str
    image: str
    container_name: str
    host_port: int
    ready_path: str
    start_timeout_s: int
    env: tuple[tuple[str, str], ...]
    volumes: tuple[str, ...]
    opencode_provider: str
    served_name: str | None
    source: Source


@dataclass(frozen=True)
class ClientMeta:
    """A recipe's [clients] table: what the clients send and how they choose thinking."""

    output: int
    thinking: str  # "toggle", "effort", "always" or "none"
    default: str | None  # "off" or "on" for toggle, else None
    efforts: tuple[str, ...]  # for effort, in validate.EFFORTS order, else ()
    medium_kwargs: tuple[tuple[str, object], ...]  # toggle only, in file order


@dataclass(frozen=True)
class Recipe:
    runtime: str
    id: str
    name: str
    aliases: tuple[str, ...]
    doc: str
    opencode: bool
    chat_only: bool
    memory_gib: int
    context: int
    args: tuple[str, ...]
    needs_gpu_mib: int | None
    needs_ram_gib: float | None
    model: object
    evidence: Evidence
    source: Source
    clients: ClientMeta


@dataclass(frozen=True)
class Target:
    """What a client entry describes: the recipe, its runtime, the served model name and the runtime's label."""

    recipe: Recipe
    runtime: RuntimeConfig
    served: str
    label: str  # "llama.cpp" or "vLLM"


@dataclass(frozen=True)
class ContainerSpec:
    user: str | None
    env: tuple[tuple[str, str], ...]
    mounts: tuple[str, ...]
    extra_flags: tuple[str, ...]
    image: str
    args: tuple[str, ...]
    container_port: int


@dataclass(frozen=True)
class Problem:
    runtime: str
    path: Path
    recipe_id: str | None
    message: str


@dataclass(frozen=True)
class HostEnv:
    hf_home: Path
    uid: int
    gid: int


@dataclass(frozen=True)
class GpuInfo:
    total_mib: int
    reserved_mib: int
    used_mib: int
    free_mib: int
