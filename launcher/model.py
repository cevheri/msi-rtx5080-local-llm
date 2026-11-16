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
