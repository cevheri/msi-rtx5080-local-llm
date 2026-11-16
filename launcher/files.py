"""Paths from the environment mapping, plain JSON, atomic writes and one-time backups.

Every path comes from the mapping a caller passes in, never from the process environment.
"""

import json
import math
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from launcher.errors import ClientError, UsageError

TMP_SUFFIX = ".llmctl-tmp"
BACKUP_SUFFIX = ".llmctl-backup"
_VAR = re.compile(r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))")


def home_dir(env: Mapping[str, str]) -> Path:
    home = env.get("HOME", "")
    if not home:
        raise UsageError("HOME is not set, so llmctl cannot find the client configs")
    return Path(home)


def env_path(env: Mapping[str, str], value: str) -> Path:
    """Expand a leading `~` with the mapping's HOME and `$NAME` or `${NAME}` from the mapping."""
    if value == "~" or value.startswith("~/"):
        value = str(home_dir(env)) + value[1:]
    elif value.startswith("~"):
        raise UsageError(f"{value}: llmctl expands only ~ and ~/, not ~user")

    def expand(match: re.Match) -> str:
        name = match.group("braced") or match.group("plain")
        return env[name] if name in env else match.group(0)

    return Path(_VAR.sub(expand, value))


def _refuse_constant(name: str) -> object:
    raise ValueError(f"{name} is not plain JSON")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"{text} is out of range for a JSON number")
    return value


def parse_json(text: str) -> object:
    return json.loads(text, parse_constant=_refuse_constant, parse_float=_finite_float)


def load_json_object(path: Path, client: str) -> tuple[str | None, dict]:
    try:
        with open(path, encoding="utf-8", newline="") as handle:
            text = handle.read()
    except FileNotFoundError:
        return None, {}
    except (OSError, UnicodeDecodeError) as exc:
        raise ClientError(f"cannot read {path}: {exc}") from exc
    try:
        data = parse_json(text)
    except ValueError as exc:
        raise ClientError(
            f"{path} is not plain JSON ({exc}); llmctl edits only plain JSON: "
            f"remove comments and trailing commas, or leave {client} out with LLMCTL_CLIENTS"
        ) from exc
    if not isinstance(data, dict):
        raise ClientError(f"{path} is not a JSON object; fix it, or leave {client} out with LLMCTL_CLIENTS")
    try:
        json.dumps(data, ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ClientError(f"{path} holds an escaped lone surrogate (\\ud800 to \\udfff) that llmctl cannot write "
                          f"back as UTF-8; fix it, or leave {client} out with LLMCTL_CLIENTS") from exc
    return text, data


@dataclass(frozen=True)
class FileChange:
    """One planned file write: old_text None means a new file; preview is the dry-run line."""

    path: Path
    old_text: str | None
    new_text: str
    summary: str
    preview: str
    mode: int = 0o600
    backup: bool = False
    dirs: tuple[Path, ...] = ()


def backup_path(path: Path) -> Path:
    return path.with_name(path.name + BACKUP_SUFFIX)


def _fsync_dir(directory: str) -> None:
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_atomic(path: Path, text: str, *, new_mode: int = 0o600) -> None:
    """Replace the file (a symlink's target) through a private temporary file beside it."""
    real = os.path.realpath(path)
    directory, name = os.path.split(real)
    tmp = os.path.join(directory, name + TMP_SUFFIX)
    try:
        mode = os.stat(real).st_mode & 0o7777
    except FileNotFoundError:
        mode = new_mode
    if os.path.lexists(tmp):
        os.remove(tmp)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        try:
            _write_all(fd, text.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        os.chmod(tmp, mode)
        os.replace(tmp, real)
    except BaseException:
        if os.path.lexists(tmp):
            os.remove(tmp)
        raise
    _fsync_dir(directory)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def backup_once(path: Path, text: str) -> Path | None:
    """Write text to <path>.llmctl-backup beside the path as given, unless a backup exists."""
    backup = backup_path(path)
    if os.path.lexists(backup):
        return None
    data = text.encode("utf-8")
    mode = os.stat(path).st_mode & 0o7777
    fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        _write_all(fd, data)
        os.fchmod(fd, mode)
        os.fsync(fd)
    finally:
        os.close(fd)
    return backup


def make_dirs(dirs: Iterable[Path]) -> None:
    for directory in dirs:
        if not os.path.isdir(directory):
            os.mkdir(directory, 0o700)
            os.chmod(directory, 0o700)
