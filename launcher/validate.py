"""Reading recipe files and checking their values. Generic: no runtime's settings live here."""

import difflib
import hashlib
import tomllib
from collections.abc import Iterable
from pathlib import Path

from launcher.errors import RecipeError
from launcher.model import Evidence, Source

_MISSING = object()


def read_toml(path: Path, kind: str) -> tuple[dict, Source]:
    """Parse a TOML file and return its data with the SHA-256 of the bytes as read."""
    raw = path.read_bytes()
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise RecipeError(f"{path}: not valid TOML: {exc}") from None
    return data, Source(kind, path, hashlib.sha256(raw).hexdigest())


def check_keys(data: dict, allowed: Iterable[str], where: str) -> None:
    allowed = set(allowed)
    for key in data:
        if key in allowed:
            continue
        message = f"{where}: unknown key '{key}'"
        close = difflib.get_close_matches(key, sorted(allowed), n=1, cutoff=0.6)
        if close:
            message += f" (did you mean '{close[0]}'?)"
        raise RecipeError(message)


def _get(data: dict, key: str, where: str, required: bool):
    value = data.get(key, _MISSING)
    if value is _MISSING and required:
        raise RecipeError(f"{where}: missing key '{key}'")
    return value


def get_str(data: dict, key: str, where: str, *, required: bool = True) -> str | None:
    value = _get(data, key, where, required)
    if value is _MISSING:
        return None
    if not isinstance(value, str) or not value:
        raise RecipeError(f"{where}: '{key}' must be a non-empty string")
    return value


def get_int(data: dict, key: str, where: str, *, lo: int, hi: int, required: bool = True) -> int | None:
    value = _get(data, key, where, required)
    if value is _MISSING:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecipeError(f"{where}: '{key}' must be an integer")
    if not lo <= value <= hi:
        raise RecipeError(f"{where}: '{key}' must be between {lo} and {hi}, found {value}")
    return value


def get_float(data: dict, key: str, where: str, *, lo: float, hi: float, required: bool = True) -> float | None:
    value = _get(data, key, where, required)
    if value is _MISSING:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecipeError(f"{where}: '{key}' must be a number")
    if not lo <= value <= hi:
        raise RecipeError(f"{where}: '{key}' must be between {lo} and {hi}, found {value}")
    return float(value)


def get_bool(data: dict, key: str, where: str, *, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise RecipeError(f"{where}: '{key}' must be true or false")
    return value


def get_str_list(data: dict, key: str, where: str, *, required: bool = True) -> tuple[str, ...]:
    value = _get(data, key, where, required)
    if value is _MISSING:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise RecipeError(f"{where}: '{key}' must be a list of strings")
    return tuple(value)


def get_table(data: dict, key: str, where: str, *, required: bool = True) -> dict:
    value = _get(data, key, where, required)
    if value is _MISSING:
        return {}
    if not isinstance(value, dict):
        raise RecipeError(f"{where}: '{key}' must be a table")
    return value


_EVIDENCE_KEYS = {"kind", "summary", "date", "hardware", "record", "doc", "decode_tok_s"}


def parse_evidence(data: dict, where: str) -> Evidence:
    check_keys(data, _EVIDENCE_KEYS, where)
    kind = get_str(data, "kind", where)
    if kind not in ("measured", "estimate"):
        raise RecipeError(f"{where}: 'kind' must be 'measured' or 'estimate', found '{kind}'")
    measured = kind == "measured"
    if not measured and "decode_tok_s" in data:
        raise RecipeError(f"{where}: 'decode_tok_s' is a measurement, and an estimate may not state one")
    return Evidence(
        kind=kind,
        summary=get_str(data, "summary", where),
        date=get_str(data, "date", where, required=measured),
        hardware=get_str(data, "hardware", where, required=measured),
        record=get_str(data, "record", where, required=measured),
        doc=get_str(data, "doc", where, required=not measured),
        decode_tok_s=get_float(data, "decode_tok_s", where, lo=0.1, hi=10000, required=False),
    )
