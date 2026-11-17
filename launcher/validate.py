"""Reading recipe files and checking their values. Generic: no runtime's settings live here."""

import difflib
import hashlib
import math
import tomllib
from collections.abc import Iterable
from pathlib import Path

from launcher.errors import RecipeError
from launcher.model import ClientMeta, Evidence, Source

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


THINKING = ("toggle", "effort", "always", "none")
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
_CLIENTS_KEYS = {"output", "thinking", "default", "efforts", "medium_kwargs"}


def _applies_only(data: dict, key: str, thinking: str, owner: str, where: str) -> None:
    if key in data and thinking != owner:
        raise RecipeError(f'{where}: \'{key}\' applies only to thinking = "{owner}"')


def _efforts(data: dict, where: str) -> tuple[str, ...]:
    value = _get(data, "efforts", where, True)
    shape = f"a non-empty list of {', '.join(EFFORTS)} in that order, without duplicates, that contains medium"
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise RecipeError(f"{where}: 'efforts' must be {shape}")
    unknown = [item for item in value if item not in EFFORTS]
    if unknown:
        raise RecipeError(f"{where}: 'efforts' must be {shape}, found unknown '{unknown[0]}'")
    positions = [EFFORTS.index(item) for item in value]
    if positions != sorted(set(positions)):
        raise RecipeError(f"{where}: 'efforts' must be {shape}, found {value}")
    if "medium" not in value:
        raise RecipeError(f"{where}: 'efforts' must contain medium, the server's default effort")
    return tuple(value)


def _medium_kwargs(data: dict, where: str) -> tuple[tuple[str, object], ...]:
    table = get_table(data, "medium_kwargs", where, required=False)
    if "enable_thinking" in table:
        raise RecipeError(f"{where}: 'medium_kwargs' may not set enable_thinking")
    for key, value in table.items():
        scalar = isinstance(value, (str, bool, int)) or (isinstance(value, float) and math.isfinite(value))
        if not scalar:
            raise RecipeError(f"{where}: 'medium_kwargs.{key}' must be a string, boolean, integer or finite number")
    return tuple(table.items())


def parse_clients(data: dict, where: str, *, context: int) -> ClientMeta:
    """The [clients] table: generic checks only; each adapter checks it against its own server args."""
    check_keys(data, _CLIENTS_KEYS, where)
    output = get_int(data, "output", where, lo=1024, hi=context // 2)
    thinking = get_str(data, "thinking", where)
    if thinking not in THINKING:
        raise RecipeError(f"{where}: 'thinking' must be one of {', '.join(THINKING)}, found '{thinking}'")
    _applies_only(data, "default", thinking, "toggle", where)
    _applies_only(data, "efforts", thinking, "effort", where)
    _applies_only(data, "medium_kwargs", thinking, "toggle", where)
    default = None
    if thinking == "toggle":
        default = get_str(data, "default", where)
        if default not in ("off", "on"):
            raise RecipeError(f"{where}: 'default' must be 'off' or 'on', found '{default}'")
    efforts = _efforts(data, where) if thinking == "effort" else ()
    medium_kwargs = _medium_kwargs(data, where) if thinking == "toggle" else ()
    return ClientMeta(output, thinking, default, efforts, medium_kwargs)


SECRET_ENV_SUFFIXES = ("_TOKEN", "_KEY", "_SECRET", "_PASSWORD")


def check_env_secrets(env: dict, where: str) -> None:
    """Refuses an [env] key that names a secret, since [env] goes into the docker run command line."""
    for key in env:
        if key.upper().endswith(SECRET_ENV_SUFFIXES):
            raise RecipeError(f"{where} [env]: '{key}' would put a secret into the docker run command line; llmctl "
                              "takes a Hugging Face token only from HF_TOKEN or --hf-env-file at pull time "
                              "(docs/llmctl.md#what-llmctl-enforces)")


def flag_value(args: tuple[str, ...], flag: str) -> str | None:
    """The value of the first `flag` in args: the next token, or the text after `flag=`; "" when flag ends args."""
    for index, arg in enumerate(args):
        if arg == flag:
            return args[index + 1] if index + 1 < len(args) else ""
        if arg.startswith(flag + "="):
            return arg[len(flag) + 1:]
    return None


def flag_present(args: tuple[str, ...], flag: str) -> bool:
    return flag_value(args, flag) is not None
