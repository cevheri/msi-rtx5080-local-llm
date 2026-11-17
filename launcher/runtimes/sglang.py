"""SGLang: the recipe schema, the runtime file and the checks a recipe's server args must pass.

This module reads only recipes/sglang/ and imports no other adapter or client module.
"""

import json
import re
from dataclasses import dataclass

from launcher.errors import RecipeError
from launcher.hfcache import HfRepoModel
from launcher.model import ClientMeta, GpuInfo, Recipe, RuntimeConfig, Source
from launcher.validate import (
    check_keys,
    flag_present,
    flag_value,
    get_bool,
    get_float,
    get_int,
    get_str,
    get_str_list,
    get_table,
    parse_clients,
    parse_evidence,
)

NAME = "sglang"
LABEL = "SGLang"
SERVED_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
CONTAINER_PORT = 30000  # sglang.launch_server's default port
CONTAINER_HF_HOME = "/root/.cache/huggingface"  # the server runs as root, with the cache mounted read-only
MODEL_MOUNT = "/llmctl/model"  # the recipe's model overlay, mounted read-only
LAUNCH = ("python3", "-m", "sglang.launch_server")
HF_COMMAND = ("-m", "huggingface_hub.cli.hf")  # the image has no hf executable
MODELS_PATH = "/v1/models"
SERVER_INFO_PATH = "/server_info"
FLUSH_PATH = "/flush_cache"
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty")
MAX_FRACTION = 0.90
OWNED_FLAGS = frozenset({"--model-path", "--model", "--served-model-name", "--context-length", "--host", "--port",
                         "--revision"})
SECRET_FLAGS = frozenset({"--api-key", "--admin-api-key", "--config"})

_FRACTION = "--mem-fraction-static"
_RUNNING = "--max-running-requests"
_PREFILL_GRAPH = "--disable-prefill-cuda-graph"
_GRAPH = "--disable-cuda-graph"
_SAMPLING_DEFAULTS = "--sampling-defaults"
_PREFERRED_SAMPLING = "--preferred-sampling-params"
_TOOL_PARSER = "--tool-call-parser"
_REASONING_PARSER = "--reasoning-parser"
_PAGED = "--enable-paged-experts"
_RESIDENT = "--paged-experts-num-resident"
_MOE_BACKEND = "--moe-runner-backend"
_TEMPLATE_KWARGS = "--default-chat-template-kwargs"
# Every flag a rule reads: SGLang's argparse expands abbreviations and keeps the last value of a repeated flag,
# so these must be written in full and once.
_CHECKED_FLAGS = OWNED_FLAGS | SECRET_FLAGS | frozenset({
    _FRACTION, _RUNNING, _PREFILL_GRAPH, _GRAPH, _SAMPLING_DEFAULTS, _PREFERRED_SAMPLING, _TOOL_PARSER,
    _REASONING_PARSER, _PAGED, _RESIDENT, _MOE_BACKEND, _TEMPLATE_KWARGS,
})

_RUNTIME_KEYS = {"schema", "runtime", "image", "container_name", "host_port", "ready_path",
                 "start_timeout_s", "opencode_provider", "env", "volumes", "min_free_at_ready_mib"}
_RECIPE_KEYS = {"schema", "id", "name", "aliases", "doc", "opencode", "chat_only", "model",
                "container", "server", "needs", "evidence", "clients"}
_MODEL_KEYS = {"repo", "revision", "include", "exclude", "size_bytes", "served_name"}
_CONTAINER_KEYS = {"memory_gib"}
_SERVER_KEYS = {"context", "args", "sampling"}
_NEEDS_KEYS = {"gpu_mib", "ram_gib"}
_ALIAS = re.compile(r"[a-z0-9][a-z0-9._-]*")
_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*")
_REPO = re.compile(r"[A-Za-z0-9][\w.-]*/[\w.-]+")
_REVISION = re.compile(r"[0-9a-f]{40}")
_VOLUME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*:/[^:]+(:ro)?")


@dataclass(frozen=True)
class SglangRuntimeConfig(RuntimeConfig):
    min_free_at_ready_mib: int = 0  # free GPU memory the post-ready check requires; 0 switches the check off


@dataclass(frozen=True)
class SglangModel(HfRepoModel):
    sampling: tuple[tuple[str, int | float], ...] = ()  # [server] sampling, in SAMPLING_KEYS order


def parse_runtime(data: dict, source: Source) -> SglangRuntimeConfig:
    where = str(source.path)
    check_keys(data, _RUNTIME_KEYS, where)
    get_int(data, "schema", where, lo=1, hi=1)
    if data.get("runtime") != NAME:
        raise RecipeError(f"{where}: 'runtime' must be '{NAME}'")
    container_name = get_str(data, "container_name", where)
    if not _NAME.fullmatch(container_name):
        raise RecipeError(f"{where}: 'container_name' is not a valid Docker container name")
    ready_path = get_str(data, "ready_path", where)
    if not ready_path.startswith("/"):
        raise RecipeError(f"{where}: 'ready_path' must start with '/'")
    env = get_table(data, "env", where, required=False)
    for key, value in env.items():
        if not isinstance(value, str):
            raise RecipeError(f"{where} [env]: '{key}' must be a string")
    volumes = get_str_list(data, "volumes", where, required=False)
    for volume in volumes:
        if not _VOLUME.fullmatch(volume):
            raise RecipeError(f"{where}: volume '{volume}' must be a named volume such as "
                              "'sglang-cache:/root/.cache/sglang'")
    return SglangRuntimeConfig(
        name=NAME,
        image=get_str(data, "image", where),
        container_name=container_name,
        host_port=get_int(data, "host_port", where, lo=1024, hi=65535),
        ready_path=ready_path,
        start_timeout_s=get_int(data, "start_timeout_s", where, lo=10, hi=7200),
        env=tuple(env.items()),
        volumes=volumes,
        opencode_provider=get_str(data, "opencode_provider", where),
        served_name=None,
        source=source,
        refuses_tools_without_parser=False,  # until a measurement shows HTTP 400 (plan ruling 4)
        effort_enables_toggle_thinking=True,  # any reasoning_effort but none turns a toggle model's thinking on
        min_free_at_ready_mib=get_int(data, "min_free_at_ready_mib", where, lo=0, hi=65536),
    )


def _flag_kind_text(flag: str, kind: str) -> str:
    if kind == "owned":
        return f"'{flag}' is set by llmctl and may not appear in 'args'"
    if kind == "--config":
        return f"'{flag}' could carry a key in its YAML file and may not appear in 'args'"
    return f"'{flag}' would put a secret on the command line and may not appear in 'args'"


def _check_flags(args: tuple[str, ...], where: str) -> None:
    """Owned, secret, abbreviated and repeated checked flags, in every form SGLang's argparse accepts."""
    seen: set[str] = set()
    repeated = None
    for arg in args:
        if not arg.startswith("--") or arg == "--":
            continue
        flag = arg.split("=", 1)[0]
        if flag in OWNED_FLAGS:
            raise RecipeError(f"{where}: {_flag_kind_text(flag, 'owned')}")
        if flag in SECRET_FLAGS:
            raise RecipeError(f"{where}: {_flag_kind_text(flag, '--config' if flag == '--config' else 'secret')}")
        if flag in _CHECKED_FLAGS:
            if flag in seen and repeated is None:
                repeated = flag
            seen.add(flag)
            continue
        matches = sorted(checked for checked in _CHECKED_FLAGS if checked.startswith(flag))
        if not matches:
            continue
        expands = " or ".join(matches)
        if all(match in OWNED_FLAGS for match in matches):
            raise RecipeError(f"{where}: {_flag_kind_text(flag, 'owned')} (it abbreviates {expands})")
        if all(match in SECRET_FLAGS for match in matches):
            kind = "--config" if matches == ["--config"] else "secret"
            raise RecipeError(f"{where}: {_flag_kind_text(flag, kind)} (it abbreviates {expands})")
        raise RecipeError(f"{where}: '{flag}' abbreviates {expands}; write the flag in full")
    if repeated is not None:
        raise RecipeError(f"{where}: '{repeated}' appears more than once in 'args'; "
                          "SGLang uses the last one, so write it once")


def _check_args(args: tuple[str, ...], measured: bool, where: str) -> None:
    _check_flags(args, where)
    text = flag_value(args, _FRACTION)
    try:
        fraction = float(text) if text is not None else None
    except ValueError:
        fraction = None
    if fraction is None or not 0 < fraction <= MAX_FRACTION:
        raise RecipeError(f"{where}: {_FRACTION} is required, a number above 0 and at most {MAX_FRACTION:.2f} "
                          "(sglang/README.md#memory-budget)")
    if flag_value(args, _RUNNING) != "1":
        raise RecipeError(f"{where}: SGLang recipes run one request at a time: {_RUNNING} 1 is required")
    if not measured and not flag_present(args, _PREFILL_GRAPH):
        raise RecipeError(f"{where}: keep {_PREFILL_GRAPH} until the prefill CUDA graph is measured with this recipe "
                          "([evidence] kind = \"measured\")")
    if flag_present(args, _GRAPH):
        raise RecipeError(f"{where}: {_GRAPH} is deprecated in SGLang v0.5.23 and also turns the decode graph off; "
                          f"use {_PREFILL_GRAPH}")
    defaults = flag_value(args, _SAMPLING_DEFAULTS)
    if defaults is not None and defaults != "model":
        raise RecipeError(f"{where}: {_SAMPLING_DEFAULTS} must stay model, or the overlay's sampling is ignored")
    if flag_present(args, _PREFERRED_SAMPLING):
        raise RecipeError(f"{where}: {_PREFERRED_SAMPLING} does not reach chat requests in SGLang v0.5.23; "
                          "pin sampling in [server] sampling")
    resident = flag_value(args, _RESIDENT)
    if not flag_present(args, _PAGED):
        if resident is not None:
            raise RecipeError(f"{where}: {_RESIDENT} works only with {_PAGED}")
        return
    if resident is None:
        raise RecipeError(f"{where}: {_PAGED} needs {_RESIDENT} <K>")
    if not (resident.isascii() and resident.isdigit() and int(resident) > 0):
        raise RecipeError(f"{where}: {_RESIDENT} needs a positive integer, found '{resident}'")
    if flag_value(args, _MOE_BACKEND) != "marlin":
        raise RecipeError(f"{where}: {_PAGED} needs {_MOE_BACKEND} marlin")


def _check_clients(clients: ClientMeta, args: tuple[str, ...], chat_only: bool, where: str) -> None:
    """The parser names, and the [clients] table and chat_only against what these server args make the server do."""
    for parser_flag in (_TOOL_PARSER, _REASONING_PARSER):
        value = flag_value(args, parser_flag)
        if value in ("", "auto") or (value or "").startswith("--"):  # "--..." is the next flag, not a name
            raise RecipeError(f"{where}: {parser_flag} needs an explicit parser name, not auto")
    parser = flag_present(args, _REASONING_PARSER)
    if parser != (clients.thinking != "none"):
        need = f"needs {_REASONING_PARSER}" if clients.thinking != "none" else f"may not set {_REASONING_PARSER}"
        raise RecipeError(f"{where}: thinking = \"{clients.thinking}\" {need}")
    text = flag_value(args, _TEMPLATE_KWARGS)
    if clients.thinking == "toggle" and text is not None:
        try:
            kwargs = json.loads(text)
        except ValueError:
            kwargs = None
        if not isinstance(kwargs, dict):
            raise RecipeError(f"{where}: '{_TEMPLATE_KWARGS}' must be followed by a JSON object, found '{text}'")
        if "enable_thinking" in kwargs and kwargs["enable_thinking"] is not (clients.default == "on"):
            raise RecipeError(f"{where}: default = \"{clients.default}\" but {_TEMPLATE_KWARGS} sets "
                              f"enable_thinking to {json.dumps(kwargs['enable_thinking'])}")
    tools = flag_present(args, _TOOL_PARSER)
    if not chat_only and not tools:
        raise RecipeError(f"{where}: a recipe with tools (chat_only false) needs {_TOOL_PARSER}")
    if chat_only and tools:
        raise RecipeError(f"{where}: a chat-only recipe may not set {_TOOL_PARSER}")


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


_SAMPLING_RULES = {
    "temperature": ("a number from 0 to 2", lambda v: _number(v) and 0 <= v <= 2),
    "top_p": ("a number above 0 and at most 1", lambda v: _number(v) and 0 < v <= 1),
    "top_k": ("-1 or an integer from 1 to 1000000",
              lambda v: isinstance(v, int) and not isinstance(v, bool) and (v == -1 or 1 <= v <= 1000000)),
    "min_p": ("a number from 0 to 1", lambda v: _number(v) and 0 <= v <= 1),
    "repetition_penalty": ("a number above 0 and at most 2", lambda v: _number(v) and 0 < v <= 2),
}


def _sampling(server_data: dict, where: str) -> tuple[tuple[str, int | float], ...]:
    """[server] sampling: the keys SGLang reads from generation_config.json, within SGLang's own bounds."""
    sampling_where = f"{where} [server] sampling"
    table = get_table(server_data, "sampling", f"{where} [server]", required=False)
    check_keys(table, SAMPLING_KEYS, sampling_where)
    for key, value in table.items():
        bound, allowed = _SAMPLING_RULES[key]
        if not allowed(value):  # a NaN fails every comparison, so every bound refuses it
            shown = json.dumps(value, default=str)  # str shows a TOML date or time, which JSON has no type for
            raise RecipeError(f"{sampling_where}: '{key}' must be {bound}, found {shown}")
    return tuple((key, table[key]) for key in SAMPLING_KEYS if key in table)


def parse_recipe(data: dict, source: Source, stem: str) -> Recipe:
    where = str(source.path)
    check_keys(data, _RECIPE_KEYS, where)
    get_int(data, "schema", where, lo=1, hi=1)
    recipe_id = get_str(data, "id", where)
    if recipe_id != stem:
        raise RecipeError(f"{where}: id '{recipe_id}' must match the file name '{stem}'")
    aliases = get_str_list(data, "aliases", where, required=False)
    for alias in aliases:
        if not _ALIAS.fullmatch(alias):
            raise RecipeError(f"{where}: alias '{alias}' may use only lower-case letters, digits, '.', '_' and '-'")

    model_where = f"{where} [model]"
    model_data = get_table(data, "model", where)
    check_keys(model_data, _MODEL_KEYS, model_where)
    repo = get_str(model_data, "repo", model_where)
    if not _REPO.fullmatch(repo):
        raise RecipeError(f"{model_where}: 'repo' must look like 'owner/name'")
    revision = get_str(model_data, "revision", model_where)
    if not _REVISION.fullmatch(revision):
        raise RecipeError(f"{model_where}: 'revision' must be a 40-character commit hash")
    include = get_str_list(model_data, "include", model_where, required=False)
    exclude = get_str_list(model_data, "exclude", model_where, required=False)
    size_bytes = get_int(model_data, "size_bytes", model_where, lo=1, hi=10**13)
    served = get_str(model_data, "served_name", model_where)
    if not SERVED_NAME.fullmatch(served):
        raise RecipeError(f"{model_where}: 'served_name' may use only letters, digits, '.', '_' and '-', "
                          f"and must start with a letter or digit, found '{served}'")

    container_where = f"{where} [container]"
    container_data = get_table(data, "container", where)
    check_keys(container_data, _CONTAINER_KEYS, container_where)
    memory_gib = get_int(container_data, "memory_gib", container_where, lo=4, hi=56)

    evidence = parse_evidence(get_table(data, "evidence", where), f"{where} [evidence]")

    server_where = f"{where} [server]"
    server_data = get_table(data, "server", where)
    check_keys(server_data, _SERVER_KEYS, server_where)
    context = get_int(server_data, "context", server_where, lo=1024, hi=1048576)
    args = get_str_list(server_data, "args", server_where)
    _check_args(args, evidence.kind == "measured", server_where)
    sampling = _sampling(server_data, where)
    chat_only = get_bool(data, "chat_only", where, default=False)

    clients_where = f"{where} [clients]"
    clients = parse_clients(get_table(data, "clients", where), clients_where, context=context)
    _check_clients(clients, args, chat_only, clients_where)

    needs_where = f"{where} [needs]"
    needs_data = get_table(data, "needs", where)
    check_keys(needs_data, _NEEDS_KEYS, needs_where)
    gpu_mib = get_int(needs_data, "gpu_mib", needs_where, lo=256, hi=262144)
    ram_gib = get_float(needs_data, "ram_gib", needs_where, lo=0.1, hi=1024)
    if ram_gib > memory_gib:
        raise RecipeError(f"{needs_where}: ram_gib {ram_gib:g} exceeds [container] memory_gib {memory_gib}, "
                          "so the container would be killed at its measured peak")

    return Recipe(
        runtime=NAME,
        id=recipe_id,
        name=get_str(data, "name", where),
        aliases=aliases,
        doc=get_str(data, "doc", where),
        opencode=get_bool(data, "opencode", where, default=False),
        chat_only=chat_only,
        memory_gib=memory_gib,
        context=context,
        args=args,
        needs_gpu_mib=gpu_mib,
        needs_ram_gib=ram_gib,
        model=SglangModel(repo=repo, revision=revision, include=include, exclude=exclude, size_bytes=size_bytes,
                          served_name=served, sampling=sampling),
        evidence=evidence,
        source=source,
        clients=clients,
    )


def served_name(recipe: Recipe, runtime: RuntimeConfig) -> str:
    return recipe.model.served_name


def gpu_need_mib(recipe: Recipe, runtime: RuntimeConfig, gpu: GpuInfo) -> int:
    """SGLang sizes itself from whatever is free, so the need is the measured peak the recipe states."""
    return recipe.needs_gpu_mib


def gpu_need_text(recipe: Recipe, runtime: RuntimeConfig) -> str:
    return f"{recipe.needs_gpu_mib} MiB"


def gpu_advice(recipe: Recipe) -> str:
    if flag_present(recipe.args, _PAGED):
        return ("write an override with fewer resident experts (--paged-experts-num-resident) and the gpu_mib "
                "you measure with it (sglang/README.md#paged-experts)")
    return ("write an override with a smaller --mem-fraction-static or context and the gpu_mib you measure with it "
            "(sglang/README.md#memory-budget)")
