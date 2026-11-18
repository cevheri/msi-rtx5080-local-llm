"""vLLM: the recipe schema, the container a recipe runs in, and how to tell the server is ready.

This module reads only recipes/vllm/ and does not import the llama.cpp adapter.
"""

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from launcher.errors import DownloadError, PreflightError, RecipeError, UsageError
from launcher.hfcache import (
    HfAuth,
    HfRepoModel,
    build_overlay,
    cli_download,
    repo_dirname,
    snapshot_dir,
    snapshot_problems,
)
from launcher.model import ClientMeta, ContainerSpec, GpuInfo, HostEnv, Recipe, RuntimeConfig, Source
from launcher.validate import (
    check_env_secrets,
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

NAME = "vllm"
LABEL = "vLLM"
SERVED_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
CONTAINER_PORT = 8000  # vllm serve's default port
CONTAINER_HF_HOME = "/root/.cache/huggingface"  # the server runs as root, with the cache mounted read-only
OWNED_FLAGS = frozenset({"--host", "--port", "--model", "--revision", "--served-model-name", "--max-model-len"})
SECRET_FLAGS = frozenset({"--api-key", "--hf-token"})
UTILIZATION = "--gpu-memory-utilization"
TOKENIZER_MOUNT = "/llmctl/tokenizer"  # the tokenizer overlay, for a recipe with [model] tokenizer_class
TOKENIZER_FLAG = "--tokenizer"

_RUNTIME_KEYS = {"schema", "runtime", "image", "container_name", "host_port", "ready_path",
                 "start_timeout_s", "opencode_provider", "env", "volumes", "default_gpu_memory_utilization"}
_RECIPE_KEYS = {"schema", "id", "name", "aliases", "doc", "opencode", "chat_only", "model",
                "container", "server", "needs", "evidence", "clients"}
_TEMPLATE_KWARGS = "--default-chat-template-kwargs"
_TOOL_FLAGS = ("--enable-auto-tool-choice", "--tool-call-parser")
_MODEL_KEYS = {"repo", "revision", "include", "exclude", "size_bytes", "served_name", "tokenizer_class"}
_CONTAINER_KEYS = {"memory_gib"}
_SERVER_KEYS = {"context", "args"}
_NEEDS_KEYS = {"ram_gib"}
_ALIAS = re.compile(r"[a-z0-9][a-z0-9._-]*")
_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*")
_REPO = re.compile(r"[A-Za-z0-9][\w.-]*/[\w.-]+")
_REVISION = re.compile(r"[0-9a-f]{40}")
_VOLUME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*:/[^:]+(:ro)?")
_CLASS = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TOKENIZER_CONFIG = "tokenizer_config.json"


@dataclass(frozen=True)
class VllmModel(HfRepoModel):
    tokenizer_class: str | None = None  # set: llmctl mounts a tokenizer overlay whose tokenizer_config.json names it


@dataclass(frozen=True)
class VllmRuntimeConfig(RuntimeConfig):
    default_gpu_memory_utilization: float = 0.0  # what the image's vLLM uses when a recipe does not set it


def parse_runtime(data: dict, source: Source) -> RuntimeConfig:
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
    check_env_secrets(env, where)
    volumes = get_str_list(data, "volumes", where, required=False)
    for volume in volumes:
        if not _VOLUME.fullmatch(volume):
            raise RecipeError(f"{where}: volume '{volume}' must be a named volume such as 'vllm-cache:/root/.cache/vllm'")
    return VllmRuntimeConfig(
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
        refuses_tools_without_parser=True,
        default_gpu_memory_utilization=get_float(data, "default_gpu_memory_utilization", where, lo=0.05, hi=1.0),
    )


def _utilization(args: tuple[str, ...], where: str) -> float | None:
    """The recipe's --gpu-memory-utilization, None when it relies on the runtime default."""
    for index, arg in enumerate(args):
        if arg == UTILIZATION:
            value = args[index + 1] if index + 1 < len(args) else ""
        elif arg.startswith(UTILIZATION + "="):
            value = arg.split("=", 1)[1]
        else:
            continue
        try:
            fraction = float(value)
        except ValueError:
            break
        if 0 < fraction <= 1:
            return fraction
        break
    else:
        return None
    raise RecipeError(f"{where}: {UTILIZATION} must be a fraction between 0 and 1")


def _check_args(args: tuple[str, ...], where: str) -> None:
    for arg in args:
        flag = arg.split("=", 1)[0]
        if flag in OWNED_FLAGS:
            raise RecipeError(f"{where}: '{flag}' is set by llmctl and may not appear in 'args'")
        if flag in SECRET_FLAGS:
            raise RecipeError(f"{where}: '{flag}' would put a secret on the command line and may not appear in 'args'")
    _utilization(args, where)


def _check_clients(clients: ClientMeta, args: tuple[str, ...], chat_only: bool, where: str) -> None:
    """The [clients] table and chat_only must describe what these server args make the server do."""
    parser = flag_present(args, "--reasoning-parser")
    if parser != (clients.thinking != "none"):
        need = "needs --reasoning-parser" if clients.thinking != "none" else "may not set --reasoning-parser"
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
    present = [flag for flag in _TOOL_FLAGS if flag_present(args, flag)]
    if chat_only and present:
        raise RecipeError(f"{where}: a chat-only recipe may not set {present[0]}")
    if not chat_only and len(present) != len(_TOOL_FLAGS):
        missing = [flag for flag in _TOOL_FLAGS if flag not in present]
        raise RecipeError(f"{where}: a recipe with tools (chat_only false) needs {' and '.join(missing)}")


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
    tokenizer_class = get_str(model_data, "tokenizer_class", model_where, required=False)
    if tokenizer_class is not None and not _CLASS.fullmatch(tokenizer_class):
        raise RecipeError(f"{model_where}: 'tokenizer_class' must be a Python class name such as "
                          f"'PreTrainedTokenizerFast', found '{tokenizer_class}'")
    model = VllmModel(
        repo=repo,
        revision=revision,
        include=get_str_list(model_data, "include", model_where, required=False),
        exclude=get_str_list(model_data, "exclude", model_where, required=False),
        size_bytes=get_int(model_data, "size_bytes", model_where, lo=1, hi=10**13),
        served_name=get_str(model_data, "served_name", model_where),
        tokenizer_class=tokenizer_class,
    )
    if not SERVED_NAME.fullmatch(model.served_name):
        raise RecipeError(f"{model_where}: 'served_name' may use only letters, digits, '.', '_' and '-', "
                          f"and must start with a letter or digit, found '{model.served_name}'")

    container_where = f"{where} [container]"
    container_data = get_table(data, "container", where)
    check_keys(container_data, _CONTAINER_KEYS, container_where)
    memory_gib = get_int(container_data, "memory_gib", container_where, lo=4, hi=56)

    server_where = f"{where} [server]"
    server_data = get_table(data, "server", where)
    check_keys(server_data, _SERVER_KEYS, server_where)
    context = get_int(server_data, "context", server_where, lo=1024, hi=1048576)
    args = get_str_list(server_data, "args", server_where)
    _check_args(args, server_where)
    if tokenizer_class is not None and flag_present(args, TOKENIZER_FLAG):
        # vLLM would take the last of two --tokenizer flags, so the recipe's would replace the overlay.
        raise RecipeError(f"{server_where}: '{TOKENIZER_FLAG}' is set by llmctl when [model] tokenizer_class is set "
                          "and may not appear in 'args'")
    chat_only = get_bool(data, "chat_only", where, default=False)

    clients_where = f"{where} [clients]"
    clients = parse_clients(get_table(data, "clients", where), clients_where, context=context)
    _check_clients(clients, args, chat_only, clients_where)

    needs_where = f"{where} [needs]"
    needs_data = get_table(data, "needs", where, required=False)
    check_keys(needs_data, _NEEDS_KEYS, needs_where)
    ram_gib = get_float(needs_data, "ram_gib", needs_where, lo=0.1, hi=1024, required=False)

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
        needs_gpu_mib=None,
        needs_ram_gib=ram_gib,
        model=model,
        evidence=parse_evidence(get_table(data, "evidence", where), f"{where} [evidence]"),
        source=source,
        clients=clients,
    )


def _tokenizer_config(snapshot: Path, revision: str) -> tuple[dict | None, str | None]:
    """The snapshot's top-level tokenizer_config.json as an object, or the problem that keeps the overlay from it."""
    path = snapshot / _TOKENIZER_CONFIG
    where = f"{_TOKENIZER_CONFIG} in snapshot {revision[:12]}"
    if not path.is_file():
        return None, (f"{where} is missing, so the tokenizer overlay cannot set tokenizer_class; remove "
                      "tokenizer_class from the recipe, or make the download include that file, then pull again")
    try:
        config = json.loads(path.read_bytes())
        reason = None if isinstance(config, dict) else "its top level is not an object"
    except (OSError, ValueError) as exc:
        reason = str(exc)
    if reason is None:
        return config, None
    return None, (f"{where} is not a JSON object ({reason}), so the tokenizer overlay cannot set tokenizer_class; "
                  "delete that file and its blob from the Hugging Face cache, then pull again")


def file_problems(recipe: Recipe, hf_home: Path) -> list[str]:
    problems = snapshot_problems(recipe.model, hf_home)
    if problems or recipe.model.tokenizer_class is None:
        return problems
    snapshot = snapshot_dir(hf_home, recipe.model.repo, recipe.model.revision)
    _, problem = _tokenizer_config(snapshot, recipe.model.revision)
    return [problem] if problem else []


def served_name(recipe: Recipe, runtime: RuntimeConfig) -> str:
    return recipe.model.served_name


def overlay_dir(recipe: Recipe, env: HostEnv) -> Path:
    """The recipe's tokenizer overlay, mounted read-only at TOKENIZER_MOUNT."""
    if env.cache_dir is None:
        raise ValueError("a vLLM recipe with tokenizer_class needs HostEnv.cache_dir")
    if not env.cache_dir.is_absolute():
        raise UsageError(f"llmctl's cache folder {env.cache_dir} is not an absolute path; "
                         "set XDG_CACHE_HOME (or HOME) to one")
    return env.cache_dir / "vllm" / "tokenizers" / recipe.id


def _container_snapshot(model: VllmModel) -> str:
    return f"{CONTAINER_HF_HOME}/hub/{repo_dirname(model.repo)}/snapshots/{model.revision}"


def container_spec(recipe: Recipe, runtime: RuntimeConfig, env: HostEnv) -> ContainerSpec:
    model = recipe.model
    args = (_container_snapshot(model), "--served-model-name", model.served_name,
            "--max-model-len", str(recipe.context))
    mounts = (f"{env.hf_home}:{CONTAINER_HF_HOME}:ro",)
    if model.tokenizer_class is not None:
        args += (TOKENIZER_FLAG, TOKENIZER_MOUNT)
        mounts += (f"{overlay_dir(recipe, env)}:{TOKENIZER_MOUNT}:ro",)
    args += recipe.args
    return ContainerSpec(
        user=None,
        env=runtime.env,
        mounts=mounts + runtime.volumes,
        extra_flags=("--ipc=host",),
        image=runtime.image,
        args=args,
        container_port=CONTAINER_PORT,
    )


def gpu_need_mib(recipe: Recipe, runtime: VllmRuntimeConfig, gpu: GpuInfo) -> int:
    """vLLM refuses to start unless free memory covers this share of what CUDA can use."""
    fraction = _utilization(recipe.args, recipe.id) or runtime.default_gpu_memory_utilization
    return math.ceil(fraction * (gpu.total_mib - gpu.reserved_mib))


def gpu_advice(recipe: Recipe) -> str:
    return "write an override with a smaller --gpu-memory-utilization or a smaller context (--max-model-len)"


def ready(host, runtime: RuntimeConfig, recipe: Recipe) -> bool:
    try:
        response = host.http("GET", f"http://127.0.0.1:{runtime.host_port}{runtime.ready_path}", timeout=2.0)
    except OSError:
        return False
    if response.status != 200:
        return False
    try:
        listed = {entry.get("id") for entry in json.loads(response.body).get("data", [])}
    except (ValueError, AttributeError):
        return False
    return recipe.model.served_name in listed


def prepare_start(recipe: Recipe, runtime: RuntimeConfig, env: HostEnv, *, dry_run: bool) -> tuple[str, ...]:
    """Build the tokenizer overlay of a recipe with tokenizer_class from the snapshot as it is now, or with dry_run
    only describe it; return up's lines. A recipe without the key needs nothing."""
    model = recipe.model
    if model.tokenizer_class is None:
        return ()
    folder = overlay_dir(recipe, env)
    ending = (f"into snapshot {model.revision[:12]} of {model.repo}, and {_TOKENIZER_CONFIG} with tokenizer_class "
              f"{model.tokenizer_class}")
    if dry_run:
        return (f"would write {folder}: tokenizer overlay with links {ending}",)
    snapshot = snapshot_dir(env.hf_home, model.repo, model.revision)
    if not snapshot.is_dir():
        raise PreflightError(f"cannot write the tokenizer overlay {folder}: missing snapshot {model.revision} of "
                             f"{model.repo}")
    config, problem = _tokenizer_config(snapshot, model.revision)
    if problem:
        raise PreflightError(f"cannot write the tokenizer overlay {folder}: {problem}")
    config["tokenizer_class"] = model.tokenizer_class  # replaced in place, or appended last
    data = (json.dumps(config, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    try:
        links = build_overlay(folder, snapshot, _container_snapshot(model), {_TOKENIZER_CONFIG: data})
    except OSError as exc:
        raise PreflightError(f"cannot write the tokenizer overlay {folder}: {exc}; check the free space and "
                             f"permissions of {env.cache_dir}") from None
    return (f"wrote {folder}: tokenizer overlay with {links} links {ending}",)


def pull(host, recipe: Recipe, runtime: RuntimeConfig, hf_home: Path, auth: HfAuth, out) -> None:
    """Download the repository at its pinned revision with the hf CLI from the pinned vLLM image, then check it."""
    cli_download(host, recipe.id, recipe.model, runtime.image, "hf", (), hf_home, auth, out)
    problems = file_problems(recipe, hf_home)
    if problems:
        raise DownloadError(f"{recipe.id}: " + "; ".join(problems))


def gpu_need_text(recipe: Recipe, runtime: VllmRuntimeConfig) -> str:
    fraction = _utilization(recipe.args, recipe.id) or runtime.default_gpu_memory_utilization
    return f"{fraction * 100:g}%"
