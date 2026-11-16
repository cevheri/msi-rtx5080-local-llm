"""llama.cpp: the recipe schema, the container a recipe runs in, and how to tell the server is ready.

This module reads only recipes/llamacpp/ and does not import the vLLM adapter.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from launcher.errors import DownloadError, RecipeError
from launcher.hfcache import HfAuth, download_file, repo_dirname
from launcher.model import ClientMeta, ContainerSpec, GpuInfo, HostEnv, Recipe, RuntimeConfig, Source
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

NAME = "llamacpp"
LABEL = "llama.cpp"
SERVED_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
CONTAINER_PORT = 8080  # the image's LLAMA_ARG_PORT
CONTAINER_HF_HOME = "/hf"
OWNED_FLAGS = frozenset({
    "--host", "--port", "-m", "--model", "-hf", "--hf-repo", "-hfr", "--hf-file", "-hff",
    "-hfv", "--hf-repo-v", "-hffv", "--hf-file-v", "-c", "--ctx-size", "-a", "--alias",
})
SECRET_FLAGS = frozenset({"--api-key", "--hf-token", "-hft"})
# CUDA graphs hang RTX 50 laptop GPUs (https://github.com/ggml-org/llama.cpp/issues/27330): always off, whatever a file says.
GRAPHS_OFF = ("GGML_CUDA_DISABLE_GRAPHS", "1")

_RUNTIME_KEYS = {"schema", "runtime", "image", "container_name", "host_port", "ready_path",
                 "start_timeout_s", "opencode_provider", "served_name", "env"}
_RECIPE_KEYS = {"schema", "id", "name", "aliases", "doc", "opencode", "chat_only", "model",
                "container", "server", "needs", "evidence", "clients"}
_TEMPLATE_KWARGS = "--chat-template-kwargs"
_REASONING_FLAGS = ("--reasoning", "--reasoning-effort", "--reasoning-format", _TEMPLATE_KWARGS)
_MODEL_KEYS = {"repo", "revision", "files", "size_bytes"}
_CONTAINER_KEYS = {"memory_gib"}
_SERVER_KEYS = {"context", "args"}
_NEEDS_KEYS = {"gpu_mib", "ram_gib"}
_ALIAS = re.compile(r"[a-z0-9][a-z0-9._-]*")
_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*")
_REPO = re.compile(r"[A-Za-z0-9][\w.-]*/[\w.-]+")
_REVISION = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class GgufModel:
    repo: str
    revision: str
    files: tuple[str, ...]
    size_bytes: int


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
    if GRAPHS_OFF[0] in env:
        raise RecipeError(f"{where} [env]: {GRAPHS_OFF[0]} is set by llmctl (always 1) and may not appear here")
    served = get_str(data, "served_name", where)
    if not SERVED_NAME.fullmatch(served):
        raise RecipeError(f"{where}: 'served_name' may use only letters, digits, '.', '_' and '-', "
                          f"and must start with a letter or digit, found '{served}'")
    return RuntimeConfig(
        name=NAME,
        image=get_str(data, "image", where),
        container_name=container_name,
        host_port=get_int(data, "host_port", where, lo=1024, hi=65535),
        ready_path=ready_path,
        start_timeout_s=get_int(data, "start_timeout_s", where, lo=10, hi=7200),
        env=tuple(env.items()),
        volumes=(),
        opencode_provider=get_str(data, "opencode_provider", where),
        served_name=served,
        source=source,
    )


def _check_args(args: tuple[str, ...], where: str) -> None:
    for arg in args:
        flag = arg.split("=", 1)[0]
        if flag in OWNED_FLAGS:
            raise RecipeError(f"{where}: '{flag}' is set by llmctl and may not appear in 'args'")
        if flag in SECRET_FLAGS:
            raise RecipeError(f"{where}: '{flag}' would put a secret on the command line and may not appear in 'args'")


def _template_kwargs(args: tuple[str, ...], where: str) -> dict:
    """The JSON object of --chat-template-kwargs, {} when the flag is absent."""
    text = flag_value(args, _TEMPLATE_KWARGS)
    if text is None:
        return {}
    try:
        value = json.loads(text)
    except ValueError:
        value = None
    if not isinstance(value, dict):
        raise RecipeError(f"{where}: '{_TEMPLATE_KWARGS}' must be followed by a JSON object, found '{text}'")
    return value


def _check_clients(clients: ClientMeta, args: tuple[str, ...], server_where: str, where: str) -> None:
    """The [clients] table must describe what these server args make the server do."""
    kwargs = _template_kwargs(args, server_where)
    effort = flag_value(args, "--reasoning-effort")
    if clients.thinking == "toggle":
        off = flag_value(args, "--reasoning") == "off" or kwargs.get("enable_thinking") is False
        server = "off" if off else "on"
        if clients.default != server:
            raise RecipeError(
                f"{where}: default = \"{clients.default}\" but the server args leave thinking {server} by default "
                f"(off only with --reasoning off or enable_thinking false in {_TEMPLATE_KWARGS})")
    elif clients.thinking == "effort":
        if effort not in (None, "medium"):
            raise RecipeError(f"{where}: thinking = \"effort\" needs --reasoning-effort absent or medium, "
                              f"found --reasoning-effort '{effort}'")
    elif clients.thinking == "always":
        if not (effort == "high" or kwargs.get("reasoning_effort") == "high"
                or flag_value(args, "--reasoning-format") == "deepseek"):
            raise RecipeError(f"{where}: thinking = \"always\" needs --reasoning-effort high, reasoning_effort high "
                              f"in {_TEMPLATE_KWARGS}, or --reasoning-format deepseek")
    else:
        present = [flag for flag in _REASONING_FLAGS if flag_present(args, flag)]
        if present:
            raise RecipeError(f"{where}: thinking = \"none\" allows none of {', '.join(_REASONING_FLAGS)}, "
                              f"found {present[0]}")


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
    files = get_str_list(model_data, "files", model_where)
    if not files or any("/" in name or not name.endswith(".gguf") for name in files):
        raise RecipeError(f"{model_where}: 'files' must list one or more .gguf file names")
    size_bytes = get_int(model_data, "size_bytes", model_where, lo=1, hi=10**13)

    container_where = f"{where} [container]"
    container_data = get_table(data, "container", where)
    check_keys(container_data, _CONTAINER_KEYS, container_where)
    memory_gib = get_int(container_data, "memory_gib", container_where, lo=4, hi=56)

    server_where = f"{where} [server]"
    server_data = get_table(data, "server", where)
    check_keys(server_data, _SERVER_KEYS, server_where)
    context = get_int(server_data, "context", server_where, lo=1024, hi=1048576)
    args = get_str_list(server_data, "args", server_where, required=False)
    _check_args(args, server_where)

    clients_where = f"{where} [clients]"
    clients = parse_clients(get_table(data, "clients", where), clients_where, context=context)
    _check_clients(clients, args, server_where, clients_where)

    evidence = parse_evidence(get_table(data, "evidence", where), f"{where} [evidence]")
    needs_where = f"{where} [needs]"
    needs_data = get_table(data, "needs", where)
    check_keys(needs_data, _NEEDS_KEYS, needs_where)
    gpu_mib = get_int(needs_data, "gpu_mib", needs_where, lo=256, hi=262144)
    ram_gib = get_float(needs_data, "ram_gib", needs_where, lo=0.1, hi=1024, required=False)

    return Recipe(
        runtime=NAME,
        id=recipe_id,
        name=get_str(data, "name", where),
        aliases=aliases,
        doc=get_str(data, "doc", where),
        opencode=get_bool(data, "opencode", where, default=False),
        chat_only=get_bool(data, "chat_only", where, default=False),
        memory_gib=memory_gib,
        context=context,
        args=args,
        needs_gpu_mib=gpu_mib,
        needs_ram_gib=ram_gib,
        model=GgufModel(repo, revision, files, size_bytes),
        evidence=evidence,
        source=source,
        clients=clients,
    )


def model_paths(recipe: Recipe, hf_home: Path) -> list[Path]:
    model = recipe.model
    snapshot = hf_home / "hub" / repo_dirname(model.repo) / "snapshots" / model.revision
    return [snapshot / name for name in model.files]


def file_problems(recipe: Recipe, hf_home: Path) -> list[str]:
    paths = model_paths(recipe, hf_home)
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        return [f"missing {name}" for name in missing]
    total = sum(path.stat().st_size for path in paths)
    if total != recipe.model.size_bytes:
        return [f"the files add up to {total} bytes, the recipe expects {recipe.model.size_bytes}"]
    return []


def served_name(recipe: Recipe, runtime: RuntimeConfig) -> str:
    return runtime.served_name


def container_spec(recipe: Recipe, runtime: RuntimeConfig, env: HostEnv) -> ContainerSpec:
    model = recipe.model
    snapshot = f"{CONTAINER_HF_HOME}/hub/{repo_dirname(model.repo)}/snapshots/{model.revision}"
    args = ("-m", f"{snapshot}/{model.files[0]}", "--alias", runtime.served_name,
            "-c", str(recipe.context)) + recipe.args
    return ContainerSpec(
        user=f"{env.uid}:{env.gid}",
        env=(GRAPHS_OFF,) + runtime.env + (("HF_HOME", CONTAINER_HF_HOME),),
        mounts=(f"{env.hf_home}:{CONTAINER_HF_HOME}",),
        extra_flags=(),
        image=runtime.image,
        args=args,
        container_port=CONTAINER_PORT,
    )


def gpu_need_mib(recipe: Recipe, runtime: RuntimeConfig, gpu: GpuInfo) -> int:
    return recipe.needs_gpu_mib


def gpu_advice(recipe: Recipe) -> str:
    if "--moe-cache-mib" in recipe.args:
        return ("write an override with a smaller --moe-cache-mib, then a higher --n-cpu-moe "
                "(llamacpp/README.md#moe-expert-offload)")
    return "write an override with a smaller context (-c) or a smaller quantization"


def ready(host, runtime: RuntimeConfig, recipe: Recipe) -> bool:
    try:
        response = host.http("GET", f"http://127.0.0.1:{runtime.host_port}{runtime.ready_path}", timeout=2.0)
    except OSError:
        return False
    return response.status == 200


def pull(host, recipe: Recipe, runtime: RuntimeConfig, hf_home: Path, auth: HfAuth, out) -> None:
    """Download the recipe's missing GGUF files into the cache, each checked against its published SHA-256."""
    for path in model_paths(recipe, hf_home):
        if path.is_file():
            out.write(f"{path.name}: already in the cache\n")
            continue
        download_file(host, hf_home, recipe.model.repo, recipe.model.revision, path.name, auth.token, out)
    problems = file_problems(recipe, hf_home)
    if problems:
        raise DownloadError(f"{recipe.id}: " + "; ".join(problems))


def gpu_need_text(recipe: Recipe, runtime: RuntimeConfig) -> str:
    return f"{recipe.needs_gpu_mib} MiB"
