"""The commands that change containers: up, down, and the read-only status and logs."""

import shlex

from launcher import docker_state
from launcher.catalog import resolve
from launcher.command import LABEL_PREFIX, docker_run_argv
from launcher.errors import DownloadError, PreflightError, StartError, UsageError
from launcher.gpu import gpu_info
from launcher.hfcache import hf_home, read_auth
from launcher.lock import hold_lock
from launcher.model import HostEnv
from launcher.preflight import check_containers, check_port, check_resources, check_static
from launcher.runtimes import ADAPTERS


def describe(container) -> str:
    recipe = container.labels.get(LABEL_PREFIX + "recipe", "?")
    return f"{recipe} in container '{container.name}'"


def _wait_ready(host, recipe, runtime, adapter) -> None:
    deadline = host.monotonic() + runtime.start_timeout_s
    host.out.write(f"waiting for {recipe.id} to load (up to {runtime.start_timeout_s} s)\n")
    while not adapter.ready(host, runtime, recipe):
        status = docker_state.state(host, runtime.container_name)
        if status != "running":
            tail = docker_state.logs_tail(host, runtime.container_name, 40)
            raise StartError(f"the container stopped ({status}) before the server was ready; last log lines:\n{tail}\n"
                             f"the container is kept for inspection: ./llmctl logs, ./llmctl down --remove")
        if host.monotonic() >= deadline:
            tail = docker_state.logs_tail(host, runtime.container_name, 40)
            raise StartError(f"the server was not ready after {runtime.start_timeout_s} s; last log lines:\n{tail}\n"
                             "it is left running for inspection: ./llmctl logs, ./llmctl down")
        host.sleep(1)


def up(host, catalog, name: str, runtime_name: str | None, *, replace: bool, dry_run: bool) -> int:
    recipe, runtime = resolve(catalog, name, runtime_name)
    adapter = ADAPTERS[recipe.runtime]
    home = hf_home(host.env)
    uid, gid = host.uid_gid()
    argv = docker_run_argv(recipe, runtime, adapter.container_spec(recipe, runtime, HostEnv(home, uid, gid)))
    out = host.out
    out.write(f"{recipe.id} on {recipe.runtime}: {recipe.source.kind} recipe, {recipe.evidence.kind}\n")
    if recipe.source.kind == "override":
        out.write(f"recipe file: {recipe.source.path}\n")
    if runtime.source.kind == "override":
        out.write(f"runtime settings from {runtime.source.path}: the OpenCode provider's baseURL must use port "
                  f"{runtime.host_port}, and an image other than the shipped one has no measurement behind it\n")
    if recipe.evidence.kind == "estimate":
        out.write("warning: this recipe was never run on the reference laptop; it may not start or fit\n")
    if dry_run:
        out.write(shlex.join(argv) + "\n")
        return 0
    with hold_lock(host.env):
        check_static(host, recipe, runtime, adapter, home)
        running = check_containers(host, runtime)
        if running and not replace:
            names = ", ".join(describe(container) for container in running)
            raise PreflightError(f"another model server is running: {names}; stop it with ./llmctl down, "
                                 "or start this one with --replace")
        for container in running:
            out.write(f"stopping {describe(container)}\n")
            docker_state.stop(host, container.name)
        check_port(host, runtime)
        existing = docker_state.by_name(host, runtime.container_name)
        if existing is not None:
            out.write(f"removing the stopped container '{existing.name}' from an earlier llmctl run\n")
            docker_state.remove(host, existing.name)
        check_resources(host, recipe, runtime, adapter)
        result = host.run(argv)
        if result.returncode != 0:
            raise StartError(f"docker run failed: {(result.stderr or result.stdout).strip()}")
        try:
            _wait_ready(host, recipe, runtime, adapter)
        except KeyboardInterrupt:
            host.err.write(f"interrupted; the server keeps starting in the background, check ./llmctl status\n")
            return 130
    served = adapter.served_name(recipe, runtime)
    out.write(f"ready: http://127.0.0.1:{runtime.host_port}/v1, served as '{served}', context {recipe.context}\n")
    if recipe.opencode:
        model = f"{runtime.opencode_provider}/{recipe.id}"
        out.write(f"OpenCode model: {model}\n")
        out.write("  in the OpenCode screen: type /models (or press ctrl+x m) and pick it\n")
        out.write(f'  for a single prompt: opencode run --model {model} "your prompt"\n')
    else:
        out.write("OpenCode: no OpenCode entry for this recipe in docs/opencode.md\n")
    if recipe.chat_only:
        out.write("note: chat only, no tool calling\n")
    evidence = recipe.evidence
    where = f"{evidence.date}, {evidence.record}" if evidence.kind == "measured" else evidence.doc
    out.write(f"evidence: {evidence.summary} ({where})\n")
    out.write("stop with ./llmctl down\n")
    return 0


def down(host, *, remove: bool) -> int:
    with hold_lock(host.env):
        containers = docker_state.managed(host)
        if not containers:
            host.out.write("no model server started by llmctl is running\n")
            return 0
        for container in containers:
            if container.state == "running":
                docker_state.stop(host, container.name)
                host.out.write(f"stopped {describe(container)}\n")
            if remove:
                docker_state.remove(host, container.name)
                host.out.write(f"removed container '{container.name}'\n")
    return 0


def status(host, catalog) -> int:
    containers = docker_state.managed(host)
    if not containers:
        host.out.write("no container started by llmctl\n")
        return 0
    for container in containers:
        labels = container.labels
        runtime_name = labels.get(LABEL_PREFIX + "runtime", "?")
        recipe_id = labels.get(LABEL_PREFIX + "recipe", "?")
        host.out.write(f"{recipe_id} on {runtime_name}, container '{container.name}', {container.state}\n")
        host.out.write(f"  recipe: {labels.get(LABEL_PREFIX + 'source', '?')} file, sha256 "
                       f"{labels.get(LABEL_PREFIX + 'sha256', '?')[:12]}, {labels.get(LABEL_PREFIX + 'evidence', '?')}\n")
        if container.state != "running":
            continue
        port = host.run(["docker", "port", container.name]).stdout.strip()
        entry = catalog.runtimes.get(runtime_name)
        recipe = entry.recipes.get(recipe_id) if entry else None
        health = "unknown"
        if entry and entry.runtime and recipe:
            health = "ready" if ADAPTERS[runtime_name].ready(host, entry.runtime, recipe) else "starting"
        memory = host.run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container.name]).stdout.strip()
        gpu = gpu_info(host)
        host.out.write(f"  port: {port or 'none'}, health: {health}\n")
        host.out.write(f"  container memory: {memory}, GPU memory in use: {gpu.used_mib} MiB of {gpu.total_mib} MiB\n")
    return 0


def logs(host, *, follow: bool, tail: int) -> int:
    containers = docker_state.managed(host)
    if not containers:
        host.out.write("no container started by llmctl\n")
        return 0
    chosen = next((c for c in containers if c.state == "running"), containers[0])
    argv = ["docker", "logs", "--tail", str(tail)] + (["-f"] if follow else []) + [chosen.name]
    return host.run_attached(argv)


def pull(host, catalog, name: str, runtime_name: str | None, *, yes: bool, env_file) -> int:
    recipe, runtime = resolve(catalog, name, runtime_name)
    adapter = ADAPTERS[recipe.runtime]
    home = hf_home(host.env)
    auth = read_auth(host.env, env_file)
    files_needed = bool(adapter.file_problems(recipe, home))
    image_needed = not docker_state.image_present(host, runtime.image)
    out = host.out
    if not files_needed and not image_needed:
        out.write(f"{recipe.id}: already pulled; start it with ./llmctl up {recipe.id}\n")
        return 0
    parts = []
    if files_needed:
        parts.append(f"{recipe.model.size_bytes / 1e9:.1f} GB of model files into {home}")
    if image_needed:
        parts.append(f"the image {runtime.image} (several GB)")
    out.write(f"{recipe.id}: {' and '.join(parts)} to download\n")
    if not yes:
        if not host.isatty():
            raise UsageError("pull downloads a lot, so it asks first; run it in a terminal or pass --yes")
        if host.ask("Download? [y/N] ").strip().lower() not in ("y", "yes"):
            out.write("aborted, nothing downloaded\n")
            return 1
    with hold_lock(host.env):
        # Created as the calling user, before a bind mount can make Docker create it as root.
        (home / "hub").mkdir(parents=True, exist_ok=True)
        if image_needed:
            docker_state.docker_ok(host)
            if host.run_attached(["docker", "pull", runtime.image]) != 0:
                raise DownloadError(f"docker pull {runtime.image} failed")
        if files_needed:
            adapter.pull(host, recipe, runtime, home, auth, out)
    out.write(f"ready: ./llmctl up {recipe.id}{'' if recipe.runtime == 'llamacpp' else ' --runtime ' + recipe.runtime}\n")
    return 0
