"""The commands that change containers: up, down, and the read-only status and logs."""

import shlex

from launcher import clients, docker_state
from launcher.catalog import cache_dir, resolve
from launcher.command import LABEL_PREFIX, docker_run_argv
from launcher.errors import ClientWriteError, DownloadError, PreflightError, StartError, UsageError
from launcher.gpu import gpu_info
from launcher.hfcache import hf_home, read_auth
from launcher.lock import hold_lock
from launcher.model import HostEnv, Recipe, RuntimeConfig, Target
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


def _same_recipe(running, recipe, runtime) -> bool:
    """True when the one server running was started from this very recipe and runtime file."""
    if len(running) != 1:
        return False
    labels = running[0].labels
    return (labels.get(LABEL_PREFIX + "runtime") == recipe.runtime and labels.get(LABEL_PREFIX + "recipe") == recipe.id
            and labels.get(LABEL_PREFIX + "sha256") == recipe.source.sha256
            and labels.get(LABEL_PREFIX + "runtime-sha256") == runtime.source.sha256)


def _start(host, recipe, runtime, adapter, argv, running, replace, env: HostEnv) -> None:
    out = host.out
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
    prepare_start = getattr(adapter, "prepare_start", None)
    if prepare_start is not None:
        for line in prepare_start(recipe, runtime, env, dry_run=False):
            out.write(line + "\n")
    result = host.run(argv)
    if result.returncode != 0:
        raise StartError(f"docker run failed: {(result.stderr or result.stdout).strip()}")


def _apply_check(host, runtime, check) -> int | None:
    """Print the post-ready check's lines; a refusal stops the container, keeps it, and fails the start.

    Returns 130 when an interrupt lands during the refusal's stop, after one stderr line that keeps the refusal."""
    for line in check.lines:
        host.out.write(line + "\n")
    if check.refusal is None:
        return None
    try:
        docker_state.stop(host, runtime.container_name)
    except KeyboardInterrupt:
        host.err.write(f"{check.refusal}; the stop was interrupted, so the container may still be running: "
                       "./llmctl status, ./llmctl down\n")
        return 130
    except PreflightError as exc:
        raise StartError(f"{check.refusal}; llmctl could not stop the container ({exc}), so stop it with "
                         "./llmctl down") from None
    raise StartError(f"{check.refusal}; llmctl stopped the container and kept it for its logs: ./llmctl logs, "
                     "./llmctl down --remove")


def client_target(recipe: Recipe, runtime: RuntimeConfig) -> Target:
    adapter = ADAPTERS[recipe.runtime]
    return Target(recipe, runtime, adapter.served_name(recipe, runtime), adapter.LABEL)


def target_from(catalog, containers) -> Target | None:
    """The Target of the first running container whose runtime and recipe labels name a recipe in the catalog."""
    for container in containers:
        if container.state != "running":
            continue
        entry = catalog.runtimes.get(container.labels.get(LABEL_PREFIX + "runtime"))
        recipe = entry.recipes.get(container.labels.get(LABEL_PREFIX + "recipe")) if entry else None
        if recipe is not None and entry.runtime is not None:
            return client_target(recipe, entry.runtime)
    return None


def running_target(host, catalog) -> Target | None:
    """The model llmctl is running, as a client Target; None when no managed container runs a catalog recipe."""
    return target_from(catalog, docker_state.managed(host))


def up_command(recipe: Recipe) -> str:
    """The up command for this recipe; --runtime only when it is not the default runtime."""
    command = f"./llmctl up {recipe.id}"
    return command if recipe.runtime == next(iter(ADAPTERS)) else f"{command} --runtime {recipe.runtime}"


def up(host, catalog, name: str, runtime_name: str | None, *, replace: bool, dry_run: bool,
       set_default: bool = False) -> int:
    recipe, runtime = resolve(catalog, name, runtime_name)
    target = client_target(recipe, runtime)
    # Every client refusal comes before any check or start (spec D8); only a real start lets Pi check the entry.
    plan = clients.plan(host, catalog, target, set_default=set_default, check=not dry_run)
    adapter = ADAPTERS[recipe.runtime]
    home = hf_home(host.env)
    uid, gid = host.uid_gid()
    env = HostEnv(home, uid, gid, cache_dir(host.env))
    argv = docker_run_argv(recipe, runtime, adapter.container_spec(recipe, runtime, env))
    prepare_start = getattr(adapter, "prepare_start", None)
    check_started = getattr(adapter, "check_started", None)
    out = host.out
    out.write(f"{recipe.id} on {recipe.runtime}: {recipe.source.kind} recipe, {recipe.evidence.kind}\n")
    if recipe.source.kind == "override":
        out.write(f"recipe file: {recipe.source.path}\n")
    if runtime.source.kind == "override":
        out.write(f"runtime settings from {runtime.source.path}: the OpenCode provider's baseURL must use port "
                  f"{runtime.host_port}, and an image other than the shipped one has no measurement behind it\n")
    if recipe.evidence.kind == "estimate":
        out.write("warning: this recipe has no measurement on the reference laptop; it may not start or fit\n")
    out.write(clients.clients_line(host, list(plan.states)) + "\n")
    for line in plan.warnings:
        out.write(line + "\n")
    if dry_run:
        for line in prepare_start(recipe, runtime, env, dry_run=True) if prepare_start else ():
            out.write(line + "\n")
        out.write(shlex.join(argv) + "\n")
        for line in clients.dry_run_lines(plan):
            out.write(line + "\n")
        return 0
    with hold_lock(host.env):
        check_static(host, recipe, runtime, adapter, home)
        running = check_containers(host, runtime)
        started = not (_same_recipe(running, recipe, runtime) and not replace)
        if not started:
            out.write(f"{recipe.id} is already running from this recipe file; nothing to start\n")
            if check_started is not None:
                out.write(f"note: {recipe.id} was already running from this recipe file, so llmctl did not check "
                          f"its memory again; {up_command(recipe)} --replace restarts it and checks it\n")
        else:
            _start(host, recipe, runtime, adapter, argv, running, replace, env)
        try:
            _wait_ready(host, recipe, runtime, adapter)
        except KeyboardInterrupt:
            host.err.write(f"interrupted; the server keeps starting in the background, check ./llmctl status\n")
            return 130
        if started and check_started is not None:
            try:
                check = check_started(host, runtime, recipe)
            except KeyboardInterrupt:
                host.err.write("interrupted; the server is running, but llmctl did not check its memory or write the "
                               f"clients; {up_command(recipe)} --replace restarts it and checks it\n")
                return 130
            interrupted = _apply_check(host, runtime, check)
            if interrupted is not None:
                return interrupted
        out.write(f"ready: http://127.0.0.1:{runtime.host_port}/v1, served as '{target.served}', "
                  f"context {recipe.context}, output {recipe.clients.output}\n")
        result = clients.apply(host, catalog, target, set_default=set_default, dsh_checked=plan.dsh_checked,
                               kimi_checked=plan.kimi_checked)
        for line in result.lines:
            out.write(line + "\n")
    for line in clients.up_lines(host, target, plan.states, up_command=up_command(recipe), changed=result.changed,
                                 failed=result.failed, default_failed=result.default_failed, dsh_old=result.dsh_old,
                                 kimi_old=result.kimi_old):
        out.write(line + "\n")
    if recipe.chat_only:
        out.write("note: chat only, no tool calling\n")
    evidence = recipe.evidence
    where = f"{evidence.date}, {evidence.record}" if evidence.kind == "measured" else evidence.doc
    out.write(f"evidence: {evidence.summary} ({where})\n")
    out.write("stop with ./llmctl down\n")
    if result.failures:
        raise ClientWriteError(clients.failure_message(result.failures))
    return 0


def down(host, catalog, *, remove: bool) -> int:
    with hold_lock(host.env):
        containers = docker_state.managed(host)
        if not containers:
            host.out.write("no model server started by llmctl is running\n")
            return 0
        stopped = next((c for c in containers if c.state == "running"), None)
        for container in containers:
            if container.state == "running":
                docker_state.stop(host, container.name)
                host.out.write(f"stopped {describe(container)}\n")
            if remove:
                docker_state.remove(host, container.name)
                host.out.write(f"removed container '{container.name}'\n")
        # Read after the stop, so a bad LLMCTL_CLIENTS cannot keep the GPU busy; no client file changes (D6 rule 4).
        lines = clients.down_lines(host, catalog, stopped.labels.get(LABEL_PREFIX + "recipe", "?")) if stopped else []
        for line in lines:
            host.out.write(line + "\n")
    return 0


def status(host, catalog) -> int:
    containers = docker_state.managed(host)
    if not containers:
        host.out.write("no container started by llmctl\n")
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
    target = target_from(catalog, containers)
    # A running container whose recipe the catalog lacks is still a running model, so no "no model is running".
    unknown_running = target is None and any(container.state == "running" for container in containers)
    for line in clients.report(host, catalog, target, with_paths=True, running_known=not unknown_running):
        host.out.write(line + "\n")
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
        out.write(f"{recipe.id}: already pulled; start it with {up_command(recipe)}\n")
        return 0
    parts = []
    if files_needed:
        parts.append(f"{recipe.model.size_bytes / 1e9:.1f} GB of model files into {home}")
    if image_needed:
        parts.append(f"the image {runtime.image} (see docs/host-setup.md#disk-budget for its size)")
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
    out.write(f"ready: {up_command(recipe)}\n")
    return 0
