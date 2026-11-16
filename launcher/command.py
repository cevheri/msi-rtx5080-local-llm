"""The rules every model container follows, whatever the runtime or the recipe says.

Ports are published on 127.0.0.1 only, memory is capped with no swap, the GPU is requested through CDI,
and there is no restart policy. The labels record which recipe file, and which version of it, started the container.
"""

from launcher.model import ContainerSpec, Recipe, RuntimeConfig

LABEL_PREFIX = "io.llmctl."
_FORBIDDEN = ("--restart", "--privileged", "--network", "--net", "--rm", "-p", "--publish", "-P",
              "--publish-all", "--memory", "-m", "--memory-swap", "--gpus", "--device", "--cap-add",
              "--security-opt", "--pid", "--userns")


def labels(recipe: Recipe, runtime: RuntimeConfig) -> list[tuple[str, str]]:
    return [
        (LABEL_PREFIX + "managed", "true"),
        (LABEL_PREFIX + "runtime", recipe.runtime),
        (LABEL_PREFIX + "recipe", recipe.id),
        (LABEL_PREFIX + "source", recipe.source.kind),
        (LABEL_PREFIX + "sha256", recipe.source.sha256),
        (LABEL_PREFIX + "evidence", recipe.evidence.kind),
        (LABEL_PREFIX + "runtime-source", runtime.source.kind),
        (LABEL_PREFIX + "runtime-sha256", runtime.source.sha256),
    ]


def docker_run_argv(recipe: Recipe, runtime: RuntimeConfig, spec: ContainerSpec) -> list[str]:
    for flag in spec.extra_flags:
        if flag.split("=", 1)[0] in _FORBIDDEN:
            raise ValueError(f"adapter flag '{flag}' would break a container rule")
    argv = ["docker", "run", "-d", "--name", runtime.container_name]
    for key, value in labels(recipe, runtime):
        argv += ["--label", f"{key}={value}"]
    memory = f"{recipe.memory_gib}g"
    argv += ["--device", "nvidia.com/gpu=all", "--memory", memory, "--memory-swap", memory,
             "-p", f"127.0.0.1:{runtime.host_port}:{spec.container_port}"]
    if spec.user:
        argv += ["--user", spec.user]
    for key, value in spec.env:
        argv += ["-e", f"{key}={value}"]
    for mount in spec.mounts:
        argv += ["-v", mount]
    argv += list(spec.extra_flags)
    argv.append(spec.image)
    argv += list(spec.args)
    return argv
