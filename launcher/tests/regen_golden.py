"""The exact `docker run` argument list each shipped recipe produces, kept under launcher/tests/golden/, and the
client entries generated from it (the Hermes block and Pi's models.json), kept under launcher/tests/golden/clients/.

Rewrite them on purpose, and read the diff before keeping it:
    python3 -m launcher.tests.regen_golden
"""

import json
from pathlib import Path

from launcher import hermes_config, pi_config
from launcher.catalog import load_catalog
from launcher.command import docker_run_argv
from launcher.model import HostEnv, Target
from launcher.runtimes import ADAPTERS

REPO = Path(__file__).resolve().parents[2]
GOLDEN = Path(__file__).resolve().parent / "golden"
CLIENT_GOLDEN = GOLDEN / "clients"
ENV = HostEnv(Path("/home/user/.cache/huggingface"), 1000, 1000)


def golden_argv(recipe_id: str, runtime: str, recipes_dir: Path) -> list[str]:
    entry = load_catalog(recipes_dir, None).runtimes[runtime]
    recipe = entry.recipes[recipe_id]
    spec = ADAPTERS[runtime].container_spec(recipe, entry.runtime, ENV)
    return docker_run_argv(recipe, entry.runtime, spec)


def target_for(recipe_id: str, runtime: str, recipes_dir: Path) -> Target:
    entry = load_catalog(recipes_dir, None).runtimes[runtime]
    recipe = entry.recipes[recipe_id]
    adapter = ADAPTERS[runtime]
    return Target(recipe, entry.runtime, adapter.served_name(recipe, entry.runtime), adapter.LABEL)


def main() -> None:
    catalog = load_catalog(REPO / "recipes", None)
    for runtime, entry in catalog.runtimes.items():
        problems = [problem.message for problem in entry.problems]
        if problems:
            raise SystemExit(f"{runtime} recipes have problems, fix them first:\n" + "\n".join(problems))
        directory = GOLDEN / runtime
        directory.mkdir(parents=True, exist_ok=True)
        for old in directory.glob("*.json"):
            old.unlink()
        for recipe_id in sorted(entry.recipes):
            argv = golden_argv(recipe_id, runtime, REPO / "recipes")
            (directory / f"{recipe_id}.json").write_text(json.dumps(argv, indent=1) + "\n")
        print(f"{runtime}: {len(entry.recipes)} golden files")
        clients = CLIENT_GOLDEN / runtime
        clients.mkdir(parents=True, exist_ok=True)
        for old in [*clients.glob("*.hermes.yaml"), *clients.glob("*.pi.json")]:
            old.unlink()
        for recipe_id in sorted(entry.recipes):
            target = target_for(recipe_id, runtime, REPO / "recipes")
            (clients / f"{recipe_id}.hermes.yaml").write_text(hermes_config.render_block(target))
            (clients / f"{recipe_id}.pi.json").write_text(pi_config.new_models_text(target))
        print(f"{runtime}: {len(entry.recipes)} client golden files of each kind")


if __name__ == "__main__":
    main()
