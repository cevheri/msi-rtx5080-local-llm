"""Loading the shipped recipes and the user's overrides, one runtime at a time, and finding a recipe by name.

An override file replaces the shipped file of the same name whole; nothing is merged across files.
A file that does not read becomes a Problem, and the recipe it names refuses to resolve until it is fixed.
"""

import difflib
from dataclasses import dataclass, field
from pathlib import Path

from launcher.errors import LlmctlError, RecipeError, UsageError
from launcher.model import Problem, Recipe, RuntimeConfig
from launcher.runtimes import ADAPTERS
from launcher.validate import read_toml

RUNTIME_FILE = "_runtime.toml"
_AMBIGUOUS = object()


@dataclass
class RuntimeCatalog:
    runtime: RuntimeConfig | None
    recipes: dict[str, Recipe] = field(default_factory=dict)
    problems: list[Problem] = field(default_factory=list)
    overridden: frozenset[str] = frozenset()
    broken: dict[str, Problem] = field(default_factory=dict)
    names: dict[str, object] = field(default_factory=dict)  # lower-case ID or alias -> ID, or _AMBIGUOUS
    claims: dict[str, list[str]] = field(default_factory=dict)  # lower-case alias -> IDs that use it


@dataclass
class Catalog:
    runtimes: dict[str, RuntimeCatalog]


def override_dir(env) -> Path:
    if env.get("LLMCTL_RECIPES_DIR"):
        return Path(env["LLMCTL_RECIPES_DIR"])
    if env.get("XDG_CONFIG_HOME"):
        return Path(env["XDG_CONFIG_HOME"]) / "llmctl" / "recipes"
    return Path(env.get("HOME", "~")).expanduser() / ".config" / "llmctl" / "recipes"


def _toml_files(directory: Path | None) -> dict[str, Path]:
    if directory is None or not directory.is_dir():
        return {}
    return {path.name: path for path in sorted(directory.glob("*.toml")) if path.is_file()}


def _load_runtime(name: str, adapter, shipped: Path, overrides: Path | None) -> RuntimeCatalog:
    shipped_files = _toml_files(shipped / name)
    override_files = _toml_files(overrides / name if overrides else None)
    catalog = RuntimeCatalog(runtime=None)

    runtime_path = override_files.get(RUNTIME_FILE) or shipped_files.get(RUNTIME_FILE)
    if runtime_path is None:
        catalog.problems.append(Problem(name, shipped / name / RUNTIME_FILE, None, f"{shipped / name / RUNTIME_FILE}: missing"))
    else:
        kind = "override" if RUNTIME_FILE in override_files else "shipped"
        try:
            data, source = read_toml(runtime_path, kind)
            catalog.runtime = adapter.parse_runtime(data, source)
        except LlmctlError as exc:
            catalog.problems.append(Problem(name, runtime_path, None, str(exc)))

    def parse(path: Path, kind: str) -> Recipe:
        data, source = read_toml(path, kind)
        return adapter.parse_recipe(data, source, path.stem)

    shipped_recipes: dict[str, Recipe] = {}
    for file_name, path in shipped_files.items():
        if file_name == RUNTIME_FILE:
            continue
        try:
            shipped_recipes[path.stem] = parse(path, "shipped")
        except LlmctlError as exc:
            problem = Problem(name, path, path.stem, str(exc))
            catalog.problems.append(problem)
            catalog.broken[path.stem] = problem

    recipes = dict(shipped_recipes)
    overridden = set()
    for file_name, path in override_files.items():
        if file_name == RUNTIME_FILE:
            continue
        stem = path.stem
        recipes.pop(stem, None)
        try:
            recipes[stem] = parse(path, "override")
            overridden.add(stem)
            catalog.broken.pop(stem, None)
        except LlmctlError as exc:
            problem = Problem(name, path, stem, str(exc))
            catalog.problems = [p for p in catalog.problems if p.recipe_id != stem] + [problem]
            catalog.broken[stem] = problem
            # The shipped aliases still lead to this model, so that resolving one reports the broken override.
            for alias in shipped_recipes[stem].aliases if stem in shipped_recipes else ():
                catalog.names.setdefault(alias.lower(), stem)
    catalog.recipes = recipes
    catalog.overridden = frozenset(overridden)

    for stem in catalog.broken:
        catalog.names[stem.lower()] = stem
    for recipe in recipes.values():
        catalog.names[recipe.id.lower()] = recipe.id
    for recipe in recipes.values():
        for alias in recipe.aliases:
            catalog.claims.setdefault(alias.lower(), []).append(recipe.id)
    for alias, ids in catalog.claims.items():
        if len(ids) > 1 or catalog.names.get(alias, ids[0]) != ids[0]:
            users = sorted(set(ids) | ({catalog.names[alias]} if alias in catalog.names else set()))
            catalog.names[alias] = _AMBIGUOUS
            catalog.claims[alias] = users
            for recipe_id in users:
                path = recipes[recipe_id].source.path if recipe_id in recipes else None
                catalog.problems.append(Problem(name, path, recipe_id, f"alias '{alias}' is used by {', '.join(users)}"))
        else:
            catalog.names[alias] = ids[0]
    return catalog


def load_catalog(shipped: Path, overrides: Path | None) -> Catalog:
    return Catalog({name: _load_runtime(name, adapter, shipped, overrides) for name, adapter in ADAPTERS.items()})


def resolve(catalog: Catalog, name: str, runtime: str | None) -> tuple[Recipe, RuntimeConfig]:
    if runtime is not None and runtime not in ADAPTERS:
        raise UsageError(f"unknown runtime '{runtime}'; choose {' or '.join(ADAPTERS)}")
    key = name.lower()
    searched = [runtime] if runtime else list(ADAPTERS)
    for runtime_name in searched:
        entry = catalog.runtimes[runtime_name]
        target = entry.names.get(key)
        if target is None:
            continue
        if target is _AMBIGUOUS:
            raise RecipeError(f"{runtime_name}: alias '{key}' is used by {', '.join(entry.claims[key])}; use the ID instead")
        if target in entry.broken:
            problem = entry.broken[target]
            raise RecipeError(f"{problem.message}\nfix or remove {problem.path} to use '{target}'")
        if entry.runtime is None:
            problem = next(p for p in entry.problems if p.recipe_id is None)
            raise RecipeError(f"{problem.message}\nfix or remove {problem.path} to use {runtime_name}")
        return entry.recipes[target], entry.runtime
    known = sorted({k for r in searched for k, v in catalog.runtimes[r].names.items() if v is not _AMBIGUOUS})
    close = difflib.get_close_matches(key, known, n=3, cutoff=0.5)
    hint = f"; did you mean {', '.join(close)}?" if close else ""
    raise UsageError(f"unknown model '{name}'{hint} (./llmctl list shows every recipe)")
