"""A change to one runtime's files must not change any command generated for the other runtime."""

import ast
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path

from launcher import dsh_config, hermes_config, kimi_config, pi_config
from launcher.catalog import load_catalog
from launcher.tests.fakes import REPO
from launcher.tests.regen_golden import CLIENT_GOLDEN, GOLDEN, golden_argv, target_for

CLIENT_MODULES = ["files.py", "hermes_config.py", "pi_config.py", "opencode_config.py", "clients.py", "dsh_config.py",
                  "kimi_config.py"]
ADAPTERS = ["runtimes/llamacpp.py", "runtimes/vllm.py", "runtimes/sglang.py"]
CLIENT_IMPORTS = {"launcher.files", "launcher.clients", "launcher.hermes_config", "launcher.pi_config",
                  "launcher.opencode_config", "launcher.dsh_config", "launcher.kimi_config"}


def edit_runtime(directory: Path, port: int, extra_arg: str) -> None:
    """Change the runtime file and every recipe of one runtime."""
    runtime = directory / "_runtime.toml"
    text = runtime.read_text()
    text = re.sub(r'(?m)^image = "([^"]+):[^"]+"$', r'image = "\1:edited"', text)
    text = re.sub(r"(?m)^host_port = \d+$", f"host_port = {port}", text)
    text = re.sub(r"(?m)^start_timeout_s = \d+$", "start_timeout_s = 42", text)
    runtime.write_text(text)
    for recipe in directory.glob("*.toml"):
        if recipe.name == "_runtime.toml":
            continue
        text = recipe.read_text()
        assert "args = [\n" in text, recipe
        text = text.replace("args = [\n", f'args = [\n  "{extra_arg}",\n', 1)
        text = re.sub(r"(?m)^memory_gib = \d+$", "memory_gib = 7", text)
        text = re.sub(r"(?m)^output = \d+$", "output = 12288", text)
        recipe.write_text(text)


class IsolationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.recipes = Path(self.tmp.name) / "recipes"
        shutil.copytree(REPO / "recipes", self.recipes)

    def tearDown(self):
        self.tmp.cleanup()

    def assert_runtime_untouched(self, edited: str, untouched: str) -> None:
        catalog = load_catalog(self.recipes, None)
        self.assertEqual([p.message for p in catalog.runtimes[edited].problems], [])
        changed = 0
        for recipe_id in catalog.runtimes[edited].recipes:
            expected = json.loads((GOLDEN / edited / f"{recipe_id}.json").read_text())
            changed += golden_argv(recipe_id, edited, self.recipes) != expected
        self.assertEqual(changed, len(catalog.runtimes[edited].recipes), "the edit did not reach every recipe")
        for recipe_id in catalog.runtimes[untouched].recipes:
            expected = json.loads((GOLDEN / untouched / f"{recipe_id}.json").read_text())
            self.assertEqual(golden_argv(recipe_id, untouched, self.recipes), expected, f"{untouched}/{recipe_id}")
        for recipe_id in catalog.runtimes[edited].recipes:
            texts = self.client_texts(recipe_id, edited)
            self.assertNotEqual(texts[0], (CLIENT_GOLDEN / edited / f"{recipe_id}.hermes.yaml").read_text(),
                                f"the edit did not reach the Hermes block of {edited}/{recipe_id}")
            self.assertNotEqual(texts[1], (CLIENT_GOLDEN / edited / f"{recipe_id}.pi.json").read_text(),
                                f"the edit did not reach the Pi entry of {edited}/{recipe_id}")
            self.assertNotEqual(texts[2], (CLIENT_GOLDEN / edited / f"{recipe_id}.dsh.yml").read_text(),
                                f"the edit did not reach the dsh block of {edited}/{recipe_id}")
            self.assertNotEqual(texts[3], (CLIENT_GOLDEN / edited / f"{recipe_id}.kimi.toml").read_text(),
                                f"the edit did not reach the Kimi tables of {edited}/{recipe_id}")
        for recipe_id in catalog.runtimes[untouched].recipes:
            texts = self.client_texts(recipe_id, untouched)
            self.assertEqual(texts[0], (CLIENT_GOLDEN / untouched / f"{recipe_id}.hermes.yaml").read_text(),
                             f"{untouched}/{recipe_id}.hermes.yaml")
            self.assertEqual(texts[1], (CLIENT_GOLDEN / untouched / f"{recipe_id}.pi.json").read_text(),
                             f"{untouched}/{recipe_id}.pi.json")
            self.assertEqual(texts[2], (CLIENT_GOLDEN / untouched / f"{recipe_id}.dsh.yml").read_text(),
                             f"{untouched}/{recipe_id}.dsh.yml")
            self.assertEqual(texts[3], (CLIENT_GOLDEN / untouched / f"{recipe_id}.kimi.toml").read_text(),
                             f"{untouched}/{recipe_id}.kimi.toml")

    def client_texts(self, recipe_id: str, runtime: str) -> tuple[str, str, str, str]:
        target = target_for(recipe_id, runtime, self.recipes)
        return (hermes_config.render_block(target), pi_config.new_models_text(target),
                dsh_config.render_block(target, default=True), kimi_config.render_tables(target))

    def test_llamacpp_edits_leave_vllm_alone(self):
        edit_runtime(self.recipes / "llamacpp", 8090, "--metrics")
        self.assert_runtime_untouched("llamacpp", "vllm")

    def test_vllm_edits_leave_llamacpp_alone(self):
        edit_runtime(self.recipes / "vllm", 8001, "--disable-log-requests")
        self.assert_runtime_untouched("vllm", "llamacpp")


def parse_module(name: str) -> ast.Module:
    path = REPO / "launcher" / name
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def imported_modules(tree: ast.Module) -> list[str]:
    """Every module an import names, with `from launcher import x` as `launcher.x`."""
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
            names += [f"{node.module}.{alias.name}" for alias in node.names]
    return names


class ClientBoundaryTest(unittest.TestCase):
    def test_client_modules_read_no_process_environment(self):
        for name in CLIENT_MODULES:
            with self.subTest(module=name):
                found = []
                for node in ast.walk(parse_module(name)):
                    if isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv", "expanduser"):
                        found.append(f"line {node.lineno}: .{node.attr}")
                    elif (isinstance(node, ast.Attribute) and node.attr == "home"
                          and isinstance(node.value, ast.Name) and node.value.id == "Path"):
                        found.append(f"line {node.lineno}: Path.home")
                    elif isinstance(node, ast.ImportFrom) and node.module in ("os", "os.path"):
                        for alias in node.names:
                            if alias.name in ("environ", "getenv", "expanduser"):
                                found.append(f"line {node.lineno}: from {node.module} import {alias.name}")
                self.assertEqual(found, [])

    def test_adapters_and_client_modules_stay_apart(self):
        for name in CLIENT_MODULES:
            with self.subTest(module=name):
                bad = [m for m in imported_modules(parse_module(name))
                       if m == "launcher.runtimes" or m.startswith("launcher.runtimes.")]
                self.assertEqual(bad, [])
        for name in ADAPTERS:
            with self.subTest(module=name):
                bad = [m for m in imported_modules(parse_module(name))
                       if m.startswith("launcher.runtimes.") or m in CLIENT_IMPORTS]
                self.assertEqual(bad, [])

    def test_dsh_config_imports_no_other_client_module(self):
        others = {"launcher.clients", "launcher.hermes_config", "launcher.pi_config", "launcher.opencode_config"}
        bad = [m for m in imported_modules(parse_module("dsh_config.py"))
               if m in others or m == "launcher.runtimes" or m.startswith("launcher.runtimes.")]
        self.assertEqual(bad, [])

    def test_kimi_config_imports_no_other_client_module(self):
        others = {"launcher.clients", "launcher.hermes_config", "launcher.pi_config", "launcher.opencode_config",
                  "launcher.dsh_config"}
        bad = [m for m in imported_modules(parse_module("kimi_config.py"))
               if m in others or m == "launcher.runtimes" or m.startswith("launcher.runtimes.")]
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
