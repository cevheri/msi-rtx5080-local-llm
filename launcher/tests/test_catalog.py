import shutil
import tempfile
import unittest
from pathlib import Path

from launcher.catalog import load_catalog, override_dir, resolve
from launcher.command import docker_run_argv
from launcher.errors import RecipeError, UsageError
from launcher.model import HostEnv
from launcher.runtimes import ADAPTERS
from launcher.tests.fakes import REPO

ENV = HostEnv(Path("/home/user/.cache/huggingface"), 1000, 1000)
QWEN_FILE = "qwen3.6-35b-a3b.toml"


def argv_for(catalog, recipe_id, runtime):
    recipe, config = resolve(catalog, recipe_id, runtime)
    spec = ADAPTERS[runtime].container_spec(recipe, config, ENV)
    return docker_run_argv(recipe, config, spec)


class CatalogTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.shipped = root / "shipped"
        shutil.copytree(REPO / "recipes", self.shipped)
        self.overrides = root / "overrides"

    def tearDown(self):
        self.tmp.cleanup()

    def write_override(self, runtime, name, text):
        path = self.overrides / runtime / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def shipped_text(self, runtime, name):
        return (self.shipped / runtime / name).read_text()

    def test_alias_resolves(self):
        recipe, runtime = resolve(load_catalog(self.shipped, None), "QWEN3.6", None)
        self.assertEqual((recipe.id, recipe.runtime, runtime.name), ("qwen3.6-35b-a3b", "llamacpp", "llamacpp"))

    def test_default_runtime_is_llamacpp(self):
        text = self.shipped_text("vllm", "gpt-oss-20b.toml")
        llama_twin = self.shipped_text("llamacpp", QWEN_FILE).replace('id = "qwen3.6-35b-a3b"', 'id = "gpt-oss-20b"')
        llama_twin = llama_twin.replace('aliases = ["qwen3.6"]', "aliases = []")
        (self.shipped / "llamacpp" / "gpt-oss-20b.toml").write_text(llama_twin)
        catalog = load_catalog(self.shipped, None)
        self.assertEqual(resolve(catalog, "gpt-oss-20b", None)[0].runtime, "llamacpp")
        self.assertEqual(resolve(catalog, "gpt-oss-20b", "vllm")[0].runtime, "vllm")
        self.assertIn("gpt-oss-20b", text)

    def test_unknown_runtime(self):
        with self.assertRaises(UsageError) as ctx:
            resolve(load_catalog(self.shipped, None), "qwen3.6", "ollama")
        self.assertIn("unknown runtime 'ollama'", str(ctx.exception))

    def test_unknown_name_suggests(self):
        with self.assertRaises(UsageError) as ctx:
            resolve(load_catalog(self.shipped, None), "qwen3.7", None)
        self.assertIn("unknown model 'qwen3.7'", str(ctx.exception))
        self.assertIn("qwen3.6", str(ctx.exception))

    def test_override_replaces_whole(self):
        text = self.shipped_text("llamacpp", QWEN_FILE)
        text = text.replace('"--moe-cache-mib", "3200", ', "").replace('kind = "measured"', 'kind = "estimate"\ndoc = "mine"')
        text = "\n".join(line for line in text.splitlines()
                         if not line.startswith(("date =", "hardware =", "record =")))
        self.write_override("llamacpp", QWEN_FILE, text)
        catalog = load_catalog(self.shipped, self.overrides)
        recipe, _ = resolve(catalog, "qwen3.6", None)
        self.assertNotIn("--moe-cache-mib", recipe.args)
        self.assertEqual((recipe.source.kind, recipe.evidence.kind), ("override", "estimate"))
        self.assertIn("qwen3.6-35b-a3b", catalog.runtimes["llamacpp"].overridden)

    def test_override_adds_model(self):
        text = self.shipped_text("llamacpp", QWEN_FILE).replace('id = "qwen3.6-35b-a3b"', 'id = "my-model"')
        text = text.replace('aliases = ["qwen3.6"]', 'aliases = ["mine"]')
        self.write_override("llamacpp", "my-model.toml", text)
        catalog = load_catalog(self.shipped, self.overrides)
        self.assertEqual(resolve(catalog, "mine", None)[0].id, "my-model")
        self.assertEqual(resolve(catalog, "qwen3.6", None)[0].source.kind, "shipped")

    def test_broken_override_blocks_model(self):
        text = self.shipped_text("llamacpp", QWEN_FILE).replace("memory_gib = 20", "memmory_gib = 20")
        path = self.write_override("llamacpp", QWEN_FILE, text)
        catalog = load_catalog(self.shipped, self.overrides)
        for name in ("qwen3.6", "qwen3.6-35b-a3b"):
            with self.assertRaises(RecipeError) as ctx:
                resolve(catalog, name, None)
            self.assertIn(str(path), str(ctx.exception))
            self.assertIn("memmory_gib", str(ctx.exception))
        self.assertEqual(resolve(catalog, "gpt-oss-20b", "vllm")[0].id, "gpt-oss-20b")
        self.assertEqual([p.recipe_id for p in catalog.runtimes["llamacpp"].problems], ["qwen3.6-35b-a3b"])

    def test_duplicate_alias_is_reported_and_ambiguous(self):
        text = self.shipped_text("llamacpp", QWEN_FILE).replace('id = "qwen3.6-35b-a3b"', 'id = "other"')
        self.write_override("llamacpp", "other.toml", text)
        catalog = load_catalog(self.shipped, self.overrides)
        with self.assertRaises(RecipeError) as ctx:
            resolve(catalog, "qwen3.6", None)
        self.assertIn("'qwen3.6' is used by", str(ctx.exception))
        self.assertEqual(resolve(catalog, "other", None)[0].id, "other")
        self.assertTrue(any("qwen3.6" in p.message for p in catalog.runtimes["llamacpp"].problems))

    def test_runtime_override_changes_only_its_runtime(self):
        before_vllm = argv_for(load_catalog(self.shipped, None), "gpt-oss-20b", "vllm")
        text = self.shipped_text("llamacpp", "_runtime.toml").replace("host_port = 8080", "host_port = 8081")
        self.write_override("llamacpp", "_runtime.toml", text)
        catalog = load_catalog(self.shipped, self.overrides)
        llama = argv_for(catalog, "qwen3.6", "llamacpp")
        self.assertIn("127.0.0.1:8081:8080", llama)
        self.assertIn("io.llmctl.runtime-source=override", llama)
        self.assertEqual(argv_for(catalog, "gpt-oss-20b", "vllm"), before_vllm)

    def test_runtime_override_cannot_unbind(self):
        for extra in ('bind_address = "0.0.0.0"', 'restart = "always"'):
            text = extra + "\n" + self.shipped_text("llamacpp", "_runtime.toml")  # top level, before [env]
            path = self.write_override("llamacpp", "_runtime.toml", text)
            catalog = load_catalog(self.shipped, self.overrides)
            problems = catalog.runtimes["llamacpp"].problems
            self.assertEqual(len(problems), 1)
            self.assertIn(extra.split(" ")[0], problems[0].message)
            with self.assertRaises(RecipeError) as ctx:
                resolve(catalog, "qwen3.6", None)
            self.assertIn(str(path), str(ctx.exception))

    def test_example_override_is_valid(self):
        catalog = load_catalog(self.shipped, REPO / "recipes" / "examples")
        self.assertEqual([p for rc in catalog.runtimes.values() for p in rc.problems], [])
        recipe, _ = resolve(catalog, "qwen3.6", None)
        self.assertEqual((recipe.source.kind, recipe.memory_gib), ("override", 24))

    def test_shipped_recipes_have_no_problems(self):
        catalog = load_catalog(REPO / "recipes", None)
        self.assertEqual([p for rc in catalog.runtimes.values() for p in rc.problems], [])


class OverrideDirTest(unittest.TestCase):
    def test_override_dir_env(self):
        self.assertEqual(override_dir({"LLMCTL_RECIPES_DIR": "/x", "XDG_CONFIG_HOME": "/c", "HOME": "/h"}), Path("/x"))
        self.assertEqual(override_dir({"XDG_CONFIG_HOME": "/c", "HOME": "/h"}), Path("/c/llmctl/recipes"))
        self.assertEqual(override_dir({"XDG_CONFIG_HOME": "", "HOME": "/h"}), Path("/h/.config/llmctl/recipes"))


if __name__ == "__main__":
    unittest.main()
