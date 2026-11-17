import dataclasses
import json
import os
import stat
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from launcher import kimi_config
from launcher.catalog import load_catalog
from launcher.errors import ClientError, ClientWriteError, UsageError
from launcher.files import FileChange
from launcher.model import Target
from launcher.tests.fakes import REPO, FakeHost, fail, ok
from launcher.tests.regen_golden import target_for
from launcher.tests.sglang_fixture import recipes_with_sglang, registered

RECIPES = REPO / "recipes"

EMPTY_MESSAGE = ("KIMI_CODE_HOME is set but empty, and Kimi 2.5.3 then reads config.toml from the working folder in "
                 "some commands and from ~/.kimi-code in others; unset it, or set it to an absolute path")

# The design is git-ignored, so its text is copied here (docs/superpowers/specs/2026-11-17-llmctl-kimi-design.md).
# Lines 325-331: the agent file, with the body's three sentences on their own lines (the lead's ruling in the ledger
# docs/superpowers/plans/2026-11-17-llmctl-kimi-ledger.md).
DESIGN_AGENT_FILE = """\
---
name: llmctl-chat-only
description: Chat-only agent for llmctl's chat-only recipes; it has no tools.
tools: []
---

You are a helpful assistant.
Answer the user directly in plain text.
You have no tools.
"""

# Lines 609-621: qwen3.6-35b-a3b on llama.cpp.
DESIGN_QWEN36 = """\
[providers.llmctl]
type = "openai"
base_url = "http://127.0.0.1:8080/v1"
api_key = "no-key-required"
model_source = "static"

[models."llmctl/local"]
provider = "llmctl"
model = "local"
display_name = "qwen3.6-35b-a3b on llama.cpp (llmctl)"
max_context_size = 131072
max_output_size = 16384
capabilities = ["tool_use"]
"""

# Lines 627-641: gpt-oss-20b on vLLM.
DESIGN_GPTOSS = """\
[providers.llmctl]
type = "openai"
base_url = "http://127.0.0.1:8000/v1"
api_key = "no-key-required"
model_source = "static"

[models."llmctl/gpt-oss-20b"]
provider = "llmctl"
model = "gpt-oss-20b"
display_name = "gpt-oss-20b on vLLM (llmctl)"
max_context_size = 65536
max_output_size = 16384
capabilities = ["tool_use", "thinking"]
support_efforts = ["low", "medium", "high"]
default_effort = "medium"
"""

DESIGN_PROVIDER = """\
[providers.llmctl]
type = "openai"
base_url = "http://127.0.0.1:{port}/v1"
api_key = "no-key-required"
model_source = "static"
"""

# Lines 648-689: the model table of the other kinds.
DESIGN_MODEL_TABLES = {
    ("vllm", "mellum2.1-12b"): """\
[models."llmctl/mellum2.1"]
provider = "llmctl"
model = "mellum2.1"
display_name = "mellum2.1-12b on vLLM (llmctl)"
max_context_size = 131072
max_output_size = 16384
capabilities = ["tool_use"]
""",
    ("llamacpp", "qwen3.8-27b"): """\
[models."llmctl/local"]
provider = "llmctl"
model = "local"
display_name = "qwen3.8-27b on llama.cpp (llmctl)"
max_context_size = 65536
max_output_size = 16384
capabilities = ["tool_use", "thinking"]
support_efforts = ["low", "medium"]
default_effort = "medium"
off_effort = "none"
""",
    ("llamacpp", "k2-horizon-7b"): """\
[models."llmctl/local"]
provider = "llmctl"
model = "local"
display_name = "k2-horizon-7b on llama.cpp (llmctl)"
max_context_size = 81920
max_output_size = 32768
capabilities = ["tool_use"]
""",
    ("llamacpp", "qwen3-coder-next"): """\
[models."llmctl/local"]
provider = "llmctl"
model = "local"
display_name = "qwen3-coder-next on llama.cpp (llmctl)"
max_context_size = 131072
max_output_size = 16384
capabilities = ["tool_use"]
""",
    ("vllm", "deepseek-r1-0528-qwen3-8b"): """\
[models."llmctl/deepseek-r1-8b"]
provider = "llmctl"
model = "deepseek-r1-8b"
display_name = "deepseek-r1-0528-qwen3-8b on vLLM (llmctl)"
max_context_size = 65536
max_output_size = 16384
capabilities = []
""",
}


KIMI_FIXTURES = REPO / "launcher/tests/fixtures/kimi"
INPUTS = ("owner", "f0-template-plus-llmctl", "f2-default-at-top-plus-other", "a-web-set-default", "a2-tui-model-pick",
          "a3-web-switch-default", "b-provider-add", "c-provider-remove", "d-web-start-refresh", "p3a", "p3b")


def kimi_fixture(name: str) -> str:
    with open(KIMI_FIXTURES / name, encoding="utf-8", newline="") as handle:
        return handle.read()


OWNER_TEXT = kimi_fixture("owner.toml")
TEMPLATE_LINES = ["# ~/.kimi-code/config.toml", "# Environment settings for Kimi Code.",
                  "# This file starts empty so built-in defaults can apply.",
                  "# Login will populate managed Kimi provider and model entries."]

# A hand-written file that recreates the spec's separate prototype run (multi-line values, a header inside a string,
# a sub-table below an alias), and the prototype's plan of it for qwen3.6-35b-a3b on llama.cpp without --set-default.
HAND_WRITTEN = '''\
"quoted root" = 1
list = [
  "a", # a comment with ] and "
  "b",
]
default_model = 'llmctl/old' # pinned by hand
notes = """
[providers.llmctl]
not a header
"""

[providers.other]
type = "openai"

[[hooks]]
event = "x"

[providers.llmctl]
# a comment inside llmctl's table
type = "openai"
extra.nested = 1

[models."llmctl/old"]
provider = "llmctl"

[models."llmctl/old".sub]
x = 1

[models."llmctl/older"]
provider = "llmctl"
'''

HAND_WRITTEN_PLANNED = '''\
"quoted root" = 1
list = [
  "a", # a comment with ] and "
  "b",
]
default_model = "llmctl/local" # pinned by hand
notes = """
[providers.llmctl]
not a header
"""

[providers.other]
type = "openai"

[[hooks]]
event = "x"

[providers.llmctl]
type = "openai"
base_url = "http://127.0.0.1:8080/v1"
api_key = "no-key-required"
model_source = "static"

[models."llmctl/local"]
provider = "llmctl"
model = "local"
display_name = "qwen3.6-35b-a3b on llama.cpp (llmctl)"
max_context_size = 131072
max_output_size = 16384
capabilities = ["tool_use"]


'''


def all_targets() -> list[Target]:
    catalog = load_catalog(RECIPES, None)
    return [target_for(recipe_id, runtime, RECIPES)
            for runtime, entry in catalog.runtimes.items() for recipe_id in sorted(entry.recipes)]


class PathsTest(unittest.TestCase):
    env = {"HOME": "/h"}

    def home(self, value: str) -> Path:
        return kimi_config.kimi_home({**self.env, "KIMI_CODE_HOME": value})

    def test_kimi_home_default(self):
        self.assertEqual(kimi_config.kimi_home(self.env), Path("/h/.kimi-code"))
        with self.assertRaises(UsageError) as caught:
            kimi_config.kimi_home({})
        self.assertTrue(str(caught.exception).startswith("HOME is not set"), str(caught.exception))

    def test_kimi_home_absolute_is_normalized(self):
        self.assertEqual(self.home("/a/b/../c/"), Path("/a/c"))
        self.assertEqual(self.home("/a/./b"), Path("/a/b"))
        self.assertEqual(str(self.home("//a")), "/a")
        self.assertEqual(str(self.home("///a")), "/a")
        self.assertEqual(kimi_config.kimi_home({"KIMI_CODE_HOME": "/abs"}), Path("/abs"))

    def test_kimi_home_keeps_dollar_names(self):
        env = {**self.env, "X": "y", "KIMI_CODE_HOME": "/a/$X/b"}
        self.assertEqual(kimi_config.kimi_home(env), Path("/a/$X/b"))

    def test_empty_kimi_home_refused(self):
        with self.assertRaises(UsageError) as caught:
            self.home("")
        self.assertEqual(caught.exception.exit_code, 1)
        self.assertEqual(str(caught.exception), EMPTY_MESSAGE)

    def test_relative_kimi_home_refused(self):
        for value in ("rel", "./x", "~", "~/x", " /a", "  ", "$HOME/.kimi-code"):
            with self.subTest(value=value):
                with self.assertRaises(UsageError) as caught:
                    self.home(value)
                self.assertEqual(str(caught.exception),
                                 f"KIMI_CODE_HOME is relative ({value}), and Kimi uses it as given, against each "
                                 "command's working folder; set it to an absolute path, or unset it for ~/.kimi-code")

    def test_config_path(self):
        self.assertEqual(kimi_config.config_path(self.env), Path("/h/.kimi-code/config.toml"))


class GenerateTest(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(kimi_config.PROVIDER, "llmctl")
        self.assertEqual(kimi_config.ALIAS_PREFIX, "llmctl/")
        self.assertEqual(kimi_config.VERIFIED_VERSION, "2.5.3")
        self.assertEqual(kimi_config.API_KEY, "no-key-required")
        self.assertEqual(kimi_config.SWITCHES, (("KIMI_DISABLE_TELEMETRY", "1"), ("KIMI_CODE_NO_AUTO_UPDATE", "1"),
                                                ("KIMI_CODE_REGION_MARKER", "off")))
        self.assertEqual(kimi_config.AGENT_FILE, REPO / "launcher/kimi-chat-only.md")
        self.assertTrue(kimi_config.AGENT_FILE.is_absolute())
        self.assertEqual(kimi_config.OUT, "leave Kimi out with LLMCTL_CLIENTS")
        self.assertEqual(kimi_config.RETRY_NOTE,
                         "  Kimi retries a stopped server for about two and a half minutes before a prompt fails")

    def test_agent_file(self):
        text = kimi_config.AGENT_FILE.read_text(encoding="utf-8")
        self.assertEqual(text, DESIGN_AGENT_FILE)
        lines = text.split("\n")
        self.assertEqual(lines[0], "---")
        end = lines.index("---", 1)
        front = dict(line.split(": ", 1) for line in lines[1:end])
        self.assertEqual(front["tools"], "[]")
        self.assertRegex(front["name"], r"^[a-z0-9]+(-[a-z0-9]+)*$")
        self.assertTrue(front["description"].strip())
        self.assertTrue("\n".join(lines[end + 1:]).strip())

    def test_qwen36_tables_are_the_spec_text(self):
        target = target_for("qwen3.6-35b-a3b", "llamacpp", RECIPES)
        self.assertEqual(kimi_config.render_tables(target), DESIGN_QWEN36)

    def test_gptoss_tables_are_the_spec_text(self):
        target = target_for("gpt-oss-20b", "vllm", RECIPES)
        self.assertEqual(kimi_config.render_tables(target), DESIGN_GPTOSS)

    def test_model_tables_of_the_other_kinds(self):
        for (runtime, recipe_id), model_text in DESIGN_MODEL_TABLES.items():
            with self.subTest(recipe=f"{runtime}/{recipe_id}"):
                target = target_for(recipe_id, runtime, RECIPES)
                text = kimi_config.render_tables(target)
                before, after = text.split("\n\n")
                port = target.runtime.host_port
                self.assertEqual(before + "\n", DESIGN_PROVIDER.format(port=port))
                self.assertEqual(after, model_text)

    def test_every_recipe_describes_its_server(self):
        targets = all_targets()
        self.assertEqual(len(targets), 22)
        for target in targets:
            with self.subTest(recipe=f"{target.recipe.runtime}/{target.recipe.id}"):
                provider = kimi_config.provider_table(target)
                model = kimi_config.model_table(target)
                port = target.runtime.host_port
                self.assertEqual(provider["base_url"], f"http://127.0.0.1:{port}/v1")
                self.assertEqual(model["model"], target.served)
                self.assertEqual(model["max_context_size"], target.recipe.context)
                self.assertEqual(model["max_output_size"], target.recipe.clients.output)
                self.assertEqual(kimi_config.alias(target), "llmctl/" + target.served)

    def test_thinking_kinds(self):
        kinds: dict[str, int] = {}
        for target in all_targets():
            kind = target.recipe.clients.thinking
            kinds[kind] = kinds.get(kind, 0) + 1
            model = kimi_config.model_table(target)
            with self.subTest(recipe=f"{target.recipe.runtime}/{target.recipe.id}"):
                self.assertEqual("support_efforts" in model, kind == "effort")
                self.assertEqual("default_effort" in model, kind == "effort")
                self.assertEqual("off_effort" in model, target.recipe.id == "qwen3.8-27b")
                if target.recipe.id == "deepseek-r1-0528-qwen3-8b":
                    self.assertEqual(model["capabilities"], [])
        self.assertEqual(kinds, {"toggle": 11, "effort": 4, "always": 6, "none": 1})

    def test_tables_reparse(self):
        for target in all_targets():
            with self.subTest(recipe=f"{target.recipe.runtime}/{target.recipe.id}"):
                text = kimi_config.render_tables(target)
                self.assertEqual(tomllib.loads(text), {
                    "providers": {"llmctl": kimi_config.provider_table(target)},
                    "models": {kimi_config.alias(target): kimi_config.model_table(target)}})
                self.assertTrue(text.isascii())
                self.assertTrue(text.endswith("\n"))
                self.assertEqual(text.split("\n").count(""), 2)  # the one empty line and the final newline
                self.assertNotIn("\n\n\n", text)

    def test_toml_value(self):
        for value in ("a", "modèle", "\U00010348", "\x7f", 'a"b', "a\\b", "\x01", "a\nb", 0, 131072, [], ["a", "b"]):
            with self.subTest(value=value):
                self.assertEqual(tomllib.loads("v = " + kimi_config.toml_value(value))["v"], value)
        self.assertEqual(kimi_config.toml_value("\x7f"), '"\\u007f"')
        for value in (True, 1.5, None, {}, ["a", 1]):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    kimi_config.toml_value(value)

    def test_odd_recipe_id_renders(self):
        target = target_for("qwen3.6-35b-a3b", "llamacpp", RECIPES)
        recipe = dataclasses.replace(target.recipe, id='modèle#1 "x"\x7f')
        target = dataclasses.replace(target, recipe=recipe)
        parsed = tomllib.loads(kimi_config.render_tables(target))
        self.assertEqual(parsed["models"]["llmctl/local"]["display_name"], 'modèle#1 "x"\x7f on llama.cpp (llmctl)')

    def test_dotted_served_names_are_quoted(self):
        lines = kimi_config.render_tables(target_for("qwen3.5-9b", "vllm", RECIPES)).split("\n")
        self.assertIn('[models."llmctl/qwen3.5-9b"]', lines)
        lines = kimi_config.render_tables(target_for("mellum2.1-12b", "vllm", RECIPES)).split("\n")
        self.assertIn('[models."llmctl/mellum2.1"]', lines)



class ReadTest(unittest.TestCase):
    path = Path("/h/.kimi-code/config.toml")

    def setUp(self):
        self.gemma = target_for("gemma-4-26b-a4b", "llamacpp", RECIPES)
        self.qwen38 = target_for("qwen3.8-27b", "llamacpp", RECIPES)

    def state(self, name: str) -> kimi_config.State:
        return kimi_config.parse_state(self.path, kimi_fixture(name + ".toml"))

    def refusal(self, text: str) -> str:
        with self.assertRaises(ClientError) as caught:
            kimi_config.parse_state(self.path, text)
        return str(caught.exception)

    def gemma_tables(self) -> tuple[dict, tuple]:
        return (kimi_config.provider_table(self.gemma),
                (("llmctl/local", kimi_config.model_table(self.gemma)),))

    def test_fixtures(self):
        for name in INPUTS:
            self.assertTrue((KIMI_FIXTURES / f"{name}.toml").is_file(), name)
        self.assertEqual(len(INPUTS), 11)
        self.assertEqual(INPUTS[0], "owner")
        self.assertEqual(len(OWNER_TEXT.encode()), 204)

    def test_read_state_reads_the_file_as_it_is(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {"HOME": tmp}
            self.assertIsNone(kimi_config.read_state(env))
            path = Path(tmp) / ".kimi-code" / "config.toml"
            path.parent.mkdir()
            text = OWNER_TEXT.replace("\n", "\r\n")
            with open(path, "w", encoding="utf-8", newline="") as handle:
                handle.write(text)
            state = kimi_config.read_state(env)
        self.assertEqual(state.path, path)
        self.assertEqual(state.text, text)
        self.assertEqual(state.data, {"telemetry": False})

    def test_owner_state(self):
        state = self.state("owner")
        self.assertEqual(state.path, self.path)
        self.assertEqual(state.text, OWNER_TEXT)
        self.assertIsNone(state.provider)
        self.assertEqual(state.aliases, ())
        self.assertIs(state.default, kimi_config.MISSING)
        self.assertIsNone(state.thinking)
        self.assertIsNone(state.description)
        self.assertIsNone(state.served)

    def test_state_of_a_file_with_llmctl_tables(self):
        state = self.state("f2-default-at-top-plus-other")
        self.assertEqual((state.provider, state.aliases), self.gemma_tables())
        self.assertEqual(state.default, "llmctl/local")
        self.assertEqual(state.description, "gemma-4-26b-a4b on llama.cpp")
        self.assertEqual(state.served, "local")

    def test_kimi_written_files_are_read(self):
        for name in ("a-web-set-default", "a2-tui-model-pick", "b-provider-add", "c-provider-remove",
                     "d-web-start-refresh"):
            state = self.state(name)
            self.assertEqual((state.provider, state.aliases), self.gemma_tables(), name)
        self.assertIn("dummy/dummy-model", self.state("d-web-start-refresh").data["models"])
        self.assertEqual(self.state("p3b").thinking, {"enabled": True, "effort": "low"})
        self.assertEqual(self.state("a2-tui-model-pick").thinking, {"enabled": False})

    def test_foreign_lines(self):
        def foreign(name: str) -> list[str]:
            return kimi_config.foreign_lines(self.path, kimi_fixture(name + ".toml"))

        self.assertEqual(foreign("owner"), TEMPLATE_LINES + ["telemetry = false"])
        self.assertEqual(foreign("f0-template-plus-llmctl"), TEMPLATE_LINES)
        lines = kimi_fixture("b-provider-add.toml").split("\n")
        expected = [line for line in lines[6:15] + lines[24:31] if line]
        self.assertEqual(len(expected), 15)
        self.assertEqual((expected[0], expected[-1]), ("[providers.dummy]", 'display_name = "Dummy Model"'))
        self.assertEqual(foreign("b-provider-add"), expected)
        self.assertEqual(foreign("p3a"), ["[thinking]", "enabled = true"])
        lines = kimi_fixture("a3-web-switch-default.toml").split("\n")
        self.assertEqual(foreign("a3-web-switch-default"),
                         TEMPLATE_LINES + ["# owner comment: a second, hand-written provider"] + lines[21:25]
                         + lines[26:30])
        self.assertEqual(foreign("c-provider-remove"), [])

    def test_header_forms(self):
        for header in ("[ models . 'llmctl/local' ]", '[models."llmctl/local"]'):
            text = f'{header}\nprovider = "llmctl"\n'
            state = kimi_config.parse_state(self.path, text)
            self.assertEqual(state.aliases, (("llmctl/local", {"provider": "llmctl"}),), header)
            self.assertEqual(kimi_config.foreign_lines(self.path, text), [], header)
        text = '[models.llmctl.local]\nprovider = "llmctl"\n'
        self.assertEqual(kimi_config.parse_state(self.path, text).aliases, ())
        self.assertEqual(kimi_config.foreign_lines(self.path, text), ["[models.llmctl.local]", 'provider = "llmctl"'])
        text = '[providers.llmctl]\ntype = "openai"\n\n[[hooks]]\nevent = "x"\n\n[providers.other]\ntype = "openai"\n'
        self.assertEqual(kimi_config.foreign_lines(self.path, text),
                         ["[[hooks]]", 'event = "x"', "[providers.other]", 'type = "openai"'])

    def test_multi_line_values_are_one_statement(self):
        state = kimi_config.parse_state(self.path, HAND_WRITTEN)
        self.assertEqual(state.aliases, (("llmctl/old", {"provider": "llmctl", "sub": {"x": 1}}),
                                         ("llmctl/older", {"provider": "llmctl"})))
        self.assertEqual(state.provider, {"type": "openai", "extra": {"nested": 1}})
        self.assertEqual(state.default, "llmctl/old")
        lines = HAND_WRITTEN.split("\n")
        numbers = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 15, 16]
        self.assertEqual(kimi_config.foreign_lines(self.path, HAND_WRITTEN), [lines[n - 1] for n in numbers])
        self.assertEqual(lines[7], "[providers.llmctl]")

    def test_unreadable_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {"HOME": tmp}
            path = Path(tmp) / ".kimi-code" / "config.toml"
            path.parent.mkdir()
            path.write_bytes(b"\xff\n")
            with self.assertRaises(ClientError) as caught:
                kimi_config.read_state(env)
            self.assertTrue(str(caught.exception).startswith(f"cannot read {path}: "), str(caught.exception))
            path.unlink()
            path.mkdir()
            with self.assertRaises(ClientError) as caught:
                kimi_config.read_state(env)
            self.assertTrue(str(caught.exception).startswith(f"cannot read {path}: "), str(caught.exception))

    def test_invalid_toml(self):
        self.assertEqual(self.refusal("a = \n"),
                         f"{self.path} is not valid TOML (Invalid value (at line 1, column 5)); fix it, or leave Kimi "
                         "out with LLMCTL_CLIENTS")

    def test_byte_order_mark_refuses(self):
        self.assertEqual(self.refusal("﻿" + OWNER_TEXT),
                         f"{self.path} is not valid TOML (it starts with a byte-order mark, which TOML does not "
                         "allow); fix it, or leave Kimi out with LLMCTL_CLIENTS")

    def test_providers_or_models_not_a_table(self):
        providers = (f'{self.path}: "providers" is not a table; Kimi needs [providers.<name>] tables; fix it, or '
                     "leave Kimi out with LLMCTL_CLIENTS")
        models = (f'{self.path}: "models" is not a table; Kimi needs [models."<alias>"] tables; fix it, or '
                  "leave Kimi out with LLMCTL_CLIENTS")
        self.assertEqual(self.refusal('[[providers]]\ntype = "x"\n'), providers)
        self.assertEqual(self.refusal("providers = 1\n"), providers)
        self.assertEqual(self.refusal('models = "x"\n'), models)

    def test_owned_names_without_their_header(self):
        provider = (f"{self.path}: llmctl's provider is written with dotted keys or as an inline table, so llmctl "
                    "cannot replace it line by line; write it as a [providers.llmctl] table, or delete it")
        model = (f"{self.path}: the model llmctl/local is written with dotted keys or as an inline table, so llmctl "
                 'cannot replace it line by line; write it as a [models."llmctl/local"] table, or delete it')
        self.assertEqual(self.refusal('[providers]\nllmctl = { type = "openai" }\n'), provider)
        self.assertEqual(self.refusal('providers.llmctl.type = "openai"\n'), provider)
        self.assertEqual(self.refusal('[models]\n"llmctl/local".provider = "llmctl"\n'), model)
        self.assertEqual(self.refusal('[models]\n"llmctl/local" = { provider = "llmctl" }\n'), model)

    def test_entry_matches(self):
        f2 = kimi_fixture("f2-default-at-top-plus-other.toml")
        self.assertTrue(kimi_config.entry_matches(self.state("f2-default-at-top-plus-other"), self.gemma))
        self.assertFalse(kimi_config.entry_matches(self.state("f2-default-at-top-plus-other"), self.qwen38))
        self.assertTrue(kimi_config.entry_matches(self.state("a-web-set-default"), self.gemma))
        text = f2.replace('capabilities = ["tool_use"]\n', 'capabilities = ["tool_use"]\ncontext_window_tokens = 5\n')
        self.assertNotEqual(text, f2)
        self.assertTrue(kimi_config.entry_matches(kimi_config.parse_state(self.path, text), self.gemma))
        text = f2 + '\n[models."llmctl/x"]\nprovider = "llmctl"\n'
        self.assertFalse(kimi_config.entry_matches(kimi_config.parse_state(self.path, text), self.gemma))
        text = f2.replace("max_output_size = 16384", "max_output_size = true")
        self.assertFalse(kimi_config.entry_matches(kimi_config.parse_state(self.path, text), self.gemma))
        self.assertFalse(kimi_config.entry_matches(None, self.gemma))

    def test_points_at_llmctl(self):
        self.assertTrue(kimi_config.points_at_llmctl(self.state("f2-default-at-top-plus-other")))
        self.assertFalse(kimi_config.points_at_llmctl(self.state("a3-web-switch-default")))
        self.assertFalse(kimi_config.points_at_llmctl(self.state("owner")))
        self.assertFalse(kimi_config.points_at_llmctl(None))
        self.assertFalse(kimi_config.points_at_llmctl(kimi_config.parse_state(self.path, "default_model = 5\n")))

    def test_description_and_served(self):
        for display_name in ('"gemma"', '" (llmctl)"', "5"):
            text = f'[models."llmctl/local"]\nmodel = "local"\ndisplay_name = {display_name}\n'
            state = kimi_config.parse_state(self.path, text)
            self.assertIsNone(state.description, display_name)
            self.assertEqual(state.served, "local")
        state = kimi_config.parse_state(self.path, '[models."llmctl/local"]\nmodel = 5\n')
        self.assertIsNone(state.served)



TARGETS = {"gemma": ("gemma-4-26b-a4b", "llamacpp"), "qwen38": ("qwen3.8-27b", "llamacpp"),
           "gptoss-vllm": ("gpt-oss-20b", "vllm"), "deepseek-vllm": ("deepseek-r1-0528-qwen3-8b", "vllm")}
EXPECTED = KIMI_FIXTURES / "expected"
# The ten plans of gemma-4-26b-a4b over files that already hold its tables in Kimi's spacing: llmctl keeps them,
# while the prototype re-rendered them, so their reference plans are not kept (design D3).
KEPT_AS_THEY_ARE = ("a-web-set-default", "a2-tui-model-pick", "b-provider-add", "c-provider-remove",
                    "d-web-start-refresh")

# Design lines 695-714: the owner's file after the first up of gemma-4-26b-a4b without --set-default.
DESIGN_OWNER_FIRST_WRITE = """\
# ~/.kimi-code/config.toml
# Environment settings for Kimi Code.
# This file starts empty so built-in defaults can apply.
# Login will populate managed Kimi provider and model entries.

telemetry = false

[providers.llmctl]
type = "openai"
base_url = "http://127.0.0.1:8080/v1"
api_key = "no-key-required"
model_source = "static"

[models."llmctl/local"]
provider = "llmctl"
model = "local"
display_name = "gemma-4-26b-a4b on llama.cpp (llmctl)"
max_context_size = 131072
max_output_size = 16384
capabilities = ["tool_use"]
"""

# Design lines 191-224: b-provider-add.toml after up of gpt-oss-20b on vLLM without --set-default.
DESIGN_B_PROVIDER_ADD = """\
[providers.llmctl]
type = "openai"
base_url = "http://127.0.0.1:8000/v1"
api_key = "no-key-required"
model_source = "static"

[providers.dummy]
base_url = "http://127.0.0.1:47401/v1"
type = "openai"
api_key = ""

[providers.dummy.source]
kind = "apiJson"
url = "http://127.0.0.1:47400/api.json"
apiKey = ""

[models."llmctl/gpt-oss-20b"]
provider = "llmctl"
model = "gpt-oss-20b"
display_name = "gpt-oss-20b on vLLM (llmctl)"
max_context_size = 65536
max_output_size = 16384
capabilities = ["tool_use", "thinking"]
support_efforts = ["low", "medium", "high"]
default_effort = "medium"

[models."dummy/dummy-model"]
provider = "dummy"
model = "dummy-model"
max_context_size = 32768
max_output_size = 4096
capabilities = [ "tool_use" ]
display_name = "Dummy Model"

"""

# Review Focus 1: llmctl's tables for qwen3.6-35b-a3b as Kimi or a hand edit may rewrite them.
REWRITTEN_TABLES = """\
default_model = "llmctl/local"

[providers.llmctl]
api_key = 'no-key-required'
base_url = "http://127.0.0.1:8080/v1" # moved by hand
model_source = "static"
type = "openai"

[providers.other]
type = "openai"

[models."llmctl/local"]
capabilities = [
  "tool_use",
]
display_name = "qwen3.6-35b-a3b on llama.cpp (llmctl)"
max_context_size = 131_072
max_output_size = 16_384
model = 'local'
provider = "llmctl"
"""

# Review Focus 3: tables under llmctl's names that llmctl did not write.
OWNER_MADE_TABLES = """\
[providers.llmctl]
type = "anthropic"
base_url = "https://example.invalid/v1"
api_key = "own-placeholder"

[models."llmctl/mine"]
provider = "llmctl"
model = "mine"
display_name = "Mine"
max_context_size = 8192
"""


def plan_cases() -> list[tuple[str, str, bool]]:
    return [(name, target, set_default) for name in INPUTS for target in TARGETS for set_default in (False, True)]


def case_file(name: str, target: str, set_default: bool) -> Path:
    return EXPECTED / f"{name}--{target}--{'default' if set_default else 'nodefault'}.toml"


def write_config(home: str, text: str | None) -> Path:
    folder = Path(home) / ".kimi-code"
    folder.mkdir()
    path = folder / "config.toml"
    if text is not None:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
    return path


class PlanTest(unittest.TestCase):
    path = Path("/h/.kimi-code/config.toml")

    def setUp(self):
        self.targets = {key: target_for(recipe_id, runtime, RECIPES) for key, (recipe_id, runtime) in TARGETS.items()}
        self.qwen36 = target_for("qwen3.6-35b-a3b", "llamacpp", RECIPES)
        self.gptoss = self.targets["gptoss-vllm"]

    def text(self, old: str | None, target: Target, *, set_default: bool = False) -> str:
        return kimi_config.config_text(self.path, old, target, set_default=set_default)

    def refusal(self, old: str, target: Target, *, set_default: bool = False) -> str:
        with self.assertRaises(ClientError) as caught:
            self.text(old, target, set_default=set_default)
        return str(caught.exception)

    def plan(self, old: str | None, target: Target, *, set_default: bool = False):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp, old)
            change = kimi_config.plan_config({"HOME": tmp}, target, set_default=set_default)
        if change is not None:
            self.assertEqual(change.path, path)
        return change

    def model_lines(self, target: Target) -> str:
        return kimi_config.render_tables(target).split("\n\n", 1)[1]

    def test_owner_first_write(self):
        gemma = self.targets["gemma"]
        self.assertEqual(self.text(OWNER_TEXT, gemma), DESIGN_OWNER_FIRST_WRITE)
        with_default = DESIGN_OWNER_FIRST_WRITE.replace("telemetry = false\n",
                                                        'telemetry = false\ndefault_model = "llmctl/local"\n')
        self.assertEqual(self.text(OWNER_TEXT, gemma, set_default=True), with_default)

    def test_new_file(self):
        tables = kimi_config.render_tables(self.qwen36)
        for old in (None, ""):
            self.assertEqual(self.text(old, self.qwen36), tables)
            self.assertEqual(self.text(old, self.qwen36, set_default=True),
                             'default_model = "llmctl/local"\n\n' + tables)

    def test_reference_plans(self):
        self.assertEqual(len(list(EXPECTED.iterdir())), 78)
        compared = 0
        for name, target, set_default in plan_cases():
            with self.subTest(name=name, target=target, set_default=set_default):
                old = kimi_fixture(name + ".toml")
                new = self.text(old, self.targets[target], set_default=set_default)
                expected_file = case_file(name, target, set_default)
                if expected_file.exists():
                    with open(expected_file, encoding="utf-8", newline="") as handle:
                        self.assertEqual(new, handle.read())
                    compared += 1
                    continue
                self.assertEqual(target, "gemma")
                self.assertIn(name, KEPT_AS_THEY_ARE)
                if set_default and name in ("b-provider-add", "c-provider-remove", "d-web-start-refresh"):
                    old = 'default_model = "llmctl/local"\n' + old
                self.assertEqual(new, old)
        self.assertEqual(compared, 78)

    def test_planning_is_idempotent_and_composes(self):
        gemma = self.targets["gemma"]
        for name, target, set_default in plan_cases():
            with self.subTest(name=name, target=target, set_default=set_default):
                old = kimi_fixture(name + ".toml")
                new = self.text(old, self.targets[target], set_default=set_default)
                self.assertEqual(self.text(new, self.targets[target], set_default=set_default), new)
                first = self.text(old, gemma)
                self.assertEqual(self.text(first, self.targets[target], set_default=set_default), new)

    def test_spec_example_b_provider_add(self):
        self.assertEqual(self.text(kimi_fixture("b-provider-add.toml"), self.gptoss), DESIGN_B_PROVIDER_ADD)

    def test_hand_written_file(self):
        self.assertEqual(self.text(HAND_WRITTEN, self.qwen36), HAND_WRITTEN_PLANNED)
        change = self.plan(HAND_WRITTEN, self.qwen36)
        self.assertEqual(change.new_text, HAND_WRITTEN_PLANNED)
        self.assertEqual(change.summary, "provider llmctl and model llmctl/local for qwen3.6-35b-a3b on llama.cpp (was "
                                         "an unknown model); default_model llmctl/local (was llmctl/old); removed the "
                                         "model llmctl/older")
        self.assertTrue(change.backup)

    def test_insertion_points(self):
        provider, model = kimi_config.render_tables(self.qwen36).split("\n\n")
        provider += "\n"
        other = '[providers.other]\ntype = "openai"\n'
        other_model = '[models."other/m"]\nprovider = "other"\n'
        self.assertEqual(self.text(other + "\n" + other_model, self.qwen36),
                         other + "\n" + provider + "\n" + other_model + "\n" + model)
        self.assertEqual(self.text(other + other_model + "\n# end\n", self.qwen36),
                         other + "\n" + provider + "\n" + other_model + "\n" + model + "\n# end\n")
        self.assertEqual(self.text('[providers.other]\ntype = "openai"', self.qwen36),
                         other + "\n" + provider + "\n" + model)
        self.assertEqual(self.text("telemetry = false\n", self.qwen36),
                         "telemetry = false\n\n" + provider + "\n" + model)
        self.assertEqual(self.text("telemetry = false\n\n", self.qwen36),
                         "telemetry = false\n\n" + provider + "\n" + model)
        self.assertEqual(self.text("telemetry = false", self.qwen36),
                         "telemetry = false\n\n" + provider + "\n" + model)
        self.assertEqual(self.text(provider + "[thinking]\nenabled = true\n", self.qwen36),
                         provider + "[thinking]\nenabled = true\n\n" + model)

    def test_kimi_rewritten_tables_are_kept(self):
        for set_default in (False, True):
            self.assertEqual(self.text(REWRITTEN_TABLES, self.qwen36, set_default=set_default), REWRITTEN_TABLES)
            self.assertIsNone(self.plan(REWRITTEN_TABLES, self.qwen36, set_default=set_default))
        self.assertTrue(kimi_config.entry_matches(kimi_config.parse_state(self.path, REWRITTEN_TABLES), self.qwen36))
        added = REWRITTEN_TABLES + "context_window_tokens = 5\n"
        self.assertTrue(kimi_config.entry_matches(kimi_config.parse_state(self.path, added), self.qwen36))
        start = REWRITTEN_TABLES.index('[models."llmctl/local"]')
        change = self.plan(added, self.qwen36)
        self.assertEqual(change.new_text, REWRITTEN_TABLES[:start] + self.model_lines(self.qwen36))
        self.assertEqual(change.summary, "provider llmctl and model llmctl/local for qwen3.6-35b-a3b on llama.cpp "
                                         "(same recipe, tables rewritten)")

    def test_crlf_files_get_crlf_lines(self):
        def crlf(text: str) -> str:
            return text.replace("\n", "\r\n")

        for name, target, set_default in plan_cases():
            with self.subTest(name=name, target=target, set_default=set_default):
                old = kimi_fixture(name + ".toml")
                new = self.text(old, self.targets[target], set_default=set_default)
                self.assertEqual(self.text(crlf(old), self.targets[target], set_default=set_default), crlf(new))
        for name in ("owner", "c-provider-remove"):
            with self.subTest(name=name):
                old = kimi_fixture(name + ".toml")
                inserted = kimi_config.default_text(self.path, old, "llmctl/local")
                self.assertEqual(kimi_config.default_text(self.path, crlf(old), "llmctl/local"), crlf(inserted))
                self.assertEqual(kimi_config.default_text(self.path, crlf(inserted), None), crlf(old))

    def test_owner_made_tables_under_llmctl_names_are_backed_up(self):
        change = self.plan(OWNER_MADE_TABLES, self.qwen36)
        self.assertEqual(change.new_text, kimi_config.render_tables(self.qwen36))
        self.assertTrue(change.backup)
        self.assertEqual(change.summary, "provider llmctl and model llmctl/local for qwen3.6-35b-a3b on llama.cpp (was "
                                         "an unknown model)")
        change = self.plan(kimi_config.render_tables(self.gptoss), self.qwen36)
        self.assertEqual(change.new_text, kimi_config.render_tables(self.qwen36))
        self.assertFalse(change.backup)

    def test_unusual_default_model_forms(self):
        tables = "\n" + kimi_config.render_tables(self.qwen36)
        refusal = (f'could not change the top-level default_model in {self.path} safely; set it to "llmctl/local" '
                   "yourself")
        follows = [('default_model = "llmctl/old"', 'default_model = "llmctl/local"'),
                   ("default_model='llmctl/old'", 'default_model="llmctl/local"'),
                   ('"default_model" = "llmctl/old"  # pinned "here"',
                    '"default_model" = "llmctl/local"  # pinned "here"'),
                   ('default_model = "llmctl\\u002fold"', 'default_model = "llmctl/local"')]
        for line, expected in follows:
            with self.subTest(line=line):
                self.assertEqual(self.text(line + "\n" + tables, self.qwen36), expected + "\n" + tables)
        for lines in ('default_model = """llmctl/old"""\n', 'default_model = """\nllmctl/old"""\n'):
            with self.subTest(lines=lines):
                self.assertEqual(self.refusal(lines + tables, self.qwen36), refusal)
        others = ["default_model = 5\n" + tables, tables + "\n[default_model]\nx = 1\n",
                  "default_model.x = 1\n" + tables]
        for text in others:
            with self.subTest(text=text):
                self.assertEqual(self.refusal(text, self.qwen36, set_default=True), refusal)
                self.assertEqual(self.text(text, self.qwen36), text)
        text = 'default_model = "other/m" # "x"\n' + tables
        self.assertEqual(self.text(text, self.qwen36), text)
        self.assertEqual(self.text(text, self.qwen36, set_default=True),
                         'default_model = "llmctl/local" # "x"\n' + tables)

    def test_root_inline_or_dotted_refused(self):
        self.assertEqual(self.refusal('providers = { other = { type = "openai" } }\n', self.qwen36),
                         f"{self.path} line 1: providers is written as an inline table or with dotted keys, so llmctl "
                         "cannot add its [providers.llmctl] table after it; write each provider as a "
                         "[providers.<name>] table")
        self.assertEqual(self.refusal('"x" = 1\nmodels.other.provider = "x"\n', self.qwen36),
                         f"{self.path} line 2: models is written as an inline table or with dotted keys, so llmctl "
                         'cannot add its [models."llmctl/local"] table after it; write each model as a '
                         '[models."<alias>"] table')

    def test_extra_key_subtable_and_second_alias(self):
        tables = kimi_config.render_tables(self.qwen36)
        old = (tables.replace('model_source = "static"\n', 'model_source = "static"\nextra = 1\n')
               + '\n[models."llmctl/local".sub]\nx = 1\n\n' + kimi_config.render_tables(self.gptoss).split("\n\n")[1])
        change = self.plan(old, self.qwen36)
        self.assertEqual(change.new_text, tables + "\n\n")
        self.assertEqual(change.summary, "provider llmctl and model llmctl/local for qwen3.6-35b-a3b on llama.cpp "
                                         "(same recipe, tables rewritten); removed the model llmctl/gpt-oss-20b")

    def test_aliases_follow_across_runtimes(self):
        text = 'default_model = "llmctl/gpt-oss-20b"\n\n' + kimi_config.render_tables(self.gptoss)
        qwen38 = self.targets["qwen38"]
        for target in (self.qwen36, self.gptoss, qwen38, self.gptoss):
            text = self.text(text, target)
            state = kimi_config.parse_state(self.path, text)
            self.assertEqual(state.default, kimi_config.alias(target))
            self.assertEqual([name for name, _ in state.aliases], [kimi_config.alias(target)])

    def test_own_check_refuses_a_wrong_rendering(self):
        real = kimi_config.toml_value
        message = f"{self.path}: llmctl's own check of the new config.toml failed; nothing was written"
        with mock.patch("launcher.kimi_config.toml_value",
                        side_effect=lambda value: "0" if value == 131072 else real(value)):
            for old in (None, OWNER_TEXT):
                with self.assertRaises(ClientError) as caught:
                    self.text(old, self.qwen36)
                self.assertEqual(str(caught.exception), message)

    def test_unencodable_value_refuses(self):
        target = dataclasses.replace(self.qwen36, recipe=dataclasses.replace(self.qwen36.recipe, id="bad\udcff"))
        with self.assertRaises(ClientError) as caught:
            self.text(OWNER_TEXT, target)
        self.assertEqual(str(caught.exception),
                         f"{self.path}: llmctl's own check of the new config.toml failed; nothing was written")

    def test_summaries(self):
        head = "provider llmctl and model llmctl/local for qwen3.6-35b-a3b on llama.cpp"
        qwen36 = kimi_config.render_tables(self.qwen36)
        gptoss = 'default_model = "llmctl/gpt-oss-20b"\n\n' + kimi_config.render_tables(self.gptoss)
        cases = [
            (None, False, "created it with " + head),
            (None, True, "created it with " + head + "; default_model llmctl/local"),
            ("", False, "created it with " + head),
            (OWNER_TEXT, False, head + " (was absent)"),
            (OWNER_TEXT, True, head + " (was absent); default_model llmctl/local (was unset)"),
            (OWNER_MADE_TABLES, False, head + " (was an unknown model)"),
            (gptoss, False, head + " (was gpt-oss-20b on vLLM); default_model llmctl/local (was llmctl/gpt-oss-20b)"),
            (REWRITTEN_TABLES + "context_window_tokens = 5\n", False, head + " (same recipe, tables rewritten)"),
            (qwen36, True, head + " (same recipe); default_model llmctl/local (was unset)"),
            ('default_model = "other/m"\n\n' + qwen36, True, head + " (same recipe); default_model llmctl/local (was "
                                                                    "other/m)"),
            (qwen36 + '\n[models."llmctl/a"]\nprovider = "llmctl"\n', False,
             head + " (same recipe, tables rewritten); removed the model llmctl/a"),
        ]
        for old, set_default, summary in cases:
            with self.subTest(old=old, set_default=set_default):
                change = self.plan(old, self.qwen36, set_default=set_default)
                self.assertEqual(change.summary, summary)
                self.assertEqual(change.preview, f"would write {change.path}: {summary}")

    def test_plan_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp, None)
            change = kimi_config.plan_config({"HOME": tmp}, self.qwen36, set_default=False)
        self.assertEqual(change.path, path)
        self.assertIsNone(change.old_text)
        self.assertEqual(change.new_text, kimi_config.render_tables(self.qwen36))
        self.assertEqual(change.mode, 0o600)
        self.assertFalse(change.backup)
        self.assertEqual(change.dirs, ())
        self.assertTrue(change.summary.startswith("created it with "))
        self.assertEqual(change.preview, f"would write {path}: {change.summary}")
        change = self.plan(OWNER_TEXT, self.qwen36)
        self.assertEqual(change.old_text, OWNER_TEXT)
        self.assertTrue(change.backup)
        self.assertTrue(change.summary.endswith(" (was absent)"))
        self.assertIsNone(self.plan(kimi_config.render_tables(self.qwen36), self.qwen36))

    def test_plan_config_reads_as_read_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp, None)
            path.write_bytes(b"\xff\n")
            with self.assertRaises(ClientError) as caught:
                kimi_config.plan_config({"HOME": tmp}, self.qwen36, set_default=False)
        self.assertTrue(str(caught.exception).startswith(f"cannot read {path}: "), str(caught.exception))


class DefaultTest(unittest.TestCase):
    path = Path("/h/.kimi-code/config.toml")

    def test_default_insert_and_remove_round_trip(self):
        for name in ("owner", "f0-template-plus-llmctl", "b-provider-add", "c-provider-remove",
                     "d-web-start-refresh"):
            with self.subTest(name=name):
                old = kimi_fixture(name + ".toml")
                new = kimi_config.default_text(self.path, old, "llmctl/local")
                line = 'default_model = "llmctl/local"\n'
                if name == "owner":
                    self.assertEqual(new, old.replace("telemetry = false\n", "telemetry = false\n" + line))
                else:
                    self.assertEqual(new, line + old)
                self.assertEqual(kimi_config.default_text(self.path, new, None), old)

    def test_default_edit_in_place_and_compare(self):
        f2 = kimi_fixture("f2-default-at-top-plus-other.toml")
        new = kimi_config.default_text(self.path, f2, "other/m")
        self.assertEqual(new, f2.replace('default_model = "llmctl/local"\n', 'default_model = "other/m"\n'))
        self.assertNotEqual(new, f2)
        self.assertEqual(kimi_config.default_text(self.path, f2, "llmctl/local"), f2)
        a = kimi_fixture("a-web-set-default.toml")
        first, rest = a.split("\n", 1)
        self.assertTrue(first.startswith("default_model = "), first)
        self.assertEqual(kimi_config.default_text(self.path, a, None), rest)

    def test_removal_of_any_form(self):
        text = 'default_model = """\nllmctl/old"""\ntelemetry = false\n'
        self.assertEqual(kimi_config.default_text(self.path, text, None), "telemetry = false\n")
        with self.assertRaises(ClientError) as caught:
            kimi_config.default_text(self.path, "telemetry = false\n\n[default_model]\nx = 1\n", None)
        self.assertEqual(str(caught.exception),
                         f"could not remove the top-level default_model in {self.path} safely; remove it yourself")
        self.assertEqual(kimi_config.default_text(self.path, OWNER_TEXT, None), OWNER_TEXT)

    def test_plan_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {"HOME": tmp}
            path = write_config(tmp, None)
            self.assertIsNone(kimi_config.plan_default(env, None))
            with self.assertRaises(ClientError) as caught:
                kimi_config.plan_default(env, "llmctl/local")
            self.assertEqual(str(caught.exception), f"cannot read {path}: it does not exist")
            path.write_text(OWNER_TEXT)
            self.assertIsNone(kimi_config.plan_default(env, None))
            change = kimi_config.plan_default(env, "llmctl/local")
            self.assertEqual(change.summary, "default_model llmctl/local (was unset)")
            self.assertEqual(change.preview, f"would write {path}: {change.summary}")
            self.assertEqual((change.old_text, change.mode, change.backup), (OWNER_TEXT, 0o600, True))
            path.write_text(change.new_text)
            self.assertIsNone(kimi_config.plan_default(env, "llmctl/local"))
            change = kimi_config.plan_default(env, "other/m")
            self.assertEqual(change.summary, "default_model other/m (was llmctl/local)")
            change = kimi_config.plan_default(env, None)
            self.assertEqual(change.summary, "removed default_model (was llmctl/local)")
            self.assertEqual(change.new_text, OWNER_TEXT)
            path.write_text('default_model = "llmctl/local"\n\n' + kimi_fixture("c-provider-remove.toml"))
            change = kimi_config.plan_default(env, None)
            self.assertEqual(change.new_text, "\n" + kimi_fixture("c-provider-remove.toml"))
            self.assertFalse(change.backup)


OUT_TAIL = "; check that kimi runs in this shell, or leave Kimi out with LLMCTL_CLIENTS"
SWITCH_ENV = {"KIMI_DISABLE_TELEMETRY": "1", "KIMI_CODE_NO_AUTO_UPDATE": "1", "KIMI_CODE_REGION_MARKER": "off"}
RECORDS = {("gemma-4-26b-a4b", "llamacpp"): "provider-list-llamacpp-gemma-4-26b-a4b.json",
           ("qwen3.8-27b", "llamacpp"): "provider-list-llamacpp-qwen3.8-27b.json",
           ("deepseek-r1-0528-qwen3-8b", "vllm"): "provider-list-vllm-deepseek-r1-0528-qwen3-8b.json"}


class CheckTest(unittest.TestCase):
    path = Path("/h/.kimi-code/config.toml")

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.host = FakeHost(env={"HOME": tmp.name})
        self.gemma = target_for("gemma-4-26b-a4b", "llamacpp", RECIPES)
        self.qwen36 = target_for("qwen3.6-35b-a3b", "llamacpp", RECIPES)
        self.folders: list[str] = []

    def answer(self, host: FakeHost, *, version: str | None = None, record: str | None = None, data=None,
               result=None, extra=None):
        host.on(["kimi", "--version"], ok(kimi_fixture("version.txt") if version is None else version))

        def route(argv):
            home = host.env_extras[-1]["KIMI_CODE_HOME"]
            self.folders.append(os.path.dirname(home))
            if extra is not None:
                extra(home)
            if result is not None:
                if isinstance(result, BaseException):
                    raise result
                return result
            return ok(kimi_fixture(record) if data is None else json.dumps(data))

        host.on(["kimi", "provider", "list", "--json"], route)

    def check(self, target: Target, text: str | None = None):
        return kimi_config.check_config(self.host, kimi_config.render_tables(target) if text is None else text,
                                        target, path=self.path)

    def refusal(self, target: Target, cls=ClientError) -> str:
        with self.assertRaises(cls) as caught:
            self.check(target)
        return str(caught.exception)

    def gemma_record(self) -> dict:
        return json.loads(kimi_fixture(RECORDS[("gemma-4-26b-a4b", "llamacpp")]))

    def test_check_passes_on_each_e6_record(self):
        for (recipe_id, runtime), record in RECORDS.items():
            with self.subTest(recipe_id=recipe_id, runtime=runtime):
                self.host = FakeHost(env=self.host.env)
                self.answer(self.host, record=record)
                self.assertIsNone(self.check(target_for(recipe_id, runtime, RECIPES)))
                self.assertEqual(self.host.calls, [["kimi", "--version"], ["kimi", "provider", "list", "--json"]])
                first, second = self.host.env_extras
                self.assertEqual(first, second)
                folder = self.folders[-1]
                self.assertTrue(os.path.basename(folder).startswith("llmctl-kimi-"))
                self.assertEqual(first, {"KIMI_CODE_HOME": os.path.join(folder, "home"),
                                         "KIMI_CODE_CACHE_DIR": os.path.join(folder, "cache"), **SWITCH_ENV})

    def test_candidate_is_checked_privately(self):
        text = kimi_config.render_tables(self.gemma).replace("\n", "\r\n")
        seen = {}

        def look(home):
            folder = os.path.dirname(home)
            config = os.path.join(home, "config.toml")
            seen["folder"] = folder
            seen["modes"] = [stat.S_IMODE(os.stat(p).st_mode) for p in (folder, home, os.path.join(folder, "cache"),
                                                                         config)]
            with open(config, "rb") as handle:
                seen["bytes"] = handle.read()

        self.answer(self.host, record=RECORDS[("gemma-4-26b-a4b", "llamacpp")], extra=look)
        self.assertIsNone(self.check(self.gemma, text))
        self.assertTrue(os.path.basename(seen["folder"]).startswith("llmctl-kimi-"))
        self.assertEqual(seen["modes"], [0o700, 0o700, 0o700, 0o600])
        self.assertEqual(seen["bytes"], text.encode("utf-8"))
        self.assertFalse(os.path.exists(seen["folder"]))

    def test_private_folder_removed_on_every_path(self):
        record = self.gemma_record()
        del record["models"]["llmctl/local"]
        cases = {"pass": {}, "version": {"version": ""}, "list": {"result": fail("boom")},
                 "alias": {"data": record}, "oserror": {"result": OSError(5, "Input/output error")}}
        for name, kwargs in cases.items():
            with self.subTest(name):
                self.host = FakeHost(env=self.host.env)
                self.folders.clear()
                created = []
                real = tempfile.mkdtemp

                def mkdtemp(*args, **kw):
                    created.append(real(*args, **kw))
                    return created[-1]

                self.answer(self.host, record=RECORDS[("gemma-4-26b-a4b", "llamacpp")], **kwargs)
                with mock.patch("launcher.kimi_config.tempfile.mkdtemp", side_effect=mkdtemp):
                    try:
                        self.check(self.gemma)
                    except ClientError:
                        pass
                self.assertEqual(len(created), 1)
                self.assertFalse(os.path.exists(created[0]))

    def test_version_warning(self):
        self.answer(self.host, version="2.6.2\n", record=RECORDS[("gemma-4-26b-a4b", "llamacpp")])
        self.assertEqual(self.check(self.gemma),
                         "warning: Kimi 2.6.2 is not 2.5.3, the version llmctl's Kimi entry was checked with; run the "
                         "Kimi steps of the e2e before relying on it (docs/clients.md#maintenance)")
        self.assertEqual(len(self.host.calls), 2)

    def test_version_failures(self):
        cases = [(fail("kimi: command not found", 127), "kimi: command not found"), (fail("", 2), "exit 2"),
                 (fail("\n  first line  \nsecond\n", 1), "first line"), (ok(""), "it printed no version"),
                 (ok("\n  \n"), "it printed no version")]
        for result, detail in cases:
            with self.subTest(detail=detail):
                self.host = FakeHost(env=self.host.env)
                self.host.on(["kimi", "--version"], result)
                self.assertEqual(self.refusal(self.gemma), f"kimi --version failed: {detail}{OUT_TAIL}")
                self.assertEqual(self.host.calls, [["kimi", "--version"]])

    def test_list_failures(self):
        cases = [(fail("boom\n", 1), "boom"), (fail("", 1), "exit 1"),
                 (ok("not json"), "its output is not a JSON object"), (ok("[]"), "its output is not a JSON object")]
        for result, detail in cases:
            with self.subTest(detail=detail):
                self.host = FakeHost(env=self.host.env)
                self.answer(self.host, result=result)
                self.assertEqual(self.refusal(self.gemma), f"kimi provider list --json failed: {detail}{OUT_TAIL}")

    def test_empty_maps_refuse(self):
        self.answer(self.host, record="provider-list-invalid-toml.json")
        self.assertEqual(self.refusal(self.qwen36),
                         "Kimi 2.5.3 does not read llmctl's entry for qwen3.6-35b-a3b from the new config.toml: no "
                         "provider llmctl; run kimi doctor config /h/.kimi-code/config.toml to see whether Kimi reads "
                         "the rest of the file")

    def test_named_problems(self):
        start = "Kimi 2.5.3 does not read llmctl's entry for gemma-4-26b-a4b from the new config.toml: "
        tail = "; run kimi doctor config /h/.kimi-code/config.toml to see whether Kimi reads the rest of the file"
        no_model = self.gemma_record()
        del no_model["models"]["llmctl/local"]
        short = self.gemma_record()
        short["models"]["llmctl/local"]["maxOutputSize"] = 4096
        other_key = self.gemma_record()
        other_key["providers"]["llmctl"]["apiKey"] = "other"
        no_base = self.gemma_record()
        del no_base["providers"]["llmctl"]["baseUrl"]
        efforts = self.gemma_record()
        efforts["models"]["llmctl/local"]["capabilities"] = ["tool_use", "thinking"]
        as_float = self.gemma_record()
        as_float["models"]["llmctl/local"]["maxContextSize"] = 131072.0
        cases = [(no_model, "no model llmctl/local"), (short, "max_output_size is 4096 instead of 16384"),
                 (other_key, "api_key differs from llmctl's"), (no_base, "base_url is missing"),
                 (efforts, 'capabilities is ["tool_use", "thinking"] instead of ["tool_use"]'),
                 (as_float, "max_context_size is 131072.0 instead of 131072")]
        for data, problem in cases:
            with self.subTest(problem=problem):
                self.host = FakeHost(env=self.host.env)
                self.answer(self.host, data=data)
                self.assertEqual(self.refusal(self.gemma), start + problem + tail)
        own = self.gemma_record()
        own["providers"]["llmctl"]["apiKey"] = "sk-secret-value-123456"
        other = self.gemma_record()
        other["providers"]["other"] = {"type": "openai", "apiKey": "sk-secret-value-123456"}
        other["models"] = {}
        for data, problem in ((own, "api_key differs from llmctl's"), (other, "no model llmctl/local")):
            with self.subTest(problem=problem, secret=True):
                self.host = FakeHost(env=self.host.env)
                self.answer(self.host, data=data)
                message = self.refusal(self.gemma)
                self.assertNotIn("sk-secret", message)
                self.assertEqual(message, start + problem + tail)

    def test_keys_kimi_adds_are_ignored(self):
        data = self.gemma_record()
        data["models"]["llmctl/local"]["contextWindowTokens"] = 5
        self.answer(self.host, data=data)
        self.assertIsNone(self.check(self.gemma))

    def test_other_version_tail(self):
        self.answer(self.host, version="2.6.2\n", record="provider-list-invalid-toml.json")
        self.assertEqual(self.refusal(self.qwen36),
                         "Kimi 2.6.2 does not read llmctl's entry for qwen3.6-35b-a3b from the new config.toml: no "
                         "provider llmctl; llmctl's entry was checked with Kimi 2.5.3, so leave Kimi out with "
                         "LLMCTL_CLIENTS until llmctl supports this version")

    def test_private_folder_failure(self):
        with mock.patch("launcher.kimi_config.tempfile.mkdtemp", side_effect=OSError(28, "No space left on device")):
            self.assertEqual(self.refusal(self.gemma),
                             "cannot check /h/.kimi-code/config.toml with kimi: [Errno 28] No space left on device")
        self.assertEqual(self.host.calls, [])

    def test_read_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            change = FileChange(path, None, "a = 1\r\n", "summary", "preview")
            path.write_bytes(b"a = 1\r\n")
            self.assertIsNone(kimi_config.read_back(change))
            path.write_bytes(b"a = 1\n")
            with self.assertRaises(ClientWriteError) as caught:
                kimi_config.read_back(change)
            self.assertEqual(str(caught.exception), f"{path}: the file changed while llmctl wrote it, probably "
                                                    "through Kimi; check it, then run ./llmctl up again")
            self.assertEqual(caught.exception.exit_code, 6)
            path.unlink()
            with self.assertRaises(ClientWriteError) as caught:
                kimi_config.read_back(change)
            self.assertTrue(str(caught.exception).startswith(f"{path}: cannot read it back after writing: "))
            path.write_bytes(b"\xff\xfe")
            with self.assertRaises(ClientWriteError) as caught:
                kimi_config.read_back(change)
            self.assertTrue(str(caught.exception).startswith(f"{path}: cannot read it back after writing: "))

    def test_read_text_keeps_line_endings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            self.assertIsNone(kimi_config.read_text(path))
            path.write_bytes(b"a = 1\r\n")
            self.assertEqual(kimi_config.read_text(path), "a = 1\r\n")
            path.write_bytes(b"\xff\xfe")
            with self.assertRaises(ClientError) as caught:
                kimi_config.read_text(path)
            self.assertTrue(str(caught.exception).startswith(f"cannot read {path}: "), str(caught.exception))

    def test_unchanged_since_plan(self):
        # Plan ruling 9: a Kimi save between the plan and the rename fails with the read-back message.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            change = FileChange(path, "a = 1\n", "a = 2\n", "summary", "preview")
            path.write_bytes(b"a = 1\n")
            self.assertIsNone(kimi_config.unchanged_since_plan(change))
            path.write_bytes(b"a = 1\nb = 2\n")
            with self.assertRaises(ClientWriteError) as caught:
                kimi_config.unchanged_since_plan(change)
            self.assertEqual(str(caught.exception), f"{path}: the file changed while llmctl wrote it, probably "
                                                    "through Kimi; check it, then run ./llmctl up again")
            self.assertEqual(caught.exception.exit_code, 6)


class LinesTest(unittest.TestCase):
    path = Path("/h/.kimi-code/config.toml")
    cmd = "./llmctl up qwen3.6-35b-a3b"

    def setUp(self):
        self.qwen36 = target_for("qwen3.6-35b-a3b", "llamacpp", RECIPES)
        self.qwen38 = target_for("qwen3.8-27b", "llamacpp", RECIPES)
        self.gptoss = target_for("gpt-oss-20b", "vllm", RECIPES)

    def state(self, old: str | None, target: Target, *, default: str | None = None) -> kimi_config.State:
        text = kimi_config.config_text(self.path, old, target, set_default=False)
        if default is not None:
            text = kimi_config.default_text(self.path, text, default)
        return kimi_config.parse_state(self.path, text)

    def lines(self, state: kimi_config.State | None, target: Target | None, env: dict | None = None) -> list[str]:
        return kimi_config.state_lines(state, target, env or {}, up_command=self.cmd)

    def test_thinking_text(self):
        cases = {("mellum2.1-12b", "vllm"): "on, as the server sets it",
                 ("qwen3.6-35b-a3b", "llamacpp"): "off, as the server sets it",
                 ("qwen3.8-27b", "llamacpp"): "medium effort; /effort in kimi for another (off, low, medium)",
                 ("gpt-oss-20b", "vllm"): "medium effort; /effort in kimi for another (low, medium, high)",
                 ("k2-horizon-7b", "llamacpp"): "always on, at the effort the server sets",
                 ("qwen3-coder-next", "llamacpp"): "none"}
        for (recipe_id, runtime), text in cases.items():
            with self.subTest(recipe_id):
                meta = target_for(recipe_id, runtime, RECIPES).recipe.clients
                self.assertEqual(kimi_config.thinking_text(meta), text)
        odd = dataclasses.replace(self.qwen36.recipe.clients, thinking="sometimes")
        with self.assertRaises(ValueError):
            kimi_config.thinking_text(odd)

    def test_usage_lines(self):
        self.assertEqual(kimi_config.usage_lines(self.qwen36, is_default=True, up_command=self.cmd),
                         ["Kimi: llmctl/local, the default",
                          '  kimi, or for a single prompt: kimi -p "your prompt"',
                          "  thinking: off, as the server sets it"])
        self.assertEqual(kimi_config.usage_lines(self.qwen36, is_default=False, up_command=self.cmd),
                         ["Kimi: llmctl/local, not the default (./llmctl up qwen3.6-35b-a3b --set-default makes it "
                          "one)",
                          "  kimi -m llmctl/local, or pick qwen3.6-35b-a3b on llama.cpp (llmctl) with /model in kimi",
                          "  kimi -p needs Kimi's default even with -m, so for a single prompt run "
                          "./llmctl up qwen3.6-35b-a3b --set-default first",
                          "  thinking: off, as the server sets it"])

    def test_chat_only_lines(self):
        file = str(kimi_config.AGENT_FILE)
        starts = {"llamacpp": "  chat only: Kimi sends its tools with every request, also in its TUI",
                  "vllm": "  chat only: vLLM rejects every Kimi request that carries tools for this recipe (HTTP 400), "
                          "and Kimi's TUI always sends them"}
        for runtime, start in starts.items():
            target = target_for("deepseek-r1-0528-qwen3-8b", runtime, RECIPES)
            # Design D7: Kimi's TUI always sends its tools (P1), so only the print-mode route is printed.
            with self.subTest(runtime, is_default=True):
                lines = kimi_config.usage_lines(target, is_default=True, up_command=self.cmd)
                self.assertEqual(lines, [
                    f"Kimi: {kimi_config.alias(target)}, the default",
                    "  thinking: always on, at the effort the server sets",
                    f'{start}; for a prompt without them: kimi -p "your prompt" --agent-file {file}'])
            with self.subTest(runtime, is_default=False):
                lines = kimi_config.usage_lines(target, is_default=False, up_command=self.cmd)
                self.assertEqual(lines, [
                    f"Kimi: {kimi_config.alias(target)}, not the default (./llmctl up qwen3.6-35b-a3b --set-default "
                    "makes it one)",
                    "  thinking: always on, at the effort the server sets",
                    f"{start}; for a prompt without them, make it the default "
                    f'(./llmctl up qwen3.6-35b-a3b --set-default), then kimi -p "your prompt" --agent-file {file}'])

    def test_chat_only_lines_follow_the_runtime(self):
        # Design D11: the runtime fact refuses_tools_without_parser picks the line, not the label.
        file = str(kimi_config.AGENT_FILE)
        with tempfile.TemporaryDirectory() as tmp, registered():
            sglang = target_for("deepseek-r1-0528-qwen3-8b", "sglang", recipes_with_sglang(Path(tmp)))
        refusing = dataclasses.replace(sglang, runtime=dataclasses.replace(sglang.runtime,
                                                                           refuses_tools_without_parser=True))
        cases = [("vLLM", target_for("deepseek-r1-0528-qwen3-8b", "vllm", RECIPES),
                  "  chat only: vLLM rejects every Kimi request that carries tools for this recipe (HTTP 400), and "
                  "Kimi's TUI always sends them"),
                 ("SGLang", sglang, "  chat only: Kimi sends its tools with every request, also in its TUI"),
                 ("SGLang refusing tools", refusing,
                  "  chat only: SGLang rejects every Kimi request that carries tools for this recipe (HTTP 400), and "
                  "Kimi's TUI always sends them")]
        for name, target, start in cases:
            with self.subTest(name):
                lines = kimi_config.usage_lines(target, is_default=True, up_command=self.cmd)
                self.assertEqual(lines[-1], f'{start}; for a prompt without them: kimi -p "your prompt" '
                                            f"--agent-file {file}")

    def test_sglang_toggle_effort_warning(self):
        # Design D11: SGLang reads every reasoning_effort except none as thinking on for a toggle model.
        with tempfile.TemporaryDirectory() as tmp, registered():
            sglang = target_for("qwen3.5-9b", "sglang", recipes_with_sglang(Path(tmp)))
        vllm = target_for("qwen3.5-9b", "vllm", RECIPES)
        existing = ('warning: [thinking] effort = "{effort}" in /h/.kimi-code/config.toml makes Kimi send '
                    "reasoning_effort {effort} to llmctl/qwen3.5-9b, which sets its thinking on the server (toggle); "
                    "the server may refuse it or think differently; remove that effort line, or choose another with "
                    "/effort in kimi")
        rows = [
            ("low", sglang,
             ['warning: [thinking] effort = "low" in /h/.kimi-code/config.toml makes Kimi send reasoning_effort low '
              "to llmctl/qwen3.5-9b, which turns thinking on for qwen3.5-9b, because SGLang reads every effort except "
              "none as thinking on; remove that effort line, or choose another with /effort in kimi"]),
            ("none", sglang, [existing.format(effort="none")]),
            ("on", sglang, []),
            ("off", sglang, []),
            ("low", vllm, [existing.format(effort="low")]),
        ]
        for effort, target, lines in rows:
            with self.subTest(effort=effort, runtime=target.label):
                state = self.state(f'[thinking]\neffort = "{effort}"\n', target)
                self.assertEqual(self.lines(state, target), lines)

    def test_thinking_warnings(self):
        refuse = ("warning: [thinking] {what} in /h/.kimi-code/config.toml makes Kimi refuse every prompt to "
                  "llmctl/gpt-oss-20b before sending it, because the model reasons by default and declares no off "
                  "effort; set enabled = true there, or choose an effort with /effort in kimi")
        rows = [
            ("enabled = false\n", self.gptoss, refuse.format(what="enabled = false")),
            ('effort = "off"\n', self.gptoss, refuse.format(what='effort = "off"')),
            ("enabled = false\n", self.qwen38,
             "warning: [thinking] enabled = false in /h/.kimi-code/config.toml makes Kimi send reasoning_effort none "
             "to llmctl/local, so qwen3.8-27b runs without thinking; set enabled = true there, or choose an effort "
             "with /effort in kimi"),
            ('effort = "max"\n', self.qwen38,
             'warning: [thinking] effort = "max" in /h/.kimi-code/config.toml makes Kimi send reasoning_effort max '
             "to llmctl/local, which lists only low, medium; the server may refuse it or think differently; remove "
             "that effort line, or choose another with /effort in kimi"),
            ('effort = "high"\n', self.qwen36,
             'warning: [thinking] effort = "high" in /h/.kimi-code/config.toml makes Kimi send reasoning_effort high '
             "to llmctl/local, which sets its thinking on the server (toggle); the server may refuse it or think "
             "differently; remove that effort line, or choose another with /effort in kimi"),
            ('effort = "low"\n', self.gptoss,
             'note: [thinking] effort = "low" in /h/.kimi-code/config.toml makes Kimi use effort low for '
             "llmctl/gpt-oss-20b instead of medium"),
        ]
        for body, target, line in rows:
            with self.subTest(body=body, target=target.recipe.id):
                self.assertEqual(self.lines(self.state("[thinking]\n" + body, target), target), [line])
        quiet = [('effort = "medium"\n', self.qwen38), ('effort = "on"\n', self.qwen36),
                 ("effort = 5\n", self.qwen38), ('enabled = false\neffort = "high"\n', self.qwen36)]
        for body, target in quiet:
            with self.subTest(body=body, target=target.recipe.id):
                self.assertEqual(self.lines(self.state("[thinking]\n" + body, target), target), [])
        with self.subTest("no [thinking] table"):
            self.assertEqual(self.lines(self.state(None, self.gptoss), self.gptoss), [])
        with self.subTest("another recipe's entry"):
            self.assertEqual(self.lines(self.state("[thinking]\nenabled = false\n", self.qwen38), self.gptoss), [])
        with self.subTest("no target"):
            self.assertEqual(self.lines(self.state("[thinking]\nenabled = false\n", self.gptoss), None), [])

    def test_missing_alias_warning(self):
        state = self.state(None, self.qwen36, default="llmctl/gone")
        self.assertEqual(self.lines(state, self.qwen36),
                         ["warning: Kimi's default_model llmctl/gone names no model in /h/.kimi-code/config.toml, so "
                          'every kimi -p fails (Model "llmctl/gone" is not configured in config.toml.); pick a model '
                          "with /model in kimi, or run ./llmctl up qwen3.6-35b-a3b --set-default"])
        other = self.state('default_model = "other/m"\n\n[models."other/m"]\nprovider = "x"\nmodel = "m"\n',
                           self.qwen36)
        self.assertEqual(other.default, "other/m")
        self.assertEqual(self.lines(other, self.qwen36), [])
        self.assertEqual(self.lines(self.state(None, self.qwen36), self.qwen36), [])

    def test_kimi_model_name_note(self):
        state = self.state(None, self.qwen36)
        note = ("note: KIMI_MODEL_NAME is set in this shell, so kimi uses that model instead of Kimi's default_model; "
                "unset it, or pass -m {name}")
        self.assertEqual(self.lines(state, self.qwen36, {"KIMI_MODEL_NAME": "x"}),
                         [note.format(name="llmctl/local")])
        self.assertEqual(self.lines(state, self.qwen36, {"KIMI_MODEL_NAME": ""}), [])
        self.assertEqual(self.lines(self.state(None, self.gptoss), None, {"KIMI_MODEL_NAME": "x"}),
                         [note.format(name="llmctl/gpt-oss-20b")])
        self.assertEqual(self.lines(None, self.gptoss, {"KIMI_MODEL_NAME": "x"}),
                         [note.format(name="llmctl/gpt-oss-20b")])
        self.assertEqual(self.lines(None, None, {"KIMI_MODEL_NAME": "x"}),
                         [note.format(name="llmctl/<served name>")])

    def test_state_lines_order(self):
        state = self.state("[thinking]\nenabled = false\n", self.gptoss, default="llmctl/gone")
        lines = self.lines(state, self.gptoss, {"KIMI_MODEL_NAME": "x"})
        self.assertEqual([line.split(" ", 2)[:2] for line in lines],
                         [["warning:", "[thinking]"], ["warning:", "Kimi's"], ["note:", "KIMI_MODEL_NAME"]])

    def test_session_note(self):
        self.assertIsNone(kimi_config.session_note(None, self.qwen36))
        self.assertIsNone(kimi_config.session_note(kimi_config.parse_state(self.path, "telemetry = false\n"),
                                                   self.qwen36))
        self.assertIsNone(kimi_config.session_note(self.state(None, self.qwen36), self.qwen36))
        self.assertEqual(kimi_config.session_note(self.state(None, self.gptoss), self.qwen36),
                         "note: an open Kimi session on llmctl/gpt-oss-20b fails its next turn until you pick "
                         "llmctl/local with /model or start a new session")
        self.assertEqual(kimi_config.session_note(self.state(None, self.qwen38), self.qwen36),
                         "note: an open Kimi session on llmctl/local sends its next turn to qwen3.6-35b-a3b on "
                         "llama.cpp but keeps its old thinking level; start a new session")

    def test_switch_back_and_retry_note(self):
        self.assertEqual(kimi_config.switch_back({}), "Kimi /model")
        self.assertEqual(kimi_config.RETRY_NOTE,
                         "  Kimi retries a stopped server for about two and a half minutes before a prompt fails")


if __name__ == "__main__":
    unittest.main()
