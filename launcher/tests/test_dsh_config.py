import dataclasses
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from launcher import dsh_config, files, hermes_config
from launcher.catalog import load_catalog
from launcher.errors import ClientError, UsageError
from launcher.host import Result
from launcher.tests.fakes import REPO, FakeHost, fail, ok
from launcher.tests.regen_golden import target_for
from launcher.tests.sglang_fixture import recipes_with_sglang, registered

RECIPES = REPO / "recipes"
DSH_FIXTURES = REPO / "launcher/tests/fixtures/dsh"


def dsh_fixture(name: str, patch: str) -> str:
    """A fixture cut from the design's experiments, with every {patch} replaced by the given home patch path."""
    return (DSH_FIXTURES / name).read_text(encoding="utf-8").replace("{patch}", patch)

# The design is git-ignored, so its text is copied here verbatim (docs/superpowers/specs/2026-11-16-llmctl-dsh-design.md).
# D2, lines 138-141: the header of a new home patch.
DESIGN_HEADER = """\
# dsh rows for the model llmctl runs (docs/llmctl.md#clients).
# llmctl rewrites the lines between the two markers on every ./llmctl up and keeps everything else in this file.
# While the block holds the agent-default-model line, the running model is dsh's default in every profile;
# delete that line and the acp line to give the default back to dsh's model picker.
"""

# D3, lines 193-206: the no-tools rows of a chat-only prompt.
DESIGN_NO_TOOLS = """\
- {"id": "tool-bash", "disabled": true}
- {"id": "tool-goal", "disabled": true}
- {"id": "tool-fs", "disabled": true}
- {"id": "tool-fs-search", "disabled": true}
- {"id": "tool-jobs", "disabled": true}
- {"id": "tool-subagent-control", "disabled": true}
- {"id": "tool-skill", "disabled": true}
- {"id": "tool-subagent", "disabled": true}
- {"id": "tool-subagent-fork", "disabled": true}
- {"id": "tool-todo", "disabled": true}
- {"id": "tool-web", "disabled": true}
- {"id": "tool-workflow", "disabled": true}
- {"id": "tool-working-directory", "disabled": true}
- {"id": "tool-agent-team", "disabled": true}
"""

# Generated rows, lines 417-427: the owner's home patch after ./llmctl up gemma-4-26b-a4b --set-default.
DESIGN_OWNER_EXAMPLE = """\
- id: session-telemetry-otel
  disabled: true
# dsh rows for the model llmctl runs (docs/llmctl.md#clients).
# llmctl rewrites the lines between the two markers on every ./llmctl up and keeps everything else in this file.
# While the block holds the agent-default-model line, the running model is dsh's default in every profile;
# delete that line and the acp line to give the default back to dsh's model picker.
# >>> llmctl managed block: rewritten by llmctl on every start; edits here are lost >>>
- {"id": "llm-pi-ai", "config": {"providers": {"llmctl": {"displayName": "llmctl", "api": "openai-completions", "baseURL": "http://127.0.0.1:8080/v1", "headers": {"Authorization": "Bearer no-key-required"}, "compat": {"supportsStore": false, "supportsDeveloperRole": false, "maxTokensField": "max_tokens"}, "reasoning": "off", "streamIdleTimeoutMs": 900000, "retryPolicy": {"mode": "normal", "maxRetries": 5, "retryableCodes": ["EMPTY_RESPONSE", "RATE_LIMIT", "SERVER", "TRANSPORT"]}, "models": [{"id": "local", "name": "gemma-4-26b-a4b on llama.cpp (llmctl)", "contextWindow": 131072, "maxTokens": 16384, "input": ["text"], "reasoningEfforts": {"off": null, "medium": "medium"}, "compat": {"thinkingFormat": "chat-template", "chatTemplateKwargs": {"enable_thinking": {"$var": "thinking.enabled"}}}}]}}}}
- {"id": "agent-default-model", "config": {"provider": "llmctl", "model": "local"}}
- {"id": "acp", "config": {"provider": "llmctl", "model": "local"}}
# <<< llmctl managed block <<<
"""

# Generated rows, lines 435-437: qwen3.6-35b-a3b on llama.cpp.
DESIGN_QWEN36 = """\
- {"id": "llm-pi-ai", "config": {"providers": {"llmctl": {"displayName": "llmctl", "api": "openai-completions", "baseURL": "http://127.0.0.1:8080/v1", "headers": {"Authorization": "Bearer no-key-required"}, "compat": {"supportsStore": false, "supportsDeveloperRole": false, "maxTokensField": "max_tokens"}, "reasoning": "off", "streamIdleTimeoutMs": 900000, "retryPolicy": {"mode": "normal", "maxRetries": 5, "retryableCodes": ["EMPTY_RESPONSE", "RATE_LIMIT", "SERVER", "TRANSPORT"]}, "models": [{"id": "local", "name": "qwen3.6-35b-a3b on llama.cpp (llmctl)", "contextWindow": 131072, "maxTokens": 16384, "input": ["text"], "reasoningEfforts": {"off": null, "medium": "medium"}, "compat": {"thinkingFormat": "chat-template", "chatTemplateKwargs": {"enable_thinking": {"$var": "thinking.enabled"}}}}]}}}}
- {"id": "agent-default-model", "config": {"provider": "llmctl", "model": "local"}}
- {"id": "acp", "config": {"provider": "llmctl", "model": "local"}}
"""

# Generated rows, lines 443-445: gpt-oss-20b on vLLM.
DESIGN_GPTOSS = """\
- {"id": "llm-pi-ai", "config": {"providers": {"llmctl": {"displayName": "llmctl", "api": "openai-completions", "baseURL": "http://127.0.0.1:8000/v1", "headers": {"Authorization": "Bearer no-key-required"}, "compat": {"supportsStore": false, "supportsDeveloperRole": false, "maxTokensField": "max_tokens"}, "reasoning": "medium", "streamIdleTimeoutMs": 900000, "retryPolicy": {"mode": "normal", "maxRetries": 5, "retryableCodes": ["EMPTY_RESPONSE", "RATE_LIMIT", "SERVER", "TRANSPORT"]}, "models": [{"id": "gpt-oss-20b", "name": "gpt-oss-20b on vLLM (llmctl)", "contextWindow": 65536, "maxTokens": 16384, "input": ["text"], "reasoningEfforts": {"low": "low", "medium": "medium", "high": "high"}, "compat": {"thinkingFormat": "openai", "supportsReasoningEffort": true}}]}}}}
- {"id": "agent-default-model", "config": {"provider": "llmctl", "model": "gpt-oss-20b"}}
- {"id": "acp", "config": {"provider": "llmctl", "model": "gpt-oss-20b"}}
"""

# Generated rows, the llm-pi-ai line of the other thinking types.
DESIGN_OTHER_TYPES = (
    ("llamacpp", "nemotron-3.5-bolt", """\
- {"id": "llm-pi-ai", "config": {"providers": {"llmctl": {"displayName": "llmctl", "api": "openai-completions", "baseURL": "http://127.0.0.1:8080/v1", "headers": {"Authorization": "Bearer no-key-required"}, "compat": {"supportsStore": false, "supportsDeveloperRole": false, "maxTokensField": "max_tokens"}, "reasoning": "off", "thinkingBudgets": {"medium": 1, "high": 0}, "streamIdleTimeoutMs": 900000, "retryPolicy": {"mode": "normal", "maxRetries": 5, "retryableCodes": ["EMPTY_RESPONSE", "RATE_LIMIT", "SERVER", "TRANSPORT"]}, "models": [{"id": "local", "name": "nemotron-3.5-bolt on llama.cpp (llmctl)", "contextWindow": 131072, "maxTokens": 16384, "input": ["text"], "reasoningEfforts": {"off": null, "medium": "medium", "high": "high"}, "compat": {"thinkingFormat": "chat-template", "chatTemplateKwargs": {"enable_thinking": {"$var": "thinking.enabled"}, "medium_effort": {"$var": "thinking.budget"}}}}]}}}}
"""),  # line 452
    ("llamacpp", "qwen3.8-27b", """\
- {"id": "llm-pi-ai", "config": {"providers": {"llmctl": {"displayName": "llmctl", "api": "openai-completions", "baseURL": "http://127.0.0.1:8080/v1", "headers": {"Authorization": "Bearer no-key-required"}, "compat": {"supportsStore": false, "supportsDeveloperRole": false, "maxTokensField": "max_tokens"}, "reasoning": "medium", "streamIdleTimeoutMs": 900000, "retryPolicy": {"mode": "normal", "maxRetries": 5, "retryableCodes": ["EMPTY_RESPONSE", "RATE_LIMIT", "SERVER", "TRANSPORT"]}, "models": [{"id": "local", "name": "qwen3.8-27b on llama.cpp (llmctl)", "contextWindow": 65536, "maxTokens": 16384, "input": ["text"], "reasoningEfforts": {"off": "none", "low": "low", "medium": "medium"}, "compat": {"thinkingFormat": "openai", "supportsReasoningEffort": true}}]}}}}
"""),  # line 454
    ("llamacpp", "k2-horizon-7b", """\
- {"id": "llm-pi-ai", "config": {"providers": {"llmctl": {"displayName": "llmctl", "api": "openai-completions", "baseURL": "http://127.0.0.1:8080/v1", "headers": {"Authorization": "Bearer no-key-required"}, "compat": {"supportsStore": false, "supportsDeveloperRole": false, "maxTokensField": "max_tokens"}, "reasoning": "high", "streamIdleTimeoutMs": 900000, "retryPolicy": {"mode": "normal", "maxRetries": 5, "retryableCodes": ["EMPTY_RESPONSE", "RATE_LIMIT", "SERVER", "TRANSPORT"]}, "models": [{"id": "local", "name": "k2-horizon-7b on llama.cpp (llmctl)", "contextWindow": 81920, "maxTokens": 32768, "input": ["text"], "reasoningEfforts": {"high": "high"}, "compat": {"supportsReasoningEffort": false}}]}}}}
"""),  # line 456
    ("llamacpp", "qwen3-coder-next", """\
- {"id": "llm-pi-ai", "config": {"providers": {"llmctl": {"displayName": "llmctl", "api": "openai-completions", "baseURL": "http://127.0.0.1:8080/v1", "headers": {"Authorization": "Bearer no-key-required"}, "compat": {"supportsStore": false, "supportsDeveloperRole": false, "maxTokensField": "max_tokens"}, "streamIdleTimeoutMs": 900000, "retryPolicy": {"mode": "normal", "maxRetries": 5, "retryableCodes": ["EMPTY_RESPONSE", "RATE_LIMIT", "SERVER", "TRANSPORT"]}, "models": [{"id": "local", "name": "qwen3-coder-next on llama.cpp (llmctl)", "contextWindow": 131072, "maxTokens": 16384, "input": ["text"], "reasoningEfforts": false}]}}}}
"""),  # line 458
    ("llamacpp", "deepseek-r1-0528-qwen3-8b", """\
- {"id": "llm-pi-ai", "config": {"providers": {"llmctl": {"displayName": "llmctl", "api": "openai-completions", "baseURL": "http://127.0.0.1:8080/v1", "headers": {"Authorization": "Bearer no-key-required"}, "compat": {"supportsStore": false, "supportsDeveloperRole": false, "maxTokensField": "max_tokens"}, "reasoning": "high", "streamIdleTimeoutMs": 900000, "retryPolicy": {"mode": "normal", "maxRetries": 5, "retryableCodes": ["EMPTY_RESPONSE", "RATE_LIMIT", "SERVER", "TRANSPORT"]}, "models": [{"id": "local", "name": "deepseek-r1-0528-qwen3-8b on llama.cpp (llmctl)", "contextWindow": 65536, "maxTokens": 16384, "input": ["text"], "reasoningEfforts": {"high": "high"}, "compat": {"supportsReasoningEffort": false}}]}}}}
"""),  # line 460
    ("vllm", "mellum2.1-12b", """\
- {"id": "llm-pi-ai", "config": {"providers": {"llmctl": {"displayName": "llmctl", "api": "openai-completions", "baseURL": "http://127.0.0.1:8000/v1", "headers": {"Authorization": "Bearer no-key-required"}, "compat": {"supportsStore": false, "supportsDeveloperRole": false, "maxTokensField": "max_tokens"}, "reasoning": "medium", "streamIdleTimeoutMs": 900000, "retryPolicy": {"mode": "normal", "maxRetries": 5, "retryableCodes": ["EMPTY_RESPONSE", "RATE_LIMIT", "SERVER", "TRANSPORT"]}, "models": [{"id": "mellum2.1", "name": "mellum2.1-12b on vLLM (llmctl)", "contextWindow": 131072, "maxTokens": 16384, "input": ["text"], "reasoningEfforts": {"off": null, "medium": "medium"}, "compat": {"thinkingFormat": "chat-template", "chatTemplateKwargs": {"enable_thinking": {"$var": "thinking.enabled"}}}}]}}}}
"""),  # line 462
)


def all_targets():
    catalog = load_catalog(RECIPES, None)
    for runtime, entry in catalog.runtimes.items():
        for recipe_id in sorted(entry.recipes):
            yield runtime, recipe_id, target_for(recipe_id, runtime, RECIPES)


def with_medium_kwargs(target, medium_kwargs):
    recipe = target.recipe
    clients = dataclasses.replace(recipe.clients, medium_kwargs=medium_kwargs)
    return dataclasses.replace(target, recipe=dataclasses.replace(recipe, clients=clients))


class PathsTest(unittest.TestCase):
    env = {"HOME": "/h"}

    def home(self, value, env=None):
        mapping = dict(self.env if env is None else env)
        if value is not None:
            mapping["DSH_HOME"] = value
        return dsh_config.dsh_home(mapping)

    def test_dsh_home_default(self):
        for value in (None, "", "   ", "\t\n"):
            with self.subTest(value=value):
                self.assertEqual(self.home(value), Path("/h/.dsh"))

    def test_dsh_home_tilde_forms(self):
        for value, expected in (("~", "/h"), ("~/x", "/h/x"), ("~\\x", "/h/x"), ("~/", "/h"), ("~//x", "/h/x")):
            with self.subTest(value=value):
                self.assertEqual(self.home(value), Path(expected))
        self.assertEqual(self.home("~/x", {"HOME": "/h/"}), Path("/h/x"))

    def test_dsh_home_absolute_is_normalized(self):
        for value, expected in (("/a/b/../c/", "/a/c"), ("/a/./b", "/a/b"), ("//a", "/a")):
            with self.subTest(value=value):
                self.assertEqual(self.home(value), Path(expected))
                self.assertEqual(str(self.home(value)), expected)
        self.assertEqual(dsh_config.dsh_home({"DSH_HOME": "/abs"}), Path("/abs"))

    def test_dsh_home_keeps_dollar_names(self):
        self.assertEqual(self.home("/a/$X/b", {"HOME": "/h", "X": "y"}), Path("/a/$X/b"))

    def test_dsh_home_relative_refused(self):
        for value in ("rel", "./x", "~bob/x", " /a", "$HOME/.dsh"):
            with self.subTest(value=value):
                with self.assertRaises(UsageError) as caught:
                    self.home(value)
                self.assertEqual(caught.exception.exit_code, 1)
                self.assertEqual(str(caught.exception),
                                 f"DSH_HOME is relative ({value}), and dsh resolves it against each command's working "
                                 "folder; set it to an absolute path, ~ or ~/...")

    def test_dsh_home_needs_home_for_the_default(self):
        for env in ({}, {"DSH_HOME": "~/x"}):
            with self.subTest(env=env):
                with self.assertRaises(UsageError) as caught:
                    dsh_config.dsh_home(env)
                self.assertTrue(str(caught.exception).startswith("HOME is not set"), str(caught.exception))

    def test_paths(self):
        self.assertEqual(dsh_config.home_patch(self.env), Path("/h/.dsh/cordis.patch.yml"))
        self.assertEqual(dsh_config.profiles_dir(self.env), Path("/h/.dsh/profiles"))


class GenerateTest(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(dsh_config.PROVIDER, "llmctl")
        self.assertEqual(dsh_config.VERIFIED_VERSION, "0.2.2-rc.3")
        self.assertEqual(dsh_config.OWNED_ROWS, ("llm-pi-ai", "agent-default-model", "acp"))
        self.assertEqual(dsh_config.STREAM_IDLE_TIMEOUT_MS, 900000)
        self.assertEqual(dsh_config.RETRYABLE_CODES, ("EMPTY_RESPONSE", "RATE_LIMIT", "SERVER", "TRANSPORT"))
        self.assertEqual(dsh_config.COMPACTION_HEADROOM, 65536)
        self.assertEqual(dsh_config.NO_TOOLS_ROWS, (
            "tool-bash", "tool-goal", "tool-fs", "tool-fs-search", "tool-jobs", "tool-subagent-control", "tool-skill",
            "tool-subagent", "tool-subagent-fork", "tool-todo", "tool-web", "tool-workflow", "tool-working-directory",
            "tool-agent-team"))
        self.assertEqual(dsh_config.OUT, "leave dsh out with LLMCTL_CLIENTS")
        self.assertEqual(dsh_config.HEADER, DESIGN_HEADER)
        self.assertEqual(dsh_config.BEGIN, "# >>> llmctl managed block: rewritten by llmctl on every start; edits here are lost >>>")
        self.assertEqual(dsh_config.END, "# <<< llmctl managed block <<<")
        self.assertEqual(dsh_config.BEGIN.encode(), hermes_config.BEGIN.encode())
        self.assertEqual(dsh_config.END.encode(), hermes_config.END.encode())

    def test_qwen36_block_is_the_design_block(self):
        target = target_for("qwen3.6-35b-a3b", "llamacpp", RECIPES)
        self.assertEqual(dsh_config.render_block(target, default=True), DESIGN_QWEN36)

    def test_gptoss_block_is_the_design_block(self):
        target = target_for("gpt-oss-20b", "vllm", RECIPES)
        self.assertEqual(dsh_config.render_block(target, default=True), DESIGN_GPTOSS)

    def test_gemma_home_text_is_the_owner_example(self):
        target = target_for("gemma-4-26b-a4b", "llamacpp", RECIPES)
        owner = "- id: session-telemetry-otel\n  disabled: true\n"
        self.assertEqual(owner + dsh_config.new_home_text(target, default=True), DESIGN_OWNER_EXAMPLE)

    def test_llm_pi_ai_lines_match_the_design(self):
        for runtime, recipe_id, line in DESIGN_OTHER_TYPES:
            with self.subTest(recipe=f"{runtime}/{recipe_id}"):
                target = target_for(recipe_id, runtime, RECIPES)
                self.assertEqual(dsh_config.render_block(target, default=False), line)

    def test_default_lines(self):
        qwen36 = target_for("qwen3.6-35b-a3b", "llamacpp", RECIPES)
        self.assertEqual(dsh_config.rows(qwen36, default=True)[1:], [
            {"id": "agent-default-model", "config": {"provider": "llmctl", "model": "local"}},
            {"id": "acp", "config": {"provider": "llmctl", "model": "local"}}])
        self.assertEqual(len(dsh_config.rows(qwen36, default=False)), 1)
        for runtime, recipe_id, target in all_targets():
            for default in (True, False):
                with self.subTest(recipe=f"{runtime}/{recipe_id}", default=default):
                    texts = (dsh_config.render_block(target, default=default),
                             dsh_config.new_home_text(target, default=default), dsh_config.default_overlay(target))
                    for text in texts:
                        self.assertNotIn('"reasoningEffort"', text)

    def test_every_recipe_describes_its_server(self):
        count = 0
        for runtime, recipe_id, target in all_targets():
            with self.subTest(recipe=f"{runtime}/{recipe_id}"):
                route = dsh_config.route(target)
                self.assertEqual(route["baseURL"], f"http://127.0.0.1:{target.runtime.host_port}/v1")
                model = route["models"][0]
                self.assertEqual(model["id"], target.served)
                self.assertEqual(model["contextWindow"], target.recipe.context)
                self.assertEqual(model["maxTokens"], target.recipe.clients.output)
            count += 1
        self.assertEqual(count, 25)

    def test_lines_are_ascii_and_reparse(self):
        for runtime, recipe_id, target in all_targets():
            for default in (True, False):
                with self.subTest(recipe=f"{runtime}/{recipe_id}", default=default):
                    self.assert_lines(target, default)

    def assert_lines(self, target, default):
        lines = dsh_config.render_block(target, default=default).splitlines(keepends=True)
        rows = dsh_config.rows(target, default=default)
        self.assertEqual(len(lines), len(rows))
        for line, row in zip(lines, rows):
            self.assertTrue(line.startswith("- "), line)
            self.assertTrue(line.endswith("\n"), line)
            self.assertTrue(line.isascii(), line)
            self.assertNotIn("\\/", line)
            self.assertEqual(json.loads(line[2:-1]), row)
            self.assertEqual(dsh_config.row_line(row), line)

    def test_odd_recipe_id_renders(self):
        target = target_for("qwen3.6-35b-a3b", "llamacpp", RECIPES)
        odd = dataclasses.replace(target, recipe=dataclasses.replace(target.recipe, id='modèle#1 "x"'))
        for default in (True, False):
            with self.subTest(default=default):
                self.assert_lines(odd, default)
        first = dsh_config.render_block(odd, default=True).splitlines()[0]
        model = json.loads(first[2:])["config"]["providers"]["llmctl"]["models"][0]
        self.assertEqual(model["name"], 'modèle#1 "x" on llama.cpp (llmctl)')

    def test_medium_kwargs_must_be_true(self):
        bolt = target_for("nemotron-3.5-bolt", "llamacpp", RECIPES)
        with self.assertRaises(ClientError) as caught:
            dsh_config.render_block(with_medium_kwargs(bolt, (("medium_effort", "high"),)), default=False)
        self.assertEqual(caught.exception.exit_code, 3)
        self.assertEqual(str(caught.exception),
                         'dsh cannot send medium_effort = "high" of nemotron-3.5-bolt per level, because llmctl maps a '
                         "middle level only to true; leave dsh out with LLMCTL_CLIENTS for this recipe")
        with self.assertRaises(ClientError) as caught:
            dsh_config.route(with_medium_kwargs(bolt, (("medium_effort", 1),)))
        self.assertIn("medium_effort = 1 ", str(caught.exception))
        self.assertEqual(bolt.recipe.clients.medium_kwargs, (("medium_effort", True),))
        dsh_config.route(bolt)

    def test_no_tools_patch(self):
        self.assertEqual(dsh_config.no_tools_patch(), DESIGN_NO_TOOLS)

    def test_chat_only_wording_follows_the_runtime(self):
        # Design D11: the runtime fact refuses_tools_without_parser picks the line, not the label.
        rows = "use dsh --profile headless --patch <file> with the rows in docs/clients.md#chat-only-recipes"
        with tempfile.TemporaryDirectory() as tmp, registered():
            sglang = target_for("deepseek-r1-0528-qwen3-8b", "sglang", recipes_with_sglang(Path(tmp)))
        refusing = dataclasses.replace(sglang, runtime=dataclasses.replace(sglang.runtime,
                                                                           refuses_tools_without_parser=True))
        cases = [("vLLM", target_for("deepseek-r1-0528-qwen3-8b", "vllm", RECIPES),
                  f"  chat only: vLLM rejects every dsh request that carries tools for this recipe (HTTP 400); {rows}"),
                 ("SGLang", sglang, f"  chat only: dsh still sends its tools; for a prompt without them {rows}"),
                 ("SGLang refusing tools", refusing,
                  f"  chat only: SGLang rejects every dsh request that carries tools for this recipe (HTTP 400); "
                  f"{rows}")]
        for name, target, line in cases:
            with self.subTest(name):
                lines = dsh_config.usage_lines(target, held=True, up_command="./llmctl up deepseek-r1")
                self.assertEqual([item for item in lines if item.startswith("  chat only: ")], [line])

    def test_default_overlay(self):
        target = target_for("gpt-oss-20b", "vllm", RECIPES)
        self.assertEqual(dsh_config.default_overlay(target),
                         '- {"id": "agent-default-model", "config": {"provider": "llmctl", "model": "gpt-oss-20b"}}\n')


TELEMETRY = "- id: session-telemetry-otel\n  disabled: true\n"
BOM = "﻿"

# dsh's profile template, $SCRATCH/dsh/templates/web.cordis.patch.yml (the headless one is equal), byte for byte.
PROFILE_TEMPLATE_HEADER = """\
# Your patch layer for this dsh profile, applied after every bundle layer:
# a top-level YAML array of loader patch entries (id-targeted config
# overrides, disables, and insert lists; `!!js` expressions allowed).
"""
PROFILE_TEMPLATE = PROFILE_TEMPLATE_HEADER + "[]\n"

# Rows in the style the Web UI re-emits (design L2, A2-profile).
WEB_UI_SETTINGS = """\
- id: ui-settings-general
  config:
    theme: dark
"""
WEB_UI_LLM_PI_AI = """\
- id: llm-pi-ai
  config:
    providers:
      llamacpp:
        api: openai-completions
        baseURL: http://127.0.0.1:8080/v1
        models:
          - id: local
            name: gemma-4-26b-a4b
            input: [ text ]
"""
WEB_UI_DEFAULT = """\
- id: agent-default-model
  config:
    provider: llamacpp
    model: local
    reasoningEffort: 'off'
"""

UNKNOWN_LINE = ("{path} line {line}: the llmctl block holds a line llmctl did not write; delete it, or move your row "
                "after the end marker")
TOP_LEVEL = "{path} line {line}: llmctl cannot check this top-level line; write each row as a '- ' list item"
EMPTY_LIST = ("{path} line {line}: dsh reads this file as one YAML list, so llmctl cannot add its rows after '[]'; "
              "delete that line")
OUTSIDE_PI_AI = ("{path} line {line}: the llm-pi-ai row is set outside the llmctl block, and dsh applies only one of "
                 "the two; delete it (docs/clients.md#the-home-patch)")
OUTSIDE_DEFAULT = ("{path} line {line}: the {row} row is set outside the llmctl block while llmctl holds or sets dsh's "
                   "default; delete that row, or delete llmctl's agent-default-model and acp lines and run up without "
                   "--set-default")


def block_text(lines: str) -> str:
    """A home patch holding only llmctl's header and a block of the given lines."""
    return dsh_config.HEADER + dsh_config.BEGIN + "\n" + lines + dsh_config.END + "\n"


class HomeCase(unittest.TestCase):
    """A temporary HOME whose .dsh/profiles exists, passed only through the mapping."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = {"HOME": self.tmp.name}
        self.path = Path(self.tmp.name) / ".dsh" / "cordis.patch.yml"
        (Path(self.tmp.name) / ".dsh" / "profiles").mkdir(parents=True)
        self.qwen36 = target_for("qwen3.6-35b-a3b", "llamacpp", RECIPES)
        self.gptoss = target_for("gpt-oss-20b", "vllm", RECIPES)
        self.gemma = target_for("gemma-4-26b-a4b", "llamacpp", RECIPES)

    def tearDown(self):
        self.tmp.cleanup()

    def write_home(self, text: str) -> None:
        with open(self.path, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)

    def write_profile(self, name: str, text: str) -> Path:
        path = Path(self.tmp.name) / ".dsh" / "profiles" / name / "cordis.patch.yml"
        path.parent.mkdir(exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        return path

    def home_bytes(self) -> bytes | None:
        return self.path.read_bytes() if self.path.exists() else None

    def refuses(self, message: str, *, set_default: bool = False, target=None) -> None:
        before = self.home_bytes()
        with self.assertRaises(ClientError) as caught:
            dsh_config.plan_home(self.env, target or self.qwen36, set_default=set_default)
        self.assertEqual(str(caught.exception), message)
        self.assertEqual(caught.exception.exit_code, 3)
        self.assertEqual(self.home_bytes(), before)
        self.assertEqual(sorted(os.listdir(self.path.parent)), sorted(["profiles"] + (["cordis.patch.yml"]
                                                                                      if before is not None else [])))

    def plan(self, *, set_default: bool = False, target=None):
        return dsh_config.plan_home(self.env, target or self.qwen36, set_default=set_default)


class BlockTest(HomeCase):
    def test_read_block_without_block(self):
        self.assertIsNone(dsh_config.read_block(self.env))
        self.assertFalse(dsh_config.holds_default(self.env))
        self.write_home(TELEMETRY)
        self.assertIsNone(dsh_config.read_block(self.env))
        self.assertFalse(dsh_config.holds_default(self.env))
        self.assertIsNone(dsh_config.parse_block(self.path, TELEMETRY))

    def test_read_block_rows(self):
        self.write_home(dsh_config.new_home_text(self.qwen36, default=True))
        block = dsh_config.read_block(self.env)
        self.assertEqual(block.rows, tuple(dsh_config.rows(self.qwen36, default=True)))
        self.assertTrue(block.holds_default)
        self.assertTrue(dsh_config.holds_default(self.env))
        self.assertEqual(block.description, "qwen3.6-35b-a3b on llama.cpp")
        self.assertEqual(block.served, "local")
        self.write_home(dsh_config.new_home_text(self.gptoss, default=False))
        block = dsh_config.read_block(self.env)
        self.assertFalse(block.holds_default)
        self.assertFalse(dsh_config.holds_default(self.env))
        self.assertEqual((block.description, block.served), ("gpt-oss-20b on vLLM", "gpt-oss-20b"))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            block.served = "x"

    def test_block_without_readable_llm_pi_ai_row(self):
        for lines in ("", '- {"id": "llm-pi-ai", "config": {}}\n',
                      '- {"id": "llm-pi-ai", "config": {"providers": {"llmctl": {"models": '
                      '[{"id": 1, "name": "x"}]}}}}\n'):
            with self.subTest(lines=lines):
                block = dsh_config.parse_block(self.path, block_text(lines))
                self.assertEqual((block.description, block.served), (None, None))

    def test_block_skips_blank_and_comment_lines(self):
        lines = dsh_config.render_block(self.qwen36, default=True).splitlines(keepends=True)
        text = block_text(lines[0] + "\n" + "  # note\n" + "".join(lines[1:]))
        block = dsh_config.parse_block(self.path, text)
        self.assertEqual(block.rows, tuple(dsh_config.rows(self.qwen36, default=True)))

    def test_unknown_block_lines_refused(self):
        pi_ai, default, acp = dsh_config.render_block(self.qwen36, default=True).splitlines(keepends=True)
        surrogate = pi_ai.replace('"id": "local"', '"id": "\\ud800"')
        self.assertIn("\\ud800", surrogate)
        first = 6  # the header's four lines and the begin marker come first
        for lines, bad in (('  - {"id": "acp", "config": {}}\n', 0), ("- not json\n", 0), ("- [1]\n", 0),
                           ('- {"id": "tool-web", "disabled": true}\n', 0), ('-{"id": "acp"}\n', 0),
                           (pi_ai + acp + default, 2), (pi_ai + pi_ai, 1),
                           ('- {"id": "acp"}\r- {"id": "acp"}\n', 0), ('# note\r- {"id": "acp"}\n', 0),
                           (surrogate, 0), (pi_ai + "\n" + "x\n", 2)):
            with self.subTest(lines=lines[:40]):
                with self.assertRaises(ClientError) as caught:
                    dsh_config.parse_block(self.path, block_text(lines))
                self.assertEqual(str(caught.exception), UNKNOWN_LINE.format(path=self.path, line=first + bad))

    def test_holds_default(self):
        self.write_home(block_text('- {"id": "acp", "config": {"provider": "llmctl", "model": "local"}}\n'))
        self.assertFalse(dsh_config.holds_default(self.env))
        self.write_home(block_text('- {"id": "agent-default-model", "config": {"provider": "llmctl", "model": "x"}}\n'))
        self.assertTrue(dsh_config.holds_default(self.env))

    def test_finding_is_frozen(self):
        finding = dsh_config.Finding(Path("/p"), 1, "acp", "web")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            finding.line = 2


class PlanTest(HomeCase):
    QWEN36 = "llmctl rows for qwen3.6-35b-a3b on llama.cpp"

    def test_new_file(self):
        change = self.plan()
        summary = ("created it with the llmctl rows for qwen3.6-35b-a3b on llama.cpp; dsh's default stays with its "
                   "model picker")
        self.assertIsNone(change.old_text)
        self.assertEqual(change.path, self.path)
        self.assertEqual(change.new_text, dsh_config.new_home_text(self.qwen36, default=False))
        self.assertEqual((change.mode, change.backup, change.dirs), (0o600, False, ()))
        self.assertEqual(change.summary, summary)
        self.assertEqual(change.preview, f"would write {self.path}: {summary}")
        change = self.plan(set_default=True)
        self.assertEqual(change.new_text, dsh_config.new_home_text(self.qwen36, default=True))
        self.assertTrue(change.summary.endswith(
            "; dsh's default llmctl/local in every profile (was not held by llmctl)"), change.summary)
        self.assertFalse(self.path.exists())

    def test_append_after_user_rows(self):
        self.write_home(TELEMETRY)
        change = self.plan()
        self.assertEqual(change.old_text, TELEMETRY)
        self.assertEqual(change.new_text, TELEMETRY + dsh_config.HEADER + dsh_config.BEGIN + "\n"
                         + dsh_config.render_block(self.qwen36, default=False) + dsh_config.END + "\n")
        self.assertTrue(change.backup)
        self.assertEqual(change.summary, "added the llmctl rows for qwen3.6-35b-a3b on llama.cpp after your rows; "
                         "dsh's default stays with its model picker")

    def test_append_without_final_newline(self):
        old = "- id: session-telemetry-otel\n  disabled: true"
        self.write_home(old)
        change = self.plan()
        self.assertTrue(change.new_text.startswith(old + "\n" + dsh_config.HEADER), change.new_text[:120])

    def test_append_to_crlf_file(self):
        old = "- id: session-telemetry-otel\r\n  disabled: true\r\n"
        self.write_home(old)
        change = self.plan()
        self.assertEqual(change.new_text, old + dsh_config.new_home_text(self.qwen36, default=False))

    def test_empty_file_counts_as_new(self):
        self.write_home("")
        change = self.plan()
        self.assertEqual(change.old_text, "")
        self.assertEqual(change.new_text, dsh_config.new_home_text(self.qwen36, default=False))
        self.assertFalse(change.backup)
        self.assertTrue(change.summary.startswith("created it with the llmctl rows for "), change.summary)

    def test_rewrite_keeps_outside_bytes(self):
        before = "# mine\n" + TELEMETRY
        after = "- id: tool-web\n  disabled: true\n"
        old = before + dsh_config.new_home_text(self.gptoss, default=True) + after
        self.write_home(old)
        change = self.plan()
        head = before + dsh_config.HEADER + dsh_config.BEGIN + "\n"
        tail = dsh_config.END + "\n" + after
        self.assertEqual(change.new_text, head + dsh_config.render_block(self.qwen36, default=True) + tail)
        self.assertTrue(change.backup)
        self.assertEqual(change.summary, "llmctl rows for qwen3.6-35b-a3b on llama.cpp (was gpt-oss-20b on vLLM); "
                         "dsh's default llmctl/local in every profile (was llmctl/gpt-oss-20b)")

    def test_unchanged_is_none(self):
        self.write_home(dsh_config.new_home_text(self.qwen36, default=True))
        self.assertIsNone(self.plan())
        self.assertIsNone(self.plan(set_default=True))

    def test_default_rules(self):
        pi_ai, default, acp = dsh_config.render_block(self.qwen36, default=True).splitlines(keepends=True)
        short = pi_ai.replace('"maxTokens": 16384', '"maxTokens": 1')
        self.assertNotEqual(short, pi_ai)
        cases = (
            ("held default kept", pi_ai + default + acp, False, None),
            ("flag adds the default", pi_ai, True, (pi_ai + default + acp, self.QWEN36 + " (same recipe); dsh's "
                                                   "default llmctl/local in every profile (was not held by llmctl)")),
            ("no flag adds nothing", pi_ai, False, None),
            ("missing acp written back", pi_ai + default, False,
             (pi_ai + default + acp, self.QWEN36 + " (same recipe); restored the acp row")),
            ("lone acp dropped", pi_ai + acp, False,
             (pi_ai, self.QWEN36 + " (same recipe); removed the acp row, because the block no longer holds "
                                   "agent-default-model")),
            ("empty block", "", False, (pi_ai, self.QWEN36)),
            ("rows rewritten", short, False, (pi_ai, self.QWEN36 + " (same recipe, rows rewritten)")),
        )
        for name, lines, set_default, expected in cases:
            with self.subTest(name):
                self.write_home(block_text(lines))
                change = self.plan(set_default=set_default)
                if expected is None:
                    self.assertIsNone(change)
                    continue
                self.assertEqual(change.new_text, block_text(expected[0]))
                self.assertEqual(change.summary, expected[1])
                self.assertFalse(change.backup)

    def test_release_default(self):
        self.write_home(dsh_config.new_home_text(self.qwen36, default=True))
        change = dsh_config.plan_release_default(self.env, self.qwen36)
        self.assertEqual(change.new_text, dsh_config.new_home_text(self.qwen36, default=False))
        self.assertEqual(change.summary, "llmctl rows for qwen3.6-35b-a3b on llama.cpp (same recipe); dsh's default "
                         "goes back to its model picker (was llmctl/local)")
        self.write_home(dsh_config.new_home_text(self.qwen36, default=False))
        self.assertIsNone(dsh_config.plan_release_default(self.env, self.qwen36))
        self.path.unlink()
        self.assertIsNone(dsh_config.plan_release_default(self.env, self.qwen36))
        self.assertFalse(self.path.exists())
        self.write_home(TELEMETRY)
        self.assertIsNone(dsh_config.plan_release_default(self.env, self.qwen36))
        self.assertEqual(self.path.read_text(), TELEMETRY)

    def test_byte_order_mark_is_kept_and_skipped(self):
        old = BOM + TELEMETRY
        self.write_home(old)
        change = self.plan()
        self.assertTrue(change.new_text.startswith(BOM + "- id: "))
        self.assertEqual(change.new_text, old + dsh_config.new_home_text(self.qwen36, default=False))
        self.write_home(BOM + "[]\n")
        self.refuses(EMPTY_LIST.format(path=self.path, line=1))

    def test_byte_order_mark_before_the_begin_marker(self):
        block = dsh_config.render_block(self.qwen36, default=False)
        old = BOM + dsh_config.BEGIN + "\n" + block + dsh_config.END + "\n" + TELEMETRY
        self.write_home(old)
        self.assertEqual(dsh_config.read_block(self.env).rows, tuple(dsh_config.rows(self.qwen36, default=False)))
        self.assertIsNone(self.plan())
        change = self.plan(set_default=True)
        expected = (BOM + dsh_config.BEGIN + "\n" + dsh_config.render_block(self.qwen36, default=True)
                    + dsh_config.END + "\n" + TELEMETRY)
        self.assertEqual(change.new_text, expected)
        self.assertTrue(change.backup)
        self.assertEqual(dsh_config.home_text(self.path, old, self.gptoss, default=False),
                         BOM + dsh_config.BEGIN + "\n" + dsh_config.render_block(self.gptoss, default=False)
                         + dsh_config.END + "\n" + TELEMETRY)

    def test_lone_cr_outside_refused(self):
        self.write_home("- id: a\r  disabled: true\n")
        self.refuses(TOP_LEVEL.format(path=self.path, line=1))
        self.write_home("- id: a\r\n  disabled: true\r\n")
        self.assertIsNotNone(self.plan())

    def test_self_check_failure_refuses(self):
        for old in (None, TELEMETRY, dsh_config.new_home_text(self.gptoss, default=False)):
            with self.subTest(old=old and old[:20]):
                if old is not None:
                    self.write_home(old)
                with mock.patch("launcher.dsh_config.render_block", return_value='- {"id": "llm-pi-ai"}\n'):
                    self.refuses(f"{self.path}: llmctl's own check of the new home patch failed; nothing was written")

    def test_hand_edited_home_patch_is_kept(self):
        old = (BOM + "- id: session-telemetry-otel\r\n  disabled: true\r\n# my rows\r\n"
               "- {\r\n    \"id\": \"tool-web\",\r\n    \"disabled\": true\r\n  }")
        self.write_home(old)
        change = self.plan()
        self.assertEqual(change.new_text, old + "\n" + dsh_config.new_home_text(self.qwen36, default=False))
        self.assertTrue(change.new_text.startswith(old))
        lines = files.split_lines(change.new_text)
        begin = lines.index(dsh_config.BEGIN + "\n")
        end = lines.index(dsh_config.END + "\n")
        edited = lines[:begin + 1] + [files.bare(line) + "   \r\n" for line in lines[begin + 1:end]] + lines[end:]
        self.write_home("".join(edited))
        change = self.plan(target=self.gptoss)
        new = files.split_lines(change.new_text)
        self.assertEqual(new[:begin + 1], edited[:begin + 1])
        self.assertEqual(new[-len(edited[end:]):], edited[end:])
        self.assertEqual("".join(new[begin + 1:-len(edited[end:])]),
                         dsh_config.render_block(self.gptoss, default=False))
        self.assertEqual(change.summary, "llmctl rows for gpt-oss-20b on vLLM (was qwen3.6-35b-a3b on llama.cpp)")


class RefusalTest(HomeCase):
    def test_unreadable_home_patch(self):
        self.path.write_bytes(b"\xff\n")
        with self.assertRaises(ClientError) as caught:
            self.plan()
        self.assertTrue(str(caught.exception).startswith(f"cannot read {self.path}: "), str(caught.exception))
        self.assertEqual(caught.exception.exit_code, 3)
        with self.assertRaises(ClientError):
            dsh_config.read_block(self.env)
        if os.geteuid() == 0:
            self.skipTest("root reads a file of mode 0o000")
        self.write_home(TELEMETRY)
        self.path.chmod(0o000)
        try:
            with self.assertRaises(ClientError) as caught:
                self.plan()
            self.assertTrue(str(caught.exception).startswith(f"cannot read {self.path}: "), str(caught.exception))
        finally:
            self.path.chmod(0o600)
        self.assertEqual(self.path.read_text(), TELEMETRY)

    def test_damaged_markers(self):
        for text, found in ((TELEMETRY + dsh_config.BEGIN + "\n", "1 begin marker, 0 end markers"),
                            (dsh_config.END + "\n" + dsh_config.BEGIN + "\n", "the end marker comes first")):
            with self.subTest(found=found):
                self.write_home(text)
                self.refuses(f"{self.path}: the llmctl block markers are damaged ({found}); restore the two marker "
                             "lines exactly")

    def test_empty_list_line_refused(self):
        for text in ("[]\n", "[ ] # none\n"):
            with self.subTest(text=text):
                self.write_home(text)
                self.refuses(EMPTY_LIST.format(path=self.path, line=1))

    def test_other_top_level_lines_refused(self):
        for text in ("id: x\n", "{a: 1}\n", "---\n", '"- quoted": 1\n'):
            with self.subTest(text=text):
                self.write_home(text)
                self.refuses(TOP_LEVEL.format(path=self.path, line=1))
        for text in ("-\n  id: x\n", '- {"id": "x"}\n', "\n\n", "# c\n", " \t\n"):
            with self.subTest(text=text):
                self.write_home(text)
                self.assertIsNotNone(self.plan())

    def test_outside_llm_pi_ai_refused(self):
        block = dsh_config.new_home_text(self.qwen36, default=False)
        for text, line in (("- id: llm-pi-ai\n", 1), ('- {"id": "llm-pi-ai", "config": {}}\n', 1),
                           ('- {\n    "id": "llm-pi-ai",\n    "config": {}\n  }\n', 2),
                           ("- insert: llm\n  id: llm-pi-ai\n", 2), ("- 'id': 'llm-pi-ai'\n", 1)):
            for where in ("before", "after"):
                with self.subTest(text=text, where=where):
                    if where == "before":
                        self.write_home(text + block)
                        number = line
                    else:
                        self.write_home(block + text)
                        number = len(files.split_lines(block)) + line
                    self.refuses(OUTSIDE_PI_AI.format(path=self.path, line=number))

    def test_scan_skips_comments_and_other_names(self):
        for text in ("# - id: llm-pi-ai\n", "  # id: acp\n",
                     "- name: '@deepseek-ai/dsh-llm-pi-ai'\n  disabled: false\n",
                     "- id: acp-app-startup\n  disabled: true\n", "- id: llm-pi-ai-x\n"):
            with self.subTest(text=text):
                self.write_home(text)
                change = self.plan(set_default=True)
                self.assertEqual(change.new_text, text + dsh_config.new_home_text(self.qwen36, default=True))

    def test_outside_default_rows_only_with_the_default(self):
        user = "- id: agent-default-model\n  config:\n    provider: deepseek-official\n"
        self.write_home(user)
        self.assertIsNotNone(self.plan())
        self.refuses(OUTSIDE_DEFAULT.format(path=self.path, line=1, row="agent-default-model"), set_default=True)
        block = dsh_config.new_home_text(self.qwen36, default=True)
        self.write_home(block + "- id: acp\n")
        self.refuses(OUTSIDE_DEFAULT.format(path=self.path, line=len(files.split_lines(block)) + 1, row="acp"))

    def test_profile_llm_pi_ai_rows_refused(self):
        rows = PROFILE_TEMPLATE_HEADER + "- id: llm-pi-ai\n  config: {}\n"
        web = self.write_profile("web", rows)
        headless = self.write_profile("headless", rows)
        self.refuses(f"dsh's profile patches set the llm-pi-ai row ({headless} line 4; {web} line 4), which llmctl's "
                     f"row in {self.path} would hide without a message; delete those rows "
                     "(docs/clients.md#moving-a-hand-made-setup), or leave dsh out with LLMCTL_CLIENTS")

    def test_profile_findings(self):
        self.assertEqual(dsh_config.profile_findings(self.env), [])
        acp = self.write_profile("acp", "# x\n- id: acp\n")
        web = self.write_profile("web", "# x\n- id: llm-pi-ai\n  config: {}\n# - id: acp\n- id: agent-default-model\n")
        self.write_profile("headless", PROFILE_TEMPLATE)
        F = dsh_config.Finding
        self.assertEqual(dsh_config.profile_findings(self.env),
                         [F(acp, 2, "acp", "acp"), F(web, 2, "llm-pi-ai", "web"),
                          F(web, 5, "agent-default-model", "web")])
        self.write_profile("web", BOM + "# x\r- id: llm-pi-ai\r\n")
        self.assertEqual(dsh_config.profile_findings(self.env),
                         [F(acp, 2, "acp", "acp"), F(web, 2, "llm-pi-ai", "web")])
        bad = self.write_profile("web", "")
        bad.write_bytes(b"\xff")
        with self.assertRaises(ClientError) as caught:
            dsh_config.profile_findings(self.env)
        self.assertTrue(str(caught.exception).startswith(f"cannot read {bad}: "), str(caught.exception))

    def test_profile_findings_without_profiles_folder(self):
        (Path(self.tmp.name) / ".dsh" / "profiles").rmdir()
        self.assertEqual(dsh_config.profile_findings(self.env), [])

    def test_owner_migration_sequence(self):
        self.write_home(TELEMETRY)
        web = self.write_profile("web", PROFILE_TEMPLATE_HEADER + WEB_UI_SETTINGS + "# llmctl rows by hand\n"
                                 + WEB_UI_LLM_PI_AI + WEB_UI_DEFAULT)
        headless = self.write_profile("headless", PROFILE_TEMPLATE_HEADER + WEB_UI_LLM_PI_AI + WEB_UI_DEFAULT)
        self.refuses(f"dsh's profile patches set the llm-pi-ai row ({headless} line 4; {web} line 8), which llmctl's "
                     f"row in {self.path} would hide without a message; delete those rows "
                     "(docs/clients.md#moving-a-hand-made-setup), or leave dsh out with LLMCTL_CLIENTS",
                     set_default=True, target=self.gemma)
        self.write_profile("headless", PROFILE_TEMPLATE)
        self.write_profile("web", PROFILE_TEMPLATE_HEADER + WEB_UI_SETTINGS)
        change = self.plan(set_default=True, target=self.gemma)
        self.assertTrue(change.backup)
        self.assertEqual(change.new_text, TELEMETRY + dsh_config.new_home_text(self.gemma, default=True))
        self.assertEqual(change.new_text, DESIGN_OWNER_EXAMPLE)
        self.write_profile("headless", PROFILE_TEMPLATE_HEADER + WEB_UI_DEFAULT)
        self.write_profile("web", PROFILE_TEMPLATE_HEADER + WEB_UI_SETTINGS + WEB_UI_DEFAULT)
        change = self.plan(set_default=True, target=self.gemma)
        self.assertEqual(change.new_text, DESIGN_OWNER_EXAMPLE)
        F = dsh_config.Finding
        self.assertEqual(dsh_config.profile_findings(self.env),
                         [F(headless, 4, "agent-default-model", "headless"), F(web, 7, "agent-default-model", "web")])


class HomeTextTest(unittest.TestCase):
    def test_home_text_transforms(self):
        qwen36 = target_for("qwen3.6-35b-a3b", "llamacpp", RECIPES)
        path = Path("/h/.dsh/cordis.patch.yml")
        self.assertEqual(dsh_config.home_text(path, None, qwen36, default=True),
                         dsh_config.new_home_text(qwen36, default=True))
        self.assertEqual(dsh_config.home_text(path, "", qwen36, default=False),
                         dsh_config.new_home_text(qwen36, default=False))
        self.assertEqual(dsh_config.home_text(path, "x", qwen36, default=False),
                         "x\n" + dsh_config.new_home_text(qwen36, default=False))
        old = "a\n" + block_text("- {}\n") + "b\n"
        self.assertEqual(dsh_config.home_text(path, old, qwen36, default=True),
                         "a\n" + dsh_config.new_home_text(qwen36, default=True) + "b\n")


S1_LLM_PI_AI = "# == @deepseek-ai/dsh-base, patched by {patch}\n- id: llm-pi-ai"
S1_DEFAULT = "# == @deepseek-ai/dsh-base, patched by {patch}\n- id: agent-default-model"
REJECTED = "dsh rejects the home patch llmctl would write for qwen3.6-35b-a3b: dsh: failed to parse patches {path}: "
NOT_APPLIED = "dsh did not apply llmctl's {row} row from {path}; leave dsh out with LLMCTL_CLIENTS and report it"


class CheckTest(unittest.TestCase):
    """The dsh check of design D5, against a FakeHost whose dump routes answer from the fixtures."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.host = FakeHost(env={"HOME": self.tmp.name})
        self.qwen36 = target_for("qwen3.6-35b-a3b", "llamacpp", RECIPES)
        self.candidate = dsh_config.new_home_text(self.qwen36, default=True)
        self.path = dsh_config.home_patch(self.host.env)
        self.seen: list[str] = []

    def tearDown(self):
        self.tmp.cleanup()

    def private_patch(self) -> str:
        return self.host.env_extras[-1]["DSH_HOME"] + "/cordis.patch.yml"

    def dump(self, stdout_name: str | None = "s1-dump.txt", stderr: str = "", code: int = 0, edit=None):
        """A route that answers from a fixture with {patch} as the private home patch, and records the folder."""

        def answer(argv):
            patch = self.private_patch()
            self.seen.append(str(Path(patch).parent.parent))
            text = dsh_fixture(stdout_name, patch) if stdout_name else ""
            if edit is not None:
                text = edit(text, patch)
            err = stderr.replace("{patch}", patch)
            return Result(code, text, err)

        return answer

    def check(self, written=dsh_config.OWNED_ROWS):
        return dsh_config.check_home(self.host, self.candidate, self.qwen36, path=self.path, written=written)

    def refuses(self, message: str, written=dsh_config.OWNED_ROWS) -> None:
        with self.assertRaises(ClientError) as caught:
            self.check(written)
        self.assertEqual(str(caught.exception), message)
        self.assertEqual(caught.exception.exit_code, 3)

    def test_installed_version_and_warning(self):
        self.host.on(["dsh", "--version"], ok((DSH_FIXTURES / "s6-version.txt").read_text()))
        self.assertEqual(dsh_config.installed_version(self.host), "0.2.2-rc.3")
        self.assertIsNone(dsh_config.version_warning(self.host))
        self.assertEqual(self.host.calls[-1], ["dsh", "--version"])
        self.host.on(["dsh", "--version"], ok("0.2.3\n"))
        self.assertEqual(dsh_config.version_warning(self.host),
                         "warning: dsh 0.2.3 is not 0.2.2-rc.3, the version llmctl's dsh rows were checked with; "
                         "run the dsh steps of the e2e before relying on them (docs/clients.md#maintenance)")
        self.host.on(["dsh", "--version"], ok("\n  0.2.4  \nmore\n"))
        self.assertEqual(dsh_config.installed_version(self.host), "0.2.4")
        cases = [(fail("dsh: command not found", 127),
                  "dsh --version failed: dsh: command not found; check that dsh runs in this shell, or leave dsh out "
                  "with LLMCTL_CLIENTS"),
                 (fail("", 2), "dsh --version failed: exit 2; check that dsh runs in this shell, or leave dsh out "
                               "with LLMCTL_CLIENTS"),
                 (ok(""), "dsh --version failed: it printed no version; check that dsh runs in this shell, or leave "
                          "dsh out with LLMCTL_CLIENTS")]
        for result, message in cases:
            with self.subTest(message=message):
                self.host.on(["dsh", "--version"], result)
                for call in (dsh_config.installed_version, dsh_config.version_warning):
                    with self.assertRaises(ClientError) as caught:
                        call(self.host)
                    self.assertEqual(str(caught.exception), message)

    def test_check_passes_on_the_s1_dump(self):
        self.host.on(["dsh", "--profile", "acp", "--dump-config"], self.dump())
        self.assertIsNone(self.check())
        self.assertEqual(self.host.calls[-1], ["dsh", "--profile", "acp", "--dump-config"])
        self.assertEqual(self.host.env_extras[-1], {"DSH_HOME": self.seen[0] + "/home", "DSH_TELEMETRY_DISABLED": "1"})

    def test_candidate_is_checked_privately(self):
        observed = {}

        def route(argv):
            home = Path(self.host.env_extras[-1]["DSH_HOME"])
            patch = home / "cordis.patch.yml"
            observed.update(folder=home.parent, folder_mode=home.parent.stat().st_mode & 0o777,
                            home=home.name, home_mode=home.stat().st_mode & 0o777,
                            patch_mode=patch.stat().st_mode & 0o777, patch=patch.read_bytes(),
                            listing=sorted(os.listdir(home.parent)))
            return ok(dsh_fixture("s1-dump.txt", str(patch)))

        self.candidate = "﻿a\r\n" + self.candidate
        self.host.on(["dsh", "--profile", "acp", "--dump-config"], route)
        self.check()
        self.assertTrue(observed["folder"].name.startswith("llmctl-dsh-"))
        self.assertEqual(observed["home"], "home")
        self.assertEqual(observed["listing"], ["home"])
        self.assertEqual((observed["folder_mode"], observed["home_mode"], observed["patch_mode"]),
                         (0o700, 0o700, 0o600))
        self.assertEqual(observed["patch"], self.candidate.encode("utf-8"))
        self.assertFalse(observed["folder"].exists())

    def test_private_folder_removed_on_every_path(self):
        def raises(argv):
            self.seen.append(str(Path(self.private_patch()).parent.parent))
            raise OSError(5, "Input/output error")

        unapplied = (lambda text, patch: text.replace(S1_LLM_PI_AI.replace("{patch}", patch),
                                                      "# == @deepseek-ai/dsh-base\n- id: llm-pi-ai"))
        cases = [("pass", self.dump(), False),
                 ("s3 exit", self.dump(None, (DSH_FIXTURES / "s3.stderr").read_text(), 1), True),
                 ("s2 missing rows", self.dump("s2-dump.txt", (DSH_FIXTURES / "s2.stderr").read_text()), True),
                 ("rows not applied", self.dump(edit=unapplied), True),
                 ("route raises OSError", raises, True)]
        self.host.on(["dsh", "--version"], ok("0.2.4\n"))
        for name, route, refused in cases:
            with self.subTest(name):
                self.seen.clear()
                self.host.on(["dsh", "--profile", "acp", "--dump-config"], route)
                if refused:
                    with self.assertRaises(ClientError):
                        self.check()
                else:
                    self.check()
                self.assertEqual(len(self.seen), 1)
                self.assertTrue(Path(self.seen[0]).name.startswith("llmctl-dsh-"))
                self.assertFalse(Path(self.seen[0]).exists())

    def test_route_oserror_names_the_real_path(self):
        def raises(argv):
            raise OSError(5, "Input/output error")

        self.host.on(["dsh", "--profile", "acp", "--dump-config"], raises)
        self.refuses(f"cannot check {self.path} with dsh: [Errno 5] Input/output error")

    def test_dsh_rejects_the_candidate(self):
        cases = [("s3.stderr", "YAMLException: end of the stream or a document separator is expected (7:1)"),
                 ("s4.stderr", "YAMLException: missed comma between flow collection entries (8:1)")]
        for name, tail in cases:
            with self.subTest(name):
                self.seen.clear()
                self.host.on(["dsh", "--profile", "acp", "--dump-config"],
                             self.dump(None, (DSH_FIXTURES / name).read_text(), 1))
                with self.assertRaises(ClientError) as caught:
                    self.check()
                self.assertEqual(str(caught.exception), REJECTED.format(path=self.path) + tail)
                self.assertNotIn(self.seen[0], str(caught.exception))

    def test_rejection_names_the_real_home(self):
        stderr = "Error: dsh: cannot read {home}/profiles/acp: EACCES\n"

        def route(argv):
            home = self.host.env_extras[-1]["DSH_HOME"]
            return fail(stderr.replace("{home}", home), 1)

        self.host.on(["dsh", "--profile", "acp", "--dump-config"], route)
        self.refuses("dsh rejects the home patch llmctl would write for qwen3.6-35b-a3b: dsh: cannot read "
                     f"{self.path.parent}/profiles/acp: EACCES")

    def test_dump_failure_without_error_line(self):
        self.host.on(["dsh", "--profile", "acp", "--dump-config"], fail("boom\n", 1))
        self.refuses("dsh --dump-config failed: boom; check that dsh runs in this shell, or leave dsh out with "
                     "LLMCTL_CLIENTS")
        self.host.on(["dsh", "--profile", "acp", "--dump-config"], fail("", 1))
        self.refuses("dsh --dump-config failed: exit 1; check that dsh runs in this shell, or leave dsh out with "
                     "LLMCTL_CLIENTS")

    def test_missing_row_id_refuses(self):
        self.host.on(["dsh", "--version"], ok("0.2.4\n"))
        self.host.on(["dsh", "--profile", "acp", "--dump-config"],
                     self.dump("s2-dump.txt", (DSH_FIXTURES / "s2.stderr").read_text()))
        self.refuses("dsh 0.2.4 has no llm-pi-ai row, so llmctl's row would not apply; llmctl's dsh rows were "
                     "checked with dsh 0.2.2-rc.3; leave dsh out with LLMCTL_CLIENTS until llmctl supports this "
                     "version")

    def test_rows_not_applied_refuses(self):
        def edit(text, patch):
            old = S1_LLM_PI_AI.replace("{patch}", patch)
            self.assertIn(old, text)
            return text.replace(old, "# == @deepseek-ai/dsh-base\n- id: llm-pi-ai")

        self.host.on(["dsh", "--profile", "acp", "--dump-config"], self.dump(edit=edit))
        self.refuses(NOT_APPLIED.format(row="llm-pi-ai", path=self.path))

    def test_only_written_rows_are_checked(self):
        def edit(text, patch):
            old = S1_DEFAULT.replace("{patch}", patch)
            self.assertIn(old, text)
            return text.replace(old, "# == @deepseek-ai/dsh-base\n- id: agent-default-model")

        self.host.on(["dsh", "--profile", "acp", "--dump-config"], self.dump(edit=edit))
        self.assertIsNone(self.check(("llm-pi-ai",)))
        self.refuses(NOT_APPLIED.format(row="agent-default-model", path=self.path))

    def test_patch_path_must_be_a_whole_item(self):
        def edit(text, patch):
            return text.replace(f"patched by {patch}", f"patched by {patch}.old")

        self.host.on(["dsh", "--profile", "acp", "--dump-config"], self.dump(edit=edit))
        self.refuses(NOT_APPLIED.format(row="llm-pi-ai", path=self.path))

    def test_other_stderr_lines_do_not_fail(self):
        self.host.on(["dsh", "--profile", "acp", "--dump-config"],
                     self.dump(stderr='dsh: [{patch}] patch: entry "tool-x" not found\n'))
        self.assertIsNone(self.check())

    def test_private_folder_failure(self):
        with mock.patch("launcher.dsh_config.tempfile.mkdtemp", side_effect=OSError(28, "No space left on device")):
            self.refuses(f"cannot check {self.path} with dsh: [Errno 28] No space left on device")
        self.assertEqual(self.host.calls, [])


if __name__ == "__main__":
    unittest.main()
