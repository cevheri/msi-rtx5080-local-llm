import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from launcher import pi_config
from launcher.catalog import load_catalog
from launcher.errors import ClientError, ClientWriteError, UsageError
from launcher.host import Result
from launcher.tests.fakes import REPO, FakeHost, fail, ok
from launcher.tests.regen_golden import target_for

RECIPES = REPO / "recipes"

# The qwen3.6-35b-a3b models.json of the spec section "Generated client entries"
# (docs/superpowers/specs/2026-11-16-llmctl-clients-design.md).
QWEN36_MODELS_JSON = """\
{
  "providers": {
    "llmctl": {
      "name": "llmctl",
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "no-key-required",
      "compat": { "supportsStore": false, "supportsDeveloperRole": false, "maxTokensField": "max_tokens" },
      "models": [
        { "id": "local", "name": "qwen3.6-35b-a3b on llama.cpp (llmctl)", "reasoning": true, "input": ["text"], "contextWindow": 131072, "maxTokens": 16384,
          "thinkingLevelMap": { "minimal": null, "low": null, "high": null },
          "compat": { "thinkingFormat": "chat-template", "chatTemplateKwargs": { "enable_thinking": { "$var": "thinking.enabled" } } } }
      ]
    }
  }
}
"""

# The model objects of the same section, one per line.
SPEC_MODELS = (
    ("llamacpp", "nemotron-3.5-bolt", r'{"id": "local", "name": "nemotron-3.5-bolt on llama.cpp (llmctl)", "reasoning": true, "input": ["text"], "contextWindow": 131072, "maxTokens": 16384, "thinkingLevelMap": {"minimal": null, "low": null}, "compat": {"thinkingFormat": "chat-template", "chatTemplateKwargs": {"enable_thinking": {"$var": "thinking.enabled"}}}, "samplingParamsByThinkingLevel": {"medium": {"chat_template_kwargs": {"enable_thinking": true, "medium_effort": true}}}}'),
    ("llamacpp", "qwen3.8-27b", r'{"id": "local", "name": "qwen3.8-27b on llama.cpp (llmctl)", "reasoning": true, "input": ["text"], "contextWindow": 65536, "maxTokens": 16384, "thinkingLevelMap": {"off": "none", "minimal": null, "low": "low", "medium": "medium", "high": null}, "compat": {"thinkingFormat": "openai", "supportsReasoningEffort": true}}'),
    ("llamacpp", "k2-horizon-7b", r'{"id": "local", "name": "k2-horizon-7b on llama.cpp (llmctl)", "reasoning": true, "input": ["text"], "contextWindow": 81920, "maxTokens": 32768, "thinkingLevelMap": {"off": null, "minimal": null, "low": null, "medium": null}, "compat": {"supportsReasoningEffort": false}}'),
    ("llamacpp", "qwen3-coder-next", r'{"id": "local", "name": "qwen3-coder-next on llama.cpp (llmctl)", "reasoning": false, "input": ["text"], "contextWindow": 131072, "maxTokens": 16384}'),
    ("llamacpp", "deepseek-r1-0528-qwen3-8b", r'{"id": "local", "name": "deepseek-r1-0528-qwen3-8b on llama.cpp (llmctl)", "reasoning": true, "input": ["text"], "contextWindow": 65536, "maxTokens": 16384, "thinkingLevelMap": {"off": null, "minimal": null, "low": null, "medium": null}, "compat": {"supportsReasoningEffort": false}}'),
    ("vllm", "gpt-oss-20b", r'{"id": "gpt-oss-20b", "name": "gpt-oss-20b on vLLM (llmctl)", "reasoning": true, "input": ["text"], "contextWindow": 65536, "maxTokens": 16384, "thinkingLevelMap": {"off": null, "minimal": null, "low": "low", "medium": "medium", "high": "high"}, "compat": {"thinkingFormat": "openai", "supportsReasoningEffort": true}}'),
)


def shipped_targets():
    catalog = load_catalog(RECIPES, None)
    for runtime, entry in catalog.runtimes.items():
        for recipe_id in sorted(entry.recipes):
            yield target_for(recipe_id, runtime, RECIPES)


class EntryTest(unittest.TestCase):
    def test_qwen36_models_json_is_the_spec_example(self):
        target = target_for("qwen3.6-35b-a3b", "llamacpp", RECIPES)
        text = pi_config.new_models_text(target)
        self.assertEqual(json.loads(text), json.loads(QWEN36_MODELS_JSON))
        self.assertEqual(text, json.dumps(json.loads(text), indent=2, ensure_ascii=False) + "\n")

    def test_model_objects_match_spec(self):
        for runtime, recipe_id, line in SPEC_MODELS:
            with self.subTest(runtime=runtime, recipe=recipe_id):
                expected = json.loads(line)
                model = pi_config.provider_entry(target_for(recipe_id, runtime, RECIPES))["models"][0]
                self.assertEqual(model, expected)
                self.assertEqual(list(model), list(expected))

    def test_provider_part_is_shared(self):
        reference = pi_config.provider_entry(target_for("qwen3.6-35b-a3b", "llamacpp", RECIPES))
        del reference["models"]
        checked = 0
        for target in shipped_targets():
            with self.subTest(runtime=target.runtime.name, recipe=target.recipe.id):
                entry = pi_config.provider_entry(target)
                self.assertEqual(len(entry["models"]), 1)
                del entry["models"]
                expected = dict(reference)
                if target.runtime.name == "vllm":
                    expected["baseUrl"] = "http://127.0.0.1:8000/v1"
                self.assertEqual(entry, expected)
                checked += 1
        self.assertEqual(checked, 19)

    def test_default_levels(self):
        expected = {
            ("llamacpp", "qwen3.6-35b-a3b"): "off",
            ("llamacpp", "nemotron-3.5-bolt"): "off",
            ("llamacpp", "qwen3.8-27b"): "medium",
            ("llamacpp", "k2-horizon-7b"): "high",
            ("llamacpp", "qwen3-coder-next"): "off",
            ("llamacpp", "deepseek-r1-0528-qwen3-8b"): "high",
            ("vllm", "gpt-oss-20b"): "medium",
            ("vllm", "mellum2.1-12b"): "medium",
        }
        for (runtime, recipe_id), level in expected.items():
            with self.subTest(runtime=runtime, recipe=recipe_id):
                meta = target_for(recipe_id, runtime, RECIPES).recipe.clients
                self.assertEqual(pi_config.default_level(meta), level)


# E7 fixtures (pi/runs/val-* and neg-*).
PASSING_STDOUT = ("provider  model  context  max-out  thinking  images\n"
                  "llmctl    local  131.1K   16.4K    yes       no    \n")
NO_MODELS_STDOUT = ("No models available. Use /login to log into a provider via OAuth or API key. See:\n")
SCHEMA_STDERR = ("Warning: errors loading models.json:\nInvalid models.json schema:\n"
                 "  - providers.llmctl.models.0.compat.thinkingFormat: must be equal to constant\n")


def mode_of(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


class PiFilesCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self.agent = self.home / ".pi" / "agent"
        self.agent.mkdir(parents=True)
        self.env = {"HOME": str(self.home)}
        self.models = self.agent / "models.json"
        self.settings = self.agent / "settings.json"
        self.qwen36 = target_for("qwen3.6-35b-a3b", "llamacpp", RECIPES)
        self.gpt_oss = target_for("gpt-oss-20b", "vllm", RECIPES)

    def write_json(self, path, doc, indent=2):
        path.write_text(json.dumps(doc, indent=indent, ensure_ascii=False) + "\n", encoding="utf-8")


class PathsTest(PiFilesCase):
    def test_agent_dir(self):
        self.assertEqual(pi_config.agent_dir(self.env), self.home / ".pi" / "agent")
        self.assertEqual(pi_config.agent_dir({**self.env, "PI_CODING_AGENT_DIR": ""}), self.home / ".pi" / "agent")
        self.assertEqual(pi_config.agent_dir({**self.env, "PI_CODING_AGENT_DIR": "~/pa"}), self.home / "pa")
        self.assertEqual(pi_config.models_path(self.env), self.home / ".pi" / "agent" / "models.json")
        self.assertEqual(pi_config.settings_path(self.env), self.home / ".pi" / "agent" / "settings.json")
        with self.assertRaises(UsageError):
            pi_config.agent_dir({})
        with self.assertRaises(UsageError):
            pi_config.agent_dir({"PI_CODING_AGENT_DIR": "~/pa"})


class ModelsPlanTest(PiFilesCase):
    def test_create_models_json(self):
        change = pi_config.plan_models(self.env, self.qwen36)
        self.assertEqual(change.path, self.models)
        self.assertIsNone(change.old_text)
        self.assertEqual(json.loads(change.new_text), json.loads(QWEN36_MODELS_JSON))
        self.assertEqual(change.new_text, pi_config.new_models_text(self.qwen36))
        self.assertEqual(change.mode, 0o600)
        self.assertFalse(change.backup)
        self.assertEqual(change.summary, "provider llmctl for qwen3.6-35b-a3b on llama.cpp, model local (was absent)")
        self.assertEqual(change.preview, f"would write {self.models}: {change.summary}")

    def test_replace_in_place_keeps_other_providers(self):
        a = {"name": "A", "baseUrl": "https://a.invalid/v1", "models": [{"id": "m"}]}
        z = {"name": "Z", "models": []}
        old = {"providers": {"a": a, "llmctl": pi_config.provider_entry(self.gpt_oss), "z": z}}
        self.write_json(self.models, old)
        change = pi_config.plan_models(self.env, self.qwen36)
        new = json.loads(change.new_text)
        self.assertEqual(list(new["providers"]), ["a", "llmctl", "z"])
        self.assertEqual(new["providers"]["a"], a)
        self.assertEqual(new["providers"]["z"], z)
        self.assertEqual(new["providers"]["llmctl"]["models"][0]["id"], "local")
        self.assertEqual(new["providers"]["llmctl"]["baseUrl"], "http://127.0.0.1:8080/v1")
        self.assertEqual(change.old_text, self.models.read_text())
        self.assertTrue(change.summary.endswith("(was gpt-oss-20b on vLLM)"), change.summary)
        self.assertTrue(change.backup)

    def test_append_when_absent(self):
        a = {"name": "A", "models": []}
        self.write_json(self.models, {"providers": {"a": a}})
        new = json.loads(pi_config.plan_models(self.env, self.qwen36).new_text)
        self.assertEqual(list(new["providers"]), ["a", "llmctl"])
        self.assertEqual(new["providers"]["a"], a)
        self.write_json(self.models, {"other": 1})
        new = json.loads(pi_config.plan_models(self.env, self.qwen36).new_text)
        self.assertEqual(list(new), ["other", "providers"])

    def test_only_llmctl_means_no_backup(self):
        self.write_json(self.models, {"providers": {"llmctl": pi_config.provider_entry(self.gpt_oss)}})
        change = pi_config.plan_models(self.env, self.qwen36)
        self.assertFalse(change.backup)
        self.write_json(self.models, {"providers": {}})
        self.assertFalse(pi_config.plan_models(self.env, self.qwen36).backup)

    def test_unchanged_entry_is_not_written(self):
        self.write_json(self.models, {"providers": {"llmctl": pi_config.provider_entry(self.qwen36)}}, indent=4)
        self.assertIsNone(pi_config.plan_models(self.env, self.qwen36))

    def test_models_refusals(self):
        for text in ('{"providers": {}} // x\n', '{"providers": {"a": {"name": "a",}}}\n'):
            with self.subTest(text=text):
                self.models.write_text(text)
                with self.assertRaises(ClientError) as caught:
                    pi_config.plan_models(self.env, self.qwen36)
                self.assertIn("leave Pi out with LLMCTL_CLIENTS", str(caught.exception))
        self.models.write_text("[]\n")
        with self.assertRaises(ClientError):
            pi_config.plan_models(self.env, self.qwen36)
        self.models.write_text('{"providers": []}\n')
        with self.assertRaises(ClientError) as caught:
            pi_config.plan_models(self.env, self.qwen36)
        self.assertEqual(str(caught.exception), f'{self.models}: "providers" is not a JSON object; fix it, '
                                                "or leave Pi out with LLMCTL_CLIENTS")

    def test_unusual_models_json_is_kept(self):
        text = ('{"$schema": "https://example.invalid/pi.json", "providers": {"cloud": {"name": "Mod\u00e8le cloud", '
                '"baseUrl": "https://api.example.invalid/v1", "api": "openai-completions", "apiKey": "!pass show cloud", '
                '"models": [{"id": "big", "name": "Grand mod\u00e8le"}]}, "llmctl": {"name": "llmctl", "models": []}, '
                '"zeta": {"name": "z", "models": []}}}')
        self.models.write_text(text, encoding="utf-8")
        old = json.loads(text)
        change = pi_config.plan_models(self.env, self.qwen36)
        new = json.loads(change.new_text)
        self.assertEqual(list(new), ["$schema", "providers"])
        self.assertEqual(list(new["providers"]), ["cloud", "llmctl", "zeta"])
        self.assertEqual(new["providers"]["cloud"], old["providers"]["cloud"])
        self.assertEqual(new["providers"]["zeta"], old["providers"]["zeta"])
        self.assertEqual(new["providers"]["cloud"]["apiKey"], "!pass show cloud")
        self.assertIn("Mod\u00e8le cloud", change.new_text)
        self.assertIn("Grand mod\u00e8le", change.new_text)
        self.assertTrue(change.summary.endswith("(was another llmctl entry)"), change.summary)
        self.assertTrue(change.backup)


class ModelsReaderTest(PiFilesCase):
    def test_installed_entry_and_description(self):
        self.assertIsNone(pi_config.installed_entry(self.env))
        self.write_json(self.models, {"providers": {"a": {}}})
        self.assertIsNone(pi_config.installed_entry(self.env))
        entry = pi_config.provider_entry(self.gpt_oss)
        self.write_json(self.models, {"providers": {"llmctl": entry}})
        self.assertEqual(pi_config.installed_entry(self.env), entry)
        self.assertEqual(pi_config.entry_description(entry), "gpt-oss-20b on vLLM")
        self.assertIsNone(pi_config.entry_description({"name": "llmctl", "models": []}))
        self.assertIsNone(pi_config.entry_description({"models": [{"id": "x"}]}))


class ValidateTest(PiFilesCase):
    def setUp(self):
        super().setUp()
        self.host = FakeHost(env=self.env)
        self.text = pi_config.new_models_text(self.qwen36)
        self.seen = {}

    def route(self, result):
        def answer(argv):
            directory = self.host.env_extras[-1]["PI_CODING_AGENT_DIR"]
            models = Path(directory) / "models.json"
            self.seen.update(directory=directory, dir_mode=mode_of(directory),
                             text=models.read_text(encoding="utf-8"), file_mode=mode_of(models))
            return result
        self.host.on(["pi"], answer)

    def validate(self, target=None, text=None):
        return pi_config.validate_models(self.host, self.text if text is None else text,
                                         target or self.qwen36, self.models)

    def test_validation_passes_with_row(self):
        self.route(ok(PASSING_STDOUT))
        self.assertIsNone(self.validate())
        self.assertEqual(self.host.calls, [["pi", "--offline", "--list-models", "llmctl"]])
        extra = self.host.env_extras[-1]
        self.assertEqual(set(extra), {"PI_CODING_AGENT_DIR", "PI_OFFLINE", "PI_SKIP_VERSION_CHECK", "PI_TELEMETRY"})
        self.assertEqual((extra["PI_OFFLINE"], extra["PI_SKIP_VERSION_CHECK"], extra["PI_TELEMETRY"]), ("1", "1", "0"))
        self.assertTrue(os.path.basename(self.seen["directory"]).startswith("llmctl-pi-"))
        self.assertEqual(self.seen["dir_mode"], 0o700)
        self.assertEqual(self.seen["text"], self.text)
        self.assertEqual(self.seen["file_mode"], 0o600)
        self.assertFalse(os.path.lexists(self.seen["directory"]))

    def test_validation_removes_directory_when_pi_raises(self):
        def boom(argv):
            self.seen["directory"] = self.host.env_extras[-1]["PI_CODING_AGENT_DIR"]
            raise KeyboardInterrupt
        self.host.on(["pi"], boom)
        with self.assertRaises(KeyboardInterrupt):
            self.validate()
        self.assertFalse(os.path.lexists(self.seen["directory"]))

    def test_validation_oserror_is_a_client_error(self):
        with mock.patch("launcher.pi_config.tempfile.mkdtemp", side_effect=OSError(28, "No space left on device")):
            with self.assertRaises(ClientError) as caught:
                self.validate()
        self.assertEqual(str(caught.exception),
                         f"cannot check {self.models} with pi: [Errno 28] No space left on device")
        self.assertEqual(self.host.calls, [])

    def test_validation_unicode_error_is_a_client_error(self):
        self.route(ok(PASSING_STDOUT))
        made = []
        real_mkdtemp = tempfile.mkdtemp

        def mkdtemp(**kwargs):
            made.append(real_mkdtemp(**kwargs))
            return made[-1]

        with mock.patch("launcher.pi_config.tempfile.mkdtemp", side_effect=mkdtemp):
            with self.assertRaises(ClientError) as caught:
                self.validate(text='{"a": "\ud800"}')
        self.assertTrue(str(caught.exception).startswith(f"cannot check {self.models} with pi: "),
                        str(caught.exception))
        self.assertEqual(self.host.calls, [])
        self.assertFalse(os.path.lexists(made[0]))

    def test_validation_schema_error_refuses(self):
        self.route(Result(0, NO_MODELS_STDOUT, SCHEMA_STDERR))
        with self.assertRaises(ClientError) as caught:
            self.validate()
        self.assertEqual(str(caught.exception),
                         "Pi rejects the llmctl entry for qwen3.6-35b-a3b: Warning: errors loading models.json: / "
                         "Invalid models.json schema: / - providers.llmctl.models.0.compat.thinkingFormat: "
                         "must be equal to constant")
        self.assertFalse(os.path.lexists(self.seen["directory"]))

    def test_validation_missing_row_refuses(self):
        self.route(Result(0, NO_MODELS_STDOUT, ""))
        with self.assertRaises(ClientError) as caught:
            self.validate()
        self.assertEqual(str(caught.exception),
                         "Pi rejects the llmctl entry for qwen3.6-35b-a3b: no llmctl row in pi --list-models")

    def test_validation_warns_about_other_problems(self):
        self.route(Result(0, PASSING_STDOUT, SCHEMA_STDERR))
        self.assertEqual(self.validate(), f"warning: Pi reports problems in {self.models} outside the llmctl entry: "
                                          "Warning: errors loading models.json:")

    def test_validation_pi_cannot_run(self):
        self.route(fail("pi: command not found", code=127))
        with self.assertRaises(ClientError) as caught:
            self.validate()
        self.assertEqual(str(caught.exception), "pi --list-models failed: pi: command not found; check that pi runs "
                                                "in this shell, or leave Pi out with LLMCTL_CLIENTS")
        self.host.on(["pi"], fail("", code=2))
        with self.assertRaises(ClientError) as caught:
            self.validate()
        self.assertTrue(str(caught.exception).startswith("pi --list-models failed: exit 2;"), str(caught.exception))

    def test_validation_matches_dotted_served_name(self):
        mellum = target_for("mellum2.1-12b", "vllm", RECIPES)
        text = pi_config.new_models_text(mellum)
        header = "provider  model  context  max-out  thinking  images\n"
        self.host.on(["pi"], ok(header + "llmctl    mellum2.1  131.1K   16.4K    yes       no\n"))
        self.assertIsNone(self.validate(mellum, text))
        self.host.on(["pi"], ok(header + "llmctl    mellum2x1  131.1K   16.4K    yes       no\n"))
        with self.assertRaises(ClientError):
            self.validate(mellum, text)
        self.host.on(["pi"], ok(header + "llmctl    mellum2.1-12b  131.1K   16.4K    yes       no\n"))
        with self.assertRaises(ClientError):
            self.validate(mellum, text)
        self.host.on(["pi"], ok(header + "llmctl    mellum2.1"))
        self.assertIsNone(self.validate(mellum, text))


class SettingsPlanTest(PiFilesCase):
    def plan(self, target=None, set_default=False):
        return pi_config.plan_settings(self.env, target or self.qwen36, set_default=set_default)

    def test_settings_first_write(self):
        change = self.plan()
        self.assertEqual(change.path, self.settings)
        self.assertIsNone(change.old_text)
        self.assertEqual(change.new_text, '{\n  "modelThinkingLevels": {\n    "llmctl/local": "off"\n  }\n}\n')
        self.assertEqual(change.summary, "modelThinkingLevels llmctl/local off (was absent)")
        self.assertEqual(change.preview, f"would write {self.settings}: modelThinkingLevels llmctl/local off (was absent)")
        self.assertEqual(change.mode, 0o600)
        self.assertFalse(change.backup)

    def test_settings_set_default_on_owner_file(self):
        self.write_json(self.settings, {"lastChangelogVersion": "1.5.2"})
        change = self.plan(set_default=True)
        self.assertEqual(change.new_text, json.dumps(
            {"lastChangelogVersion": "1.5.2", "modelThinkingLevels": {"llmctl/local": "off"},
             "defaultProvider": "llmctl", "defaultModel": "local"}, indent=2) + "\n")
        self.assertEqual(list(json.loads(change.new_text)),
                         ["lastChangelogVersion", "modelThinkingLevels", "defaultProvider", "defaultModel"])
        self.assertEqual(change.summary, "modelThinkingLevels llmctl/local off (was absent); "
                                         "defaultProvider llmctl (was absent); defaultModel local (was absent)")
        self.assertTrue(change.backup)

    def test_settings_follow_and_foreign_default(self):
        self.write_json(self.settings, {"defaultProvider": "llmctl", "defaultModel": "gpt-oss-20b",
                                        "modelThinkingLevels": {"llmctl/local": "off"}})
        change = self.plan()
        new = json.loads(change.new_text)
        self.assertEqual((new["defaultProvider"], new["defaultModel"]), ("llmctl", "local"))
        self.assertEqual(change.summary, "defaultModel local (was gpt-oss-20b)")
        self.assertFalse(change.backup)
        self.write_json(self.settings, {"defaultProvider": "anthropic", "defaultModel": "claude-x"})
        new = json.loads(self.plan().new_text)
        self.assertEqual((new["defaultProvider"], new["defaultModel"]), ("anthropic", "claude-x"))
        change = self.plan(set_default=True)
        self.assertIn("defaultProvider llmctl (was anthropic)", change.summary)
        self.assertIn("defaultModel local (was claude-x)", change.summary)

    def test_other_llmctl_levels_removed(self):
        self.write_json(self.settings, {"modelThinkingLevels": {"llmctl/gpt-oss-20b": "medium", "anthropic/x": "high"}})
        change = self.plan()
        self.assertEqual(json.loads(change.new_text)["modelThinkingLevels"],
                         {"anthropic/x": "high", "llmctl/local": "off"})
        self.assertEqual(change.summary, "modelThinkingLevels llmctl/local off (was absent); "
                                         "removed modelThinkingLevels llmctl/gpt-oss-20b")
        self.write_json(self.settings, {"modelThinkingLevels": {"llmctl/local": "high", "anthropic/x": "high"}})
        change = self.plan()
        self.assertEqual(list(json.loads(change.new_text)["modelThinkingLevels"].items()),
                         [("llmctl/local", "off"), ("anthropic/x", "high")])
        self.assertEqual(change.summary, "modelThinkingLevels llmctl/local off (was high)")

    def test_trailing_newline_kept_as_found(self):
        self.settings.write_text('{"a": 1}', encoding="utf-8")
        change = self.plan()
        self.assertFalse(change.new_text.endswith("\n"))
        self.assertEqual(change.old_text, '{"a": 1}')
        self.settings.write_text('{"a": 1}\n', encoding="utf-8")
        self.assertTrue(self.plan().new_text.endswith("}\n"))

    def test_settings_unchanged_and_refused(self):
        self.settings.write_text('{"modelThinkingLevels": {"llmctl/local": "off"}, "x": [1, 2]}')
        self.assertIsNone(self.plan())
        self.settings.write_text('{"a": 1,}')
        with self.assertRaises(ClientError):
            self.plan()
        self.settings.write_text('{"modelThinkingLevels": ["x"]}')
        with self.assertRaises(ClientError) as caught:
            self.plan()
        self.assertIn('"modelThinkingLevels" is not a JSON object; fix it, or leave Pi out with LLMCTL_CLIENTS',
                      str(caught.exception))

    def test_plan_defaults_sets_and_removes(self):
        self.write_json(self.settings, {"theme": "dark", "defaultProvider": "anthropic", "defaultModel": "claude-x"})
        change = pi_config.plan_defaults(self.env, "llmctl", "local")
        self.assertEqual(json.loads(change.new_text),
                         {"theme": "dark", "defaultProvider": "llmctl", "defaultModel": "local"})
        self.assertTrue(change.backup)
        change = pi_config.plan_defaults(self.env, None, None)
        self.assertEqual(json.loads(change.new_text), {"theme": "dark"})
        self.assertIsNone(pi_config.plan_defaults(self.env, "anthropic", "claude-x"))
        self.settings.write_text('{"theme": "dark"}\n')
        self.assertIsNone(pi_config.plan_defaults(self.env, None, None))


class SettingsReaderTest(PiFilesCase):
    def test_readers(self):
        self.assertEqual(pi_config.pi_default(self.env), (None, None))
        self.assertIsNone(pi_config.stored_level(self.env, "local"))
        self.write_json(self.settings, {"defaultProvider": "llmctl", "defaultModel": "local",
                                        "modelThinkingLevels": {"llmctl/local": "high"}})
        self.assertEqual(pi_config.pi_default(self.env), ("llmctl", "local"))
        self.assertEqual(pi_config.stored_level(self.env, "local"), "high")
        self.assertIsNone(pi_config.stored_level(self.env, "gpt-oss-20b"))

    def test_enabled_models_skip_matches_like_pi(self):
        # Pi 1.5.2 keeps its default when an enabledModels entry matches it (dist/main.js:435-455, model-resolver.js).
        for enabled in (["llmctl/local"], ["anthropic/claude-x", "llmctl/local"], ["local"], ["LLMCTL/Local"],
                        ["llmctl/local:high"], ["local:off"], ["llmctl/*"], ["*local*"], ["*/LOC?L"],
                        ["llmctl/[lx]ocal"], ["llmctl/*:high"]):
            with self.subTest(enabled=enabled):
                self.assertIsNone(pi_config.enabled_models_skip({"enabledModels": enabled}, "local"))
        for settings in ({}, {"enabledModels": []}, {"enabledModels": "llmctl/local"}):
            with self.subTest(settings=settings):
                self.assertIsNone(pi_config.enabled_models_skip(settings, "local"))

    def test_enabled_models_skip_names_the_first_entry(self):
        for enabled, first in ((["anthropic/claude-x"], "anthropic/claude-x"),
                               (["openai/gpt-5", "anthropic/*"], "openai/gpt-5"),
                               (["llmctl/localx"], "llmctl/localx"), (["other/local-2"], "other/local-2"),
                               (["*gpt*", "anthropic/claude-x:high"], "*gpt*")):
            with self.subTest(enabled=enabled):
                self.assertEqual(pi_config.enabled_models_skip({"enabledModels": enabled}, "local"), first)
        self.assertEqual(pi_config.enabled_models_skip({"enabledModels": ["llmctl/local"]}, "gpt-oss-20b"),
                         "llmctl/local")
        self.assertIsNone(pi_config.enabled_models_skip({"enabledModels": ["gpt-oss-20b:medium"]}, "gpt-oss-20b"))

    def test_enabled_models_note(self):
        self.write_json(self.settings, {"defaultProvider": "llmctl", "enabledModels": ["anthropic/claude-x"]})
        self.assertEqual(pi_config.enabled_models_note(self.env, self.qwen36),
                         "note: a new Pi session without --model starts with the first of enabledModels "
                         "(anthropic/claude-x) instead of llmctl/local; add llmctl/local to enabledModels "
                         "or clear enabledModels")
        for settings in ({"defaultProvider": "llmctl", "enabledModels": ["llmctl/local"]},
                         {"defaultProvider": "llmctl", "enabledModels": ["llmctl/local:high", "a/b"]},
                         {"defaultProvider": "llmctl", "enabledModels": ["a/b", "llmctl/local"]},
                         {"defaultProvider": "llmctl", "enabledModels": ["anthropic/claude-x", "llmctl/*"]},
                         {"defaultProvider": "llmctl", "enabledModels": ["*local*"]},
                         {"defaultProvider": "llmctl", "enabledModels": []},
                         {"defaultProvider": "anthropic", "enabledModels": ["anthropic/claude-x"]}):
            with self.subTest(settings=settings):
                self.write_json(self.settings, settings)
                self.assertIsNone(pi_config.enabled_models_note(self.env, self.qwen36))
        self.write_json(self.settings, {"defaultProvider": "llmctl", "enabledModels": ["llmctl/localx"]})
        self.assertIsNotNone(pi_config.enabled_models_note(self.env, self.qwen36))


class SettingsLockTest(PiFilesCase):
    def setUp(self):
        super().setUp()
        self.host = FakeHost(env=self.env)
        self.lock = Path(str(self.settings) + ".lock")

    def test_settings_lock_acquired_and_released(self):
        with pi_config.settings_lock(self.host, self.settings):
            self.assertTrue(self.lock.is_dir())
        self.assertFalse(os.path.lexists(self.lock))
        with self.assertRaises(RuntimeError):
            with pi_config.settings_lock(self.host, self.settings):
                raise RuntimeError("write failed")
        self.assertFalse(os.path.lexists(self.lock))
        self.assertEqual(self.host.sleeps, [])

    def test_settings_lock_follows_symlink(self):
        real = self.home / "dotfiles"
        real.mkdir()
        (real / "settings.json").write_text("{}")
        self.settings.symlink_to(real / "settings.json")
        with pi_config.settings_lock(self.host, self.settings):
            self.assertTrue((real / "settings.json.lock").is_dir())
            self.assertFalse(os.path.lexists(self.lock))

    def test_settings_lock_held_times_out(self):
        self.lock.mkdir()
        entered = []
        with self.assertRaises(ClientWriteError) as caught:
            with pi_config.settings_lock(self.host, self.settings):
                entered.append(True)
        self.assertEqual(entered, [])
        self.assertEqual(str(caught.exception),
                         f"{self.settings}: Pi holds its settings lock {self.lock}; if no Pi is running, "
                         f"remove {self.lock}, then run ./llmctl up again")
        self.assertGreaterEqual(self.host.clock, 2.0)
        self.assertLess(self.host.clock, 2.2)
        self.assertTrue(self.host.sleeps)
        self.assertEqual(set(self.host.sleeps), {0.1})
        self.assertTrue(self.lock.is_dir())

    def test_settings_lock_freed_during_wait(self):
        self.lock.mkdir()
        self.host.on_sleep = lambda host: self.lock.rmdir() if host.sleeps == [0.1] else None
        with pi_config.settings_lock(self.host, self.settings):
            self.assertTrue(self.lock.is_dir())
        self.assertEqual(self.host.sleeps, [0.1])
        self.assertFalse(os.path.lexists(self.lock))

    def replace_lock(self):
        """Do what proper-lockfile does with a stale lock: a new lock directory in place of llmctl's."""
        spare = self.home / "spare-lock"
        spare.mkdir()
        self.lock.rmdir()
        spare.rename(self.lock)
        return os.stat(self.lock).st_ino

    def test_settings_lock_taken_over_is_left_alone(self):
        with self.assertRaises(ClientWriteError) as caught:
            with pi_config.settings_lock(self.host, self.settings):
                inode = self.replace_lock()
        self.assertEqual(str(caught.exception),
                         f"{self.settings}: Pi took over its settings lock while llmctl held it, so another writer "
                         "may have changed settings.json; check it")
        self.assertTrue(self.lock.is_dir())
        self.assertEqual(os.stat(self.lock).st_ino, inode)

    def test_settings_lock_removed_by_another_writer(self):
        with self.assertRaises(ClientWriteError) as caught:
            with pi_config.settings_lock(self.host, self.settings):
                self.lock.rmdir()
        self.assertIn("Pi took over its settings lock while llmctl held it", str(caught.exception))
        self.assertFalse(os.path.lexists(self.lock))

    def test_settings_lock_taken_over_keeps_the_body_error(self):
        with self.assertRaises(RuntimeError):
            with pi_config.settings_lock(self.host, self.settings):
                self.replace_lock()
                raise RuntimeError("write failed")
        self.assertTrue(self.lock.is_dir())


if __name__ == "__main__":
    unittest.main()
