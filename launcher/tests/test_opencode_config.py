import dataclasses
import json
import re
import tempfile
import unittest
from pathlib import Path

from launcher import opencode_config
from launcher.errors import ClientError, UsageError
from launcher.tests.fakes import REPO
from launcher.tests.regen_golden import target_for


def doc_config_text() -> str:
    """The provider config from docs/opencode.md, the way the KB installs it."""
    text = (REPO / "docs/opencode.md").read_text()
    return re.search(r"```json\n(.*?)\n```", text[text.index("## The validated config"):], re.S).group(1) + "\n"


QWEN = "llamacpp-docker/qwen3.6-35b-a3b"
GPT_OSS = "vllm-docker/gpt-oss-20b"


def with_model_line(text: str, value_token: str) -> str:
    """The doc config with its top-level "model" value token replaced by value_token, nothing else touched."""
    old = f'  "model": "{QWEN}",\n'
    assert text.count(old) == 1
    return text.replace(old, f'  "model": {value_token},\n')


class ModelLineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = {"HOME": "/nonexistent", "XDG_CONFIG_HOME": self.tmp.name}
        self.dir = Path(self.tmp.name) / "opencode"
        self.path = self.dir / "opencode.json"
        self.dir.mkdir()
        self.providers = {"llamacpp-docker", "vllm-docker"}
        self.qwen = target_for("qwen3.6-35b-a3b", "llamacpp", REPO / "recipes")

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, text, path=None):
        (path or self.path).write_text(text, encoding="utf-8")

    def doc_data(self) -> dict:
        return json.loads(doc_config_text())

    def plan(self, target=None, set_default=False):
        return opencode_config.plan_model_line(self.env, target or self.qwen, self.providers, set_default=set_default)

    def assert_only_model_changed(self, change, old_text, model):
        old = json.loads(old_text)
        self.assertEqual(json.loads(change.new_text), dict(old, model=model))
        self.assertEqual(list(json.loads(change.new_text)), list(old) if "model" in old else ["model", *old])

    def test_config_path_needs_home(self):
        with self.assertRaises(UsageError):
            opencode_config.config_path({})
        with self.assertRaises(UsageError):
            opencode_config.config_path({"XDG_CONFIG_HOME": "~/cfg"})
        self.assertEqual(opencode_config.config_path({"HOME": "/h"}), Path("/h/.config/opencode/opencode.json"))

    def test_jsonc_used_when_only_it_exists(self):
        jsonc = self.dir / "opencode.jsonc"
        text = with_model_line(doc_config_text(), json.dumps(GPT_OSS))
        self.write(text, jsonc)
        self.assertEqual(opencode_config.config_path(self.env), jsonc)
        change = self.plan()
        self.assertEqual(change.path, jsonc)
        self.assert_only_model_changed(change, text, QWEN)

    def test_follow_moves_llmctl_default(self):
        text = with_model_line(doc_config_text(), json.dumps(GPT_OSS))
        self.write(text)
        change = self.plan()
        self.assertEqual(change.path, self.path)
        self.assertEqual(change.old_text, text)
        self.assert_only_model_changed(change, text, QWEN)
        old_lines, new_lines = text.splitlines(), change.new_text.splitlines()
        self.assertEqual(len(old_lines), len(new_lines))
        self.assertEqual(sum(a != b for a, b in zip(old_lines, new_lines)), 1)
        self.assertEqual(change.summary, f"model {QWEN} (was {GPT_OSS})")
        self.assertEqual(change.preview, f"would set OpenCode's default model: {GPT_OSS} -> {QWEN} ({self.path})")
        self.assertTrue(change.backup)

    def test_crlf_config_changes_only_the_model_value(self):
        text = with_model_line(doc_config_text(), json.dumps(GPT_OSS)).replace("\n", "\r\n")
        self.path.write_bytes(text.encode("utf-8"))
        change = self.plan()
        self.assertEqual(change.old_text, text)
        self.assertEqual(change.new_text, text.replace(json.dumps(GPT_OSS), json.dumps(QWEN), 1))
        self.assertEqual(change.new_text.count("\r\n"), text.count("\r\n"))

    def test_foreign_default_stays_without_flag(self):
        text = with_model_line(doc_config_text(), json.dumps("openrouter/qwen/qwen3.6-plus"))
        self.write(text)
        self.assertIsNone(self.plan())
        change = self.plan(set_default=True)
        self.assert_only_model_changed(change, text, QWEN)
        self.assertEqual(change.summary, f"model {QWEN} (was openrouter/qwen/qwen3.6-plus)")

    def test_unset_default_follows_only_with_flag(self):
        data = self.doc_data()
        del data["model"]
        text = json.dumps(data, indent=2) + "\n"
        self.write(text)
        self.assertIsNone(self.plan())
        change = self.plan(set_default=True)
        self.assert_only_model_changed(change, text, QWEN)
        self.assertEqual(change.summary, f"model {QWEN} (was unset)")

    def test_recipe_without_docs_entry(self):
        self.write(doc_config_text())
        shipped = target_for("deepseek-r1-0528-qwen3-8b", "llamacpp", REPO / "recipes")
        deepseek = dataclasses.replace(shipped, recipe=dataclasses.replace(shipped.recipe, opencode=False))
        self.assertIsNone(self.plan(deepseek))
        self.assertIsNone(self.plan(deepseek, set_default=True))
        self.assertEqual(opencode_config.opencode_notes(self.env, deepseek), [
            "OpenCode: deepseek-r1-0528-qwen3-8b has no OpenCode entry in docs/opencode.md; "
            "its default stays llamacpp-docker/qwen3.6-35b-a3b"])

    def test_config_without_the_entry(self):
        note = f"OpenCode: no entry for {QWEN} in {self.path}; add the provider config from docs/opencode.md"
        self.assertIsNone(self.plan(set_default=True))
        self.assertEqual(opencode_config.opencode_notes(self.env, self.qwen), [note])
        data = self.doc_data()
        del data["providers"]["llamacpp-docker"]["models"]["qwen3.6-35b-a3b"]
        self.write(json.dumps(data, indent=2) + "\n")
        self.assertIsNone(self.plan(set_default=True))
        self.assertEqual(opencode_config.opencode_notes(self.env, self.qwen), [note])

    def test_has_entry(self):
        self.assertFalse(opencode_config.has_entry(self.env, self.qwen))
        self.write(doc_config_text())
        self.assertTrue(opencode_config.has_entry(self.env, self.qwen))
        self.write("// mine\n" + doc_config_text())
        self.assertFalse(opencode_config.has_entry(self.env, self.qwen))
        data = self.doc_data()
        del data["providers"]["llamacpp-docker"]["models"]["qwen3.6-35b-a3b"]
        self.write(json.dumps(data, indent=2) + "\n")
        self.assertFalse(opencode_config.has_entry(self.env, self.qwen))

    def test_not_plain_json(self):
        text = "// x\n" + with_model_line(doc_config_text(), json.dumps(GPT_OSS))
        self.write(text)
        with self.assertRaises(ClientError) as ctx:
            self.plan(set_default=True)
        self.assertEqual(ctx.exception.exit_code, 3)
        self.assertTrue(str(ctx.exception).startswith(f"{self.path} is not plain JSON ("), str(ctx.exception))
        self.assertIsNone(self.plan())
        notes = opencode_config.opencode_notes(self.env, self.qwen)
        self.assertEqual(len(notes), 1)
        self.assertTrue(notes[0].startswith(f"note: {self.path} is not plain JSON ("), notes[0])
        self.assertTrue(notes[0].endswith("), so llmctl leaves OpenCode's default as it is"), notes[0])
        self.assertEqual(self.path.read_text(), text)

    def test_not_a_json_object(self):
        self.write("[]\n")
        self.assertIsNone(self.plan())
        with self.assertRaises(ClientError):
            self.plan(set_default=True)
        self.assertEqual(opencode_config.opencode_notes(self.env, self.qwen),
                         [f"note: {self.path} is not a JSON object, so llmctl leaves OpenCode's default as it is"])

    def test_stale_entry_warning(self):
        data = self.doc_data()
        data["providers"]["llamacpp-docker"]["models"]["qwen3.6-35b-a3b"]["limit"]["output"] = 8192
        self.write(json.dumps(data, indent=2) + "\n")
        self.assertEqual(opencode_config.opencode_notes(self.env, self.qwen), [
            f"warning: OpenCode's entry {QWEN} in {self.path} differs from the recipe: "
            "limit.output 8192 instead of 16384; update it from docs/opencode.md"])

    def test_extra_entry_keys_are_not_stale(self):
        data = self.doc_data()
        entry = data["providers"]["llamacpp-docker"]["models"]["qwen3.6-35b-a3b"]
        entry["cost"] = {"input": 0}
        entry["name"] = "my own name"
        self.write(json.dumps(data, indent=2) + "\n")
        self.assertEqual(opencode_config.opencode_notes(self.env, self.qwen), [])

    def test_entry_difference(self):
        expected = {"a": {"b": [1, 2], "c": True}, "d": "x"}
        cases = [
            ({"z": 0, "d": "x", "a": {"c": True, "b": [1, 2]}}, None),
            ({"a": {"b": [1], "c": 1}}, "a.b [1] instead of [1, 2]"),
            ({"a": {"b": [1, 2], "c": 1}}, "a.c 1 instead of true"),
            ({"a": {"b": [1, 2], "c": True}}, 'd absent instead of "x"'),
            ({"a": 3, "d": "x"}, 'a 3 instead of {"b": [1, 2], "c": true}'),
        ]
        for installed, difference in cases:
            with self.subTest(installed=installed):
                self.assertEqual(opencode_config.entry_difference(installed, expected), difference)

    def test_points_at_llmctl(self):
        self.assertTrue(opencode_config.points_at_llmctl(GPT_OSS, self.providers))
        for model in (None, 3, "", "vllm-docker", "openrouter/vllm-docker/x", "llamacpp-dockerx/y"):
            with self.subTest(model=model):
                self.assertFalse(opencode_config.points_at_llmctl(model, self.providers))

    def test_set_top_level_model_insertion(self):
        cases = [
            ("{}", '{"model": "p/m"}'),
            ('{"a": 1}', '{"model": "p/m", "a": 1}'),
            ('{\n    "a": 1\n}\n', '{\n    "model": "p/m",\n    "a": 1\n}\n'),
            ('{\r\n\t"a": 1\r\n}', '{\r\n\t"model": "p/m",\r\n\t"a": 1\r\n}'),
            ('{"model": null, "b": {"model": 1}}', '{"model": "p/m", "b": {"model": 1}}'),
            ('{"b": {"model": 1}, "mod\\u0065l": "x"}', '{"b": {"model": 1}, "mod\\u0065l": "p/m"}'),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(opencode_config.set_top_level_model(text, "p/m"), expected)

    def test_same_model_gives_no_change(self):
        self.write(doc_config_text())
        self.assertIsNone(self.plan(set_default=True))
        self.assertIsNone(opencode_config.plan_set_model(self.env, QWEN))

    def test_plan_set_model_without_config(self):
        with self.assertRaises(ClientError) as ctx:
            opencode_config.plan_set_model(self.env, QWEN)
        self.assertEqual(str(ctx.exception), f"no OpenCode config at {self.path}")

    def test_follow_handles_unusual_valid_configs(self):
        doc = with_model_line(doc_config_text(), json.dumps(GPT_OSS))
        data = json.loads(doc)
        nested = {"agents": {"title": {"model": QWEN}}}
        nested.update(data)
        texts = {
            "minified": json.dumps(data, separators=(",", ":")),
            "nested model first": json.dumps(nested, indent=2) + "\n",
            "escaped slash": with_model_line(doc_config_text(), '"vllm-docker\\/gpt-oss-20b"'),
        }
        self.assertIn('"vllm-docker\\/gpt-oss-20b"', texts["escaped slash"])
        for name, text in texts.items():
            with self.subTest(config=name):
                self.write(text)
                change = self.plan()
                self.assertIsNotNone(change)
                self.assert_only_model_changed(change, text, QWEN)
                if name == "nested model first":
                    self.assertEqual(json.loads(change.new_text)["agents"], {"title": {"model": QWEN}})
                    self.assertTrue(change.new_text.startswith('{\n  "agents"'))


if __name__ == "__main__":
    unittest.main()
