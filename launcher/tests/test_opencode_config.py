import json
import os
import re
import stat
import tempfile
import unittest
from pathlib import Path

from launcher.errors import UsageError
from launcher.opencode_config import apply_default, plan_default
from launcher.tests.fakes import REPO


def doc_config_text() -> str:
    """The provider config from docs/opencode.md, the way the KB installs it."""
    text = (REPO / "docs/opencode.md").read_text()
    return re.search(r"```json\n(.*?)\n```", text[text.index("## The validated config"):], re.S).group(1) + "\n"


class OpencodeDefaultTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = {"HOME": "/nonexistent", "XDG_CONFIG_HOME": self.tmp.name}
        self.path = Path(self.tmp.name) / "opencode" / "opencode.json"
        self.path.parent.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, text, mode=0o664):
        self.path.write_text(text)
        os.chmod(self.path, mode)

    def test_changes_only_the_model_line(self):
        original = doc_config_text()
        self.write(original)
        plan = plan_default(self.env, "llamacpp-docker/gemma-4-26b-a4b")
        self.assertEqual((plan.path, plan.previous), (self.path, "llamacpp-docker/qwen3.6-35b-a3b"))
        apply_default(plan)
        new = self.path.read_text()
        changed = [(a, b) for a, b in zip(original.splitlines(), new.splitlines()) if a != b]
        self.assertEqual(changed, [('  "model": "llamacpp-docker/qwen3.6-35b-a3b",', '  "model": "llamacpp-docker/gemma-4-26b-a4b",')])
        self.assertEqual(len(original.splitlines()), len(new.splitlines()))
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o664)

    def test_unknown_entry_is_refused(self):
        self.write(doc_config_text())
        with self.assertRaises(UsageError) as ctx:
            plan_default(self.env, "llamacpp-docker/no-such-model")
        self.assertIn("no entry llamacpp-docker/no-such-model", str(ctx.exception))

    def test_comments_are_refused(self):
        self.write("// my settings\n" + doc_config_text())
        with self.assertRaises(UsageError) as ctx:
            plan_default(self.env, "llamacpp-docker/gemma-4-26b-a4b")
        self.assertIn("not plain JSON", str(ctx.exception))

    def test_missing_config_is_refused(self):
        with self.assertRaises(UsageError) as ctx:
            plan_default(self.env, "llamacpp-docker/gemma-4-26b-a4b")
        self.assertIn("docs/opencode.md", str(ctx.exception))

    def test_model_key_is_added_when_missing(self):
        data = json.loads(doc_config_text())
        del data["model"]
        self.write(json.dumps(data, indent=2) + "\n")
        apply_default(plan_default(self.env, "vllm-docker/gpt-oss-20b"))
        self.assertEqual(json.loads(self.path.read_text())["model"], "vllm-docker/gpt-oss-20b")

    def test_nested_model_first_is_refused_not_misedited(self):
        data = json.loads(doc_config_text())
        nested = {"$schema": data.pop("$schema"), "agents": {"title": {"model": "x/y"}}}
        nested.update(data)
        text = json.dumps(nested, indent=2) + "\n"
        self.write(text)
        with self.assertRaises(UsageError):
            plan_default(self.env, "llamacpp-docker/gemma-4-26b-a4b")
        self.assertEqual(self.path.read_text(), text)

    def test_same_model_is_a_no_op(self):
        self.write(doc_config_text())
        plan = plan_default(self.env, "llamacpp-docker/qwen3.6-35b-a3b")
        self.assertIsNone(plan.new_text)


if __name__ == "__main__":
    unittest.main()
