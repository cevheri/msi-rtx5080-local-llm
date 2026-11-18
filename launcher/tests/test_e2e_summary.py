"""The shared e2e summary format: the reader is the exact inverse of the driver's writer, and the step vocabulary."""

import unittest

from launcher import clients
from launcher.e2e import clients as e2e, summary
from launcher.tests.fakes import REPO


def rebuilt(rows):
    return [e2e._Row(r.recipe, r.step, r.passed, [r.reasons] if r.reasons else [], float(r.seconds)) for r in rows]


class SummaryTest(unittest.TestCase):
    def test_read_summary_inverts_summary(self):
        rows = [e2e._Row("01-qwen3.6-35b-a3b-llamacpp", "up", True, [], 12.4),
                e2e._Row("01-qwen3.6-35b-a3b-llamacpp", "opencode", False, ["a | b", "two\nlines"], 7.6),
                e2e._Row("round-trip", "restore", True, [], 0.0), e2e._Row("-", "down", False, ["exit status 1"], 3.0),
                e2e._Row("02-gpt-oss-20b-vllm", "kimi", False, [" cut at 120 "], 1.0)]
        text = e2e._summary(rows)
        parsed = summary.read_summary(text)
        self.assertEqual(parsed[1], summary.SummaryRow("01-qwen3.6-35b-a3b-llamacpp", "opencode", False,
                                                       "a / b; two lines", 8))
        self.assertEqual(e2e._summary(rebuilt(parsed)), text)

    def test_read_summary_rejects_a_malformed_file(self):
        head = "\n".join(summary.SUMMARY_HEAD) + "\n"
        cases = {"# other\n" + head.split("\n", 1)[1]: "line 1:",
                 head + "| 01-x-llamacpp | up | Yes |  | 3 |": "line 5:",          # no final newline
                 head + "| 01-x-llamacpp | up | Yes | 3 |\n": "line 5:",          # 4 cells
                 head + "| 01-x-llamacpp | up | Yes | why | 3 |\n": "line 5:",
                 head + "| 01-x-llamacpp | up | No |  | 3 |\n": "line 5:",
                 head + "| 01-x-llamacpp | up | Yes |  | 12.5 |\n": "line 5:",
                 head + "| x-llamacpp | up | Yes |  | 3 |\n": "line 5:",
                 head + "|x01-x-llamacpp | up | Yes |  | 3 |\n": "line 5:",
                 head + "| 01-x-llamacpp | up | Maybe | r | 3 |\n": "line 5:",
                 head + "| 01-x-vLLM | up | Yes |  | 3 |\n": "line 5:"}
        for text, start in cases.items():
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "^" + start):
                summary.read_summary(text)
        messages = {head + "| 01-x-llamacpp |\n": "^line 5: 1 cell instead of 5$",
                    "# llmctl clients e2e\nx\n": "^line 2: expected a blank line$",
                    head + "| x-llamacpp | up | Yes |  | 3 |\n": "is not '-', 'round-trip' or '<nn>-<id>-<engine>'$"}
        for text, message in messages.items():
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, message):
                summary.read_summary(text)

    def test_read_summary_reads_an_engine_llmctl_no_longer_has(self):
        text = "\n".join(summary.SUMMARY_HEAD) + "\n| 01-foo-oldengine | up | Yes |  | 3 |\n"
        self.assertEqual(summary.read_summary(text)[0].recipe, "01-foo-oldengine")
        self.assertEqual(summary.recipe_key("01-foo-oldengine"), ("foo", "oldengine"))
        self.assertEqual(summary.recipe_key("12-qwen3.6-35b-a3b-sglang"), ("qwen3.6-35b-a3b", "sglang"))
        self.assertEqual((summary.recipe_key("round-trip"), summary.recipe_key("-")), (None, None))
        with self.assertRaisesRegex(ValueError, "^Recipe 'x-llamacpp' is not '-', 'round-trip' or '<nn>-<id>-<engine>'$"):
            summary.recipe_key("x-llamacpp")

    def test_committed_summaries_read_back(self):
        found = 0
        for path in sorted((REPO / "research").rglob("summary.md")):
            text = path.read_text(encoding="utf-8")
            if text.split("\n", 1)[0] != summary.SUMMARY_HEAD[0]:
                continue
            found += 1
            with self.subTest(path=str(path.relative_to(REPO))):
                rows = summary.read_summary(text)
                if tuple(text.split("\n")[:4]) == summary.SUMMARY_HEAD:
                    self.assertEqual(e2e._summary(rebuilt(rows)), text)
        self.assertGreater(found, 0)

    def test_step_client_takes_the_longest_name(self):
        hermes = ("hermes-z", "hermes-chat", "hermes-config-model", "hermes-config-providers",
                  "hermes-config-model_overrides", "hermes-config-gateway", "hermes-config-platform_toolsets")
        expected = {"opencode": "opencode", "pi": "pi", "dsh-headless": "dsh", "dsh-web": "dsh", "dsh-config": "dsh",
                    "kimi": "kimi", "kimi-config": "kimi", **dict.fromkeys(hermes, "hermes"),
                    **dict.fromkeys(summary.SETUP_STEPS, None)}
        for step, client in expected.items():
            self.assertEqual(summary.step_client(step, clients.NAMES), client, step)
        self.assertEqual(summary.step_client("pi-agent-config", ("pi", "pi-agent")), "pi-agent")
        self.assertEqual(summary.step_client("pi-config", ("pi", "pi-agent")), "pi")
        self.assertEqual(summary.step_client("pi-agent-config", ("pi-agent", "pi")), "pi-agent")
        with self.assertRaisesRegex(ValueError, "foo-bar"):
            summary.step_client("foo-bar", clients.NAMES)


if __name__ == "__main__":
    unittest.main()
