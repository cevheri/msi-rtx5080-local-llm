import dataclasses
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from launcher import hermes_config, opencode_config, pi_config
from launcher.catalog import load_catalog
from launcher.errors import ClientError, ClientWriteError, UsageError
from launcher.files import make_dirs, write_atomic
from launcher.tests.fakes import REPO, FakeHost, fail, ok
from launcher.tests.regen_golden import target_for

RECIPES = REPO / "recipes"

# The four lines between the markers of the spec's D1 example (docs/superpowers/specs/2026-11-16-llmctl-clients-design.md).
QWEN36_BLOCK = """\
model: {"provider": "custom:llmctl", "default": "local"}
providers: {"llmctl": {"name": "llmctl: qwen3.6-35b-a3b on llama.cpp", "api": "http://127.0.0.1:8080/v1", "transport": "chat_completions", "default_model": "local", "discover_models": false, "extra_body": {"max_tokens": 16384}, "models": {"local": {"context_length": 131072}}}}
model_overrides: {"custom": {"local": {"supports_reasoning": false}}}
gateway: {"standalone": true}
"""

# The groups of the spec section "Generated client entries", without their heading comments.
SPEC_BLOCKS = (
    ("llamacpp", "nemotron-3.5-bolt", """\
model: {"provider": "custom:llmctl", "default": "local"}
providers: {"llmctl": {"name": "llmctl: nemotron-3.5-bolt on llama.cpp", "api": "http://127.0.0.1:8080/v1", "transport": "chat_completions", "default_model": "local", "discover_models": false, "extra_body": {"max_tokens": 16384}, "models": {"local": {"context_length": 131072}}}}
model_overrides: {"custom": {"local": {"supports_reasoning": false}}}
gateway: {"standalone": true}
"""),
    ("llamacpp", "qwen3.8-27b", """\
model: {"provider": "custom:llmctl", "default": "local"}
providers: {"llmctl": {"name": "llmctl: qwen3.8-27b on llama.cpp", "api": "http://127.0.0.1:8080/v1", "transport": "chat_completions", "default_model": "local", "discover_models": false, "extra_body": {"max_tokens": 16384}, "models": {"local": {"context_length": 65536}}}}
model_overrides: {}
gateway: {"standalone": true}
"""),
    ("llamacpp", "k2-horizon-7b", """\
model: {"provider": "custom:llmctl", "default": "local"}
providers: {"llmctl": {"name": "llmctl: k2-horizon-7b on llama.cpp", "api": "http://127.0.0.1:8080/v1", "transport": "chat_completions", "default_model": "local", "discover_models": false, "extra_body": {"max_tokens": 32768}, "models": {"local": {"context_length": 81920}}}}
model_overrides: {"custom": {"local": {"supports_reasoning": false}}}
gateway: {"standalone": true}
"""),
    ("llamacpp", "qwen3-coder-next", """\
model: {"provider": "custom:llmctl", "default": "local"}
providers: {"llmctl": {"name": "llmctl: qwen3-coder-next on llama.cpp", "api": "http://127.0.0.1:8080/v1", "transport": "chat_completions", "default_model": "local", "discover_models": false, "extra_body": {"max_tokens": 16384}, "models": {"local": {"context_length": 131072}}}}
model_overrides: {"custom": {"local": {"supports_reasoning": false}}}
gateway: {"standalone": true}
"""),
    ("llamacpp", "deepseek-r1-0528-qwen3-8b", """\
model: {"provider": "custom:llmctl", "default": "local"}
providers: {"llmctl": {"name": "llmctl: deepseek-r1-0528-qwen3-8b on llama.cpp", "api": "http://127.0.0.1:8080/v1", "transport": "chat_completions", "default_model": "local", "discover_models": false, "extra_body": {"max_tokens": 16384}, "models": {"local": {"context_length": 65536}}}}
model_overrides: {"custom": {"local": {"supports_reasoning": false}}}
gateway: {"standalone": true}
platform_toolsets: {"cli": []}
"""),
    ("vllm", "gpt-oss-20b", """\
model: {"provider": "custom:llmctl", "default": "gpt-oss-20b"}
providers: {"llmctl": {"name": "llmctl: gpt-oss-20b on vLLM", "api": "http://127.0.0.1:8000/v1", "transport": "chat_completions", "default_model": "gpt-oss-20b", "discover_models": false, "extra_body": {"max_tokens": 16384}, "models": {"gpt-oss-20b": {"context_length": 65536}}}}
model_overrides: {}
gateway: {"standalone": true}
"""),
)


def shipped_targets():
    catalog = load_catalog(RECIPES, None)
    for runtime, entry in catalog.runtimes.items():
        for recipe_id in sorted(entry.recipes):
            yield target_for(recipe_id, runtime, RECIPES)


def split_line(line: str) -> tuple[str, object]:
    key, sep, value = line.partition(": ")
    if not sep:
        raise AssertionError(f"not a <key>: <value> line: {line!r}")
    return key, json.loads(value)


class RenderTest(unittest.TestCase):
    def test_qwen36_block_is_the_spec_block(self):
        target = target_for("qwen3.6-35b-a3b", "llamacpp", RECIPES)
        self.assertEqual(hermes_config.render_block(target), QWEN36_BLOCK)

    def test_blocks_match_spec_examples(self):
        for runtime, recipe_id, block in SPEC_BLOCKS:
            with self.subTest(runtime=runtime, recipe=recipe_id):
                self.assertEqual(hermes_config.render_block(target_for(recipe_id, runtime, RECIPES)), block)

    def test_new_config_layout(self):
        target = target_for("qwen3.6-35b-a3b", "llamacpp", RECIPES)
        lines = hermes_config.new_config_text(target).splitlines()
        block = hermes_config.render_block(target).splitlines()
        self.assertEqual(lines, [
            "# Hermes profile for the model llmctl runs (docs/llmctl.md#clients).",
            "# llmctl rewrites the block below on every ./llmctl up and keeps everything outside it.",
            "# The keys model, providers, model_overrides, gateway and platform_toolsets belong to the block.",
            hermes_config.BEGIN, *block, hermes_config.END, "_config_version: 53"])
        self.assertTrue(hermes_config.new_config_text(target).endswith("_config_version: 53\n"))
        self.assertEqual(hermes_config.BEGIN,
                         "# >>> llmctl managed block: rewritten by llmctl on every start; edits here are lost >>>")
        self.assertEqual(hermes_config.END, "# <<< llmctl managed block <<<")

    def test_block_lines_reparse(self):
        checked = 0
        for target in shipped_targets():
            with self.subTest(runtime=target.runtime.name, recipe=target.recipe.id):
                text = hermes_config.render_block(target)
                self.assertTrue(text.endswith("\n"))
                self.assertTrue(text.isascii())
                self.assertNotIn("\\/", text)
                pairs = [split_line(line) for line in text.splitlines()]
                keys = [key for key, _ in pairs]
                owned = list(hermes_config.OWNED_KEYS)
                self.assertEqual(keys, owned if target.recipe.chat_only else owned[:-1])
                self.assertEqual(dict(pairs), hermes_config.block_values(target))
                checked += 1
        self.assertEqual(checked, 24)

    def test_odd_recipe_id_renders(self):
        target = target_for("qwen3.6-35b-a3b", "llamacpp", RECIPES)
        recipe = dataclasses.replace(target.recipe, id='mod\u00e8le#1 "x"')
        odd = dataclasses.replace(target, recipe=recipe)
        text = hermes_config.render_block(odd)
        self.assertTrue(text.isascii())
        values = dict(split_line(line) for line in text.splitlines())
        self.assertEqual(values["providers"]["llmctl"]["name"], 'llmctl: mod\u00e8le#1 "x" on llama.cpp')
        json.dumps(pi_config.provider_entry(odd))
        json.dumps(opencode_config.expected_entry(recipe, odd.served))


def qwen36():
    return target_for("qwen3.6-35b-a3b", "llamacpp", RECIPES)


def gptoss_vllm():
    return target_for("gpt-oss-20b", "vllm", RECIPES)


def apply(change):
    make_dirs(change.dirs)
    write_atomic(change.path, change.new_text, new_mode=change.mode)


class HermesHomeCase(unittest.TestCase):
    """A temporary HOME with an empty .hermes/ root, passed in through the environment mapping."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.root = self.home / ".hermes"
        self.root.mkdir()
        self.env = {"HOME": str(self.home)}
        self.path = self.root / "profiles" / "llmctl" / "config.yaml"

    def tearDown(self):
        self.tmp.cleanup()

    def write_profile(self, text):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)


class RootTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_root_default(self):
        env = {"HOME": str(self.home)}
        self.assertEqual(hermes_config.hermes_root(env), self.home / ".hermes")
        self.assertEqual(hermes_config.hermes_root({**env, "HERMES_HOME": ""}), self.home / ".hermes")
        self.assertEqual(hermes_config.profile_dir(env), self.home / ".hermes" / "profiles" / "llmctl")
        self.assertEqual(hermes_config.profile_config(env),
                         self.home / ".hermes" / "profiles" / "llmctl" / "config.yaml")

    def test_root_suffix(self):
        env = {"HOME": str(self.home), "HERMES_DATA_DIR_SUFFIX": "-dev"}
        self.assertEqual(hermes_config.hermes_root(env), self.home / ".hermes-dev")

    def test_root_profile_under_default(self):
        env = {"HOME": str(self.home), "HERMES_HOME": "~/.hermes/profiles/work"}
        self.assertEqual(hermes_config.hermes_root(env), self.home / ".hermes")

    def test_root_custom_profile_dir(self):
        env = {"HOME": str(self.home), "HERMES_HOME": "/x/profiles/work"}
        self.assertEqual(hermes_config.hermes_root(env), Path("/x"))

    def test_root_custom_home(self):
        env = {"HOME": str(self.home), "HERMES_HOME": "$HOME/hh"}
        self.assertEqual(hermes_config.hermes_root(env), self.home / "hh")

    def test_root_needs_home(self):
        with self.assertRaises(UsageError):
            hermes_config.hermes_root({})
        with self.assertRaises(UsageError):
            hermes_config.hermes_root({"HERMES_HOME": "/x/profiles/work"})


class PlanProfileTest(HermesHomeCase):
    def test_creates_profile(self):
        target = qwen36()
        change = hermes_config.plan_profile(self.env, target)
        summary = ("created the Hermes profile llmctl for qwen3.6-35b-a3b on llama.cpp "
                   "(standalone, not served by Hermes' gateway)")
        self.assertIsNone(change.old_text)
        self.assertEqual(change.path, self.path)
        self.assertEqual(change.dirs, (self.root / "profiles", self.root / "profiles" / "llmctl"))
        self.assertEqual(change.mode, 0o600)
        self.assertFalse(change.backup)
        self.assertEqual(change.summary, summary)
        self.assertEqual(change.preview, f"would write {self.path}: {summary}")
        apply(change)
        for directory in change.dirs:
            self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700, directory)
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        text = self.path.read_text(encoding="utf-8")
        self.assertEqual(text, hermes_config.new_config_text(target))
        self.assertEqual(text.splitlines()[-1], "_config_version: 53")

    def test_empty_profile_dir_is_used(self):
        (self.root / "profiles" / "llmctl").mkdir(parents=True)
        change = hermes_config.plan_profile(self.env, qwen36())
        self.assertIsNone(change.old_text)
        self.assertEqual(change.new_text, hermes_config.new_config_text(qwen36()))

    def test_foreign_profile_dir_refused(self):
        directory = self.root / "profiles" / "llmctl"
        directory.mkdir(parents=True)
        (directory / "SOUL.md").write_text("persona\n")
        with self.assertRaises(ClientError) as ctx:
            hermes_config.plan_profile(self.env, qwen36())
        self.assertEqual(ctx.exception.exit_code, 3)
        self.assertTrue(str(ctx.exception).startswith(
            f"{directory} holds a Hermes profile that llmctl did not create (SOUL.md without config.yaml); "
            "rename it with hermes profile rename llmctl <name>, or leave Hermes out with LLMCTL_CLIENTS"),
            str(ctx.exception))

    def test_replace_keeps_outside_bytes(self):
        old = (hermes_config.new_config_text(gptoss_vllm())
               + "# résumé des réglages\ndisplay:\n  compact: true\nextra_key: 1\n")
        self.write_profile(old)
        change = hermes_config.plan_profile(self.env, qwen36())
        self.assertEqual(change.old_text, old)
        begin = old.index(hermes_config.BEGIN) + len(hermes_config.BEGIN) + 1
        end = old.index(hermes_config.END)
        self.assertEqual(change.new_text, old[:begin] + hermes_config.render_block(qwen36()) + old[end:])
        self.assertEqual(change.summary, "llmctl block for qwen3.6-35b-a3b on llama.cpp (was gpt-oss-20b on vLLM)")
        self.assertEqual(change.preview, f"would write {self.path}: {change.summary}")
        self.assertEqual((change.mode, change.backup, change.dirs), (0o600, False, ()))
        apply(change)
        self.assertEqual(self.path.read_bytes(), change.new_text.encode("utf-8"))

    def test_crlf_lines_are_kept(self):
        old = hermes_config.new_config_text(gptoss_vllm()).replace("\n", "\r\n") + "extra_key: 1\r\n"
        self.write_profile(old)
        change = hermes_config.plan_profile(self.env, qwen36())
        begin = old.index(hermes_config.BEGIN) + len(hermes_config.BEGIN) + 2
        end = old.index(hermes_config.END)
        self.assertEqual(change.new_text, old[:begin] + hermes_config.render_block(qwen36()) + old[end:])

    def test_unchanged_profile_is_not_written(self):
        self.write_profile(hermes_config.new_config_text(qwen36()) + "extra_key: 1\n")
        self.assertIsNone(hermes_config.plan_profile(self.env, qwen36()))

    def test_profile_edited_by_hermes(self):
        original = hermes_config.new_config_text(qwen36())
        edited = original.replace('"default": "local"}', '"default": "other-model"}', 1)
        self.assertNotEqual(edited, original)
        edited = edited.replace("_config_version: 53\n", "_config_version: 54\n")
        edited += "agent:\n  reasoning_effort: low\ndisplay:\n  compact: true\n"
        self.write_profile(edited)
        change = hermes_config.plan_profile(self.env, qwen36())
        suffix = edited[edited.index(hermes_config.END):]
        self.assertTrue(suffix.startswith(hermes_config.END + "\n_config_version: 54\nagent:\n"))
        self.assertEqual(change.new_text, original[:original.index(hermes_config.END)] + suffix)
        self.assertEqual(change.summary, "llmctl block for qwen3.6-35b-a3b on llama.cpp (same recipe, block rewritten)")
        self.assertEqual(change.preview, f"would write {self.path}: {change.summary}")
        self.assertEqual(hermes_config.reasoning_effort_warning(self.env),
                         f"warning: {self.path} sets agent.reasoning_effort low outside the llmctl block, "
                         "so Hermes sends it to every model instead of the server's default; "
                         'clear it with: hermes -p llmctl config set agent.reasoning_effort ""')

    def test_summary_names_a_different_old_recipe(self):
        old_target = target_for("nemotron-3.5-bolt", "llamacpp", RECIPES)
        self.write_profile(hermes_config.new_config_text(old_target))
        change = hermes_config.plan_profile(self.env, qwen36())
        self.assertEqual(change.summary,
                         "llmctl block for qwen3.6-35b-a3b on llama.cpp (was nemotron-3.5-bolt on llama.cpp)")

    def test_profile_block(self):
        self.assertIsNone(hermes_config.profile_block(self.env))
        self.write_profile(hermes_config.new_config_text(gptoss_vllm()).replace("\n", "\r\n"))
        self.assertEqual(hermes_config.profile_block(self.env), hermes_config.render_block(gptoss_vllm()))
        self.write_profile(hermes_config.HEADER + hermes_config.BEGIN + "\n")
        with self.assertRaises(ClientError):
            hermes_config.profile_block(self.env)

    def test_block_description(self):
        self.assertEqual(hermes_config.block_description(hermes_config.render_block(gptoss_vllm())),
                         "gpt-oss-20b on vLLM")
        self.assertIsNone(hermes_config.block_description("model: {}\n"))
        self.assertIsNone(hermes_config.block_description('providers: {"llmctl": {"name": \n'))
        self.assertIsNone(hermes_config.block_description('providers: {"other": {"name": "x"}}\n'))


class RefusalTest(HermesHomeCase):
    def refused(self, start, target=None):
        with self.assertRaises(ClientError) as ctx:
            hermes_config.plan_profile(self.env, target or qwen36())
        self.assertEqual(ctx.exception.exit_code, 3)
        self.assertTrue(str(ctx.exception).startswith(start), str(ctx.exception))
        return str(ctx.exception)

    def test_no_markers(self):
        self.write_profile("display:\n  compact: true\n")
        self.refused(f"{self.path} has no llmctl block, so the Hermes profile llmctl was not created by llmctl; "
                     "rename it with hermes profile rename llmctl <name>, or leave Hermes out with LLMCTL_CLIENTS")

    def test_byte_order_mark_before_the_begin_marker(self):
        text = "\ufeff" + hermes_config.BEGIN + "\n" + hermes_config.render_block(qwen36()) + hermes_config.END + "\n"
        self.write_profile(text + "_config_version: 53\n")
        self.assertIsNone(hermes_config.plan_profile(self.env, qwen36()))
        self.assertEqual(hermes_config.profile_block(self.env), hermes_config.render_block(qwen36()))
        change = hermes_config.plan_profile(self.env, gptoss_vllm())
        self.assertTrue(change.new_text.startswith("\ufeff" + hermes_config.BEGIN + "\n"))
        self.assertTrue(change.new_text.endswith(hermes_config.END + "\n_config_version: 53\n"))

    def test_damaged_markers(self):
        text = hermes_config.new_config_text(qwen36())
        begin, end = hermes_config.BEGIN + "\n", hermes_config.END + "\n"
        cases = (
            (text.replace(end, ""), "1 begin marker, 0 end markers"),
            (text.replace(begin, begin + begin), "2 begin markers, 1 end marker"),
            (text.replace(begin, "").replace(end, end + begin), "the end marker comes first"),
        )
        for damaged, found in cases:
            with self.subTest(found=found):
                self.write_profile(damaged)
                self.refused(f"{self.path}: the llmctl block markers are damaged ({found}); "
                             "restore the two marker lines exactly")

    def test_owned_key_outside_block(self):
        self.write_profile(hermes_config.new_config_text(qwen36()) + "model_overrides: {}\n")
        self.refused(f"{self.path} line 11: 'model_overrides' is set outside the llmctl block; llmctl owns it "
                     "in this profile and Hermes stops on a duplicate key; delete that setting")

    def test_unreadable_top_level_line(self):
        for line in ('"quoted": 1', "? complex", "---", "{a: 1}", "- item"):
            with self.subTest(line=line):
                self.write_profile(hermes_config.new_config_text(qwen36()) + line + "\n")
                self.refused(f"{self.path} line 11: llmctl cannot check this top-level line; "
                             "write top-level keys as plain 'key:'")

    def test_yaml_line_break_outside_block(self):
        for char in ("\r", "\x85", "\u2028", "\u2029"):
            with self.subTest(char=hex(ord(char))):
                self.write_profile(hermes_config.new_config_text(qwen36()) + f"display: x{char}model: {{}}\n")
                self.refused(f"{self.path} line 11: llmctl cannot check this top-level line; "
                             "write top-level keys as plain 'key:'")

    def test_tombstone(self):
        tombstone = self.root / "profiles" / ".deleted" / "llmctl"
        tombstone.parent.mkdir(parents=True)
        tombstone.write_text("")
        self.refused(f"Hermes marks the profile llmctl as deleted ({tombstone}); delete that file to let llmctl "
                     "recreate the profile, or leave Hermes out with LLMCTL_CLIENTS")

    def test_missing_root(self):
        self.root.rmdir()
        self.refused(f"{self.root} does not exist; llmctl creates a Hermes profile only inside an existing Hermes home")

    def test_context_below_minimum(self):
        target = qwen36()
        small = dataclasses.replace(target, recipe=dataclasses.replace(target.recipe, context=32768))
        message = self.refused("Hermes refuses a context below 64,000 tokens and qwen3.6-35b-a3b has 32768; "
                               "leave Hermes out with LLMCTL_CLIENTS=opencode,pi", small)
        self.assertIn("32768", message)


class CommandsTest(HermesHomeCase):
    def test_sticky_profile(self):
        sticky = self.root / "active_profile"
        self.assertEqual(hermes_config.sticky_profile(self.env), "default")
        sticky.write_text("llmctl\n")
        self.assertEqual(hermes_config.sticky_profile(self.env), "llmctl")
        sticky.write_text("")
        self.assertEqual(hermes_config.sticky_profile(self.env), "default")

    def test_use_profile_runs_the_command(self):
        sticky = self.root / "active_profile"
        sticky.write_text("work\n")
        host = FakeHost(env=self.env)

        def use(argv):
            sticky.write_text("llmctl\n")
            return ok()

        host.on(("hermes", "profile", "use", "llmctl"), use)
        self.assertEqual(hermes_config.use_profile(host), [
            "ran hermes profile use llmctl: plain hermes commands, hermes gateway and hermes config included, "
            "now use the profile llmctl; switch back with hermes profile use work",
            "Hermes default profile: work -> llmctl",
        ])
        self.assertEqual(host.calls, [["hermes", "profile", "use", "llmctl"]])

    def test_use_profile_failure(self):
        (self.root / "active_profile").write_text("work\n")
        host = FakeHost(env=self.env)
        host.on(("hermes", "profile", "use", "llmctl"), fail("boom\n"))
        with self.assertRaises(ClientWriteError) as ctx:
            hermes_config.use_profile(host)
        self.assertEqual(ctx.exception.exit_code, 6)
        self.assertEqual(str(ctx.exception), "hermes profile use llmctl exited 1: boom")
        host = FakeHost(env=self.env)
        host.on(("hermes", "profile", "use", "llmctl"), ok())
        with self.assertRaises(ClientWriteError) as ctx:
            hermes_config.use_profile(host)
        self.assertEqual(str(ctx.exception),
                         f"hermes profile use llmctl exited 0, but {self.root}/active_profile reads 'work'")

    def test_gateway_warning(self):
        path = self.root / "gateway_state.json"
        self.assertIsNone(hermes_config.gateway_warning(self.env))
        path.write_text(json.dumps({"gateway_state": "running", "served_profiles": ["default", "llmctl"]}))
        self.assertEqual(hermes_config.gateway_warning(self.env),
                         f"warning: Hermes' gateway serves the profile llmctl ({path}), so its sessions run inside "
                         "your messaging gateway; this Hermes may no longer honour gateway.standalone "
                         "(docs/clients.md#hermes-agent)")
        path.write_text(json.dumps({"gateway_state": "running", "served_profiles": ["default"]}))
        self.assertIsNone(hermes_config.gateway_warning(self.env))
        path.write_text(json.dumps({"gateway_state": "stopped", "served_profiles": ["default", "llmctl"]}))
        self.assertIsNone(hermes_config.gateway_warning(self.env))
        path.write_text("{not json")
        self.assertTrue(hermes_config.gateway_warning(self.env).startswith(f"note: cannot read {path}: "))
        path.write_text("[]")
        self.assertTrue(hermes_config.gateway_warning(self.env).startswith(f"note: cannot read {path}: "))

    def test_reasoning_effort_warning(self):
        base = hermes_config.new_config_text(qwen36())
        self.assertIsNone(hermes_config.reasoning_effort_warning(self.env))
        self.write_profile(base + "agent:\n  reasoning_effort: medium\n")
        self.assertEqual(hermes_config.reasoning_effort_warning(self.env),
                         f"warning: {self.path} sets agent.reasoning_effort medium outside the llmctl block, "
                         "so Hermes sends it to every model instead of the server's default; "
                         'clear it with: hermes -p llmctl config set agent.reasoning_effort ""')
        self.write_profile(base + "agent:  # tuned\n\n  # note\n  max_turns: 9\n  reasoning_effort: 'high'  # x\n")
        self.assertIn("agent.reasoning_effort high outside", hermes_config.reasoning_effort_warning(self.env))
        self.write_profile(base + 'agent:\n  reasoning_effort: ""\n')
        self.assertIsNone(hermes_config.reasoning_effort_warning(self.env))
        self.write_profile(base + "agent:\n  nested:\n    reasoning_effort: high\n")
        self.assertIsNone(hermes_config.reasoning_effort_warning(self.env))
        self.write_profile(base + "agent:\n  max_turns: 9\nother:\n  reasoning_effort: high\n")
        self.assertIsNone(hermes_config.reasoning_effort_warning(self.env))

    def test_reasoning_effort_warning_after_a_yaml_line_break(self):
        base = hermes_config.new_config_text(qwen36())
        for char in ("\r", "\x85", "\u2028", "\u2029"):
            with self.subTest(char=hex(ord(char))):
                self.write_profile(base + f"agent:{char}  reasoning_effort: high\n")
                warning = hermes_config.reasoning_effort_warning(self.env)
                self.assertIsNotNone(warning)
                self.assertIn("agent.reasoning_effort high outside", warning)

    def test_hermes_home_note(self):
        self.assertIsNone(hermes_config.hermes_home_note(self.env))
        env = {**self.env, "HERMES_HOME": f"{self.root}/profiles/work"}
        self.assertEqual(hermes_config.hermes_home_note(env),
                         "note: HERMES_HOME names the profile work in this shell, so plain hermes ignores "
                         "the default profile; use hermes -p llmctl")
        self.assertIsNone(hermes_config.hermes_home_note({**self.env, "HERMES_HOME": str(self.root)}))


if __name__ == "__main__":
    unittest.main()
