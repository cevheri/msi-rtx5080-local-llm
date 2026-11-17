import dataclasses
import fcntl
import json
import os
import unittest
from pathlib import Path
from unittest import mock

from launcher import clients, dsh_config, hermes_config, kimi_config, ops, pi_config
from launcher.catalog import resolve
from launcher.errors import ClientError, ClientWriteError, PreflightError, StartError, UsageError
from launcher.lock import hold_lock, lock_path
from launcher.runtimes import llamacpp, vllm
from launcher.tests.client_fixture import ClientFixture
from launcher.tests.fakes import REPO, fail, http, ok
from launcher.tests.ops_fixture import OpsFixture, managed_line
from launcher.tests.regen_golden import target_for
from launcher.tests.sglang_fixture import place_snapshot, server_info
from launcher.tests.test_kimi_config import OWNER_TEXT

NO_CLIENTS = ("clients: none (opencode: not installed, no opencode on PATH) (hermes: not installed, no hermes on PATH) "
              "(pi: not installed, no pi on PATH) (dsh: not installed, no dsh on PATH) (kimi: not installed, no kimi on PATH)")


class UpTest(unittest.TestCase):
    def setUp(self):
        self.f = OpsFixture()

    def tearDown(self):
        self.f.cleanup()

    def up(self, name="qwen3.6", runtime=None, replace=False, dry_run=False):
        return ops.up(self.f.host, self.f.catalog, name, runtime, replace=replace, dry_run=dry_run)

    def test_up_prints_ready_and_clients_line(self):
        self.assertEqual(self.up(), 0)
        runs = self.f.calls_starting("docker", "run")
        self.assertEqual(runs, [self.f.golden("llamacpp", "qwen3.6-35b-a3b")])
        lines = self.f.host.out.getvalue().splitlines()
        self.assertIn("ready: http://127.0.0.1:8080/v1, served as 'local', context 131072, output 16384", lines)
        self.assertIn(NO_CLIENTS, lines)
        out = self.f.host.out.getvalue()
        self.assertIn("122.5 tok/s", out)
        self.assertIn("./llmctl down", out)

    def test_up_replace_stops_other(self):
        self.f.managed_lines = [managed_line("vllm", "running", "gpt-oss-20b", runtime="vllm")]
        self.assertEqual(self.up(replace=True), 0)
        calls = self.f.host.calls
        stop = calls.index(["docker", "stop", "-t", "60", "vllm"])
        run = calls.index(self.f.golden("llamacpp", "qwen3.6-35b-a3b"))
        self.assertLess(stop, run)

    def test_up_removes_own_stopped_container(self):
        self.f.managed_lines = [managed_line("llamacpp", "exited", "qwen3.8-27b")]
        self.assertEqual(self.up(), 0)
        self.assertEqual(self.f.calls_starting("docker", "rm"), [["docker", "rm", "llamacpp"]])

    def test_container_exits_during_start(self):
        self.f.health = [http(503)]
        self.f.state = "exited"
        with self.assertRaises(StartError) as ctx:
            self.up()
        self.assertEqual(ctx.exception.exit_code, 4)
        self.assertIn("exited", str(ctx.exception))
        self.assertIn(["docker", "logs", "--tail", "40", "llamacpp"], self.f.host.calls)
        self.assertEqual(self.f.calls_starting("docker", "rm"), [])

    def test_ready_timeout(self):
        self.f.health = [http(503)]
        with self.assertRaises(StartError) as ctx:
            self.up()
        self.assertIn("not ready after 300 s", str(ctx.exception))
        self.assertIn("./llmctl logs", str(ctx.exception))
        self.assertEqual(self.f.calls_starting("docker", "stop"), [])

    def test_interrupt_during_wait(self):
        self.f.health = [http(503)]

        def interrupt(host):
            raise KeyboardInterrupt

        self.f.host.on_sleep = interrupt
        self.assertEqual(self.up(), 130)
        self.assertIn("./llmctl status", self.f.host.err.getvalue())
        with hold_lock(self.f.host.env):
            pass

    def test_dry_run_starts_nothing(self):
        self.assertEqual(self.up(dry_run=True), 0)
        self.assertEqual(self.f.calls_starting("docker", "run"), [])
        self.assertIn("docker run -d --name llamacpp", self.f.host.out.getvalue())

    def test_estimate_warns(self):
        # A copy marked as an estimate, so the test holds whatever the shipped file's evidence says.
        recipes = self.f.catalog.runtimes["vllm"].recipes
        recipe = recipes["gemma-4-12b"]
        recipes["gemma-4-12b"] = dataclasses.replace(recipe, evidence=dataclasses.replace(
            recipe.evidence, kind="estimate", decode_tok_s=None))
        self.f.place_files("gemma-4-12b", runtime="vllm")
        self.f.host.on_http("GET", "http://127.0.0.1:8000/v1/models",
                            http(200, b'{"data":[{"id":"gemma-4-12b"}]}'))
        self.assertEqual(self.up("gemma-4-12b", "vllm"), 0)
        self.assertIn("warning: this recipe has no measurement on the reference laptop; it may not start or fit\n",
                      self.f.host.out.getvalue())

    def test_chat_only_model_says_so(self):
        self.f.place_files("deepseek-r1-0528-qwen3-8b")
        self.assertEqual(self.up("deepseek-r1"), 0)
        out = self.f.host.out.getvalue()
        self.assertIn("note: chat only, no tool calling", out.splitlines())

    def running_qwen(self, sha=None):
        running_qwen(self.f, sha)

    def test_same_recipe_running_is_not_restarted(self):
        self.running_qwen()
        self.assertEqual(self.up(), 0)
        self.assertEqual(self.f.calls_starting("docker", "run"), [])
        self.assertEqual(self.f.calls_starting("docker", "stop"), [])
        out = self.f.host.out.getvalue()
        self.assertIn("already running", out)
        self.assertIn(NO_CLIENTS, out.splitlines())

    def test_changed_recipe_file_still_needs_replace(self):
        self.running_qwen(sha="ef" * 32)
        with self.assertRaises(PreflightError) as ctx:
            self.up()
        self.assertIn("--replace", str(ctx.exception))

    def test_lock_held_refuses(self):
        with hold_lock(self.f.host.env):
            with self.assertRaises(PreflightError):
                self.up()


def running_qwen(fixture, sha=None, name="qwen3.6"):
    recipe, runtime = resolve(fixture.catalog, name, None)
    fixture.managed_lines = [managed_line("llamacpp", "running", recipe.id, sha=sha or recipe.source.sha256,
                                          runtime_sha=runtime.source.sha256)]
    fixture.health = [http(200)]


SAMPLE = """\
qwen3.6-35b-a3b on llamacpp: shipped recipe, measured
clients: opencode, hermes, pi, dsh, kimi
waiting for qwen3.6-35b-a3b to load (up to 300 s)
ready: http://127.0.0.1:8080/v1, served as 'local', context 131072, output 16384
backed up /home/u/.pi/agent/settings.json to /home/u/.pi/agent/settings.json.llmctl-backup before the first change
wrote /home/u/.config/opencode/opencode.json: model llamacpp-docker/qwen3.6-35b-a3b (was llamacpp-docker/gemma-4-26b-a4b); OpenCode reloaded its configuration
wrote /home/u/.hermes/profiles/llmctl/config.yaml: llmctl block for qwen3.6-35b-a3b on llama.cpp (was gpt-oss-20b on vLLM)
wrote /home/u/.pi/agent/models.json: provider llmctl for qwen3.6-35b-a3b on llama.cpp, model local (was gpt-oss-20b on vLLM)
wrote /home/u/.pi/agent/settings.json: modelThinkingLevels llmctl/local off (was absent); defaultModel local (was gpt-oss-20b)
wrote /home/u/.dsh/cordis.patch.yml: llmctl rows for qwen3.6-35b-a3b on llama.cpp (was gpt-oss-20b on vLLM); dsh's default llmctl/local in every profile (was llmctl/gpt-oss-20b)
wrote /home/u/.kimi-code/config.toml: provider llmctl and model llmctl/local for qwen3.6-35b-a3b on llama.cpp (was gpt-oss-20b on vLLM); default_model llmctl/local (was llmctl/gpt-oss-20b)
OpenCode: llamacpp-docker/qwen3.6-35b-a3b, the default
  in the OpenCode screen: type /models (or press ctrl+x m) and pick it
  for a single prompt: opencode run --model llamacpp-docker/qwen3.6-35b-a3b "your prompt"
  thinking: off; add #thinking to the model to turn it on
Hermes: profile llmctl, not the default (./llmctl up qwen3.6-35b-a3b --set-default makes it one)
  hermes -p llmctl
  thinking: off, as the server sets it
Pi: llmctl/local, the default
  pi --model llmctl/local
  thinking: off; --thinking medium turns it on
dsh: llmctl/local, the default in every dsh profile
  dsh web, or for a single prompt: dsh --profile headless "your prompt"
  thinking: off; choose Medium under Effort in dsh's model picker to turn it on
Kimi: llmctl/local, the default
  kimi, or for a single prompt: kimi -p "your prompt"
  thinking: off, as the server sets it
note: Hermes and Pi sessions that were open before keep the model they started with; start a new session
note: dsh web uses the new rows within seconds, and its open sessions on llmctl/gpt-oss-20b fail until you pick llmctl/local; dsh headless, acp and sdk processes that are already running keep the old rows until they restart
note: an open Kimi session on llmctl/gpt-oss-20b fails its next turn until you pick llmctl/local with /model or start a new session
evidence: 122.5 tok/s decode with 128k context, 15,190 MiB GPU peak, OpenCode session passed (2026-11-16, research/2026-11-16/llamacpp-qwen3.6-35b-a3b/)
stop with ./llmctl down
"""

START_SETTINGS = {"lastChangelogVersion": "1.5.2", "defaultProvider": "llmctl", "defaultModel": "gpt-oss-20b"}
PI_LIST = ["pi", "--offline", "--list-models", "llmctl"]
DSH_DUMP = ["dsh", "--profile", "acp", "--dump-config"]
KIMI_LIST = ["kimi", "provider", "list", "--json"]
TELEMETRY = "- id: session-telemetry-otel\n  disabled: true\n"
# The bytes of dsh's profile template (dsh 0.2.2-rc.3), as the lead kept them in $SCRATCH/dsh/templates/.
PROFILE_TEMPLATE = ("# Your patch layer for this dsh profile, applied after every bundle layer:\n"
                    "# a top-level YAML array of loader patch entries (id-targeted config\n"
                    "# overrides, disables, and insert lists; `!!js` expressions allowed).\n"
                    "[]\n")


class UpClientsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qwen = target_for("qwen3.6-35b-a3b", "llamacpp", REPO / "recipes")
        cls.old_target = target_for("gpt-oss-20b", "vllm", REPO / "recipes")

    def setUp(self):
        self.f = OpsFixture()
        self.addCleanup(self.f.cleanup)
        self.c = ClientFixture(self.f.host, Path(self.f.tmp.name))
        self.c.install("opencode", "hermes", "pi", "dsh", "kimi")
        self.c.route_clients()
        self.c.write_opencode("llamacpp-docker/gemma-4-26b-a4b")
        Path(str(self.c.opencode_path) + ".llmctl-backup").write_text("{}\n", encoding="utf-8")
        self.c.hermes_config.parent.mkdir(parents=True)
        self.c.hermes_config.write_text(hermes_config.new_config_text(self.old_target), encoding="utf-8")
        self.c.models_path.write_text(pi_config.new_models_text(self.old_target), encoding="utf-8")
        self.c.settings_path.write_text(json.dumps(START_SETTINGS, indent=2) + "\n", encoding="utf-8")
        self.write_patch(TELEMETRY + dsh_config.new_home_text(self.old_target, default=True))
        Path(str(self.c.home_patch) + ".llmctl-backup").write_text(TELEMETRY, encoding="utf-8")
        self.kimi_start = kimi_config.config_text(self.c.kimi_file, OWNER_TEXT, self.old_target, set_default=True)
        self.write_kimi(self.kimi_start)
        Path(str(self.c.kimi_file) + ".llmctl-backup").write_text(OWNER_TEXT, encoding="utf-8")

    def write_kimi(self, text):
        with open(self.c.kimi_file, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)

    def write_patch(self, text):
        with open(self.c.home_patch, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)

    def up(self, name="qwen3.6", runtime=None, *, dry_run=False, set_default=False):
        return ops.up(self.f.host, self.f.catalog, name, runtime, replace=False, dry_run=dry_run,
                      set_default=set_default)

    def out(self):
        return self.f.host.out.getvalue()

    def opencode_model(self):
        return json.loads(self.c.opencode_path.read_text(encoding="utf-8"))["model"]

    def client_calls(self, calls=None):
        return [call for call in (self.f.host.calls if calls is None else calls)
                if call[0] in ("opencode", "hermes", "pi", "dsh", "kimi")]

    def test_down_prints_client_lines_and_changes_nothing(self):
        running_qwen(self.f)
        self.f.host.on(("docker", "stop"), ok())
        (self.c.hermes_root / "active_profile").write_text("llmctl\n", encoding="utf-8")
        before = self.c.snapshot()
        self.assertEqual(ops.down(self.f.host, self.f.catalog, remove=False), 0)
        self.assertEqual(self.out().splitlines(), [
            "stopped qwen3.6-35b-a3b in container 'llamacpp'",
            "OpenCode, Hermes, Pi, dsh and Kimi still point at qwen3.6-35b-a3b and get no answer until the next "
            "./llmctl up",
            "  switch back: OpenCode /models; hermes profile use default; Pi /model; dsh: delete the "
            f"agent-default-model and acp lines in {self.c.home_patch}; Kimi /model",
            "  Kimi retries a stopped server for about two and a half minutes before a prompt fails",
        ])
        self.assertEqual(self.c.snapshot(), before)
        self.assertEqual(self.client_calls(), [])

    def test_status_reports_the_running_model(self):
        self.assertEqual(self.up(set_default=True), 0)
        running_qwen(self.f)
        self.f.host.on(("docker", "port"), ok("8080/tcp -> 127.0.0.1:8080\n"))
        self.f.host.on(("docker", "stats"), ok("14.2GiB / 20GiB\n"))
        self.f.host.out.truncate(0)
        self.f.host.out.seek(0)
        self.assertEqual(ops.status(self.f.host, self.f.catalog), 0)
        lines = self.out().splitlines()
        oc = self.c.opencode_path
        self.assertIn(f"  OpenCode: default llamacpp-docker/qwen3.6-35b-a3b, the running model ({oc})", lines)
        self.assertFalse([line for line in lines if line.startswith("warning:")], lines)

    def test_status_with_unknown_recipe_running_does_not_warn(self):
        (self.c.hermes_root / "active_profile").write_text("llmctl\n", encoding="utf-8")
        self.f.managed_lines = [managed_line("llamacpp", "running", "my-override-gone", source="override")]
        self.f.host.on(("docker", "port"), ok("8080/tcp -> 127.0.0.1:8080\n"))
        self.f.host.on(("docker", "stats"), ok("14.2GiB / 20GiB\n"))
        self.assertEqual(ops.status(self.f.host, self.f.catalog), 0)
        out = self.out()
        self.assertIn("my-override-gone on llamacpp, container 'llamacpp', running", out.splitlines())
        self.assertIn("  OpenCode: default llamacpp-docker/gemma-4-26b-a4b", "\n".join(out.splitlines()))
        self.assertIn("  Hermes: profile llmctl for gpt-oss-20b on vLLM; default profile: llmctl", out)
        self.assertIn("default llmctl/gpt-oss-20b", out)
        self.assertNotIn("but no model is running", out)

    def test_up_lines_toggle_recipe(self):
        self.assertEqual(self.up(), 0)
        self.assertEqual(self.out(), SAMPLE.replace("/home/u/", str(self.c.home) + "/"))

    def test_up_lines_effort_recipe(self):
        self.f.place_files("qwen3.8-27b")
        self.assertEqual(self.up("qwen3.8-27b"), 0)
        thinking = [line for line in self.out().splitlines() if line.startswith("  thinking: ")]
        self.assertEqual(thinking, [
            "  thinking: medium effort; add #<level> for another (none, low, medium)",
            "  thinking: medium effort; hermes --reasoning <level> for another (none, low, medium)",
            "  thinking: medium effort; --thinking <level> for another (off, low, medium)",
            "  thinking: medium effort; choose another under Effort (off, low, medium)",
            "  thinking: medium effort; /effort in kimi for another (off, low, medium)",
        ])

    def test_vllm_hint_names_the_runtime(self):
        self.f.place_files("gpt-oss-20b", runtime="vllm")
        self.f.host.on_http("GET", "http://127.0.0.1:8000/v1/models",
                            http(200, b'{"data":[{"id":"gpt-oss-20b"}]}'))
        self.assertEqual(self.up("gpt-oss-20b", "vllm"), 0)
        self.assertIn("Hermes: profile llmctl, not the default "
                      "(./llmctl up gpt-oss-20b --runtime vllm --set-default makes it one)", self.out().splitlines())

    def test_set_default_points_every_client(self):
        self.assertEqual(self.up(set_default=True), 0)
        self.assertEqual(self.opencode_model(), "llamacpp-docker/qwen3.6-35b-a3b")
        settings = json.loads(self.c.settings_path.read_text(encoding="utf-8"))
        self.assertEqual((settings["defaultProvider"], settings["defaultModel"]), ("llmctl", "local"))
        self.assertEqual((self.c.hermes_root / "active_profile").read_text(encoding="utf-8").strip(), "llmctl")
        self.assertEqual(self.f.host.calls.count(["hermes", "profile", "use", "llmctl"]), 1)
        lines = self.out().splitlines()
        self.assertIn("Hermes default profile: default -> llmctl", lines)
        self.assertIn("Hermes: profile llmctl, the default", lines)

    def test_client_refusal_starts_nothing(self):
        self.c.hermes_config.write_text("foo: 1\n", encoding="utf-8")
        before = self.c.snapshot()
        with self.assertRaises(ClientError) as caught:
            self.up()
        self.assertEqual(caught.exception.exit_code, 3)
        self.assertEqual(self.f.calls_starting("docker", "run"), [])
        self.assertEqual(self.c.snapshot(), before)

    def test_no_client_writes_when_start_fails(self):
        self.f.health = [http(503)]
        self.f.state = "exited"
        before = self.c.snapshot()
        with self.assertRaises(StartError):
            self.up()
        self.assertEqual(self.c.snapshot(), before)
        calls = self.f.host.calls
        self.assertEqual(calls.count(PI_LIST), 1)
        self.assertEqual(calls.count(DSH_DUMP), 1)
        self.assertEqual(calls.count(KIMI_LIST), 1)
        self.assertEqual(self.client_calls(calls[calls.index(KIMI_LIST) + 1:]), [])

    def point_at_llmctl(self):
        (self.c.hermes_root / "active_profile").write_text("llmctl\n", encoding="utf-8")
        self.assertEqual(json.loads(self.c.settings_path.read_text(encoding="utf-8"))["defaultProvider"], "llmctl")
        self.assertTrue(self.opencode_model().startswith("llamacpp-docker/"))

    def calls_after_start(self):
        calls = self.f.host.calls
        return self.client_calls(calls[calls.index(self.f.calls_starting("docker", "run")[0]) + 1:])

    def test_no_client_writes_on_readiness_timeout(self):
        self.point_at_llmctl()
        self.f.health = [http(503)]
        before = self.c.snapshot()
        with self.assertRaises(StartError) as caught:
            self.up(set_default=True)
        self.assertEqual(caught.exception.exit_code, 4)
        self.assertIn("not ready after", str(caught.exception))
        self.assertEqual(self.c.snapshot(), before)
        self.assertEqual(self.calls_after_start(), [])

    def test_no_client_writes_on_interrupt(self):
        self.point_at_llmctl()
        self.f.health = [http(503)]

        def interrupt(host):
            raise KeyboardInterrupt

        self.f.host.on_sleep = interrupt
        before = self.c.snapshot()
        self.assertEqual(self.up(set_default=True), 130)
        self.assertEqual(self.c.snapshot(), before)
        self.assertEqual(self.calls_after_start(), [])

    def test_apply_runs_inside_the_lock(self):
        real_write = clients.write_atomic
        held = []

        def spy(path, text, **kwargs):
            if not held:
                with open(lock_path(self.f.host.env), "a+") as handle:
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        held.append(True)
                    else:
                        fcntl.flock(handle, fcntl.LOCK_UN)
                        held.append(False)
            return real_write(path, text, **kwargs)

        with mock.patch("launcher.clients.write_atomic", side_effect=spy):
            self.assertEqual(self.up(), 0)
        self.assertEqual(held, [True])

    def test_dry_run_writes_and_runs_nothing(self):
        before = self.c.snapshot()
        self.assertEqual(self.up(dry_run=True, set_default=True), 0)
        self.assertEqual(self.c.snapshot(), before)
        self.assertEqual(self.f.calls_starting("docker", "run"), [])
        self.assertEqual(self.client_calls(), [])
        lines = self.out().splitlines()
        self.assertIn("would set OpenCode's default model: llamacpp-docker/gemma-4-26b-a4b -> "
                      f"llamacpp-docker/qwen3.6-35b-a3b ({self.c.opencode_path})", lines)
        self.assertIn("would run hermes profile use llmctl", lines)
        self.assertEqual(lines[-1], "dry run: nothing was written, and none of Pi, dsh and Kimi has checked llmctl's "
                                    "entries")

    def test_client_write_failure_is_exit_6(self):
        lock = Path(os.path.realpath(self.c.settings_path) + ".lock")
        lock.mkdir()
        with self.assertRaises(ClientWriteError) as caught:
            self.up(set_default=True)
        self.assertEqual(caught.exception.exit_code, 6)
        self.assertTrue(str(caught.exception).startswith(
            f"the model is running, but 1 client file(s) were not updated: {self.c.settings_path}: "
            "Pi holds its settings lock"), str(caught.exception))
        self.assertEqual(self.opencode_model(), "llamacpp-docker/qwen3.6-35b-a3b")
        self.assertEqual(self.c.hermes_config.read_text(encoding="utf-8"), hermes_config.new_config_text(self.qwen))
        self.assertTrue(lock.is_dir())
        lines = self.out().splitlines()
        self.assertIn("stop with ./llmctl down", lines)
        self.assertIn("Pi: not updated, see the error below", lines)
        self.assertFalse([line for line in lines if line.startswith("Pi: llmctl/")], lines)
        self.assertFalse([line for line in lines if line.startswith("  pi --model")], lines)
        self.assertIn("OpenCode: llamacpp-docker/qwen3.6-35b-a3b, the default", lines)
        self.assertIn('  for a single prompt: opencode run --model llamacpp-docker/qwen3.6-35b-a3b "your prompt"',
                      lines)
        self.assertIn("Hermes: profile llmctl, the default", lines)
        self.assertIn("  hermes -p llmctl", lines)
        self.assertIn("note: Hermes sessions that were open before keep the model they started with; "
                      "start a new session", lines)
        self.assertFalse([line for line in lines if "Pi sessions" in line], lines)

    def test_hermes_use_failure_keeps_the_hermes_block(self):
        self.f.host.on(["hermes", "profile", "use", "llmctl"], fail("boom"))
        with self.assertRaises(ClientWriteError) as caught:
            self.up(set_default=True)
        self.assertEqual(caught.exception.exit_code, 6)
        self.assertEqual(str(caught.exception), "the model is running, but 1 client file(s) were not updated: "
                                                "hermes profile use llmctl exited 1: boom")
        self.assertEqual(self.c.hermes_config.read_text(encoding="utf-8"), hermes_config.new_config_text(self.qwen))
        lines = self.out().splitlines()
        self.assertIn("Hermes: profile llmctl, not the default (hermes profile use llmctl failed, see the error below)",
                      lines)
        self.assertIn("  hermes -p llmctl", lines)
        self.assertNotIn("Hermes: not updated, see the error below", lines)
        self.assertIn("note: Hermes and Pi sessions that were open before keep the model they started with; "
                      "start a new session", lines)

    def test_listed_client_not_installed_refuses_up(self):
        del self.f.host.installed["pi"]
        self.f.host.env["LLMCTL_CLIENTS"] = "pi"
        before = self.c.snapshot()
        with self.assertRaises(ClientError) as caught:
            self.up()
        self.assertEqual(caught.exception.exit_code, 3)
        self.assertEqual(str(caught.exception), "LLMCTL_CLIENTS names pi, but Pi is not installed here: no pi on PATH")
        self.assertEqual([call for call in self.f.host.calls if call[0] == "docker"], [])
        self.assertEqual(self.client_calls(), [])
        self.assertEqual(self.c.snapshot(), before)

    def test_down_stops_before_a_bad_llmctl_clients(self):
        running_qwen(self.f)
        self.f.host.on(("docker", "stop"), ok())
        self.f.host.env["LLMCTL_CLIENTS"] = "opencode,hermse"
        before = self.c.snapshot()
        with self.assertRaises(UsageError) as caught:
            ops.down(self.f.host, self.f.catalog, remove=False)
        self.assertEqual(caught.exception.exit_code, 1)
        self.assertIn("unknown client 'hermse'", str(caught.exception))
        self.assertEqual(self.f.calls_starting("docker", "stop"), [["docker", "stop", "-t", "60", "llamacpp"]])
        self.assertEqual(self.out().splitlines(), ["stopped qwen3.6-35b-a3b in container 'llamacpp'"])
        self.assertEqual(self.c.snapshot(), before)
        with hold_lock(self.f.host.env):
            pass

    def test_recipe_without_entry_is_not_an_error(self):
        recipes = self.f.catalog.runtimes["llamacpp"].recipes
        recipes["deepseek-r1-0528-qwen3-8b"] = dataclasses.replace(recipes["deepseek-r1-0528-qwen3-8b"], opencode=False)
        self.f.place_files("deepseek-r1-0528-qwen3-8b")
        self.assertEqual(self.up("deepseek-r1-0528-qwen3-8b", set_default=True), 0)
        lines = self.out().splitlines()
        self.assertIn("OpenCode: deepseek-r1-0528-qwen3-8b has no OpenCode entry in docs/opencode.md; its default "
                      "stays llamacpp-docker/gemma-4-26b-a4b", lines)
        self.assertIn("  chat only: this profile has no tools", lines)
        self.assertIn("  pi --model llmctl/local --no-tools", lines)

    def test_same_recipe_running_still_applies_clients(self):
        running_qwen(self.f)
        self.assertEqual(self.up(), 0)
        self.assertEqual(self.f.calls_starting("docker", "run"), [])
        self.assertEqual(self.f.calls_starting("docker", "stop"), [])
        self.assertIn("already running", self.out())
        self.assertEqual(self.opencode_model(), "llamacpp-docker/qwen3.6-35b-a3b")

    def test_unreadable_sticky_profile_still_reports(self):
        (self.c.hermes_root / "active_profile").mkdir()
        self.assertEqual(self.up(), 0)
        lines = self.out().splitlines()
        hermes = [line for line in lines if line.startswith("Hermes: ")]
        self.assertEqual(len(hermes), 1)
        self.assertTrue(hermes[0].startswith(
            f"Hermes: profile llmctl, default unknown (cannot read {self.c.hermes_root / 'active_profile'}: "),
            hermes[0])
        self.assertEqual(lines[-1], "stop with ./llmctl down")

    def thinking_lines(self, target):
        lines = clients.up_lines(self.f.host, target, clients.client_states(self.f.host),
                                 up_command=ops.up_command(target.recipe), changed=frozenset())
        return lines, [line for line in lines if line.startswith("  thinking: ")]

    def test_thinking_lines_per_type(self):
        self.c.write_opencode("llamacpp-docker/qwen3.6-35b-a3b")
        always = "  thinking: always on, at the effort the server sets"
        dsh_on = "  thinking: on; choose Off under Effort to turn it off"
        kimi_on = "  thinking: on, as the server sets it"
        qwen_on = dataclasses.replace(self.qwen, recipe=dataclasses.replace(
            self.qwen.recipe, clients=dataclasses.replace(self.qwen.recipe.clients, default="on")))
        cases = [
            (target_for("mellum2.1-12b", "vllm", REPO / "recipes"),
             ["  thinking: on; add #none to turn it off", "  thinking: on, as the server sets it",
              "  thinking: on; --thinking off turns it off", dsh_on, kimi_on]),
            (target_for("nemotron-3.5-bolt", "llamacpp", REPO / "recipes"),
             ["  thinking: off; add #thinking to the model to turn it on, #medium for medium effort",
              "  thinking: off, as the server sets it",
              "  thinking: off; --thinking medium for medium effort, --thinking high for full thinking",
              "  thinking: off; Medium under Effort for medium effort, High for full thinking",
              "  thinking: off, as the server sets it"]),
            (target_for("k2-horizon-7b", "llamacpp", REPO / "recipes"), [always] * 5),
            (target_for("qwen3-coder-next", "llamacpp", REPO / "recipes"), ["  thinking: none"] * 5),
            (target_for("deepseek-r1-0528-qwen3-8b", "llamacpp", REPO / "recipes"), [always] * 5),
            (qwen_on, ["  thinking: on; add #none to turn it off", "  thinking: on, as the server sets it",
                       "  thinking: on; --thinking off turns it off", dsh_on, kimi_on]),
        ]
        for target, expected in cases:
            with self.subTest(f"{target.recipe.runtime} {target.recipe.id} {target.recipe.clients.default}"):
                lines, thinking = self.thinking_lines(target)
                self.assertEqual(thinking, expected)
                if target.recipe.chat_only:
                    for line in ("  chat only: this profile has no tools", "  pi --model llmctl/local --no-tools",
                                 "  chat only: Pi has no per-model tool switch, so pass --no-tools"):
                        self.assertIn(line, lines)

    def test_hermes_warnings_follow_the_hermes_block(self):
        text = hermes_config.new_config_text(self.qwen) + "agent:\n  reasoning_effort: high\n"
        self.c.hermes_config.write_text(text, encoding="utf-8")
        state = self.c.hermes_root / "gateway_state.json"
        state.write_text(json.dumps({"gateway_state": "running", "served_profiles": ["default", "llmctl"]}),
                         encoding="utf-8")
        self.f.host.env["HERMES_HOME"] = str(self.c.hermes_root / "profiles" / "work")
        lines = clients.up_lines(self.f.host, self.qwen, clients.client_states(self.f.host),
                                 up_command=ops.up_command(self.qwen.recipe), changed=frozenset())
        index = lines.index("  thinking: off, as the server sets it")
        self.assertEqual(lines[index + 1:index + 4], [
            hermes_config.gateway_warning(self.f.host.env),
            f"warning: {self.c.hermes_config} sets agent.reasoning_effort high outside the llmctl block, so Hermes "
            "sends it to every model instead of the server's default; clear it with: "
            'hermes -p llmctl config set agent.reasoning_effort ""',
            "note: HERMES_HOME names the profile work in this shell, so plain hermes ignores the default profile; "
            "use hermes -p llmctl",
        ])
        self.assertTrue(lines[index + 1].startswith("warning: Hermes' gateway serves the profile llmctl"))
        self.assertTrue(lines[index + 4].startswith("Pi: "))

    def dsh_lines(self, lines, first):
        index = lines.index(first)
        return lines[index:]

    def test_status_shows_the_dsh_line(self):
        self.assertEqual(self.up(set_default=True), 0)
        running_qwen(self.f)
        self.f.host.on(("docker", "port"), ok("8080/tcp -> 127.0.0.1:8080\n"))
        self.f.host.on(("docker", "stats"), ok("14.2GiB / 20GiB\n"))
        self.f.host.out.truncate(0)
        self.f.host.out.seek(0)
        self.assertEqual(ops.status(self.f.host, self.f.catalog), 0)
        self.assertIn("  dsh: llmctl/local for qwen3.6-35b-a3b on llama.cpp, the running model; default llmctl/local, "
                      f"held by llmctl ({self.c.home_patch})", self.out().splitlines())

    def test_up_lines_dsh_not_held_and_compaction(self):
        self.write_patch(TELEMETRY + dsh_config.new_home_text(self.old_target, default=False))
        self.f.place_files("qwen3.8-27b")
        self.assertEqual(self.up("qwen3.8-27b"), 0)
        lines = self.out().splitlines()
        first = "dsh: llmctl/local, not the default (./llmctl up qwen3.8-27b --set-default makes it one)"
        self.assertEqual(self.dsh_lines(lines, first)[:4], [
            first,
            "  dsh web, then pick qwen3.8-27b on llama.cpp (llmctl) in its model picker",
            "  thinking: medium effort; choose another under Effort (off, low, medium)",
            "  compaction: dsh compacts these sessions only after the server rejects an overlong prompt, because "
            "context 65536 minus output 16384 leaves no room for its 65,536-token headroom "
            "(docs/clients.md#context-compaction-and-timeouts)",
        ])

    def test_up_lines_dsh_default_unknown(self):
        # the home patch became unreadable after llmctl wrote it: the first dsh line says why llmctl cannot tell
        self.assertEqual(self.up(), 0)
        self.write_patch(dsh_config.BEGIN + "\n")
        with self.assertRaises(ClientError) as caught:
            dsh_config.holds_default(self.f.host.env)
        lines = clients.up_lines(self.f.host, self.qwen, clients.client_states(self.f.host),
                                 up_command=ops.up_command(self.qwen.recipe), changed=frozenset({"dsh"}))
        first = f"dsh: llmctl/local, default unknown ({caught.exception})"
        self.assertEqual(self.dsh_lines(lines, first)[1],
                         "  dsh web, then pick qwen3.6-35b-a3b on llama.cpp (llmctl) in its model picker")

    def test_dsh_notes_on_first_adoption(self):
        self.write_patch(TELEMETRY)
        self.assertEqual(self.up(set_default=True), 0)
        lines = self.out().splitlines()
        index = lines.index("note: Hermes and Pi sessions that were open before keep the model they started with; "
                            "start a new session")
        path = self.c.home_patch
        self.assertEqual(lines[index + 1:index + 4], [
            "note: dsh web uses the new rows within seconds; dsh headless, acp and sdk processes that are already "
            "running keep the old rows until they restart",
            "note: llmctl now holds dsh's default, so dsh's model picker changes only the open session and cannot save "
            "another default (dsh refuses that save without a message); delete the agent-default-model and acp lines "
            f"in {path} to give it back",
            f"note: while llmctl's llm-pi-ai row is in {path}, dsh's Models page cannot add or edit OpenAI-compatible "
            'providers (it says Configuration for "llm-pi-ai" is overridden by a home patch or command-line overlay); '
            "leave dsh out with LLMCTL_CLIENTS if you need them (docs/clients.md#the-home-patch)",
        ])

    def test_profile_default_note(self):
        self.write_patch(TELEMETRY + dsh_config.new_home_text(self.old_target, default=False))
        web = self.c.dsh_home / "profiles" / "web" / "cordis.patch.yml"
        web.parent.mkdir()
        web.write_text("- id: agent-default-model\n  config:\n    provider: llmctl\n    model: gpt-oss-20b\n",
                       encoding="utf-8")
        self.assertEqual(self.up(), 0)
        lines = self.out().splitlines()
        self.assertIn(f"note: {web} line 1 sets dsh's default model for the web profile, which llmctl does not change; "
                      "if it names llmctl/gpt-oss-20b, dsh asks for a model again (./llmctl up qwen3.6-35b-a3b "
                      "--set-default lets llmctl hold dsh's default)", lines)
        self.assertIn("note: dsh web uses the new rows within seconds, and its open sessions on llmctl/gpt-oss-20b "
                      "fail until you pick llmctl/local; dsh headless, acp and sdk processes that are already running "
                      "keep the old rows until they restart", lines)

    def test_chat_only_lines(self):
        self.f.place_files("deepseek-r1-0528-qwen3-8b")
        self.assertEqual(self.up("deepseek-r1-0528-qwen3-8b"), 0)
        lines = self.out().splitlines()
        self.assertIn("  chat only: dsh still sends its tools; for a prompt without them use dsh --profile headless "
                      "--patch <file> with the rows in docs/clients.md#chat-only-recipes", lines)
        self.assertIn("  compaction: dsh compacts these sessions only after the server rejects an overlong prompt, "
                      "because context 65536 minus output 16384 leaves no room for its 65,536-token headroom "
                      "(docs/clients.md#context-compaction-and-timeouts)", lines)
        target = target_for("deepseek-r1-0528-qwen3-8b", "vllm", REPO / "recipes")
        lines = clients.up_lines(self.f.host, target, clients.client_states(self.f.host),
                                 up_command=ops.up_command(target.recipe), changed=frozenset())
        self.assertIn("  chat only: vLLM rejects every dsh request that carries tools for this recipe (HTTP 400); use "
                      "dsh --profile headless --patch <file> with the rows in docs/clients.md#chat-only-recipes", lines)

    def test_dsh_refusal_starts_nothing(self):
        self.write_patch("[]\n")
        before = self.c.snapshot()
        with self.assertRaises(ClientError) as caught:
            self.up()
        self.assertEqual(caught.exception.exit_code, 3)
        self.assertEqual(self.f.calls_starting("docker", "run"), [])
        self.assertEqual([call for call in self.f.host.calls if call[0] == "dsh"], [])
        self.assertEqual(self.c.snapshot(), before)

    def test_dsh_write_failure_is_exit_6(self):
        os.chmod(self.c.dsh_home, 0o500)
        self.addCleanup(os.chmod, self.c.dsh_home, 0o700)
        with self.assertRaises(ClientWriteError) as caught:
            self.up()
        self.assertEqual(caught.exception.exit_code, 6)
        message = str(caught.exception)
        self.assertTrue(message.startswith(
            f"the model is running, but 1 client file(s) were not updated: {self.c.home_patch}: "), message)
        lines = self.out().splitlines()
        self.assertIn("dsh: not updated, see the error below", lines)
        self.assertFalse([line for line in lines if line.startswith("dsh: llmctl/")], lines)
        self.assertEqual(self.opencode_model(), "llamacpp-docker/qwen3.6-35b-a3b")
        self.assertEqual(self.c.hermes_config.read_text(encoding="utf-8"), hermes_config.new_config_text(self.qwen))
        self.assertEqual(self.c.models_path.read_text(encoding="utf-8"), pi_config.new_models_text(self.qwen))
        self.assertEqual(json.loads(self.c.settings_path.read_text(encoding="utf-8"))["defaultModel"], "local")

    def test_version_warning_follows_the_clients_line(self):
        self.f.host.on(["dsh", "--version"], ok("0.2.3\n"))
        self.assertEqual(self.up(), 0)
        lines = self.out().splitlines()
        index = lines.index("clients: opencode, hermes, pi, dsh, kimi")
        self.assertEqual(lines[index + 1],
                         "warning: dsh 0.2.3 is not 0.2.2-rc.3, the version llmctl's dsh rows were checked with; run "
                         "the dsh steps of the e2e before relying on them (docs/clients.md#maintenance)")

    def test_l2_migration_command_touches_only_dsh(self):
        self.f.place_files("gemma-4-26b-a4b")
        running_qwen(self.f, name="gemma-4-26b-a4b")
        self.f.host.env["LLMCTL_CLIENTS"] = "dsh"
        path = self.c.home_patch
        backup = Path(str(path) + ".llmctl-backup")
        backup.unlink()
        self.write_patch(TELEMETRY)
        for profile in ("web", "headless"):
            folder = self.c.dsh_home / "profiles" / profile
            folder.mkdir()
            (folder / "cordis.patch.yml").write_text(PROFILE_TEMPLATE, encoding="utf-8")
        before = self.c.snapshot()
        summary = ("added the llmctl rows for gemma-4-26b-a4b on llama.cpp after your rows; dsh's default llmctl/local "
                   "in every profile (was not held by llmctl)")
        self.assertEqual(self.up("gemma-4-26b-a4b", dry_run=True, set_default=True), 0)
        lines = self.out().splitlines()
        self.assertIn("clients: dsh (opencode: left out by LLMCTL_CLIENTS) (hermes: left out by LLMCTL_CLIENTS) "
                      "(pi: left out by LLMCTL_CLIENTS) (kimi: left out by LLMCTL_CLIENTS)", lines)
        self.assertEqual(lines[-3:], [f"would back up {path} to {backup}", f"would write {path}: {summary}",
                                      "dry run: nothing was written, and dsh has not checked llmctl's rows"])
        self.assertEqual(self.client_calls(), [])
        self.assertEqual(self.c.snapshot(), before)
        self.f.host.out.truncate(0)
        self.f.host.out.seek(0)
        self.assertEqual(self.up("gemma-4-26b-a4b", set_default=True), 0)
        lines = self.out().splitlines()
        self.assertIn("gemma-4-26b-a4b is already running from this recipe file; nothing to start", lines)
        self.assertEqual(self.f.calls_starting("docker", "run"), [])
        self.assertEqual(self.f.calls_starting("docker", "stop"), [])
        self.assertEqual(self.client_calls(), [["dsh", "--version"], DSH_DUMP])
        for line in (f"backed up {path} to {backup} before the first change", f"wrote {path}: {summary}",
                     "dsh: llmctl/local, the default in every dsh profile",
                     '  dsh web, or for a single prompt: dsh --profile headless "your prompt"',
                     "note: dsh web uses the new rows within seconds; dsh headless, acp and sdk processes that are "
                     "already running keep the old rows until they restart"):
            self.assertIn(line, lines)
        self.assertTrue([line for line in lines if line.startswith("note: llmctl now holds dsh's default")], lines)
        self.assertTrue([line for line in lines if line.startswith("note: while llmctl's llm-pi-ai row is in")], lines)
        golden = (REPO / "launcher/tests/golden/clients/llamacpp/gemma-4-26b-a4b.dsh.yml").read_text(encoding="utf-8")
        owner_example = TELEMETRY + dsh_config.HEADER + dsh_config.BEGIN + "\n" + golden + dsh_config.END + "\n"
        with open(path, encoding="utf-8", newline="") as handle:
            self.assertEqual(handle.read(), owner_example)
        self.assertEqual(backup.read_bytes(), TELEMETRY.encode("utf-8"))
        after = self.c.snapshot()
        outside = {key: value for key, value in before.items() if not key.startswith(".dsh")}
        self.assertEqual({key: value for key, value in after.items() if not key.startswith(".dsh")}, outside)

    def kimi_lines(self, lines, first):
        index = lines.index(first)
        return lines[index:]

    def test_status_shows_the_kimi_line(self):
        self.assertEqual(self.up(set_default=True), 0)
        running_qwen(self.f)
        self.f.host.on(("docker", "port"), ok("8080/tcp -> 127.0.0.1:8080\n"))
        self.f.host.on(("docker", "stats"), ok("14.2GiB / 20GiB\n"))
        self.f.host.out.truncate(0)
        self.f.host.out.seek(0)
        calls = len(self.f.host.calls)
        self.assertEqual(ops.status(self.f.host, self.f.catalog), 0)
        self.assertIn("  Kimi: llmctl/local for qwen3.6-35b-a3b on llama.cpp, the running model; default llmctl/local "
                      f"({self.c.kimi_file})", self.out().splitlines())
        self.assertEqual(self.client_calls(self.f.host.calls[calls:]), [])

    def test_up_lines_kimi_not_default_and_effort(self):
        other = '\n[models."other/m"]\nprovider = "other"\nmodel = "m"\nmax_context_size = 131072\n'
        self.write_kimi(kimi_config.default_text(self.c.kimi_file, self.kimi_start, "other/m") + other)
        self.f.place_files("qwen3.8-27b")
        self.assertEqual(self.up("qwen3.8-27b"), 0)
        lines = self.out().splitlines()
        first = "Kimi: llmctl/local, not the default (./llmctl up qwen3.8-27b --set-default makes it one)"
        self.assertEqual(self.kimi_lines(lines, first)[:4], [
            first,
            "  kimi -m llmctl/local, or pick qwen3.8-27b on llama.cpp (llmctl) with /model in kimi",
            "  kimi -p needs Kimi's default even with -m, so for a single prompt run ./llmctl up qwen3.8-27b "
            "--set-default first",
            "  thinking: medium effort; /effort in kimi for another (off, low, medium)",
        ])
        self.assertEqual(kimi_config.read_state(self.f.host.env).default, "other/m")

    def test_up_lines_kimi_default_unknown(self):
        # config.toml became unreadable after llmctl wrote it: the first Kimi line says why llmctl cannot tell
        self.assertEqual(self.up(), 0)
        self.write_kimi(self.c.kimi_file.read_text(encoding="utf-8") + "[broken\n")
        with self.assertRaises(ClientError) as caught:
            kimi_config.read_state(self.f.host.env)
        lines = clients.up_lines(self.f.host, self.qwen, clients.client_states(self.f.host),
                                 up_command=ops.up_command(self.qwen.recipe), changed=frozenset({"kimi"}))
        first = f"Kimi: llmctl/local, default unknown ({caught.exception})"
        self.assertEqual(self.kimi_lines(lines, first)[1],
                         "  kimi -m llmctl/local, or pick qwen3.6-35b-a3b on llama.cpp (llmctl) with /model in kimi")

    def test_kimi_chat_only_lines(self):
        self.f.place_files("deepseek-r1-0528-qwen3-8b")
        self.assertEqual(self.up("deepseek-r1-0528-qwen3-8b"), 0)
        lines = self.out().splitlines()
        self.assertIn("  chat only: Kimi sends its tools with every request, also in its TUI; for a prompt without "
                      f'them: kimi -p "your prompt" --agent-file {kimi_config.AGENT_FILE}', lines)
        target = target_for("deepseek-r1-0528-qwen3-8b", "vllm", REPO / "recipes")
        lines = clients.up_lines(self.f.host, target, clients.client_states(self.f.host),
                                 up_command=ops.up_command(target.recipe), changed=frozenset())
        self.assertIn("  chat only: vLLM rejects every Kimi request that carries tools for this recipe (HTTP 400), and "
                      "Kimi's TUI always sends them; for a prompt without them, make it the default (./llmctl up "
                      "deepseek-r1-0528-qwen3-8b --runtime vllm --set-default), then kimi -p \"your prompt\" "
                      f"--agent-file {kimi_config.AGENT_FILE}", lines)

    def test_kimi_thinking_warning_follows_the_kimi_lines(self):
        self.write_kimi(self.kimi_start + "\n[thinking]\nenabled = false\n")
        self.f.place_files("gpt-oss-20b", runtime="vllm")
        self.f.host.on_http("GET", "http://127.0.0.1:8000/v1/models",
                            http(200, b'{"data":[{"id":"gpt-oss-20b"}]}'))
        self.assertEqual(self.up("gpt-oss-20b", "vllm"), 0)
        lines = self.out().splitlines()
        self.assertFalse([line for line in lines if line.startswith(f"wrote {self.c.kimi_file}")], lines)
        self.assertEqual(self.kimi_lines(lines, "Kimi: llmctl/gpt-oss-20b, the default")[2:4], [
            "  thinking: medium effort; /effort in kimi for another (low, medium, high)",
            f"warning: [thinking] enabled = false in {self.c.kimi_file} makes Kimi refuse every prompt to "
            "llmctl/gpt-oss-20b before sending it, because the model reasons by default and declares no off effort; "
            "set enabled = true there, or choose an effort with /effort in kimi",
        ])

    def test_kimi_model_name_note(self):
        self.f.host.env["KIMI_MODEL_NAME"] = "other/m"
        self.assertEqual(self.up(), 0)
        lines = self.out().splitlines()
        self.assertEqual(self.kimi_lines(lines, "Kimi: llmctl/local, the default")[3],
                         "note: KIMI_MODEL_NAME is set in this shell, so kimi uses that model instead of Kimi's "
                         "default_model; unset it, or pass -m llmctl/local")

    def test_kimi_refusal_starts_nothing(self):
        self.write_kimi(self.kimi_start + "[broken\n")
        before = self.c.snapshot()
        with self.assertRaises(ClientError) as caught:
            self.up()
        self.assertEqual(caught.exception.exit_code, 3)
        self.assertEqual(self.f.calls_starting("docker", "run"), [])
        self.assertEqual([call for call in self.f.host.calls if call[0] == "kimi"], [])
        self.assertEqual(self.c.snapshot(), before)

    def test_kimi_write_failure_is_exit_6(self):
        os.chmod(self.c.kimi_home, 0o500)
        self.addCleanup(os.chmod, self.c.kimi_home, 0o700)
        with self.assertRaises(ClientWriteError) as caught:
            self.up()
        self.assertEqual(caught.exception.exit_code, 6)
        message = str(caught.exception)
        self.assertTrue(message.startswith(
            f"the model is running, but 1 client file(s) were not updated: {self.c.kimi_file}: "), message)
        lines = self.out().splitlines()
        self.assertIn("Kimi: not updated, see the error below", lines)
        self.assertFalse([line for line in lines if line.startswith("Kimi: llmctl/")], lines)
        self.assertFalse([line for line in lines if line.startswith("note: an open Kimi session")], lines)
        self.assertEqual(self.c.kimi_file.read_text(encoding="utf-8"), self.kimi_start)
        self.assertEqual(self.opencode_model(), "llamacpp-docker/qwen3.6-35b-a3b")
        self.assertEqual(self.c.hermes_config.read_text(encoding="utf-8"), hermes_config.new_config_text(self.qwen))
        self.assertEqual(self.c.models_path.read_text(encoding="utf-8"), pi_config.new_models_text(self.qwen))
        self.assertEqual(self.c.home_patch.read_text(encoding="utf-8"),
                         TELEMETRY + dsh_config.new_home_text(self.qwen, default=True))

    def test_down_reports_a_relative_kimi_home_after_the_stop(self):
        running_qwen(self.f)
        self.f.host.on(("docker", "stop"), ok())
        self.f.host.env["KIMI_CODE_HOME"] = "rel"
        before = self.c.snapshot()
        with self.assertRaises(UsageError) as caught:
            ops.down(self.f.host, self.f.catalog, remove=False)
        self.assertEqual(caught.exception.exit_code, 1)
        self.assertTrue(str(caught.exception).startswith("KIMI_CODE_HOME is relative (rel)"), str(caught.exception))
        self.assertEqual(self.f.calls_starting("docker", "stop"), [["docker", "stop", "-t", "60", "llamacpp"]])
        self.assertEqual(self.out().splitlines(), ["stopped qwen3.6-35b-a3b in container 'llamacpp'"])
        self.assertEqual(self.c.snapshot(), before)


SGLANG_UP = "./llmctl up qwen3.5-9b --runtime sglang"
STOPPED_SUFFIX = "; llmctl stopped the container and kept it for its logs: ./llmctl logs, ./llmctl down --remove"


class SglangUpTest(unittest.TestCase):
    def setUp(self):
        self.f = OpsFixture()
        self.addCleanup(self.f.cleanup)
        self.f.use_sglang()
        self.overlay = self.f.cache_dir / "sglang" / "overlays" / "qwen3.5-9b"

    def up(self, replace=False, dry_run=False, fixture=None):
        f = fixture or self.f
        return ops.up(f.host, f.catalog, "qwen3.5-9b", "sglang", replace=replace, dry_run=dry_run)

    def lines(self, fixture=None):
        return (fixture or self.f).host.out.getvalue().splitlines()

    def test_up_builds_the_overlay_then_runs_then_checks(self):
        seen = []
        self.f.host.on(("docker", "run"), lambda argv: seen.append(
            (self.overlay / "generation_config.json").is_file()) or ok("0123456789ab\n"))
        self.assertEqual(self.up(), 0)
        self.assertEqual(seen, [True])
        self.assertEqual(self.lines()[:6], [
            "qwen3.5-9b on sglang: shipped recipe, measured",
            NO_CLIENTS,
            f"wrote {self.overlay}: model overlay with 2 links into snapshot a4df85a20fb7 of "
            "RedHatAI/Qwen3.5-9B-NVFP4, and generation_config.json with temperature 0.6, top_p 0.95, top_k 20",
            "waiting for qwen3.5-9b to load (up to 900 s)",
            "memory: KV pool 140976 tokens for context 131072; 1013 MiB of GPU memory free after the start, "
            "at least 640 MiB needed",
            "ready: http://127.0.0.1:30000/v1, served as 'qwen3.5-9b', context 131072, output 16384",
        ])
        [run] = self.f.calls_starting("docker", "run")
        mount = run.index(f"{self.overlay}:/llmctl/model:ro")
        self.assertEqual(run[mount - 1], "-v")

    def test_refusal_stops_the_container_and_writes_no_client(self):
        self.f.host.on_http("GET", "http://127.0.0.1:30000/server_info", server_info(120000))
        with mock.patch.object(ops.clients, "apply", side_effect=AssertionError("no client write after a refusal")):
            with self.assertRaises(StartError) as ctx:
                self.up()
        self.assertEqual(ctx.exception.exit_code, 4)
        self.assertTrue(str(ctx.exception).startswith("the KV pool holds 120000 tokens, fewer than the context 131072"))
        self.assertTrue(str(ctx.exception).endswith(STOPPED_SUFFIX), str(ctx.exception))
        self.assertIn(["docker", "stop", "-t", "60", "sglang"], self.f.host.calls)
        self.assertIn("memory: KV pool 120000 tokens for context 131072; 1013 MiB of GPU memory free after the start, "
                      "at least 640 MiB needed", self.lines())
        self.assertFalse(any(line.startswith("ready:") for line in self.lines()))

    def test_refusal_when_the_stop_fails(self):
        self.f.host.on_http("GET", "http://127.0.0.1:30000/server_info", server_info(120000))
        self.f.host.on(("docker", "stop"), fail("Error response from daemon: cannot stop"))
        with self.assertRaises(StartError) as ctx:
            self.up()
        self.assertEqual(ctx.exception.exit_code, 4)
        self.assertTrue(str(ctx.exception).endswith(
            "; llmctl could not stop the container (could not stop sglang: Error response from daemon: cannot stop), "
            "so stop it with ./llmctl down"), str(ctx.exception))

    def test_memory_taken_after_preflight_refuses_then_recovers(self):
        self.f.gpu_free = [15731, 600]
        with mock.patch.object(ops.clients, "apply", side_effect=AssertionError("no client write after a refusal")):
            with self.assertRaises(StartError) as ctx:
                self.up()
        self.assertTrue(str(ctx.exception).startswith("only 600 MiB of GPU memory is free after the start"),
                        str(ctx.exception))
        self.assertTrue(str(ctx.exception).endswith(STOPPED_SUFFIX))
        self.f.managed_lines = [managed_line("sglang", "exited", "qwen3.5-9b", runtime="sglang")]
        self.f.gpu_free = [15731, 1013]
        self.f.host.calls.clear()
        self.f.host.out.truncate(0)
        self.f.host.out.seek(0)
        self.assertEqual(self.up(), 0)
        calls = self.f.host.calls
        self.assertLess(calls.index(["docker", "rm", "sglang"]), calls.index(self.f.calls_starting("docker", "run")[0]))
        self.assertTrue((self.overlay / "generation_config.json").is_file())
        lines = self.lines()
        removing = lines.index("removing the stopped container 'sglang' from an earlier llmctl run")
        wrote = [i for i, line in enumerate(lines) if line.startswith(f"wrote {self.overlay}: model overlay with ")]
        ready = [i for i, line in enumerate(lines) if line.startswith("ready: http://127.0.0.1:30000/v1")]
        self.assertEqual((len(wrote), len(ready)), (1, 1), lines)
        self.assertLess(removing, wrote[0])
        self.assertLess(wrote[0], ready[0])

    def stopped_container_with_old_overlay(self, fixture, overlay):
        overlay.mkdir(parents=True)
        (overlay / "marker").write_text("old overlay")
        fixture.managed_lines = [managed_line("sglang", "exited", "qwen3.5-9b", runtime="sglang")]

    def test_overlay_is_rebuilt_only_after_the_stopped_container_is_removed(self):
        self.stopped_container_with_old_overlay(self.f, self.overlay)
        seen = []
        self.f.host.on(("docker", "rm"), lambda argv: seen.append((self.overlay / "marker").exists()) or ok())
        self.assertEqual(self.up(), 0)
        self.assertEqual(seen, [True])
        self.assertFalse((self.overlay / "marker").exists())

        failing = OpsFixture()
        self.addCleanup(failing.cleanup)
        failing.use_sglang()
        overlay = failing.cache_dir / "sglang" / "overlays" / "qwen3.5-9b"
        self.stopped_container_with_old_overlay(failing, overlay)
        failing.host.on(("docker", "rm"), fail("Error response from daemon: removal failed"))
        with self.assertRaises(PreflightError) as ctx:
            self.up(fixture=failing)
        self.assertEqual(ctx.exception.exit_code, 3)
        self.assertEqual(failing.calls_starting("docker", "run"), [])
        self.assertTrue((overlay / "marker").exists())

    def test_already_running_skips_the_check_and_says_so(self):
        recipe, runtime = resolve(self.f.catalog, "qwen3.5-9b", "sglang")
        self.f.managed_lines = [managed_line("sglang", "running", "qwen3.5-9b", runtime="sglang",
                                             sha=recipe.source.sha256, runtime_sha=runtime.source.sha256)]
        self.f.sglang_ready = [http(200)]
        self.assertEqual(self.up(), 0)
        self.assertEqual(self.f.calls_starting("docker", "run"), [])
        self.assertFalse((self.f.cache_dir / "sglang" / "overlays").exists())
        self.assertNotIn("http://127.0.0.1:30000/server_info", [url for _, url, _ in self.f.host.http_calls])
        lines = self.lines()
        running = lines.index("qwen3.5-9b is already running from this recipe file; nothing to start")
        self.assertEqual(lines[running + 1],
                         "note: qwen3.5-9b was already running from this recipe file, so llmctl did not check its "
                         f"memory again; {SGLANG_UP} --replace restarts it and checks it")

    def test_interrupt_during_the_check(self):
        self.f.host.on_http("GET", "http://127.0.0.1:30000/server_info", KeyboardInterrupt())
        self.assertEqual(self.up(), 130)
        self.assertEqual(self.f.host.err.getvalue(),
                         "interrupted; the server is running, but llmctl did not check its memory or write the "
                         f"clients; {SGLANG_UP} --replace restarts it and checks it\n")
        self.assertEqual(self.f.calls_starting("docker", "stop"), [])
        with hold_lock(self.f.host.env):
            pass

    def test_interrupt_during_the_refusal_stop_keeps_the_refusal(self):
        self.f.gpu_free = [15731, 600]

        def interrupted_stop(argv):
            raise KeyboardInterrupt()

        self.f.host.on(("docker", "stop"), interrupted_stop)
        with mock.patch.object(ops.clients, "apply", side_effect=AssertionError("no client write after a refusal")):
            self.assertEqual(self.up(), 130)
        err = self.f.host.err.getvalue()
        self.assertEqual(err.count("\n"), 1, err)
        self.assertTrue(err.startswith("only 600 MiB of GPU memory is free after the start"), err)
        self.assertTrue(err.endswith("; the stop was interrupted, so the container may still be running: "
                                     "./llmctl status, ./llmctl down\n"), err)
        self.assertIn(["docker", "stop", "-t", "60", "sglang"], self.f.host.calls)
        self.assertFalse(any(line.startswith("ready:") for line in self.lines()))
        with hold_lock(self.f.host.env):
            pass

    def test_dry_run_prints_would_write_and_writes_nothing(self):
        self.assertEqual(self.up(dry_run=True), 0)
        lines = self.lines()
        run = next(i for i, line in enumerate(lines) if line.startswith("docker run -d --name sglang"))
        self.assertEqual(lines[run - 1],
                         f"would write {self.overlay}: model overlay with links into snapshot a4df85a20fb7 of "
                         "RedHatAI/Qwen3.5-9B-NVFP4, and generation_config.json with temperature 0.6, top_p 0.95, "
                         "top_k 20")
        self.assertFalse((self.f.cache_dir / "sglang" / "overlays").exists())
        self.assertEqual(self.f.calls_starting("docker", "run"), [])

    def test_sampling_file_problem_refuses_before_anything_stops(self):
        recipe = self.f.catalog.runtimes["sglang"].recipes["qwen3.5-9b"]
        place_snapshot(self.f.hf_home, recipe.model, {"generation_config.json": b"not json"})
        self.f.managed_lines = [managed_line("vllm", "running", "gpt-oss-20b", runtime="vllm")]
        with self.assertRaises(PreflightError) as ctx:
            self.up(replace=True)
        self.assertEqual(ctx.exception.exit_code, 3)
        self.assertIn("qwen3.5-9b: the model files are not ready", str(ctx.exception))
        self.assertIn("generation_config.json in snapshot a4df85a20fb7 is not a JSON object", str(ctx.exception))
        self.assertEqual(self.f.calls_starting("docker", "stop"), [])
        self.assertEqual(self.f.calls_starting("docker", "run"), [])

    def test_llamacpp_and_vllm_define_no_hooks(self):
        for adapter in (llamacpp, vllm):
            for name in ("prepare_start", "check_started"):
                with self.subTest(adapter=adapter.NAME, member=name):
                    self.assertIsNone(getattr(adapter, name, None))

    def pull(self, name, runtime):
        start = len(self.f.host.out.getvalue())
        self.assertEqual(ops.pull(self.f.host, self.f.catalog, name, runtime, yes=True, env_file=None), 0)
        return self.f.host.out.getvalue()[start:].splitlines()

    def test_printed_commands_name_the_sglang_runtime(self):
        self.assertEqual(self.pull("qwen3.5-9b", "sglang"),
                         [f"qwen3.5-9b: already pulled; start it with {SGLANG_UP}"])
        self.f.place_files("gpt-oss-20b", runtime="vllm")
        self.assertEqual(self.pull("gpt-oss-20b", "vllm"),
                         ["gpt-oss-20b: already pulled; start it with ./llmctl up gpt-oss-20b --runtime vllm"])
        self.assertEqual(self.pull("qwen3.6-35b-a3b", None),
                         ["qwen3.6-35b-a3b: already pulled; start it with ./llmctl up qwen3.6-35b-a3b"])
        self.f.host.on(("docker", "image", "inspect"), fail("Error: No such image"))
        self.assertEqual(self.pull("qwen3.5-9b", "sglang")[-1], f"ready: {SGLANG_UP}")
        with self.assertRaises(PreflightError) as ctx:
            self.up()
        self.assertIn("run ./llmctl pull qwen3.5-9b --runtime sglang", str(ctx.exception))


class DownStatusLogsTest(unittest.TestCase):
    def setUp(self):
        self.f = OpsFixture()

    def tearDown(self):
        self.f.cleanup()

    def test_down_only_managed(self):
        self.f.managed_lines = [managed_line("llamacpp", "running", "qwen3.6-35b-a3b")]
        self.assertEqual(ops.down(self.f.host, self.f.catalog, remove=False), 0)
        self.assertEqual(self.f.calls_starting("docker", "stop"), [["docker", "stop", "-t", "60", "llamacpp"]])
        self.assertEqual(self.f.calls_starting("docker", "rm"), [])
        self.assertEqual(ops.down(self.f.host, self.f.catalog, remove=True), 0)
        self.assertEqual(self.f.calls_starting("docker", "rm"), [["docker", "rm", "llamacpp"]])

    def test_down_with_nothing_running(self):
        self.assertEqual(ops.down(self.f.host, self.f.catalog, remove=False), 0)
        self.assertIn("no model server started by llmctl", self.f.host.out.getvalue())

    def test_status_without_container_prints_clients(self):
        self.assertEqual(ops.status(self.f.host, self.f.catalog), 0)
        self.assertEqual(self.f.host.out.getvalue().splitlines(), [
            "no container started by llmctl",
            "clients:",
            "  OpenCode: not installed (no opencode on PATH)",
            "  Hermes: not installed (no hermes on PATH)",
            "  Pi: not installed (no pi on PATH)",
            "  dsh: not installed (no dsh on PATH)",
            "  Kimi: not installed (no kimi on PATH)",
        ])

    def test_status_not_installed_client(self):
        fixture = ClientFixture(self.f.host, Path(self.f.tmp.name))
        fixture.install("opencode", "hermes")
        self.assertEqual(ops.status(self.f.host, self.f.catalog), 0)
        lines = self.f.host.out.getvalue().splitlines()
        self.assertIn("  Pi: not installed (no pi on PATH)", lines)
        self.assertIn("  Hermes: profile llmctl not created yet (the next ./llmctl up creates it)", lines)

    def test_running_target(self):
        self.assertIsNone(ops.running_target(self.f.host, self.f.catalog))
        self.f.managed_lines = [managed_line("llamacpp", "exited", "qwen3.8-27b"),
                                managed_line("other", "running", "no-such-recipe"),
                                managed_line("vllm", "running", "gpt-oss-20b", runtime="vllm")]
        target = ops.running_target(self.f.host, self.f.catalog)
        self.assertEqual((target.recipe.id, target.recipe.runtime, target.served, target.label),
                         ("gpt-oss-20b", "vllm", "gpt-oss-20b", "vLLM"))

    def test_status_reads_labels(self):
        self.f.managed_lines = [managed_line("llamacpp", "running", "qwen3.6-35b-a3b", sha="cd" * 32)]
        self.f.host.on(("docker", "port"), ok("8080/tcp -> 127.0.0.1:8080\n"))
        self.f.host.on(("docker", "stats"), ok("14.2GiB / 20GiB\n"))
        self.f.health = [http(200)]
        self.assertEqual(ops.status(self.f.host, self.f.catalog), 0)
        out = self.f.host.out.getvalue()
        for expected in ("qwen3.6-35b-a3b", "llamacpp", "shipped", "cdcdcdcdcdcd", "running", "127.0.0.1:8080",
                         "ready", "14.2GiB / 20GiB", "112 MiB"):
            self.assertIn(expected, out)
        lines = out.splitlines()
        self.assertGreater(lines.index("clients:"), max(i for i, line in enumerate(lines) if "GPU memory" in line))

    def test_logs_follow_is_attached(self):
        self.f.managed_lines = [managed_line("llamacpp", "running", "qwen3.6-35b-a3b")]
        self.assertEqual(ops.logs(self.f.host, follow=True, tail=100), 0)
        self.assertEqual(self.f.host.attached, [["docker", "logs", "--tail", "100", "-f", "llamacpp"]])


if __name__ == "__main__":
    unittest.main()
