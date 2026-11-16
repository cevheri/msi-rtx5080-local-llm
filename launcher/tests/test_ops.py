import dataclasses
import fcntl
import json
import os
import unittest
from pathlib import Path
from unittest import mock

from launcher import clients, hermes_config, ops, pi_config
from launcher.catalog import resolve
from launcher.errors import ClientError, ClientWriteError, PreflightError, StartError, UsageError
from launcher.lock import hold_lock, lock_path
from launcher.tests.client_fixture import ClientFixture
from launcher.tests.fakes import REPO, fail, http, ok
from launcher.tests.ops_fixture import OpsFixture, managed_line
from launcher.tests.regen_golden import target_for

NO_CLIENTS = ("clients: none (opencode: not installed, no opencode on PATH) (hermes: not installed, no hermes on PATH) "
              "(pi: not installed, no pi on PATH)")


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
        self.f.place_files("gemma-4-12b", runtime="vllm")
        self.f.host.on_http("GET", "http://127.0.0.1:8000/v1/models",
                            http(200, b'{"data":[{"id":"gemma-4-12b"}]}'))
        self.assertEqual(self.up("gemma-4-12b", "vllm"), 0)
        self.assertIn("never run on the reference laptop", self.f.host.out.getvalue())

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


def running_qwen(fixture, sha=None):
    recipe, runtime = resolve(fixture.catalog, "qwen3.6", None)
    fixture.managed_lines = [managed_line("llamacpp", "running", recipe.id, sha=sha or recipe.source.sha256,
                                          runtime_sha=runtime.source.sha256)]
    fixture.health = [http(200)]


SAMPLE = """\
qwen3.6-35b-a3b on llamacpp: shipped recipe, measured
clients: opencode, hermes, pi
waiting for qwen3.6-35b-a3b to load (up to 300 s)
ready: http://127.0.0.1:8080/v1, served as 'local', context 131072, output 16384
backed up /home/u/.pi/agent/settings.json to /home/u/.pi/agent/settings.json.llmctl-backup before the first change
wrote /home/u/.config/opencode/opencode.json: model llamacpp-docker/qwen3.6-35b-a3b (was llamacpp-docker/gemma-4-26b-a4b); OpenCode reloaded its configuration
wrote /home/u/.hermes/profiles/llmctl/config.yaml: llmctl block for qwen3.6-35b-a3b on llama.cpp (was gpt-oss-20b on vLLM)
wrote /home/u/.pi/agent/models.json: provider llmctl for qwen3.6-35b-a3b on llama.cpp, model local (was gpt-oss-20b on vLLM)
wrote /home/u/.pi/agent/settings.json: modelThinkingLevels llmctl/local off (was absent); defaultModel local (was gpt-oss-20b)
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
note: Hermes and Pi sessions that were open before keep the model they started with; start a new session
evidence: 122.5 tok/s decode with 128k context, 15,190 MiB GPU peak, OpenCode session passed (2026-11-16, research/2026-11-16/llamacpp-qwen3.6-35b-a3b/)
stop with ./llmctl down
"""

START_SETTINGS = {"lastChangelogVersion": "1.5.2", "defaultProvider": "llmctl", "defaultModel": "gpt-oss-20b"}
PI_LIST = ["pi", "--offline", "--list-models", "llmctl"]


class UpClientsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qwen = target_for("qwen3.6-35b-a3b", "llamacpp", REPO / "recipes")
        cls.old_target = target_for("gpt-oss-20b", "vllm", REPO / "recipes")

    def setUp(self):
        self.f = OpsFixture()
        self.addCleanup(self.f.cleanup)
        self.c = ClientFixture(self.f.host, Path(self.f.tmp.name))
        self.c.install("opencode", "hermes", "pi")
        self.c.route_clients()
        self.c.write_opencode("llamacpp-docker/gemma-4-26b-a4b")
        Path(str(self.c.opencode_path) + ".llmctl-backup").write_text("{}\n", encoding="utf-8")
        self.c.hermes_config.parent.mkdir(parents=True)
        self.c.hermes_config.write_text(hermes_config.new_config_text(self.old_target), encoding="utf-8")
        self.c.models_path.write_text(pi_config.new_models_text(self.old_target), encoding="utf-8")
        self.c.settings_path.write_text(json.dumps(START_SETTINGS, indent=2) + "\n", encoding="utf-8")

    def up(self, name="qwen3.6", runtime=None, *, dry_run=False, set_default=False):
        return ops.up(self.f.host, self.f.catalog, name, runtime, replace=False, dry_run=dry_run,
                      set_default=set_default)

    def out(self):
        return self.f.host.out.getvalue()

    def opencode_model(self):
        return json.loads(self.c.opencode_path.read_text(encoding="utf-8"))["model"]

    def client_calls(self, calls=None):
        return [call for call in (self.f.host.calls if calls is None else calls)
                if call[0] in ("opencode", "hermes", "pi")]

    def test_down_prints_client_lines_and_changes_nothing(self):
        running_qwen(self.f)
        self.f.host.on(("docker", "stop"), ok())
        (self.c.hermes_root / "active_profile").write_text("llmctl\n", encoding="utf-8")
        before = self.c.snapshot()
        self.assertEqual(ops.down(self.f.host, self.f.catalog, remove=False), 0)
        self.assertEqual(self.out().splitlines(), [
            "stopped qwen3.6-35b-a3b in container 'llamacpp'",
            "OpenCode, Hermes and Pi still point at qwen3.6-35b-a3b and get no answer until the next ./llmctl up",
            "  switch back: OpenCode /models; hermes profile use default; Pi /model",
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
        self.assertEqual(self.client_calls(calls[calls.index(PI_LIST) + 1:]), [])

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
        self.assertEqual(lines[-1], "dry run: nothing was written, and Pi has not checked the entry")

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
        qwen_on = dataclasses.replace(self.qwen, recipe=dataclasses.replace(
            self.qwen.recipe, clients=dataclasses.replace(self.qwen.recipe.clients, default="on")))
        cases = [
            (target_for("mellum2.1-12b", "vllm", REPO / "recipes"),
             ["  thinking: on, as the server sets it", "  thinking: on; --thinking off turns it off"]),
            (target_for("nemotron-3.5-bolt", "llamacpp", REPO / "recipes"),
             ["  thinking: off; add #thinking to the model to turn it on, #medium for medium effort",
              "  thinking: off, as the server sets it",
              "  thinking: off; --thinking medium for medium effort, --thinking high for full thinking"]),
            (target_for("k2-horizon-7b", "llamacpp", REPO / "recipes"), [always] * 3),
            (target_for("qwen3-coder-next", "llamacpp", REPO / "recipes"), ["  thinking: none"] * 3),
            (target_for("deepseek-r1-0528-qwen3-8b", "llamacpp", REPO / "recipes"), [always] * 2),
            (qwen_on, ["  thinking: on; add #none to turn it off", "  thinking: on, as the server sets it",
                       "  thinking: on; --thinking off turns it off"]),
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
