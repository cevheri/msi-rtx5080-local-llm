"""The opt-in e2e driver: its parsers, its step plan and its restore of the defaults, without running anything."""

import io
import json
import os
import stat
import subprocess
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from launcher import clients, hermes_config, opencode_config, pi_config
from launcher.catalog import load_catalog
from launcher.e2e import clients as e2e
from launcher.errors import UsageError
from launcher.tests.client_fixture import ClientFixture
from launcher.tests.fakes import REPO, FakeHost, fail, ok
from launcher.tests.ops_fixture import managed_line
from launcher.tests.regen_golden import target_for

FIXTURES = REPO / "launcher/tests/fixtures/e2e"


def fixture_lines(name: str) -> list[str]:
    return (FIXTURES / name).read_text(encoding="utf-8").splitlines()


def events(name: str) -> list[dict]:
    return [json.loads(line) for line in fixture_lines(name)]


def dump(items: list[dict]) -> list[str]:
    return [json.dumps(item) for item in items]


class ParserTest(unittest.TestCase):
    def test_opencode_outcome(self):
        outcome = e2e.opencode_outcome(fixture_lines("opencode-events.jsonl"))
        self.assertTrue(outcome.passed, outcome.reasons)
        self.assertEqual(outcome.reasons, ())
        without_read = [item for item in events("opencode-events.jsonl")
                        if not (item["type"] == "tool_use" and item["part"]["tool"] == "read")]
        outcome = e2e.opencode_outcome(dump(without_read))
        self.assertFalse(outcome.passed)
        self.assertTrue([reason for reason in outcome.reasons if "read" in reason], outcome.reasons)
        outcome = e2e.opencode_outcome(fixture_lines("opencode-events.jsonl") + ['{"type": "error"}'])
        self.assertFalse(outcome.passed)

    def test_opencode_models(self):
        lines = ['{"type": "step_start", "part": {"info": {"providerID": "llamacpp-docker", '
                 '"modelID": "qwen3.6-35b-a3b"}}}'] + fixture_lines("opencode-events.jsonl")
        self.assertEqual(e2e.opencode_outcome(lines).models, ("llamacpp-docker/qwen3.6-35b-a3b",))

    def test_hermes_stream_outcome(self):
        outcome = e2e.hermes_stream_outcome(fixture_lines("hermes-stream.jsonl"))
        self.assertTrue(outcome.passed, outcome.reasons)
        items = events("hermes-stream.jsonl")
        items[-1] = dict(items[-1], exit_code=1)
        self.assertFalse(e2e.hermes_stream_outcome(dump(items)).passed)
        items = events("hermes-stream.jsonl")
        items = [dict(item, is_error=True) if item["type"] == "tool_result" and item["name"] == "write_file" else item
                 for item in items]
        outcome = e2e.hermes_stream_outcome(dump(items))
        self.assertFalse(outcome.passed)
        self.assertTrue([reason for reason in outcome.reasons if "write_file" in reason], outcome.reasons)

    def test_hermes_stream_chat_only(self):
        items = [item for item in events("hermes-stream.jsonl") if item["type"] not in ("tool_use", "tool_result")]
        self.assertFalse(e2e.hermes_stream_outcome(dump(items)).passed)
        self.assertTrue(e2e.hermes_stream_outcome(dump(items), tools=False).passed)

    def test_hermes_usage_outcome(self):
        outcome = e2e.hermes_usage_outcome((FIXTURES / "hermes-usage.json").read_text(encoding="utf-8"))
        self.assertTrue(outcome.passed, outcome.reasons)
        self.assertEqual(outcome.models, ("custom/local",))
        outcome = e2e.hermes_usage_outcome(json.dumps({"completed": False, "provider": "custom", "model": "local"}))
        self.assertFalse(outcome.passed)
        self.assertFalse(e2e.hermes_usage_outcome("not json").passed)

    def test_pi_outcome(self):
        outcome = e2e.pi_outcome(fixture_lines("pi-events.jsonl"))
        self.assertTrue(outcome.passed, outcome.reasons)
        self.assertIn("llmctl/local", outcome.models)
        outcome = e2e.pi_outcome(fixture_lines("pi-events-error.jsonl"))
        self.assertFalse(outcome.passed)
        self.assertTrue([reason for reason in outcome.reasons if "error" in reason], outcome.reasons)

    def test_pi_chat_only(self):
        items = [item for item in events("pi-events.jsonl") if not item["type"].startswith("tool_execution")]
        self.assertFalse(e2e.pi_outcome(dump(items)).passed)
        self.assertTrue(e2e.pi_outcome(dump(items), tools=False).passed)

    def test_non_json_line_fails(self):
        outcome = e2e.pi_outcome(fixture_lines("pi-events.jsonl") + ["Traceback (most recent call last):"])
        self.assertFalse(outcome.passed)
        self.assertTrue([reason for reason in outcome.reasons if "not JSON" in reason], outcome.reasons)

    def test_hello_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hello.py"
            self.assertFalse(e2e.hello_valid(path))
            path.write_text("def add(a, b):\n    return a + b\n\n\nif __name__ == \"__main__\":\n"
                            "    print(add(2, 3))\n", encoding="utf-8")
            self.assertTrue(e2e.hello_valid(path))
            path.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
            self.assertFalse(e2e.hello_valid(path))

    def test_gateway_serves_llmctl(self):
        self.assertTrue(e2e.gateway_serves_llmctl(
            json.dumps({"gateway_state": "running", "served_profiles": ["default", "llmctl"]})))
        self.assertFalse(e2e.gateway_serves_llmctl(
            json.dumps({"gateway_state": "running", "served_profiles": ["default"]})))

    def test_status_reasons(self):
        text = ("qwen3.6-35b-a3b on llamacpp, container 'llamacpp', running\n"
                "clients:\n"
                "  OpenCode: default llamacpp-docker/qwen3.6-35b-a3b, the running model\n"
                "  Hermes: profile llmctl for qwen3.6-35b-a3b on llama.cpp, the running model; default profile: x\n"
                "  Pi: llmctl/local for qwen3.6-35b-a3b on llama.cpp, the running model; default x, thinking off\n")
        self.assertEqual(e2e.status_reasons(text, opencode=True), [])
        other = text.replace("qwen3.6-35b-a3b, the running model\n", "qwen3.6-35b-a3b\n")
        self.assertEqual(e2e.status_reasons(other, opencode=True),
                         ["OpenCode: default llamacpp-docker/qwen3.6-35b-a3b"])
        self.assertEqual(e2e.status_reasons(other, opencode=False), [])
        self.assertEqual(e2e.status_reasons("clients:\n", opencode=False),
                         ["no Hermes line in status", "no Pi line in status"])

    def test_status_reasons_gateway_warning(self):
        text = ("clients:\n"
                "  Hermes: profile llmctl for qwen3.6-35b-a3b on llama.cpp, the running model; default profile: x\n"
                "  Pi: llmctl/local for qwen3.6-35b-a3b on llama.cpp, the running model; default x, thinking off\n")
        warning = ("warning: Hermes' gateway serves the profile llmctl (/h/gateway_state.json), so its sessions run "
                   "inside your messaging gateway")
        self.assertEqual(e2e.status_reasons(text, opencode=False), [])
        self.assertEqual(e2e.status_reasons(text + warning + "\n", opencode=False), [warning])


class ConfigValueTest(unittest.TestCase):
    """hermes config get prints merged sections, so the block's value only has to be contained in it."""

    GATEWAY = {"unmanaged_idle_exit_seconds": 600, "signal_interrupt_grace_timeout": 1,
               "multiplex_profiles": True, "standalone": True}

    @classmethod
    def setUpClass(cls):
        cls.target = target_for("qwen3.6-35b-a3b", "llamacpp", REPO / "recipes")
        cls.wanted = hermes_config.block_values(cls.target)

    def check(self, key, value):
        return e2e._Run.config_value(None, key, json.dumps(value), self.target)

    def test_merged_gateway_passes(self):
        self.assertEqual(self.check("gateway", self.GATEWAY), [])

    def test_gateway_without_standalone_true_fails(self):
        self.assertTrue(self.check("gateway", dict(self.GATEWAY, standalone=False)))
        self.assertTrue(self.check("gateway", {k: v for k, v in self.GATEWAY.items() if k != "standalone"}))
        self.assertTrue(self.check("gateway", dict(self.GATEWAY, standalone=1)))  # JSON true is not 1

    def test_exact_values_pass(self):
        for key in ("model", "providers", "model_overrides", "gateway"):
            with self.subTest(key=key):
                self.assertEqual(self.check(key, self.wanted[key]), [])

    def test_other_keys_need_their_values(self):
        model = dict(self.wanted["model"], default="other")
        self.assertTrue(self.check("model", model))
        llmctl = dict(self.wanted["providers"]["llmctl"], api="http://127.0.0.1:1/v1")
        self.assertTrue(self.check("providers", {"llmctl": llmctl}))
        extra = dict(self.wanted["providers"]["llmctl"], note="extra")
        self.assertEqual(self.check("providers", {"llmctl": extra}), [])
        served = self.target.served
        self.assertTrue(self.check("model_overrides", {"custom": {served: {"supports_reasoning": True}}}))
        self.assertTrue(self.check("model_overrides", {"custom": {}}))
        self.assertTrue(self.check("model", "custom:llmctl"))

    def test_effort_recipe_needs_empty_model_overrides(self):
        # an override there would strip the reasoning effort the server defaults to
        target = target_for("qwen3.8-27b", "llamacpp", REPO / "recipes")
        self.assertEqual(hermes_config.block_values(target)["model_overrides"], {})

        def check(value):
            return e2e._Run.config_value(None, "model_overrides", json.dumps(value), target)

        self.assertEqual(check({}), [])
        self.assertTrue(check({"custom": {"local": {"supports_reasoning": False}}}))


class LogsSinceTest(unittest.TestCase):
    """docker logs replays the container's stderr on stderr, and llama.cpp logs there."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out = Path(tmp.name)
        (out / "records").mkdir()
        self.target = target_for("qwen3.6-35b-a3b", "llamacpp", REPO / "recipes")
        self.run_ = e2e._Run(out, None, FakeHost(env={}), {"HOME": str(out)}, out)

    def logs(self, stdout, stderr):
        done = subprocess.CompletedProcess([], 0, stdout, stderr)
        with unittest.mock.patch("launcher.e2e.clients.subprocess.run", return_value=done) as run:
            reasons = self.run_.logs_since("2026-11-16T10:00:00", self.target)
        self.assertEqual(run.call_args.args[0][:2], ["docker", "logs"])
        return reasons

    def test_request_on_stderr_passes(self):
        self.assertEqual(self.logs("", "slot print_timing: id  0 | task 1101 | prompt eval time = 1 ms\n"), [])
        self.assertEqual(self.logs("", "srv  log_server_r: request: POST /v1/chat/completions 127.0.0.1 200\n"), [])

    def test_request_on_stdout_passes(self):
        self.assertEqual(self.logs("POST /v1/chat/completions 200\n", ""), [])

    def test_no_request_on_either_stream_fails(self):
        self.assertEqual(self.logs("main: server is listening\n", "srv  update_slots: all slots are idle\n"),
                         ["the llama.cpp log shows no request since the OpenCode run started"])


def up_steps(steps):
    return [step.argv for step in steps if step.argv[:2] == ("./llmctl", "up")]


class PlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog(REPO / "recipes", None)

    def plan(self, matrix=e2e.DEFAULT_MATRIX, *, round_trip=False, keep_running=False):
        return e2e.plan_steps(self.catalog, matrix, round_trip=round_trip, keep_running=keep_running)

    def test_constants(self):
        self.assertTrue(e2e.TASK.startswith("Create a file named hello.py containing a function add(a, b)"))
        self.assertTrue(e2e.TASK.endswith("Do not run any shell commands."))
        self.assertIn(e2e.TASK, (REPO / "research/2026-11-16/llmctl-e2e/e2e-llamacpp.sh").read_text())
        self.assertEqual(e2e.CHAT, "Say hello in one sentence.")
        self.assertEqual(e2e.PERMISSIONS, {name: "deny" for name in (
            "bash", "external_directory", "webfetch", "websearch", "task", "question")})
        self.assertEqual((e2e.STEP_TIMEOUT_S, e2e.GATEWAY_WAIT_S), (900, 35))

    def test_default_matrix_order(self):
        self.assertEqual(up_steps(self.plan()), [
            ("./llmctl", "up", "qwen3.8-27b", "--replace"),
            ("./llmctl", "up", "k2-horizon-7b", "--replace"),
            ("./llmctl", "up", "gpt-oss-20b", "--runtime", "vllm", "--replace"),
            ("./llmctl", "up", "qwen3.6-35b-a3b", "--replace"),
        ])

    def test_extended_matrix_ends_on_default_model(self):
        self.assertEqual(len(e2e.EXTENDED_MATRIX), 12)
        self.assertEqual(e2e.EXTENDED_MATRIX[:3], e2e.DEFAULT_MATRIX[:3])
        self.assertEqual(e2e.EXTENDED_MATRIX[3:11], tuple((name, "llamacpp") for name in (
            "nemotron-3.5-bolt", "qwen3-coder-next", "deepseek-r1-0528-qwen3-8b", "gemma-4-26b-a4b",
            "nemotron-3.5-lightning", "groot-19b-a3b", "k2-horizon-mova-36b", "gpt-oss-20b")))
        ups = up_steps(self.plan(e2e.EXTENDED_MATRIX))
        self.assertEqual(len(ups), 12)
        self.assertEqual(ups[-1], ("./llmctl", "up", "qwen3.6-35b-a3b", "--replace"))

    def test_steps_respect_safety_rules(self):
        for matrix in (e2e.DEFAULT_MATRIX, e2e.EXTENDED_MATRIX):
            steps = self.plan(matrix)
            for step in steps:
                self.assertNotIn("--set-default", step.argv)
                self.assertNotIn("pull", step.argv)
                self.assertNotEqual(step.argv[:2], ("hermes", "gateway"))
                self.assertEqual(step.timeout_s, 900)
            client_steps = [step for step in steps if step.argv and step.argv[0] in ("opencode", "hermes", "pi")]
            self.assertTrue(client_steps)
            folders = [step.cwd for step in client_steps]
            self.assertNotIn(".", folders)
            self.assertEqual(len(folders), len(set(folders)))
            self.assertEqual(steps[-1].argv, ("./llmctl", "down"))
            self.assertEqual(steps[-1].cwd, ".")
        self.assertNotIn(("./llmctl", "down"), [step.argv for step in self.plan(keep_running=True)])

    def test_round_trip_is_opt_in(self):
        steps = self.plan(round_trip=True)
        with_flag = [argv for argv in up_steps(steps) if "--set-default" in argv]
        self.assertEqual(with_flag, [("./llmctl", "up", "qwen3.6-35b-a3b", "--set-default")])
        argvs = [step.argv for step in steps]
        last_matrix_up = argvs.index(("./llmctl", "up", "qwen3.6-35b-a3b", "--replace"))
        self.assertGreater(argvs.index(with_flag[0]), last_matrix_up)
        after = argvs[argvs.index(with_flag[0]):]
        self.assertTrue([argv for argv in after if argv[:3] == ("pi", "--mode", "json") and "--model" not in argv])
        self.assertTrue([argv for argv in after if argv[:2] == ("hermes", "-z") and "-p" not in argv])
        self.assertTrue([argv for argv in after if argv[:2] == ("opencode", "run") and "--model" not in argv])
        self.assertEqual(argvs[-1], ("./llmctl", "down"))

    def test_session_steps(self):
        steps = [step for step in self.plan() if step.name.startswith("04-qwen3.6-35b-a3b-llamacpp ")]
        argvs = [step.argv for step in steps]
        self.assertIn(("opencode", "run", "--standalone", "--archived", "--auto", "--format", "json",
                       "--model", "llamacpp-docker/qwen3.6-35b-a3b", e2e.TASK), argvs)
        self.assertIn(("hermes", "-p", "llmctl", "-z", e2e.TASK, "-t", "file", "--usage-file", "{work}/usage.json"),
                      argvs)
        self.assertIn(("hermes", "-p", "llmctl", "chat", "-q", e2e.TASK, "-t", "file", "--format", "stream-json",
                       "--source", "tool", "--max-turns", "10"), argvs)
        self.assertIn(("pi", "--mode", "json", "--no-session", "--no-approve", "--no-context-files",
                       "--tools", "read,write", "--model", "llmctl/local", e2e.TASK), argvs)
        self.assertEqual([argv[5] for argv in argvs if argv[3:5] == ("config", "get")],
                         list(hermes_config.OWNED_KEYS[:4]))
        self.assertFalse([argv for argv in argvs if "--thinking" in argv])
        self.assertEqual(steps[0].argv, ("./llmctl", "up", "qwen3.6-35b-a3b", "--replace"))
        self.assertEqual(steps[1].argv, ("./llmctl", "status"))

    def test_chat_only_steps(self):
        steps = self.plan((("deepseek-r1-0528-qwen3-8b", "llamacpp"),))
        argvs = [step.argv for step in steps]
        self.assertFalse([argv for argv in argvs if argv[:1] == ("opencode",)])
        pi = [argv for argv in argvs if argv[:1] == ("pi",)]
        self.assertEqual(len(pi), 1)
        self.assertIn("--no-tools", pi[0])
        self.assertIn(e2e.CHAT, pi[0])
        self.assertNotIn("--tools", pi[0])
        sessions = [argv for argv in argvs if argv[:4] in (("hermes", "-p", "llmctl", "-z"),
                                                            ("hermes", "-p", "llmctl", "chat"))]
        self.assertEqual(len(sessions), 2)
        for argv in sessions:
            self.assertNotIn("-t", argv)
            self.assertIn(e2e.CHAT, argv)
        self.assertIn("platform_toolsets", [argv[5] for argv in argvs if argv[3:5] == ("config", "get")])

    def test_no_opencode_step_without_entry(self):
        argvs = [step.argv for step in self.plan((("gpt-oss-20b", "llamacpp"),))]
        self.assertFalse([argv for argv in argvs if argv[:1] == ("opencode",)])

    def test_unknown_recipe_is_a_usage_error(self):
        with self.assertRaises(UsageError):
            self.plan((("no-such-model", "llamacpp"),))

    def test_probe_bodies(self):
        qwen = e2e.probe_bodies(target_for("qwen3.6-35b-a3b", "llamacpp", REPO / "recipes"))
        self.assertEqual(sorted(qwen), ["A", "B", "C", "D"])
        a = qwen["A"]
        self.assertEqual(a["reasoning_effort"], "none")
        self.assertEqual(a["response_format"]["json_schema"]["name"], "session_title")
        self.assertIs(a["response_format"]["json_schema"]["strict"], True)
        self.assertEqual(a["max_tokens"], 16384)
        self.assertIs(a["stream"], False)
        self.assertEqual(a["model"], "local")
        self.assertEqual(qwen["D"]["model"], "qwen3.6-35b-a3b")
        self.assertNotIn("reasoning_effort", qwen["D"])
        self.assertEqual(qwen["C"]["reasoning_effort"], "none")
        b = qwen["B"]
        assistant = b["messages"][1]
        self.assertEqual((assistant["content"], assistant["reasoning"], assistant["reasoning_content"]),
                         ("", "I will write the file.", "I will write the file."))
        call = assistant["tool_calls"][0]
        self.assertEqual(call["id"], "call_1")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"path": "hello.py", "content": "print(1)\n"})
        self.assertEqual(b["messages"][2], {"role": "tool", "tool_call_id": "call_1", "content": "ok"})
        self.assertEqual(b["tools"][0]["function"]["name"], "write")
        self.assertEqual(b["max_tokens"], 256)
        gpt = e2e.probe_bodies(target_for("gpt-oss-20b", "vllm", REPO / "recipes"))
        self.assertEqual(sorted(gpt), ["A", "B"])
        self.assertEqual(gpt["A"]["model"], "gpt-oss-20b")
        k2 = e2e.probe_bodies(target_for("k2-horizon-7b", "llamacpp", REPO / "recipes"))
        self.assertEqual(sorted(k2), ["A", "B", "C", "D", "E"])
        self.assertEqual(k2["A"]["max_tokens"], 32768)
        self.assertEqual(k2["E"]["model"], "local")
        self.assertEqual(k2["E"]["chat_template_kwargs"], {"tool_call_format": "xml"})
        self.assertNotIn("reasoning_effort", k2["E"])

    def test_probe_record(self):
        reply = {"choices": [{"message": {"content": json.dumps({"title": "Adding numbers"}),
                                          "reasoning_content": "think"}}]}
        self.assertEqual(e2e.probe_record("A", 200, json.dumps(reply).encode()),
                         {"probe": "A", "status": 200, "content": True, "reasoning": True, "title_ok": True})
        reply = {"choices": [{"message": {"content": "OK"}}]}
        self.assertEqual(e2e.probe_record("C", 200, json.dumps(reply).encode()),
                         {"probe": "C", "status": 200, "content": True, "reasoning": False})
        self.assertEqual(e2e.probe_record("A", 200, json.dumps(reply).encode())["title_ok"], False)
        self.assertEqual(e2e.probe_record("B", 400, b"bad request"),
                         {"probe": "B", "status": 400, "content": False, "reasoning": False})

    def test_driver_env_and_redact(self):
        self.assertEqual(e2e.driver_env({"LLMCTL_SET_DEFAULT": "1", "PATH": "/bin"}), {"PATH": "/bin"})
        self.assertEqual(e2e.redact("/home/x/a", "/home/x"), "$HOME/a")
        self.assertEqual(e2e.redact("/home/xy/a /home/x", "/home/x"), "/home/xy/a $HOME")
        self.assertEqual(e2e.redact("/tmp/llmctl-e2e-ab/01-x/usage.json /home/x/a /tmp/llmctl-e2e-abc",
                                    "/home/x", "/tmp/llmctl-e2e-ab"),
                         "$WORK/01-x/usage.json $HOME/a /tmp/llmctl-e2e-abc")
        self.assertEqual(e2e.redact("/home/x/tmp/llmctl-e2e-1/a", "/home/x", "/home/x/tmp/llmctl-e2e-1"), "$WORK/a")

    def test_main_refuses_llmctl_clients(self):
        with unittest.mock.patch.dict("os.environ", {"LLMCTL_CLIENTS": "pi"}):
            with unittest.mock.patch("sys.stderr") as err:
                self.assertEqual(e2e.main(["--out", "/nonexistent/never-created"]), 1)
        written = "".join(call.args[0] for call in err.write.call_args_list)
        self.assertIn("unset LLMCTL_CLIENTS: the e2e checks all three clients", written)
        self.assertFalse(Path("/nonexistent/never-created").exists())


class RestoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog(REPO / "recipes", None)
        cls.target = target_for("qwen3.6-35b-a3b", "llamacpp", REPO / "recipes")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.host = FakeHost(env={"HOME": "/nonexistent", "PATH": "/usr/bin"})
        self.fx = ClientFixture(self.host, Path(self.tmp.name))
        self.fx.install("opencode", "hermes", "pi")
        self.fx.route_clients()
        self.fx.write_opencode("vllm-docker/gpt-oss-20b")
        self.fx.settings_path.write_text(json.dumps({"lastChangelogVersion": "1.5.2"}) + "\n", encoding="utf-8")
        self.active = self.fx.hermes_root / "active_profile"
        self.active.write_text("work\n", encoding="utf-8")
        self.host.on(["hermes", "profile", "use", "work"], self.use_work)

    def use_work(self, argv):
        self.active.write_text("work\n", encoding="utf-8")
        return ok()

    def set_default(self):
        result = clients.apply(self.host, self.catalog, self.target, set_default=True)
        self.assertEqual(result.failures, ())
        self.assertEqual(hermes_config.sticky_profile(self.host.env), "llmctl")
        self.assertEqual(pi_config.pi_default(self.host.env), ("llmctl", "local"))

    def opencode_model(self):
        return json.loads(self.fx.opencode_path.read_text(encoding="utf-8"))["model"]

    def assert_pi_and_opencode_restored(self):
        self.assertEqual(json.loads(self.fx.settings_path.read_text(encoding="utf-8")),
                         {"lastChangelogVersion": "1.5.2", "modelThinkingLevels": {"llmctl/local": "off"}})
        self.assertEqual(self.opencode_model(), "vllm-docker/gpt-oss-20b")

    def test_snapshot_defaults(self):
        self.assertEqual(e2e.snapshot_defaults(self.host.env),
                         e2e.Defaults("vllm-docker/gpt-oss-20b", None, None, "work"))
        doc = json.loads(self.fx.opencode_path.read_text(encoding="utf-8"))
        del doc["model"]
        self.fx.opencode_path.write_text(json.dumps(doc), encoding="utf-8")
        with self.assertRaises(UsageError) as caught:
            e2e.snapshot_defaults(self.host.env)
        self.assertTrue(str(caught.exception).startswith("the round trip needs OpenCode's top-level model"))
        self.assertIn(str(opencode_config.config_path(self.host.env)), str(caught.exception))

    def test_restore_after_set_default(self):
        snapshot = e2e.snapshot_defaults(self.host.env)
        self.set_default()
        self.assertEqual(e2e.restore_defaults(self.host, snapshot), [])
        self.assert_pi_and_opencode_restored()
        self.assertEqual(self.host.calls.count(["hermes", "profile", "use", "work"]), 1)
        self.assertEqual(self.active.read_text(encoding="utf-8"), "work\n")

    def test_restore_reports_what_failed(self):
        self.host.on(["hermes", "profile", "use", "work"], fail("boom"))
        snapshot = e2e.snapshot_defaults(self.host.env)
        self.set_default()
        self.assertEqual(e2e.restore_defaults(self.host, snapshot),
                         ["Hermes: hermes profile use work exited 1: boom"])
        self.assert_pi_and_opencode_restored()

    def test_restore_reports_a_value_that_reads_back_different(self):
        self.host.on(["hermes", "profile", "use", "work"], lambda argv: ok())
        snapshot = e2e.snapshot_defaults(self.host.env)
        self.set_default()
        self.assertEqual(e2e.restore_defaults(self.host, snapshot),
                         ["Hermes: active profile reads llmctl, expected work"])

    def test_restore_with_nothing_to_do_runs_nothing(self):
        snapshot = e2e.snapshot_defaults(self.host.env)
        before = self.fx.snapshot()
        self.assertEqual(e2e.restore_defaults(self.host, snapshot), [])
        self.assertEqual(self.host.calls, [])
        self.assertEqual(self.fx.snapshot(), before)

    def test_round_trip_restores_after_a_failed_check(self):
        before = e2e.snapshot_defaults(self.host.env)

        def check():
            self.set_default()
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError) as caught:
            e2e.round_trip(self.host, check)
        self.assertEqual(str(caught.exception), "boom")
        self.assertEqual(e2e.snapshot_defaults(self.host.env), before)
        self.assertEqual(self.host.err.getvalue(), "")

    def test_round_trip_reports_lines_after_a_failed_check(self):
        self.host.on(["hermes", "profile", "use", "work"], fail("boom"))

        def check():
            self.set_default()
            raise RuntimeError("check failed")

        with self.assertRaises(RuntimeError):
            e2e.round_trip(self.host, check)
        self.assertEqual(self.host.err.getvalue(), "not restored: Hermes: hermes profile use work exited 1: boom\n")

    def test_round_trip_reports_restore_lines_on_both_paths(self):
        self.host.on(["hermes", "profile", "use", "work"], fail("boom"))
        reported = []

        def check():
            self.set_default()
            raise OSError(28, "No space left on device")

        with self.assertRaises(OSError):
            e2e.round_trip(self.host, check, report=reported.append)
        self.assertEqual(reported, [["Hermes: hermes profile use work exited 1: boom"]])
        reported.clear()
        self.assertEqual(e2e.round_trip(self.host, lambda: None, report=reported.append), [])
        self.assertEqual(reported, [[]])

    def test_round_trip_returns_problem_lines(self):
        self.host.on(["hermes", "profile", "use", "work"], fail("boom"))
        self.assertEqual(e2e.round_trip(self.host, self.set_default),
                         ["Hermes: hermes profile use work exited 1: boom"])

    def test_gateway_check(self):
        state = self.fx.hermes_root / "gateway_state.json"
        self.assertIsNone(e2e.gateway_check(self.host, created=True))
        self.assertEqual(self.host.sleeps, [35])
        self.host.sleeps.clear()
        state.write_text(json.dumps({"gateway_state": "running", "served_profiles": ["default", "llmctl"]}),
                         encoding="utf-8")
        expected = f"Hermes' gateway serves the profile llmctl ({self.fx.hermes_root}/gateway_state.json)"
        self.assertEqual(e2e.gateway_check(self.host, created=True), expected)
        self.assertEqual(self.host.sleeps, [35])
        self.host.sleeps.clear()
        self.assertEqual(e2e.gateway_check(self.host, created=False), expected)
        self.assertEqual(self.host.sleeps, [])
        state.write_text(json.dumps({"gateway_state": "running", "served_profiles": ["default"]}), encoding="utf-8")
        self.assertIsNone(e2e.gateway_check(self.host, created=False))


RUNNER_MATRIX = (("qwen3.6-35b-a3b", "llamacpp"),)
TWO_RECIPES = (("k2-horizon-7b", "llamacpp"), ("qwen3.6-35b-a3b", "llamacpp"))
HOST_NAME = "e2e-test-host"
WITHHELD = "record withheld: the output named a private value after redaction"
GATEWAY_LLMCTL = {"gateway_state": "running", "served_profiles": ["default", "llmctl"]}


class RunnerTest(unittest.TestCase):
    """main and the runner on a temporary HOME: subprocess.run and the host are fakes, so no process starts."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.home = self.root / "home"
        self.home.mkdir()
        (self.root / "tmp").mkdir()
        self.cwd = self.root / "cwd"
        self.cwd.mkdir()
        self.environ = {"HOME": str(self.home), "PATH": "/usr/bin:/bin", "LLMCTL_SET_DEFAULT": "1"}
        self.host = FakeHost(env={})
        self.managed: list[str] = []
        self.host.on(("docker", "ps"), lambda argv: ok("".join(line + "\n" for line in self.managed)))
        self.calls: list[tuple[tuple[str, ...], Path, dict]] = []
        self.failing_ups: set[str] = set()
        self.ups_create_profile = True
        self.gateway_after_up: dict | None = None
        self.on_call = None
        self.output = None  # argv -> the session's stdout, or None for ""
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        real_mkdtemp = tempfile.mkdtemp
        self.mkdtemp_calls = []

        def mkdtemp(*args, **kwargs):
            self.mkdtemp_calls.append(kwargs)
            return real_mkdtemp(*args, **dict(kwargs, dir=str(self.root / "tmp")))

        for patcher in (unittest.mock.patch.dict("os.environ", self.environ, clear=True),
                        unittest.mock.patch("launcher.e2e.clients.subprocess.run", side_effect=self.fake_run),
                        unittest.mock.patch("launcher.e2e.clients.tempfile.mkdtemp", side_effect=mkdtemp),
                        unittest.mock.patch("socket.gethostname", return_value=HOST_NAME),
                        unittest.mock.patch("sys.stdout", self.stdout),
                        unittest.mock.patch("sys.stderr", self.stderr)):
            patcher.start()
            self.addCleanup(patcher.stop)
        before = os.getcwd()
        os.chdir(self.cwd)
        self.addCleanup(os.chdir, before)

    def make_host(self, env):
        self.host.env = env
        return self.host

    def main(self, *args, matrix=RUNNER_MATRIX):
        with unittest.mock.patch.dict(e2e.MATRICES, {"default": matrix}):
            return e2e.main(["--out", "records-out", *args], make_host=self.make_host)

    @property
    def out(self) -> Path:
        return self.cwd / "records-out"

    def write_profile(self, recipe_id: str, runtime: str) -> None:
        path = hermes_config.profile_config(self.environ)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(hermes_config.new_config_text(target_for(recipe_id, runtime, REPO / "recipes")),
                        encoding="utf-8")

    def fake_run(self, argv, *, cwd, env, capture_output, text, timeout):
        argv = tuple(argv)
        self.calls.append((argv, Path(cwd), dict(env)))
        if self.on_call is not None:
            self.on_call(argv)
        code, stdout = 0, ""
        if argv[:2] == ("./llmctl", "up"):
            if argv[2] in self.failing_ups:
                code = 4
            elif self.ups_create_profile:
                runtime = argv[argv.index("--runtime") + 1] if "--runtime" in argv else "llamacpp"
                self.write_profile(argv[2], runtime)
                if self.gateway_after_up is not None:
                    (hermes_config.hermes_root(self.environ) / "gateway_state.json").write_text(
                        json.dumps(self.gateway_after_up), encoding="utf-8")
        elif argv[0] in ("opencode", "hermes", "pi") and "config" not in argv and "--version" not in argv:
            (Path(cwd) / "hello.py").write_text("def add(a, b):\n    return a + b\n\nprint(add(2, 3))\n",
                                                encoding="utf-8")
            if "--usage-file" in argv:  # Hermes resolves the usage file against its own working directory
                usage = Path(cwd) / argv[argv.index("--usage-file") + 1]
                usage.write_text(json.dumps({"completed": True, "provider": "custom", "model": "local"}),
                                 encoding="utf-8")
            if self.output is not None:
                stdout = self.output(argv) or ""
        return subprocess.CompletedProcess(list(argv), code, stdout, "")

    def rows(self) -> list[tuple[str, str, str, str]]:
        rows = []
        for line in (self.out / "summary.md").read_text(encoding="utf-8").splitlines()[4:]:
            recipe, step, passed, reasons, _ = [cell.strip() for cell in line.strip("|").split("|")]
            rows.append((recipe, step, passed, reasons))
        return rows

    def row(self, recipe: str, step: str) -> tuple[str, str]:
        found = [(passed, reasons) for row_recipe, row_step, passed, reasons in self.rows()
                 if (row_recipe, row_step) == (recipe, step)]
        self.assertEqual(len(found), 1, self.rows())
        return found[0]

    def argvs(self) -> list[tuple[str, ...]]:
        return [argv for argv, _, _ in self.calls]

    def work_root(self) -> Path:
        roots = list((self.root / "tmp").iterdir())
        self.assertEqual(len(roots), 1, roots)
        return roots[0]

    def test_step_order(self):
        self.main()
        catalog = load_catalog(REPO / "recipes", None)
        planned = e2e.plan_steps(catalog, RUNNER_MATRIX, round_trip=False, keep_running=False)
        work = str(self.work_root())
        expected = [tuple(part.replace(e2e.WORK, f"{work}/{step.cwd}") for part in step.argv)
                    for step in planned if step.argv]
        self.assertEqual(self.argvs()[:4], [("hermes", "--version"), ("pi", "--version"), ("opencode", "--version"),
                                            ("nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader")])
        self.assertEqual(self.argvs()[4:], expected)
        steps = [step for _, step, _, _ in self.rows()]
        self.assertEqual(steps[:3], ["up", "gateway", "status"])
        self.assertEqual(steps[-1], "down")

    def test_version_calls_leave_no_files_in_the_repository(self):
        self.assertFalse((REPO / "hello.py").exists())
        self.main()
        self.assertIn((("hermes", "--version"), REPO), [(argv, cwd) for argv, cwd, _ in self.calls])
        self.assertFalse((REPO / "hello.py").exists())

    def test_every_subprocess_gets_the_driver_env(self):
        def inject(argv):
            os.environ["INJECTED_AFTER_START"] = "1"

        self.on_call = inject
        self.main()
        self.assertGreater(len(self.calls), 5)
        wanted = e2e.driver_env(self.environ)
        self.assertNotIn("LLMCTL_SET_DEFAULT", wanted)
        for argv, cwd, env in self.calls:
            with self.subTest(argv=argv):
                self.assertEqual(env, dict(wanted, PWD=str(cwd.resolve())))
        self.assertNotIn("LLMCTL_SET_DEFAULT", os.environ)

    def test_every_subprocess_gets_its_own_folder_as_pwd(self):
        # OpenCode takes its working directory from PWD, not from the process cwd
        self.environ["PWD"] = str(REPO)
        with unittest.mock.patch.dict("os.environ", {"PWD": str(REPO)}):
            self.main()
        work = self.work_root()
        sessions = [(argv, cwd, env) for argv, cwd, env in self.calls
                    if argv[0] in ("opencode", "hermes", "pi") and "--version" not in argv and "config" not in argv]
        llmctl = [(argv, cwd, env) for argv, cwd, env in self.calls if argv[0] == "./llmctl"]
        self.assertTrue(sessions)
        self.assertTrue(llmctl)
        for argv, cwd, env in sessions:
            with self.subTest(argv=argv):
                self.assertEqual(cwd.parent, work)
                self.assertEqual(env["PWD"], str(cwd.resolve()))
        for argv, cwd, env in llmctl + [call for call in self.calls if "--version" in call[0]]:
            with self.subTest(argv=argv):
                self.assertEqual(cwd, REPO)
                self.assertEqual(env["PWD"], str(REPO))

    def test_relative_out_and_private_work_folders(self):
        self.assertEqual(self.main(), 1)  # the fakes print no status lines and serve no probes
        self.assertTrue((self.out / "summary.md").is_file())
        self.assertEqual(self.row("01-qwen3.6-35b-a3b-llamacpp", "hermes-z"), ("Yes", ""))
        work = self.work_root()
        self.assertEqual(self.mkdtemp_calls, [{"prefix": "llmctl-e2e-"}])
        self.assertTrue(work.name.startswith("llmctl-e2e-"))
        self.assertEqual(stat.S_IMODE(work.stat().st_mode), 0o700)
        self.assertEqual(self.stdout.getvalue().splitlines()[0],
                         f"records in {self.out}; work folders in {work}, left in place after the run")
        sessions = [(argv, cwd) for argv, cwd, _ in self.calls
                    if argv[0] in ("opencode", "hermes", "pi") and argv[1:2] != ("--version",)]
        self.assertTrue(sessions)
        for argv, cwd in sessions:
            with self.subTest(argv=argv):
                self.assertTrue(cwd.is_absolute())
                self.assertTrue(cwd.is_relative_to(work), cwd)
                self.assertFalse(cwd.is_relative_to(self.out), cwd)
                self.assertFalse(cwd.is_relative_to(REPO), cwd)
        self.assertTrue((work / "01-qwen3.6-35b-a3b-llamacpp-hermes-z" / "usage.json").is_file())
        records = {path.name: path.read_text(encoding="utf-8") for path in (self.out / "records").iterdir()}
        hermes_z = [text for name, text in records.items() if name.endswith("hermes-z")]
        self.assertEqual(len(hermes_z), 1)
        record = json.loads(hermes_z[0])
        self.assertEqual(record["cwd"], "$WORK/01-qwen3.6-35b-a3b-llamacpp-hermes-z")
        self.assertIn("--usage-file $WORK/01-qwen3.6-35b-a3b-llamacpp-hermes-z/usage.json", record["command"])
        for name, text in records.items():
            self.assertNotIn(str(work), text, name)
            self.assertNotIn(str(self.home), text, name)

    def test_failed_up_skips_the_rest_of_its_recipe(self):
        self.failing_ups = {"k2-horizon-7b"}
        self.assertEqual(self.main(matrix=TWO_RECIPES), 1)
        argvs = self.argvs()
        first = argvs.index(("./llmctl", "up", "k2-horizon-7b", "--replace"))
        self.assertEqual(argvs[first + 1], ("./llmctl", "up", "qwen3.6-35b-a3b", "--replace"))
        self.assertEqual([step for recipe, step, _, _ in self.rows() if recipe.startswith("01-")], ["up"])

    def test_gateway_check_after_the_up_that_created_the_profile(self):
        self.failing_ups = {"k2-horizon-7b"}
        self.gateway_after_up = GATEWAY_LLMCTL
        self.main(matrix=TWO_RECIPES)
        self.assertEqual([step for recipe, step, _, _ in self.rows() if recipe.startswith("01-")], ["up"])
        passed, reasons = self.row("02-qwen3.6-35b-a3b-llamacpp", "gateway")
        self.assertEqual(passed, "No")
        self.assertIn("Hermes' gateway serves the profile llmctl ($HOME/.hermes/gateway_state.json)", reasons)
        self.assertEqual(self.host.sleeps, [35])
        steps = [step for recipe, step, _, _ in self.rows() if recipe.startswith("02-")]
        self.assertEqual(steps[:2], ["up", "gateway"])

    def test_gateway_check_with_a_profile_from_before_the_run(self):
        self.write_profile("gpt-oss-20b", "vllm")
        self.failing_ups = {"k2-horizon-7b"}
        self.main(matrix=TWO_RECIPES)
        self.assertEqual(self.row("02-qwen3.6-35b-a3b-llamacpp", "gateway"), ("Yes", ""))
        self.assertEqual(self.host.sleeps, [])

    def test_gateway_check_that_never_ran_fails(self):
        self.ups_create_profile = False
        self.main()
        passed, reasons = self.row("-", "gateway")
        self.assertEqual(passed, "No")
        self.assertIn("no up created the profile llmctl", reasons)

    def round_trip_home(self):
        config = self.home / ".config" / "opencode"
        config.mkdir(parents=True)
        (config / "opencode.json").write_text(json.dumps({"model": "vllm-docker/gpt-oss-20b"}), encoding="utf-8")
        (self.home / ".pi" / "agent").mkdir(parents=True, exist_ok=True)

    def test_failed_round_trip_up_returns_and_restores(self):
        self.round_trip_home()
        self.failing_ups = {"qwen3.6-35b-a3b"}
        self.main("--set-default-round-trip", matrix=(("k2-horizon-7b", "llamacpp"),))
        argvs = self.argvs()
        up = argvs.index(("./llmctl", "up", "qwen3.6-35b-a3b", "--set-default"))
        self.assertEqual(argvs[up + 1:], [("./llmctl", "down")])
        steps = [step for recipe, step, _, _ in self.rows() if recipe == e2e.ROUND_TRIP]
        self.assertEqual(steps, ["up", "restore"])
        self.assertEqual(self.row(e2e.ROUND_TRIP, "restore"), ("Yes", ""))

    def test_round_trip_runs_unpinned_sessions_after_its_up(self):
        self.round_trip_home()
        self.main("--set-default-round-trip", matrix=(("k2-horizon-7b", "llamacpp"),))
        argvs = self.argvs()
        up = argvs.index(("./llmctl", "up", "qwen3.6-35b-a3b", "--set-default"))
        unpinned = [argv[0] for argv in argvs[up + 1:] if argv[0] in ("opencode", "hermes", "pi")]
        self.assertEqual(unpinned, ["opencode", "hermes", "pi"])
        steps = [step for recipe, step, _, _ in self.rows() if recipe == e2e.ROUND_TRIP]
        self.assertEqual(steps, ["up", "opencode", "hermes-z", "pi", "restore"])
        self.assertIn("Pi answered from no model, not llmctl/local", self.row(e2e.ROUND_TRIP, "pi")[1])

    def test_round_trip_check_error_has_its_own_row(self):
        self.round_trip_home()
        with unittest.mock.patch.object(e2e._Run, "logs_since", side_effect=OSError(28, "No space left on device")):
            self.main("--set-default-round-trip", matrix=(("k2-horizon-7b", "llamacpp"),))
        self.assertEqual(self.row(e2e.ROUND_TRIP, "check"), ("No", "OSError: [Errno 28] No space left on device"))
        self.assertEqual(self.row(e2e.ROUND_TRIP, "restore"), ("Yes", ""))

    def assert_refused(self, code, message):
        self.assertEqual(code, 1)
        self.assertEqual(self.stderr.getvalue(), f"error: {message}\n")
        self.assertEqual(self.calls, [])
        self.assertEqual(self.mkdtemp_calls, [])
        self.assertFalse(self.out.exists())

    def test_refuses_while_an_llmctl_server_runs(self):
        self.managed = [managed_line("llamacpp", "running", "qwen3.6-35b-a3b")]
        self.assert_refused(self.main(), "an llmctl model server is running (qwen3.6-35b-a3b on llamacpp); stop it "
                                         "with ./llmctl down, then run the driver again")
        self.assertEqual(self.host.calls, [["docker", "ps", "-a", "--filter", "label=io.llmctl.managed=true",
                                            "--format", "{{.Names}}|{{.ID}}|{{.State}}|{{.Labels}}"]])

    def test_a_stopped_llmctl_container_is_not_refused(self):
        self.managed = [managed_line("llamacpp", "exited", "qwen3.6-35b-a3b")]
        self.main()
        self.assertTrue(self.calls)

    def test_round_trip_refuses_a_hermes_home_profile(self):
        self.round_trip_home()
        os.environ["HERMES_HOME"] = str(self.home / ".hermes" / "profiles" / "work")
        self.assert_refused(self.main("--set-default-round-trip"),
                            "the round trip runs hermes without -p, but HERMES_HOME names the profile work in this "
                            "shell, so that session would use another model; unset HERMES_HOME, or run without "
                            "--set-default-round-trip")

    def test_round_trip_refuses_enabled_models_that_skip_llmctl(self):
        self.round_trip_home()
        agent = self.home / ".pi" / "agent"
        (agent / "settings.json").write_text(json.dumps({"enabledModels": ["anthropic/claude-x"]}), encoding="utf-8")
        self.assert_refused(self.main("--set-default-round-trip"),
                            "the round trip runs pi without --model, but Pi's enabledModels would start it with "
                            "anthropic/claude-x instead of llmctl/local; add llmctl/local to enabledModels or clear "
                            "it, or run without --set-default-round-trip")

    def session_output(self, kind: str, lines: list[str]):
        """An output for one session step: hermes-chat, pi or opencode."""
        def output(argv):
            found = ("hermes-chat" if argv[0] == "hermes" and "chat" in argv else argv[0])
            return "".join(line + "\n" for line in lines) if found == kind and "--version" not in argv else None
        self.output = output

    def record(self, step: str) -> dict:
        found = [path for path in (self.out / "records").iterdir()
                 if path.name.endswith(f"01-qwen3.6-35b-a3b-llamacpp-{step}")]
        self.assertEqual(len(found), 1, found)
        return json.loads(found[0].read_text(encoding="utf-8"))

    def assert_withheld(self, step: str, *private: str) -> None:
        self.assertEqual(self.row("01-qwen3.6-35b-a3b-llamacpp", step), ("No", WITHHELD))
        record = self.record(step)
        marker = "[withheld: the output named a private value after redaction]"
        self.assertEqual((record["stdout"], record["stderr"]), (marker, marker))
        text = json.dumps(record)
        summary = (self.out / "summary.md").read_text(encoding="utf-8")
        for value in private:
            self.assertNotIn(value, text)
            self.assertNotIn(value, summary)

    def test_streaming_text_events_are_dropped_from_the_hermes_record(self):
        lines = fixture_lines("hermes-stream.jsonl")
        self.session_output("hermes-chat", lines)
        self.main()
        self.assertEqual(self.row("01-qwen3.6-35b-a3b-llamacpp", "hermes-chat"), ("Yes", ""))
        record = self.record("hermes-chat")
        self.assertEqual(record["dropped_events"], {"text": 23})
        kept = [line for line in lines if json.loads(line)["type"] != "text"]
        self.assertEqual(record["stdout"], "".join(line + "\n" for line in kept))
        self.assertIn('"type": "result"', record["stdout"])

    def test_message_update_events_are_dropped_from_the_pi_record(self):
        lines = fixture_lines("pi-events.jsonl")
        update = {"type": "message_update", "message": {"role": "assistant", "provider": "llmctl", "model": "local"},
                  "assistantMessageEvent": {"type": "text_delta", "delta": "Hel"}}
        lines = lines[:8] + [json.dumps(update)] * 3 + lines[8:] + ["not json at the end"]
        self.session_output("pi", lines)
        self.main()
        self.assertEqual(self.row("01-qwen3.6-35b-a3b-llamacpp", "pi"),
                         ("No", f"line {len(lines)} is not JSON: not json at the end"))  # it read the full output
        record = self.record("pi")
        self.assertEqual(record["dropped_events"], {"message_update": 3})
        self.assertNotIn("message_update", record["stdout"])
        self.assertTrue(record["stdout"].endswith("not json at the end\n"))
        self.assertEqual(record["stdout"].count("\n"), len(lines) - 3)

    def test_a_home_path_split_over_kept_events_is_withheld(self):
        lines = fixture_lines("hermes-stream.jsonl")
        home = str(self.home)
        pieces = [home[:3], home[3:9], home[9:]]
        split = [json.dumps({"type": "progress", "detail": {"note": piece}}) for piece in pieces]
        self.session_output("hermes-chat", lines[:-1] + split + lines[-1:])
        self.assertEqual(self.main(), 1)
        self.assert_withheld("hermes-chat", home)

    def test_a_user_name_split_over_three_events_is_withheld(self):
        os.environ["USER"] = "alicebob"
        lines = fixture_lines("pi-events.jsonl")
        split = [json.dumps({"type": "tool_execution_update", "partial": piece}) for piece in ("ali", "ceb", "ob")]
        self.session_output("pi", lines[:-1] + split + lines[-1:])
        self.main()
        self.assert_withheld("pi", "alicebob")

    def split_after_pi(self, value) -> None:
        """Pi's output followed by two non-JSON lines that hold value(), cut in half."""
        lines = fixture_lines("pi-events.jsonl")

        def output(argv):
            if argv[0] != "pi" or "--version" in argv:
                return None
            wanted = value()
            half = len(wanted) // 2
            return "".join(line + "\n" for line in lines) + wanted[:half] + "\n" + wanted[half:] + "\n"
        self.output = output

    def test_a_host_name_split_over_lines_is_withheld(self):
        self.split_after_pi(lambda: HOST_NAME)
        self.main()
        self.assert_withheld("pi", HOST_NAME)

    def test_the_work_root_split_over_lines_is_withheld(self):
        self.split_after_pi(lambda: str(self.work_root()))
        self.main()
        self.assert_withheld("pi", self.work_root().name)

    def test_a_record_without_private_values_is_only_redacted(self):
        lines = fixture_lines("opencode-events.jsonl")

        def output(argv):
            if argv[0] != "opencode" or "--version" in argv:
                return None
            work = str(self.work_root())
            return "".join(line + "\n" for line in lines) + json.dumps({"type": "note", "path": f"{work}/x"}) + "\n"
        self.output = output
        self.main()
        record = self.record("opencode")
        self.assertNotIn("dropped_events", record)
        expected = "".join(line + "\n" for line in lines) + json.dumps({"type": "note", "path": "$WORK/x"}) + "\n"
        self.assertEqual(record["stdout"], expected)
        self.assertEqual(record["stderr"], "")
        self.assertEqual(self.row("01-qwen3.6-35b-a3b-llamacpp", "opencode"), ("Yes", ""))

if __name__ == "__main__":
    unittest.main()
