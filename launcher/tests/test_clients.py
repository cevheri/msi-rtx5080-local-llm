import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from launcher import clients, dsh_config, hermes_config, kimi_config, pi_config
from launcher.catalog import load_catalog
from launcher.errors import ClientError, UsageError
from launcher.host import Result
from launcher.tests.client_fixture import LIST_HEADER, ClientFixture
from launcher.tests.fakes import REPO, FakeHost, fail, ok
from launcher.tests.regen_golden import target_for
from launcher.tests.test_dsh_config import dsh_fixture
from launcher.tests.test_kimi_config import DESIGN_OWNER_FIRST_WRITE, EMPTY_MESSAGE, OWNER_TEXT

ALL_BINARIES = {"opencode": "/usr/bin/opencode", "hermes": "/usr/bin/hermes", "pi": "/usr/bin/pi",
                "dsh": "/usr/bin/dsh", "kimi": "/usr/bin/kimi"}


class DetectionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        for folder in (".config/opencode", ".hermes", ".pi/agent", ".dsh/profiles", ".kimi-code"):
            (self.home / folder).mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def host(self, installed, **env):
        host = FakeHost(env={"HOME": str(self.home), "PATH": "/usr/bin", **env})
        host.installed = dict(installed)
        return host

    def by_name(self, states):
        return {state.name: state for state in states}

    def test_all_installed(self):
        host = self.host(ALL_BINARIES)
        states = clients.client_states(host)
        self.assertEqual([s.name for s in states], ["opencode", "hermes", "pi", "dsh", "kimi"])
        self.assertEqual([s.active for s in states], [True, True, True, True, True])
        self.assertEqual([s.reason for s in states], [None, None, None, None, None])
        self.assertEqual([s.label for s in states], ["OpenCode", "Hermes", "Pi", "dsh", "Kimi"])
        self.assertEqual(self.by_name(states)["pi"].root, self.home / ".pi/agent")
        self.assertEqual(self.by_name(states)["dsh"].root, self.home / ".dsh")
        self.assertEqual(self.by_name(states)["kimi"].root, self.home / ".kimi-code")
        self.assertEqual(clients.clients_line(host, states), "clients: opencode, hermes, pi, dsh, kimi")

    def test_binary_missing_but_root_present(self):
        host = self.host({k: v for k, v in ALL_BINARIES.items() if k != "pi"})
        states = clients.client_states(host)
        pi = self.by_name(states)["pi"]
        self.assertTrue(pi.enabled)
        self.assertFalse(pi.installed)
        self.assertFalse(pi.active)
        self.assertEqual(pi.reason, "no pi on PATH")
        line = clients.clients_line(host, states)
        self.assertEqual(line, "clients: opencode, hermes, dsh, kimi (pi: not installed, no pi on PATH)")

    def test_root_missing(self):
        (self.home / ".hermes").rmdir()
        host = self.host(ALL_BINARIES)
        states = clients.client_states(host)
        hermes = self.by_name(states)["hermes"]
        self.assertFalse(hermes.installed)
        self.assertEqual(hermes.reason, f"no {self.home}/.hermes")
        self.assertEqual(clients.clients_line(host, states),
                         f"clients: opencode, pi, dsh, kimi (hermes: not installed, no {self.home}/.hermes)")

    def test_llmctl_clients_subset(self):
        host = self.host(ALL_BINARIES, LLMCTL_CLIENTS=" pi , opencode")
        with mock.patch("launcher.clients.hermes_config.hermes_root", side_effect=AssertionError):
            states = clients.client_states(host)
        hermes = self.by_name(states)["hermes"]
        self.assertFalse(hermes.enabled)
        self.assertFalse(hermes.active)
        self.assertEqual(hermes.reason, "left out by LLMCTL_CLIENTS")
        self.assertEqual(clients.clients_line(host, states),
                         "clients: opencode, pi (hermes: left out by LLMCTL_CLIENTS) "
                         "(dsh: left out by LLMCTL_CLIENTS) (kimi: left out by LLMCTL_CLIENTS)")

    def test_disabled_client_binary_is_not_looked_up(self):
        host = self.host(ALL_BINARIES, LLMCTL_CLIENTS="opencode")
        looked_up = []
        host.which = lambda name: looked_up.append(name) or ALL_BINARIES.get(name)
        clients.client_states(host)
        self.assertEqual(looked_up, ["opencode"])

    def test_llmctl_clients_empty(self):
        host = self.host(ALL_BINARIES, LLMCTL_CLIENTS="")
        states = clients.client_states(host)
        self.assertEqual([s.active for s in states], [False, False, False, False, False])
        self.assertEqual(clients.clients_line(host, states), "clients: none (LLMCTL_CLIENTS is empty)")

    def test_llmctl_clients_unknown_name(self):
        host = self.host(ALL_BINARIES, LLMCTL_CLIENTS="opencode,claude")
        with self.assertRaises(UsageError) as caught:
            clients.client_states(host)
        self.assertEqual(caught.exception.exit_code, 1)
        self.assertEqual(str(caught.exception),
                         "LLMCTL_CLIENTS: unknown client 'claude'; use opencode, hermes, pi, dsh or kimi, comma-separated, "
                         "or an empty value for none")

    def test_listed_client_missing_refuses(self):
        host = self.host({}, LLMCTL_CLIENTS="pi")
        states = clients.client_states(host)
        with self.assertRaises(ClientError) as caught:
            clients.require_listed(host, states)
        self.assertEqual(caught.exception.exit_code, 3)
        self.assertEqual(str(caught.exception), "LLMCTL_CLIENTS names pi, but Pi is not installed here: no pi on PATH")

    def test_unlisted_missing_client_is_fine(self):
        host = self.host({})
        states = clients.client_states(host)
        self.assertIsNone(clients.require_listed(host, states))
        self.assertEqual(clients.clients_line(host, states),
                         "clients: none (opencode: not installed, no opencode on PATH)"
                         " (hermes: not installed, no hermes on PATH) (pi: not installed, no pi on PATH)"
                         " (dsh: not installed, no dsh on PATH) (kimi: not installed, no kimi on PATH)")

    def test_dsh_needs_its_profiles_folder(self):
        (self.home / ".dsh/profiles").rmdir()
        host = self.host(ALL_BINARIES)
        states = clients.client_states(host)
        dsh = self.by_name(states)["dsh"]
        self.assertFalse(dsh.installed)
        self.assertEqual(dsh.reason, f"no {self.home}/.dsh/profiles")
        self.assertEqual(clients.clients_line(host, states),
                         f"clients: opencode, hermes, pi, kimi (dsh: not installed, no {self.home}/.dsh/profiles)")

    def test_dsh_home_comes_from_the_mapping(self):
        (self.home / "dh/profiles").mkdir(parents=True)
        host = self.host(ALL_BINARIES, DSH_HOME=str(self.home / "dh"))
        dsh = self.by_name(clients.client_states(host))["dsh"]
        self.assertTrue(dsh.active)
        self.assertEqual(dsh.root, Path(self.home / "dh"))

    def test_relative_dsh_home_is_a_usage_error(self):
        host = self.host(ALL_BINARIES, DSH_HOME="rel")
        with self.assertRaises(UsageError) as caught:
            clients.client_states(host)
        self.assertEqual(caught.exception.exit_code, 1)
        self.assertEqual(str(caught.exception), "DSH_HOME is relative (rel), and dsh resolves it against each "
                                                "command's working folder; set it to an absolute path, ~ or ~/...")
        host = self.host(ALL_BINARIES, DSH_HOME="rel", LLMCTL_CLIENTS="pi")
        self.assertEqual([s.active for s in clients.client_states(host)], [False, False, True, False, False])

    def test_listed_dsh_missing_refuses(self):
        host = self.host({}, LLMCTL_CLIENTS="dsh")
        states = clients.client_states(host)
        with self.assertRaises(ClientError) as caught:
            clients.require_listed(host, states)
        self.assertEqual(caught.exception.exit_code, 3)
        self.assertEqual(str(caught.exception),
                         "LLMCTL_CLIENTS names dsh, but dsh is not installed here: no dsh on PATH")

    def test_kimi_needs_its_home_folder(self):
        (self.home / ".kimi-code").rmdir()
        host = self.host(ALL_BINARIES)
        states = clients.client_states(host)
        kimi = self.by_name(states)["kimi"]
        self.assertFalse(kimi.installed)
        self.assertEqual(kimi.reason, f"no {self.home}/.kimi-code")
        self.assertEqual(clients.clients_line(host, states),
                         f"clients: opencode, hermes, pi, dsh (kimi: not installed, no {self.home}/.kimi-code)")

    def test_kimi_home_comes_from_the_mapping(self):
        (self.home / "kh").mkdir()
        host = self.host(ALL_BINARIES, KIMI_CODE_HOME=str(self.home / "kh"))
        kimi = self.by_name(clients.client_states(host))["kimi"]
        self.assertTrue(kimi.active)
        self.assertEqual(kimi.root, self.home / "kh")
        (self.home / "kh").rmdir()
        kimi = self.by_name(clients.client_states(host))["kimi"]
        self.assertEqual(kimi.reason, f"no {self.home}/kh")

    def test_empty_and_relative_kimi_home_are_usage_errors(self):
        relative = ("KIMI_CODE_HOME is relative (rel), and Kimi uses it as given, against each command's working "
                    "folder; set it to an absolute path, or unset it for ~/.kimi-code")
        for value, message in (("", EMPTY_MESSAGE), ("rel", relative)):
            with self.subTest(value=value):
                host = self.host(ALL_BINARIES, KIMI_CODE_HOME=value)
                with self.assertRaises(UsageError) as caught:
                    clients.client_states(host)
                self.assertEqual(caught.exception.exit_code, 1)
                self.assertEqual(str(caught.exception), message)
                host = self.host(ALL_BINARIES, KIMI_CODE_HOME=value, LLMCTL_CLIENTS="pi")
                self.assertEqual([s.active for s in clients.client_states(host)], [False, False, True, False, False])

    def test_listed_kimi_missing_refuses(self):
        host = self.host({}, LLMCTL_CLIENTS="kimi")
        states = clients.client_states(host)
        with self.assertRaises(ClientError) as caught:
            clients.require_listed(host, states)
        self.assertEqual(caught.exception.exit_code, 3)
        self.assertEqual(str(caught.exception),
                         "LLMCTL_CLIENTS names kimi, but Kimi is not installed here: no kimi on PATH")

    def test_llmctl_providers(self):
        catalog = load_catalog(REPO / "recipes", None)
        self.assertEqual(clients.llmctl_providers(catalog), frozenset({"llamacpp-docker", "vllm-docker"}))


START_SETTINGS = {"lastChangelogVersion": "1.5.2", "defaultProvider": "llmctl", "defaultModel": "gpt-oss-20b"}


class PlanApplyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog(REPO / "recipes", None)
        cls.target = target_for("qwen3.6-35b-a3b", "llamacpp", REPO / "recipes")
        cls.old_target = target_for("gpt-oss-20b", "vllm", REPO / "recipes")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.host = FakeHost(env={"HOME": "/nonexistent", "PATH": "/usr/bin"})
        self.fx = ClientFixture(self.host, Path(self.tmp.name))
        self.fx.install("opencode", "hermes", "pi")
        self.fx.route_clients()
        self.fx.write_opencode("vllm-docker/gpt-oss-20b")
        self.fx.hermes_config.parent.mkdir(parents=True)
        self.fx.hermes_config.write_text(hermes_config.new_config_text(self.old_target), encoding="utf-8")
        self.fx.models_path.write_text(pi_config.new_models_text(self.old_target), encoding="utf-8")
        self.write_settings(START_SETTINGS)

    def write_settings(self, doc):
        self.fx.settings_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

    def plan(self, *, set_default=False, check=True):
        return clients.plan(self.host, self.catalog, self.target, set_default=set_default, check=check)

    def apply(self, *, set_default=False):
        return clients.apply(self.host, self.catalog, self.target, set_default=set_default)

    def opencode_model(self):
        return json.loads(self.fx.opencode_path.read_text(encoding="utf-8"))["model"]

    def assert_opencode_written(self):
        self.assertEqual(self.opencode_model(), "llamacpp-docker/qwen3.6-35b-a3b")

    def assert_hermes_written(self):
        self.assertEqual(self.fx.hermes_config.read_text(encoding="utf-8"), hermes_config.new_config_text(self.target))

    def assert_models_written(self):
        self.assertEqual(self.fx.models_path.read_text(encoding="utf-8"), pi_config.new_models_text(self.target))

    def test_plan_without_check_runs_nothing(self):
        before = self.fx.snapshot()
        result = self.plan(check=False)
        self.assertEqual(self.host.calls, [])
        self.assertEqual(self.fx.snapshot(), before)
        self.assertEqual([change.path for change in result.changes],
                         [self.fx.opencode_path, self.fx.hermes_config, self.fx.models_path, self.fx.settings_path])
        self.assertEqual([state.name for state in result.states], ["opencode", "hermes", "pi", "dsh", "kimi"])
        self.assertFalse(result.hermes_use)
        self.assertTrue(result.pi_active)

    def test_plan_refusals_propagate(self):
        before_text = self.fx.models_path.read_text(encoding="utf-8")

        def no_markers():
            self.fx.hermes_config.write_text("foo: 1\n", encoding="utf-8")

        def comment_in_models():
            self.fx.models_path.write_text("// mine\n" + before_text, encoding="utf-8")

        def no_row():
            self.host.on(["pi", "--offline", "--list-models", "llmctl"], ok(LIST_HEADER))

        cases = [("no markers", no_markers), ("comment in models.json", comment_in_models), ("no pi row", no_row)]
        for name, breaker in cases:
            with self.subTest(name):
                self.setUp()
                breaker()
                before = self.fx.snapshot()
                with self.assertRaises(ClientError):
                    self.plan(check=True)
                self.assertEqual(self.fx.snapshot(), before)

    def test_dry_run_lines(self):
        oc, hermes = self.fx.opencode_path, self.fx.hermes_config
        models, settings = self.fx.models_path, self.fx.settings_path
        lines = clients.dry_run_lines(self.plan(set_default=True, check=False))
        self.assertEqual(lines, [
            f"would back up {oc} to {oc}.llmctl-backup",
            f"would set OpenCode's default model: vllm-docker/gpt-oss-20b -> llamacpp-docker/qwen3.6-35b-a3b ({oc})",
            f"would write {hermes}: llmctl block for qwen3.6-35b-a3b on llama.cpp (was gpt-oss-20b on vLLM)",
            f"would write {models}: provider llmctl for qwen3.6-35b-a3b on llama.cpp, model local "
            "(was gpt-oss-20b on vLLM)",
            f"would back up {settings} to {settings}.llmctl-backup",
            f"would write {settings}: modelThinkingLevels llmctl/local off (was absent); "
            "defaultModel local (was gpt-oss-20b)",
            "would run hermes profile use llmctl",
            "dry run: nothing was written, and Pi has not checked the entry",
        ])

    def test_dry_run_skips_an_existing_backup_and_ends_without_pi(self):
        oc = self.fx.opencode_path
        Path(str(oc) + ".llmctl-backup").write_text("{}\n", encoding="utf-8")
        self.host.env["LLMCTL_CLIENTS"] = "opencode"
        lines = clients.dry_run_lines(self.plan(check=False))
        self.assertEqual(lines, [
            f"would set OpenCode's default model: vllm-docker/gpt-oss-20b -> llamacpp-docker/qwen3.6-35b-a3b ({oc})",
            "dry run: nothing was written",
        ])

    def test_apply_writes_all_and_orders_lines(self):
        oc, hermes = self.fx.opencode_path, self.fx.hermes_config
        models, settings = self.fx.models_path, self.fx.settings_path
        result = self.apply()
        self.assertEqual(result.failures, ())
        self.assertEqual(result.lines, (
            f"backed up {oc} to {oc}.llmctl-backup before the first change",
            f"backed up {settings} to {settings}.llmctl-backup before the first change",
            f"wrote {oc}: model llamacpp-docker/qwen3.6-35b-a3b (was vllm-docker/gpt-oss-20b); "
            "OpenCode reloaded its configuration",
            f"wrote {hermes}: llmctl block for qwen3.6-35b-a3b on llama.cpp (was gpt-oss-20b on vLLM)",
            f"wrote {models}: provider llmctl for qwen3.6-35b-a3b on llama.cpp, model local (was gpt-oss-20b on vLLM)",
            f"wrote {settings}: modelThinkingLevels llmctl/local off (was absent); defaultModel local (was gpt-oss-20b)",
        ))
        self.assertEqual(result.changed, frozenset({"hermes", "pi"}))
        self.assertEqual(self.host.calls.count(["opencode", "reload"]), 1)
        self.assertFalse([call for call in self.host.calls if call[0] == "hermes"])
        self.assert_opencode_written()
        self.assert_hermes_written()
        self.assert_models_written()
        self.assertEqual(json.loads(settings.read_text(encoding="utf-8")),
                         {**START_SETTINGS, "defaultModel": "local", "modelThinkingLevels": {"llmctl/local": "off"}})

    def test_crlf_backup_is_byte_exact(self):
        original = (json.dumps(START_SETTINGS, indent=2) + "\n").replace("\n", "\r\n").encode("utf-8")
        self.fx.settings_path.write_bytes(original)
        result = self.apply()
        self.assertEqual(result.failures, ())
        self.assertNotEqual(self.fx.settings_path.read_bytes(), original)
        self.assertEqual(Path(str(self.fx.settings_path) + ".llmctl-backup").read_bytes(), original)

    def test_apply_rereads_files(self):
        self.plan()
        self.write_settings({**START_SETTINGS, "defaultThinkingLevel": "high"})
        result = self.apply()
        self.assertEqual(result.failures, ())
        doc = json.loads(self.fx.settings_path.read_text(encoding="utf-8"))
        self.assertEqual(doc["defaultThinkingLevel"], "high")
        self.assertEqual(doc["defaultModel"], "local")

    def test_one_client_failure_keeps_the_others(self):
        directory = self.fx.hermes_config.parent
        os.chmod(directory, 0o500)
        self.addCleanup(os.chmod, directory, 0o700)
        result = self.apply()
        self.assertEqual(len(result.failures), 1)
        self.assertTrue(result.failures[0].startswith(f"{self.fx.hermes_config}: "), result.failures[0])
        self.assertEqual(self.fx.hermes_config.read_text(encoding="utf-8"),
                         hermes_config.new_config_text(self.old_target))
        self.assert_opencode_written()
        self.assert_models_written()
        self.assertEqual(json.loads(self.fx.settings_path.read_text(encoding="utf-8"))["defaultModel"], "local")
        self.assertNotIn("hermes", result.changed)
        self.assertEqual(result.failed, frozenset({"hermes"}))

    def test_pi_lock_failure_keeps_other_clients(self):
        cases = [("both present", False, False), ("models.json absent", True, False),
                 ("settings.json absent", False, True), ("both absent", True, True)]
        for name, drop_models, drop_settings in cases:
            with self.subTest(name):
                self.setUp()
                if drop_models:
                    self.fx.models_path.unlink()
                if drop_settings:
                    self.fx.settings_path.unlink()
                models_before = self.fx.models_path.read_bytes() if not drop_models else None
                settings_before = self.fx.settings_path.read_bytes() if not drop_settings else None
                lock = Path(str(self.fx.settings_path) + ".lock")
                lock.mkdir()
                result = self.apply()
                self.assertEqual(len(result.failures), 1)
                self.assertIn(f"Pi holds its settings lock {lock}", result.failures[0])
                self.assert_opencode_written()
                self.assert_hermes_written()
                if drop_models:
                    self.assertFalse(os.path.lexists(self.fx.models_path))
                else:
                    self.assertEqual(self.fx.models_path.read_bytes(), models_before)
                if drop_settings:
                    self.assertFalse(os.path.lexists(self.fx.settings_path))
                else:
                    self.assertEqual(self.fx.settings_path.read_bytes(), settings_before)
                self.assertTrue(lock.is_dir())
                self.assertNotIn("pi", result.changed)

    def test_pi_lock_failure_keeps_pi_consistent(self):
        self.fx.models_path.write_text(pi_config.new_models_text(self.old_target), encoding="utf-8")
        self.write_settings({"defaultProvider": "llmctl", "defaultModel": "gpt-oss-20b"})
        models_before = self.fx.models_path.read_bytes()
        settings_before = self.fx.settings_path.read_bytes()
        Path(str(self.fx.settings_path) + ".lock").mkdir()
        result = self.apply(set_default=False)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(self.fx.models_path.read_bytes(), models_before)
        self.assertEqual(self.fx.settings_path.read_bytes(), settings_before)

    def test_models_json_is_written_under_the_settings_lock(self):
        lock = os.path.realpath(self.fx.settings_path) + ".lock"
        real_write = clients.write_atomic
        seen = []

        def spy(path, text, **kwargs):
            if Path(path) == self.fx.models_path:
                seen.append(os.path.isdir(lock))
            return real_write(path, text, **kwargs)

        with mock.patch("launcher.clients.write_atomic", side_effect=spy):
            result = self.apply()
        self.assertEqual(result.failures, ())
        self.assertEqual(result.failed, frozenset())
        self.assertEqual(seen, [True])

    def test_pi_models_failure_skips_settings(self):
        settings_before = self.fx.settings_path.read_bytes()
        models_before = self.fx.models_path.read_bytes()
        self.host.on(["pi", "--offline", "--list-models", "llmctl"], ok(LIST_HEADER))
        result = self.apply()
        self.assertEqual(result.failures,
                         ("Pi rejects the llmctl entry for qwen3.6-35b-a3b: no llmctl row in pi --list-models",))
        self.assertEqual(self.fx.settings_path.read_bytes(), settings_before)
        self.assertEqual(self.fx.models_path.read_bytes(), models_before)
        self.assertNotIn("pi", result.changed)
        self.assert_opencode_written()
        self.assert_hermes_written()

    def test_pi_settings_refusal_leaves_models_json(self):
        self.fx.models_path.write_text(pi_config.new_models_text(self.old_target), encoding="utf-8")
        self.write_settings({"defaultProvider": "llmctl", "defaultModel": "gpt-oss-20b", "modelThinkingLevels": []})
        models_before = self.fx.models_path.read_bytes()
        settings_before = self.fx.settings_path.read_bytes()
        result = self.apply()
        self.assertEqual(len(result.failures), 1, result.failures)
        self.assertEqual(self.fx.models_path.read_bytes(), models_before)
        self.assertEqual(self.fx.settings_path.read_bytes(), settings_before)
        self.assertNotIn("pi", result.changed)

    def test_pi_settings_write_failure_puts_models_json_back(self):
        for name, drop_models in (("models.json present", False), ("models.json absent", True)):
            with self.subTest(name):
                self.setUp()
                if drop_models:
                    self.fx.models_path.unlink()
                models_before = None if drop_models else self.fx.models_path.read_bytes()
                settings_before = self.fx.settings_path.read_bytes()
                real_write = clients.write_atomic
                attempts = []

                def failing(path, text, **kwargs):
                    if Path(path) == self.fx.settings_path:
                        attempts.append(path)
                        raise OSError(28, "No space left on device")
                    return real_write(path, text, **kwargs)

                with mock.patch("launcher.clients.write_atomic", side_effect=failing):
                    result = self.apply()
                self.assertEqual(len(attempts), 1)
                self.assertEqual(len(result.failures), 1, result.failures)
                message = result.failures[0]
                self.assertIn(f"{self.fx.settings_path}: No space left on device", message)
                self.assertIn(f"{self.fx.models_path} was put back", message)
                if drop_models:
                    self.assertFalse(os.path.lexists(self.fx.models_path))
                else:
                    self.assertEqual(self.fx.models_path.read_bytes(), models_before)
                self.assertEqual(self.fx.settings_path.read_bytes(), settings_before)
                self.assertNotIn("pi", result.changed)
                self.assertEqual(result.failed, frozenset({"pi"}))
                self.assert_opencode_written()
                self.assert_hermes_written()

    def test_pi_rollback_failure_names_both_errors(self):
        real_write = clients.write_atomic
        models_writes = []

        def failing(path, text, **kwargs):
            if Path(path) == self.fx.settings_path:
                raise OSError(28, "No space left on device")
            if Path(path) == self.fx.models_path:
                models_writes.append(text)
                if len(models_writes) > 1:
                    raise OSError(5, "Input/output error")
            return real_write(path, text, **kwargs)

        with mock.patch("launcher.clients.write_atomic", side_effect=failing):
            result = self.apply()
        self.assertEqual(len(result.failures), 1, result.failures)
        message = result.failures[0]
        self.assertIn(f"{self.fx.settings_path}: No space left on device", message)
        self.assertIn("Input/output error", message)
        self.assertIn("disagree", message)
        self.assertNotIn("was put back", message)

    def test_pi_warning_goes_to_after_lines(self):
        def warn(argv):
            directory = Path(self.host.env_extras[-1]["PI_CODING_AGENT_DIR"])
            served = json.loads((directory / "models.json").read_text())["providers"]["llmctl"]["models"][0]["id"]
            return Result(0, LIST_HEADER + f"llmctl    {served}  131.1K   16.4K    yes       no\n",
                          "errors loading models.json: other is bad\n")

        self.host.on(["pi", "--offline", "--list-models", "llmctl"], warn)
        result = self.apply()
        self.assertEqual(result.failures, ())
        self.assertEqual(result.lines[-1], f"warning: Pi reports problems in {self.fx.models_path} outside the llmctl "
                                           "entry: errors loading models.json: other is bad")

    def test_unchanged_models_are_still_validated(self):
        self.fx.models_path.write_text(pi_config.new_models_text(self.target), encoding="utf-8")
        result = self.apply()
        self.assertEqual(result.failures, ())
        self.assertEqual(self.host.calls.count(["pi", "--offline", "--list-models", "llmctl"]), 1)
        self.assertFalse([line for line in result.lines if line.startswith(f"wrote {self.fx.models_path}")])
        self.assertNotIn("pi", result.changed)

    def test_hermes_use_failure_is_reported(self):
        self.host.on(["hermes", "profile", "use", "llmctl"], fail("boom"))
        result = self.apply(set_default=True)
        self.assertEqual(result.failures, ("hermes profile use llmctl exited 1: boom",))
        self.assert_hermes_written()
        self.assertIn("hermes", result.changed)
        self.assertEqual(result.failed, frozenset())
        self.assertEqual(result.default_failed, frozenset({"hermes"}))

    def test_pi_interrupt_during_settings_write_puts_models_json_back(self):
        models_before = self.fx.models_path.read_bytes()
        settings_before = self.fx.settings_path.read_bytes()
        real_write = clients.write_atomic

        def interrupted(path, text, **kwargs):
            if Path(path) == self.fx.settings_path:
                raise KeyboardInterrupt
            return real_write(path, text, **kwargs)

        with mock.patch("launcher.clients.write_atomic", side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self.apply()
        self.assertEqual(self.fx.models_path.read_bytes(), models_before)
        self.assertEqual(self.fx.settings_path.read_bytes(), settings_before)
        self.assertFalse(os.path.lexists(os.path.realpath(self.fx.settings_path) + ".lock"))

    def test_interrupt_after_the_settings_write_landed_keeps_models_json(self):
        real_write = clients.write_atomic

        def landed_then_interrupted(path, text, **kwargs):
            real_write(path, text, **kwargs)
            if Path(path) == self.fx.settings_path:
                raise KeyboardInterrupt

        with mock.patch("launcher.clients.write_atomic", side_effect=landed_then_interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self.apply()
        self.assert_models_written()
        self.assertEqual(pi_config.pi_default(self.host.env), ("llmctl", "local"))

    def test_put_back_keeps_a_dangling_models_symlink(self):
        dotfiles = Path(self.tmp.name) / "dotfiles" / "pi"
        dotfiles.mkdir(parents=True)
        self.fx.models_path.unlink()
        self.fx.models_path.symlink_to(dotfiles / "models.json")
        real_write = clients.write_atomic
        written = []

        def failing(path, text, **kwargs):
            if Path(path) == self.fx.settings_path:
                raise OSError(28, "No space left on device")
            if Path(path) == self.fx.models_path:
                written.append(path)
            return real_write(path, text, **kwargs)

        with mock.patch("launcher.clients.write_atomic", side_effect=failing):
            result = self.apply()
        self.assertEqual(written, [self.fx.models_path])
        self.assertEqual(len(result.failures), 1, result.failures)
        self.assertTrue(result.failures[0].endswith(f"{self.fx.models_path} was put back as it was"), result.failures)
        self.assertTrue(self.fx.models_path.is_symlink())
        self.assertEqual(os.readlink(self.fx.models_path), str(dotfiles / "models.json"))
        self.assertFalse(os.path.lexists(dotfiles / "models.json"))

    def lock_seen(self):
        """Route pi --list-models so that each call records whether llmctl holds Pi's settings lock."""
        lock = os.path.realpath(self.fx.settings_path) + ".lock"
        seen = []

        def listing(argv):
            seen.append(os.path.isdir(lock))
            return self.fx._pi_list(argv)

        self.host.on(["pi", "--offline", "--list-models", "llmctl"], listing)
        return seen

    def test_pi_is_validated_before_the_settings_lock(self):
        seen = self.lock_seen()
        result = self.apply()
        self.assertEqual(result.failures, ())
        self.assertEqual(seen, [False])
        self.assert_models_written()

    def test_pi_revalidates_when_models_json_changed_before_the_lock(self):
        seen = self.lock_seen()
        listing = self.host._routes[-1][1]
        other = {"baseUrl": "http://127.0.0.1:9/v1", "api": "openai-completions", "models": [{"id": "m"}]}

        def changing(argv):
            if not seen:
                doc = json.loads(self.fx.models_path.read_text(encoding="utf-8"))
                doc["providers"]["other"] = other
                self.fx.models_path.write_text(json.dumps(doc), encoding="utf-8")
            return listing(argv)

        self.host.on(["pi", "--offline", "--list-models", "llmctl"], changing)
        result = self.apply()
        self.assertEqual(result.failures, ())
        self.assertEqual(seen, [False, True])
        providers = json.loads(self.fx.models_path.read_text(encoding="utf-8"))["providers"]
        self.assertEqual(providers["other"], other)
        self.assertEqual(providers["llmctl"], pi_config.provider_entry(self.target))

    def test_pi_unchanged_entry_is_validated_once(self):
        self.fx.models_path.write_text(pi_config.new_models_text(self.target), encoding="utf-8")
        seen = self.lock_seen()
        result = self.apply()
        self.assertEqual(result.failures, ())
        self.assertEqual(seen, [False])

    def test_pi_lock_taken_over_is_reported(self):
        lock = Path(os.path.realpath(self.fx.settings_path) + ".lock")
        spare = Path(self.tmp.name) / "spare-lock"
        real_write = clients.write_atomic
        taken = []

        def take_over(path, text, **kwargs):
            if Path(path) == self.fx.settings_path and not taken:
                taken.append(True)
                spare.mkdir()
                lock.rmdir()
                spare.rename(lock)
            return real_write(path, text, **kwargs)

        with mock.patch("launcher.clients.write_atomic", side_effect=take_over):
            result = self.apply()
        self.assertEqual(result.failures, (
            f"{self.fx.settings_path}: Pi took over its settings lock while llmctl held it, so another writer may "
            "have changed settings.json; check it",))
        self.assertTrue(lock.is_dir())
        self.assert_models_written()

    def test_set_default_runs_hermes_use(self):
        result = self.apply(set_default=True)
        self.assertEqual(result.failures, ())
        self.assertEqual(self.host.calls.count(["hermes", "profile", "use", "llmctl"]), 1)
        self.assertEqual(result.lines[-2:], (
            "ran hermes profile use llmctl: plain hermes commands, hermes gateway and hermes config included, "
            "now use the profile llmctl; switch back with hermes profile use default",
            "Hermes default profile: default -> llmctl",
        ))

    def test_set_default_with_llmctl_sticky_runs_no_hermes(self):
        (self.fx.hermes_root / "active_profile").write_text("llmctl\n", encoding="utf-8")
        planned = self.plan(set_default=True, check=False)
        self.assertFalse(planned.hermes_use)
        self.assertNotIn("would run hermes profile use llmctl", clients.dry_run_lines(planned))
        result = self.apply(set_default=True)
        self.assertEqual(result.failures, ())
        self.assertFalse([call for call in self.host.calls if call[0] == "hermes"])
        self.assertFalse([line for line in result.lines
                          if line.startswith(("ran hermes profile use", "Hermes default profile:"))])

    def test_reload_failure_note(self):
        self.host.on(["opencode", "reload"], fail("no server"))
        result = self.apply()
        oc = self.fx.opencode_path
        wrote = f"wrote {oc}: model llamacpp-docker/qwen3.6-35b-a3b (was vllm-docker/gpt-oss-20b)"
        note = "note: run `opencode reload`, or restart OpenCode, so an OpenCode that is already open sees it"
        self.assertIn(wrote, result.lines)
        self.assertIn(note, result.lines)
        last_write = max(index for index, line in enumerate(result.lines) if line.startswith("wrote "))
        self.assertGreater(result.lines.index(note), last_write)
        self.assertEqual(result.failures, ())

    def test_disabled_client_untouched(self):
        self.host.env["LLMCTL_CLIENTS"] = "opencode,pi"
        before = self.fx.hermes_config.read_bytes()
        result = self.apply(set_default=True)
        self.assertEqual(result.failures, ())
        self.assertEqual(self.fx.hermes_config.read_bytes(), before)
        self.assertFalse([call for call in self.host.calls if call[0] == "hermes"])
        self.assertNotIn("hermes", result.changed)
        self.assertFalse(clients.plan(self.host, self.catalog, self.target, set_default=True, check=False).hermes_use)

    def test_unencodable_write_is_a_client_write_error(self):
        path = self.fx.home / "x.json"
        change = clients.FileChange(path, None, '{"a": "\ud800"}\n', "s", "p")
        with self.assertRaises(clients.ClientWriteError) as caught:
            clients._Applier().write(change)
        self.assertTrue(str(caught.exception).startswith(f"{path}: "), str(caught.exception))
        self.assertFalse(os.path.lexists(path))

    def test_failure_message(self):
        self.assertEqual(clients.failure_message(["a: x", "b: y"]),
                         "the model is running, but 2 client file(s) were not updated: a: x; b: y")


class ReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog(REPO / "recipes", None)
        cls.target = target_for("qwen3.6-35b-a3b", "llamacpp", REPO / "recipes")
        cls.old_target = target_for("gpt-oss-20b", "vllm", REPO / "recipes")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.host = FakeHost(env={"HOME": "/nonexistent", "PATH": "/usr/bin"})
        self.fx = ClientFixture(self.host, Path(self.tmp.name))
        self.fx.install("opencode", "hermes", "pi")
        self.fx.route_clients()
        self.fx.write_opencode("vllm-docker/gpt-oss-20b")
        self.fx.hermes_config.parent.mkdir(parents=True)
        self.fx.hermes_config.write_text(hermes_config.new_config_text(self.old_target), encoding="utf-8")
        self.fx.models_path.write_text(pi_config.new_models_text(self.old_target), encoding="utf-8")
        self.write_settings(START_SETTINGS)

    def write_settings(self, doc):
        self.fx.settings_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

    def set_sticky(self, name):
        (self.fx.hermes_root / "active_profile").write_text(name + "\n", encoding="utf-8")

    def report(self, target="running", *, with_paths=True, running_known=True):
        target = self.target if target == "running" else target
        return clients.report(self.host, self.catalog, target, with_paths=with_paths, running_known=running_known)

    def test_status_lines_running_model(self):
        result = clients.apply(self.host, self.catalog, self.target, set_default=True)
        self.assertEqual(result.failures, ())
        agent = self.fx.models_path.parent
        self.assertEqual(self.report(), [
            "clients:",
            f"  OpenCode: default llamacpp-docker/qwen3.6-35b-a3b, the running model ({self.fx.opencode_path})",
            "  Hermes: profile llmctl for qwen3.6-35b-a3b on llama.cpp, the running model; default profile: llmctl "
            f"({self.fx.hermes_config})",
            "  Pi: llmctl/local for qwen3.6-35b-a3b on llama.cpp, the running model; default llmctl/local, "
            f"thinking off ({agent})",
            "  dsh: not installed (no dsh on PATH)",
            "  Kimi: not installed (no kimi on PATH)",
        ])

    def test_status_other_states(self):
        agent = self.fx.models_path.parent
        with self.subTest("no Hermes profile"):
            self.fx.hermes_config.unlink()
            self.assertIn("  Hermes: profile llmctl not created yet (the next ./llmctl up creates it)", self.report())
        with self.subTest("Pi entry for another model"):
            self.assertIn("  Pi: llmctl/gpt-oss-20b for gpt-oss-20b on vLLM, not the running model "
                          f"(run ./llmctl up again); default llmctl/gpt-oss-20b, thinking not set ({agent})",
                          self.report())
        with self.subTest("Pi entry without a running model"):
            self.assertIn("  Pi: llmctl/gpt-oss-20b for gpt-oss-20b on vLLM; default llmctl/gpt-oss-20b, "
                          f"thinking not set ({agent})", self.report(None))
        with self.subTest("Hermes profile for another model"):
            self.fx.hermes_config.write_text(hermes_config.new_config_text(self.old_target), encoding="utf-8")
            self.assertIn("  Hermes: profile llmctl for gpt-oss-20b on vLLM, not the running model "
                          f"(run ./llmctl up again); default profile: default ({self.fx.hermes_config})",
                          self.report())
            self.assertIn(f"  Hermes: profile llmctl for gpt-oss-20b on vLLM; default profile: default "
                          f"({self.fx.hermes_config})", self.report(None))
        with self.subTest("no Pi provider"):
            self.fx.models_path.unlink()
            self.assertIn(f"  Pi: no llmctl provider yet (the next ./llmctl up writes it) ({agent})", self.report())
        with self.subTest("broken opencode.json"):
            self.fx.opencode_path.write_text("{", encoding="utf-8")
            lines = self.report()
            line = next(line for line in lines if line.startswith("  OpenCode: "))
            self.assertTrue(line.startswith(f"  OpenCode: cannot read {self.fx.opencode_path}: "), line)
        with self.subTest("no OpenCode config"):
            self.fx.opencode_path.unlink()
            self.assertIn(f"  OpenCode: no config at {self.fx.opencode_path}", self.report())
        with self.subTest("disabled"):
            self.host.env["LLMCTL_CLIENTS"] = "pi,hermes"
            self.assertIn("  OpenCode: disabled by LLMCTL_CLIENTS", self.report())

    def test_status_not_installed(self):
        del self.host.installed["hermes"]
        self.assertIn("  Hermes: not installed (no hermes on PATH)", self.report())

    def test_no_default_model(self):
        self.fx.opencode_path.write_text('{"providers": {}}\n', encoding="utf-8")
        self.assertIn(f"  OpenCode: no default model ({self.fx.opencode_path})", self.report())

    def test_status_warnings_without_server(self):
        self.fx.write_opencode("llamacpp-docker/qwen3.6-35b-a3b")
        self.set_sticky("llmctl")
        self.write_settings({"defaultProvider": "llmctl", "defaultModel": "local"})
        warnings = [
            "warning: OpenCode's default llamacpp-docker/qwen3.6-35b-a3b points at llmctl, but no model is running",
            "warning: Hermes' default profile llmctl points at llmctl, but no model is running",
            "warning: Pi's default llmctl/local points at llmctl, but no model is running",
        ]
        lines = self.report(None)
        self.assertEqual(lines[-3:], warnings)
        self.assertEqual(len(lines), 9)
        quiet = self.report(None, running_known=False)
        self.assertEqual(len(quiet), 6)
        self.assertFalse([line for line in quiet if "but no model is running" in line])

    def test_no_warning_for_a_default_elsewhere(self):
        self.fx.write_opencode("openai/gpt-5")
        self.write_settings({"defaultProvider": "openai", "defaultModel": "gpt-5"})
        self.assertFalse([line for line in self.report(None) if line.startswith("warning:")])

    def test_opencode_default_not_running_model(self):
        self.fx.write_opencode("llamacpp-docker/gemma-4-26b-a4b")
        lines = self.report()
        self.assertIn("warning: OpenCode's default llamacpp-docker/gemma-4-26b-a4b is not the running model "
                      "qwen3.6-35b-a3b", lines)
        self.assertIn(f"  OpenCode: default llamacpp-docker/gemma-4-26b-a4b ({self.fx.opencode_path})", lines)

    def test_report_prints_hermes_warnings(self):
        (self.fx.hermes_root / "gateway_state.json").write_text(
            json.dumps({"gateway_state": "running", "served_profiles": ["llmctl"]}), encoding="utf-8")
        with open(self.fx.hermes_config, "a", encoding="utf-8") as handle:
            handle.write("agent:\n  reasoning_effort: high\n")
        lines = self.report()
        gateway = "warning: Hermes' gateway serves the profile llmctl"
        self.assertTrue([line for line in lines if line.startswith(gateway)], lines)
        self.assertTrue([line for line in lines
                         if line.startswith(f"warning: {self.fx.hermes_config} sets agent.reasoning_effort high")],
                        lines)

    def test_unreadable_hermes_profile(self):
        self.fx.hermes_config.write_text("foo: 1\n", encoding="utf-8")
        lines = self.report()
        line = next(line for line in lines if line.startswith("  Hermes: "))
        self.assertTrue(line.startswith(f"  Hermes: cannot read {self.fx.hermes_config}: "), line)

    def test_enabled_models_note_with_target(self):
        self.write_settings({"defaultProvider": "llmctl", "defaultModel": "local", "enabledModels": ["openai/gpt-5"]})
        self.assertIn("note: a new Pi session without --model starts with the first of enabledModels (openai/gpt-5) "
                      "instead of llmctl/local; add llmctl/local to enabledModels or clear enabledModels",
                      self.report())

    def test_report_without_paths(self):
        paths = [self.fx.opencode_path, self.fx.hermes_config, self.fx.models_path.parent]
        for name, prepare in (("other model", lambda: None),
                              ("running model", lambda: clients.apply(self.host, self.catalog, self.target,
                                                                      set_default=True))):
            with self.subTest(name):
                prepare()
                with_paths = self.report()
                lines = self.report(with_paths=False)
                self.assertEqual(len(lines), len(with_paths))
                for line in lines:
                    for path in paths:
                        self.assertFalse(line.endswith(f" ({path})"), line)
                self.assertIn("  OpenCode: default vllm-docker/gpt-oss-20b" if name == "other model"
                              else "  OpenCode: default llamacpp-docker/qwen3.6-35b-a3b, the running model", lines)

    def down_lines(self):
        return clients.down_lines(self.host, self.catalog, "qwen3.6-35b-a3b")

    def test_down_lines(self):
        self.set_sticky("llmctl")
        self.assertEqual(self.down_lines(), [
            "OpenCode, Hermes and Pi still point at qwen3.6-35b-a3b and get no answer until the next ./llmctl up",
            "  switch back: OpenCode /models; hermes profile use default; Pi /model",
        ])
        self.set_sticky("default")
        self.assertEqual(self.down_lines(), [
            "OpenCode and Pi still point at qwen3.6-35b-a3b and get no answer until the next ./llmctl up",
            "  switch back: OpenCode /models; Pi /model",
        ])
        self.fx.write_opencode("openai/gpt-5")
        self.assertEqual(self.down_lines(), [
            "Pi still points at qwen3.6-35b-a3b and gets no answer until the next ./llmctl up",
            "  switch back: Pi /model",
        ])
        self.write_settings({"lastChangelogVersion": "1.5.2"})
        self.assertEqual(self.down_lines(), [])

    def test_down_lines_skip_disabled_and_unreadable(self):
        self.set_sticky("llmctl")
        self.host.env["LLMCTL_CLIENTS"] = "hermes,pi"
        self.fx.settings_path.write_text("{", encoding="utf-8")
        self.assertEqual(self.down_lines(), [
            "Hermes still points at qwen3.6-35b-a3b and gets no answer until the next ./llmctl up",
            "  switch back: hermes profile use default",
        ])


TELEMETRY = "- id: session-telemetry-otel\n  disabled: true\n"
DSH_DUMP = ["dsh", "--profile", "acp", "--dump-config"]


class DshClientTest(unittest.TestCase):
    """dsh as the fourth client: the D7 sample state, with dsh alone unless a test says otherwise."""

    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog(REPO / "recipes", None)
        cls.target = target_for("qwen3.6-35b-a3b", "llamacpp", REPO / "recipes")
        cls.old_target = target_for("gpt-oss-20b", "vllm", REPO / "recipes")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.host = FakeHost(env={"HOME": "/nonexistent", "PATH": "/usr/bin", "LLMCTL_CLIENTS": "dsh"})
        self.fx = ClientFixture(self.host, Path(self.tmp.name))
        self.fx.install("opencode", "hermes", "pi", "dsh")
        self.fx.route_clients()
        self.fx.write_opencode("vllm-docker/gpt-oss-20b")
        self.fx.hermes_config.parent.mkdir(parents=True)
        self.fx.hermes_config.write_text(hermes_config.new_config_text(self.old_target), encoding="utf-8")
        self.fx.models_path.write_text(pi_config.new_models_text(self.old_target), encoding="utf-8")
        self.fx.settings_path.write_text(json.dumps(START_SETTINGS, indent=2) + "\n", encoding="utf-8")
        self.path = self.fx.home_patch
        self.start = TELEMETRY + dsh_config.new_home_text(self.old_target, default=True)
        self.write_patch(self.start)
        self.backup = Path(str(self.path) + ".llmctl-backup")
        self.backup.write_text(TELEMETRY, encoding="utf-8")

    def write_patch(self, text):
        with open(self.path, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)

    def read_patch(self):
        with open(self.path, encoding="utf-8", newline="") as handle:
            return handle.read()

    def plan(self, *, set_default=False, check=True):
        return clients.plan(self.host, self.catalog, self.target, set_default=set_default, check=check)

    def apply(self, *, set_default=False, dsh_checked=None):
        return clients.apply(self.host, self.catalog, self.target, set_default=set_default, dsh_checked=dsh_checked)

    def report(self, target="running", *, with_paths=True, running_known=True):
        target = self.target if target == "running" else target
        return clients.report(self.host, self.catalog, target, with_paths=with_paths, running_known=running_known)

    def dsh_change(self, planned):
        found = [change for change in planned.changes if change.path == self.path]
        self.assertEqual(len(found), 1, planned.changes)
        return found[0]

    def test_plan_checks_once(self):
        planned = self.plan(check=True)
        self.assertEqual(self.host.calls, [["dsh", "--version"], DSH_DUMP])
        self.assertTrue(planned.dsh_active)
        change = self.dsh_change(planned)
        self.assertEqual(planned.dsh_checked, change.new_text)
        self.assertEqual(planned.warnings, ())
        self.assertEqual(change.summary, "llmctl rows for qwen3.6-35b-a3b on llama.cpp (was gpt-oss-20b on vLLM); "
                                         "dsh's default llmctl/local in every profile (was llmctl/gpt-oss-20b)")

    def test_plan_without_check_runs_no_dsh(self):
        with mock.patch("launcher.dsh_config.tempfile.mkdtemp", side_effect=AssertionError):
            planned = self.plan(check=False)
        self.assertEqual(self.host.calls, [])
        self.assertTrue(planned.dsh_active)
        self.assertIsNone(planned.dsh_checked)

    def test_version_warning_is_planned(self):
        self.host.on(["dsh", "--version"], ok("0.2.3\n"))
        planned = self.plan(check=True)
        self.assertEqual(planned.warnings, (
            "warning: dsh 0.2.3 is not 0.2.2-rc.3, the version llmctl's dsh rows were checked with; run the dsh "
            "steps of the e2e before relying on them (docs/clients.md#maintenance)",))

    def test_dry_run_lines(self):
        self.backup.unlink()
        path = self.path
        self.assertEqual(clients.dry_run_lines(self.plan(check=False)), [
            f"would back up {path} to {path}.llmctl-backup",
            f"would write {path}: llmctl rows for qwen3.6-35b-a3b on llama.cpp (was gpt-oss-20b on vLLM); "
            "dsh's default llmctl/local in every profile (was llmctl/gpt-oss-20b)",
            "dry run: nothing was written, and dsh has not checked llmctl's rows",
        ])
        self.host.env["LLMCTL_CLIENTS"] = "pi,dsh"
        self.assertEqual(clients.dry_run_lines(self.plan(check=False))[-1],
                         "dry run: nothing was written, and neither Pi nor dsh has checked llmctl's entries")

    def test_apply_skips_an_unchanged_check(self):
        planned = self.plan(check=True)
        summary = self.dsh_change(planned).summary
        self.host.calls.clear()
        result = self.apply(dsh_checked=planned.dsh_checked)
        self.assertEqual(self.host.calls, [])
        self.assertEqual(result.failures, ())
        self.assertEqual(result.lines, (f"wrote {self.path}: {summary}",))
        self.assertEqual(result.changed, frozenset({"dsh"}))
        self.assertEqual(result.dsh_old.served, "gpt-oss-20b")
        self.assertEqual(self.read_patch(), TELEMETRY + dsh_config.new_home_text(self.target, default=True))

    def test_apply_rechecks_a_changed_candidate(self):
        planned = self.plan(check=True)
        self.host.calls.clear()
        user_row = "- id: tool-web\n  disabled: true\n"
        self.write_patch(self.start + user_row)
        result = self.apply(dsh_checked=planned.dsh_checked)
        self.assertEqual(result.failures, ())
        self.assertEqual(self.host.calls.count(DSH_DUMP), 1)
        self.assertEqual(self.read_patch(),
                         TELEMETRY + dsh_config.new_home_text(self.target, default=True) + user_row)

    def test_apply_check_failure_is_collected(self):
        planned = self.plan(check=True)
        changed = self.start + "- id: tool-web\n  disabled: true\n"
        self.write_patch(changed)

        def rejected(argv):
            patch = self.host.env_extras[-1]["DSH_HOME"] + "/cordis.patch.yml"
            return Result(1, "", dsh_fixture("s3.stderr", patch))

        self.host.on(DSH_DUMP, rejected)
        result = self.apply(dsh_checked=planned.dsh_checked)
        self.assertEqual(len(result.failures), 1, result.failures)
        self.assertTrue(result.failures[0].startswith(
            "dsh rejects the home patch llmctl would write for qwen3.6-35b-a3b: "), result.failures[0])
        self.assertEqual(result.failed, frozenset({"dsh"}))
        self.assertNotIn("dsh", result.changed)
        self.assertEqual(self.read_patch(), changed)

    def test_dsh_write_failure_keeps_the_others(self):
        del self.host.env["LLMCTL_CLIENTS"]
        os.chmod(self.fx.dsh_home, 0o500)
        self.addCleanup(os.chmod, self.fx.dsh_home, 0o700)
        result = self.apply()
        self.assertEqual(len(result.failures), 1, result.failures)
        self.assertTrue(result.failures[0].startswith(f"{self.path}: "), result.failures[0])
        self.assertEqual(result.failed, frozenset({"dsh"}))
        self.assertEqual(self.read_patch(), self.start)
        self.assertEqual(json.loads(self.fx.opencode_path.read_text(encoding="utf-8"))["model"],
                         "llamacpp-docker/qwen3.6-35b-a3b")
        self.assertEqual(self.fx.hermes_config.read_text(encoding="utf-8"), hermes_config.new_config_text(self.target))
        self.assertEqual(self.fx.models_path.read_text(encoding="utf-8"), pi_config.new_models_text(self.target))
        self.assertEqual(json.loads(self.fx.settings_path.read_text(encoding="utf-8"))["defaultModel"], "local")

    def test_unchanged_home_patch_is_still_checked(self):
        current = TELEMETRY + dsh_config.new_home_text(self.target, default=True)
        self.write_patch(current)
        seen = []

        def dump(argv):
            patch = self.host.env_extras[-1]["DSH_HOME"] + "/cordis.patch.yml"
            with open(patch, encoding="utf-8", newline="") as handle:
                seen.append(handle.read())
            return ok(dsh_fixture("s1-dump.txt", patch))

        self.host.on(DSH_DUMP, dump)
        planned = self.plan(check=True)
        self.assertEqual(planned.changes, ())
        self.assertEqual(planned.dsh_checked, current)
        self.assertEqual(seen, [current])
        result = self.apply()
        self.assertEqual(result.failures, ())
        self.assertEqual(result.lines, ())
        self.assertNotIn("dsh", result.changed)
        self.assertEqual(seen, [current, current])

    def test_symlinked_home_patch(self):
        dotfiles = self.fx.home / "dotfiles"
        dotfiles.mkdir()
        target_file = dotfiles / "dsh.yml"
        target_file.write_text(TELEMETRY, encoding="utf-8")
        os.chmod(target_file, 0o640)
        self.path.unlink()
        self.backup.unlink()
        self.path.symlink_to(target_file)
        result = self.apply()
        self.assertEqual(result.failures, ())
        self.assertTrue(self.path.is_symlink())
        self.assertEqual(os.readlink(self.path), str(target_file))
        self.assertEqual(target_file.read_text(encoding="utf-8"),
                         TELEMETRY + dsh_config.new_home_text(self.target, default=False))
        self.assertEqual(os.stat(target_file).st_mode & 0o7777, 0o640)
        self.assertFalse(os.path.islink(self.backup))
        self.assertEqual(self.backup.read_text(encoding="utf-8"), TELEMETRY)
        self.assertEqual(os.stat(self.backup).st_mode & 0o7777, 0o640)

    def test_status_lines(self):
        path = self.path
        result = self.apply(set_default=True)
        self.assertEqual(result.failures, ())
        self.assertIn(f"  dsh: llmctl/local for qwen3.6-35b-a3b on llama.cpp, the running model; default llmctl/local, "
                      f"held by llmctl ({path})", self.report())
        with self.subTest("no default lines"):
            self.write_patch(TELEMETRY + dsh_config.new_home_text(self.target, default=False))
            self.assertIn(f"  dsh: llmctl/local for qwen3.6-35b-a3b on llama.cpp, the running model; default left "
                          f"to dsh's model picker ({path})", self.report())
        with self.subTest("telemetry row only"):
            self.write_patch(TELEMETRY)
            self.assertIn(f"  dsh: no llmctl rows yet (the next ./llmctl up writes them) ({path})", self.report())
        with self.subTest("the starting state"):
            self.write_patch(self.start)
            self.assertIn("  dsh: llmctl/gpt-oss-20b for gpt-oss-20b on vLLM, not the running model (run ./llmctl up "
                          f"again); default llmctl/gpt-oss-20b, held by llmctl ({path})", self.report())
            self.assertFalse([line for line in self.report() if line.startswith("warning: dsh")])
        with self.subTest("no model running"):
            self.write_patch(self.start)
            lines = self.report(None)
            self.assertIn("  dsh: llmctl/gpt-oss-20b for gpt-oss-20b on vLLM; default llmctl/gpt-oss-20b, held by "
                          f"llmctl ({path})", lines)
            self.assertIn("warning: dsh's default llmctl/gpt-oss-20b points at llmctl, but no model is running", lines)
            self.assertFalse([line for line in self.report(None, running_known=False) if line.startswith("warning:")])
        with self.subTest("damaged markers"):
            self.write_patch(self.start + dsh_config.BEGIN + "\n")
            line = next(line for line in self.report() if line.startswith("  dsh: "))
            self.assertTrue(line.startswith(f"  dsh: cannot read {path}: "), line)
        with self.subTest("without paths"):
            self.write_patch(self.start)
            self.assertIn("  dsh: llmctl/gpt-oss-20b for gpt-oss-20b on vLLM, not the running model (run ./llmctl up "
                          "again); default llmctl/gpt-oss-20b, held by llmctl", self.report(with_paths=False))
        with self.subTest("disabled"):
            self.host.env["LLMCTL_CLIENTS"] = "pi"
            self.assertIn("  dsh: disabled by LLMCTL_CLIENTS", self.report())
        with self.subTest("not installed"):
            self.host.env["LLMCTL_CLIENTS"] = "dsh"
            del self.host.installed["dsh"]
            self.assertIn("  dsh: not installed (no dsh on PATH)", self.report())

    def down_lines(self):
        return clients.down_lines(self.host, self.catalog, "qwen3.6-35b-a3b")

    def test_down_lines_name_dsh(self):
        path = self.path
        del self.host.env["LLMCTL_CLIENTS"]
        (self.fx.hermes_root / "active_profile").write_text("llmctl\n", encoding="utf-8")
        self.assertEqual(self.down_lines(), [
            "OpenCode, Hermes, Pi and dsh still point at qwen3.6-35b-a3b and get no answer until the next ./llmctl up",
            "  switch back: OpenCode /models; hermes profile use default; Pi /model; dsh: delete the "
            f"agent-default-model and acp lines in {path}",
        ])
        self.host.env["LLMCTL_CLIENTS"] = "dsh"
        self.assertEqual(self.down_lines(), [
            "dsh still points at qwen3.6-35b-a3b and gets no answer until the next ./llmctl up",
            f"  switch back: dsh: delete the agent-default-model and acp lines in {path}",
        ])
        self.write_patch(TELEMETRY + dsh_config.new_home_text(self.old_target, default=False))
        self.assertEqual(self.down_lines(), [])


KIMI_VERSION = ["kimi", "--version"]
KIMI_LIST = ["kimi", "provider", "list", "--json"]
KIMI_CALLS = [KIMI_VERSION, KIMI_LIST]
READ_BACK = "the file changed while llmctl wrote it, probably through Kimi; check it, then run ./llmctl up again"


class KimiClientTest(unittest.TestCase):
    """Kimi as the fifth client: the D10 starting file, with Kimi alone unless a test says otherwise."""

    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog(REPO / "recipes", None)
        cls.target = target_for("qwen3.6-35b-a3b", "llamacpp", REPO / "recipes")
        cls.old_target = target_for("gpt-oss-20b", "vllm", REPO / "recipes")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.host = FakeHost(env={"HOME": "/nonexistent", "PATH": "/usr/bin", "LLMCTL_CLIENTS": "kimi"})
        self.fx = ClientFixture(self.host, Path(self.tmp.name))
        self.fx.install("opencode", "hermes", "pi", "dsh", "kimi")
        self.fx.route_clients()
        self.fx.write_opencode("vllm-docker/gpt-oss-20b")
        self.fx.hermes_config.parent.mkdir(parents=True)
        self.fx.hermes_config.write_text(hermes_config.new_config_text(self.old_target), encoding="utf-8")
        self.fx.models_path.write_text(pi_config.new_models_text(self.old_target), encoding="utf-8")
        self.fx.settings_path.write_text(json.dumps(START_SETTINGS, indent=2) + "\n", encoding="utf-8")
        with open(self.fx.home_patch, "w", encoding="utf-8", newline="") as handle:
            handle.write(TELEMETRY + dsh_config.new_home_text(self.old_target, default=True))
        Path(str(self.fx.home_patch) + ".llmctl-backup").write_text(TELEMETRY, encoding="utf-8")
        self.path = self.fx.kimi_file
        self.start = kimi_config.config_text(self.path, OWNER_TEXT, self.old_target, set_default=True)
        self.write_config(self.start)
        self.backup = Path(str(self.path) + ".llmctl-backup")
        self.backup.write_text(OWNER_TEXT, encoding="utf-8")

    def write_config(self, text):
        with open(self.path, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)

    def read_config(self):
        with open(self.path, encoding="utf-8", newline="") as handle:
            return handle.read()

    def append_config(self, text):
        with open(self.path, "a", encoding="utf-8", newline="") as handle:
            handle.write(text)

    def plan(self, *, set_default=False, check=True, target=None):
        return clients.plan(self.host, self.catalog, target or self.target, set_default=set_default, check=check)

    def apply(self, *, set_default=False, kimi_checked=None, target=None):
        return clients.apply(self.host, self.catalog, target or self.target, set_default=set_default,
                             kimi_checked=kimi_checked)

    def report(self, target="running", *, with_paths=True, running_known=True):
        target = self.target if target == "running" else target
        return clients.report(self.host, self.catalog, target, with_paths=with_paths, running_known=running_known)

    def kimi_change(self, planned):
        found = [change for change in planned.changes if change.path == self.path]
        self.assertEqual(len(found), 1, planned.changes)
        return found[0]

    def kimi_calls(self):
        return [call for call in self.host.calls if call[0] == "kimi"]

    def assert_others_written(self):
        self.assertEqual(json.loads(self.fx.opencode_path.read_text(encoding="utf-8"))["model"],
                         "llamacpp-docker/qwen3.6-35b-a3b")
        self.assertEqual(self.fx.hermes_config.read_text(encoding="utf-8"), hermes_config.new_config_text(self.target))
        self.assertEqual(self.fx.models_path.read_text(encoding="utf-8"), pi_config.new_models_text(self.target))
        self.assertEqual(self.fx.home_patch.read_text(encoding="utf-8"),
                         TELEMETRY + dsh_config.new_home_text(self.target, default=True))

    def test_plan_checks_once(self):
        planned = self.plan(check=True)
        self.assertEqual(self.host.calls, KIMI_CALLS)
        self.assertTrue(planned.kimi_active)
        change = self.kimi_change(planned)
        self.assertEqual(planned.kimi_checked, change.new_text)
        self.assertEqual(planned.warnings, ())
        self.assertEqual(change.summary, "provider llmctl and model llmctl/local for qwen3.6-35b-a3b on llama.cpp "
                                         "(was gpt-oss-20b on vLLM); default_model llmctl/local "
                                         "(was llmctl/gpt-oss-20b)")

    def test_plan_without_check_runs_no_kimi(self):
        with mock.patch("launcher.kimi_config.tempfile.mkdtemp", side_effect=AssertionError):
            planned = self.plan(check=False)
        self.assertEqual(self.host.calls, [])
        self.assertTrue(planned.kimi_active)
        self.assertIsNone(planned.kimi_checked)

    def test_version_warning_is_planned(self):
        self.host.on(KIMI_VERSION, ok("2.6.2\n"))
        planned = self.plan(check=True)
        self.assertEqual(planned.warnings, (
            "warning: Kimi 2.6.2 is not 2.5.3, the version llmctl's Kimi entry was checked with; run the Kimi steps "
            "of the e2e before relying on it (docs/clients.md#maintenance)",))

    def test_dry_run_lines(self):
        self.backup.unlink()
        path = self.path
        summary = self.kimi_change(self.plan(check=False)).summary
        self.assertEqual(clients.dry_run_lines(self.plan(check=False)), [
            f"would back up {path} to {path}.llmctl-backup",
            f"would write {path}: {summary}",
            "dry run: nothing was written, and Kimi has not checked llmctl's entry",
        ])
        for listed, ending in (
                ("pi,kimi", "dry run: nothing was written, and neither Pi nor Kimi has checked llmctl's entries"),
                ("dsh,kimi", "dry run: nothing was written, and neither dsh nor Kimi has checked llmctl's entries"),
                ("pi,dsh", "dry run: nothing was written, and neither Pi nor dsh has checked llmctl's entries"),
                ("pi,dsh,kimi", "dry run: nothing was written, and none of Pi, dsh and Kimi has checked llmctl's "
                                "entries")):
            with self.subTest(listed):
                self.host.env["LLMCTL_CLIENTS"] = listed
                self.assertEqual(clients.dry_run_lines(self.plan(check=False))[-1], ending)
        self.assertEqual(self.host.calls, [])

    def test_apply_skips_an_unchanged_check(self):
        planned = self.plan(check=True)
        summary = self.kimi_change(planned).summary
        self.host.calls.clear()
        result = self.apply(kimi_checked=planned.kimi_checked)
        self.assertEqual(self.host.calls, [])
        self.assertEqual(result.failures, ())
        self.assertEqual(result.lines, (f"wrote {self.path}: {summary}",))
        self.assertEqual(result.changed, frozenset({"kimi"}))
        self.assertEqual(result.kimi_old.served, "gpt-oss-20b")
        self.assertEqual(self.read_config(), planned.kimi_checked)

    def test_apply_rechecks_a_changed_candidate(self):
        planned = self.plan(check=True)
        self.host.calls.clear()
        self.append_config("[thinking]\nenabled = true\n")
        result = self.apply(kimi_checked=planned.kimi_checked)
        self.assertEqual(result.failures, ())
        self.assertEqual(self.host.calls, KIMI_CALLS)
        self.assertTrue(self.read_config().endswith("[thinking]\nenabled = true\n"), self.read_config())
        self.assertTrue(kimi_config.entry_matches(kimi_config.read_state(self.host.env), self.target))

    def test_apply_check_failure_is_collected(self):
        planned = self.plan(check=True)
        self.append_config("[thinking]\nenabled = true\n")
        changed = self.read_config()
        self.host.on(KIMI_LIST, ok('{"providers": {}, "models": {}}'))
        result = self.apply(kimi_checked=planned.kimi_checked)
        self.assertEqual(len(result.failures), 1, result.failures)
        self.assertTrue(result.failures[0].startswith(
            "Kimi 2.5.3 does not read llmctl's entry for qwen3.6-35b-a3b from the new config.toml: no provider llmctl"),
            result.failures[0])
        self.assertEqual(result.failed, frozenset({"kimi"}))
        self.assertNotIn("kimi", result.changed)
        self.assertEqual(self.read_config(), changed)

    def test_kimi_write_failure_keeps_the_others(self):
        del self.host.env["LLMCTL_CLIENTS"]
        os.chmod(self.fx.kimi_home, 0o500)
        self.addCleanup(os.chmod, self.fx.kimi_home, 0o700)
        result = self.apply(set_default=True)
        self.assertEqual(len(result.failures), 1, result.failures)
        self.assertTrue(result.failures[0].startswith(f"{self.path}: "), result.failures[0])
        self.assertEqual(result.failed, frozenset({"kimi"}))
        self.assertEqual(self.read_config(), self.start)
        self.assert_others_written()

    def test_kimi_save_during_the_check_is_kept(self):
        # Review Focus 5: an open Kimi TUI saves config.toml while llmctl's Kimi check runs.
        planned = self.plan(check=True)
        self.append_config("[loop_control]\nmax_steps_per_turn = 100\n")  # a changed candidate is checked again
        answer = self.fx._kimi_list

        def save_then_answer(argv):
            self.append_config("[thinking]\nenabled = true\n")
            return answer(argv)

        self.host.on(KIMI_LIST, save_then_answer)
        result = self.apply(kimi_checked=planned.kimi_checked)
        self.assertEqual(result.failures, (f"{self.path}: {READ_BACK}",))
        self.assertEqual(result.failed, frozenset({"kimi"}))
        self.assertNotIn("kimi", result.changed)
        self.assertTrue(self.read_config().endswith("[thinking]\nenabled = true\n"), self.read_config())
        self.assertTrue(self.read_config().startswith(self.start))

    def test_kimi_save_after_the_rename_fails_the_read_back(self):
        # Review Focus 5: an open Kimi TUI saves config.toml right after llmctl's rename.
        real_write = clients.write_atomic

        def write_then_save(path, text, **kwargs):
            real_write(path, text, **kwargs)
            if Path(path) == self.path:
                self.append_config("[thinking]\nenabled = false\n")

        del self.host.env["LLMCTL_CLIENTS"]
        with mock.patch("launcher.clients.write_atomic", side_effect=write_then_save):
            result = self.apply(set_default=True)
        self.assertEqual(result.failures, (f"{self.path}: {READ_BACK}",))
        self.assertEqual(result.failed, frozenset({"kimi"}))
        self.assertTrue(self.read_config().endswith("[thinking]\nenabled = false\n"), self.read_config())
        self.assert_others_written()

    def test_unchanged_config_is_still_checked(self):
        current = kimi_config.config_text(self.path, self.start, self.target, set_default=False)
        self.write_config(current)
        seen = []
        answer = self.fx._kimi_list

        def listing(argv):
            with open(Path(self.host.env_extras[-1]["KIMI_CODE_HOME"]) / "config.toml", encoding="utf-8",
                      newline="") as handle:
                seen.append(handle.read())
            return answer(argv)

        self.host.on(KIMI_LIST, listing)
        planned = self.plan(check=True)
        self.assertEqual(planned.changes, ())
        self.assertEqual(planned.kimi_checked, current)
        self.assertEqual(self.host.calls, KIMI_CALLS)
        self.assertEqual(seen, [current])
        result = self.apply()
        self.assertEqual(result.failures, ())
        self.assertEqual(result.lines, ())
        self.assertNotIn("kimi", result.changed)
        self.assertEqual(seen, [current, current])

    def test_first_write_into_the_owner_file(self):
        self.backup.unlink()
        self.write_config(OWNER_TEXT)
        os.chmod(self.path, 0o600)
        gemma = target_for("gemma-4-26b-a4b", "llamacpp", REPO / "recipes")
        result = self.apply(target=gemma)
        self.assertEqual(result.failures, ())
        self.assertEqual(result.lines, (
            f"backed up {self.path} to {self.path}.llmctl-backup before the first change",
            f"wrote {self.path}: provider llmctl and model llmctl/local for gemma-4-26b-a4b on llama.cpp (was absent)",
        ))
        self.assertEqual(self.read_config(), DESIGN_OWNER_FIRST_WRITE)
        self.assertEqual(self.backup.read_text(encoding="utf-8"), OWNER_TEXT)
        self.assertEqual(os.stat(self.backup).st_mode & 0o7777, 0o600)

    def test_symlinked_config(self):
        dotfiles = self.fx.home / "dotfiles"
        dotfiles.mkdir()
        target_file = dotfiles / "kimi.toml"
        target_file.write_text(OWNER_TEXT, encoding="utf-8")
        os.chmod(target_file, 0o640)
        self.path.unlink()
        self.backup.unlink()
        self.path.symlink_to(target_file)
        result = self.apply()
        self.assertEqual(result.failures, ())
        self.assertTrue(self.path.is_symlink())
        self.assertEqual(os.readlink(self.path), str(target_file))
        self.assertEqual(target_file.read_text(encoding="utf-8"),
                         kimi_config.config_text(self.path, OWNER_TEXT, self.target, set_default=False))
        self.assertEqual(os.stat(target_file).st_mode & 0o7777, 0o640)
        self.assertFalse(os.path.islink(self.backup))
        self.assertEqual(self.backup.read_text(encoding="utf-8"), OWNER_TEXT)
        self.assertEqual(os.stat(self.backup).st_mode & 0o7777, 0o640)

    def kimi_line(self, lines):
        found = [line for line in lines if line.startswith("  Kimi: ")]
        self.assertEqual(len(found), 1, lines)
        return found[0]

    def test_status_lines(self):
        path = self.path
        result = self.apply(set_default=True)
        self.assertEqual(result.failures, ())
        self.host.calls.clear()
        self.assertEqual(self.kimi_line(self.report()), "  Kimi: llmctl/local for qwen3.6-35b-a3b on llama.cpp, the "
                                                        f"running model; default llmctl/local ({path})")
        self.assertFalse([line for line in self.report() if line.startswith(("warning:", "note:"))])
        with self.subTest("default not set"):
            self.write_config(kimi_config.config_text(path, OWNER_TEXT, self.target, set_default=False))
            self.assertEqual(self.kimi_line(self.report()), "  Kimi: llmctl/local for qwen3.6-35b-a3b on llama.cpp, "
                                                            f"the running model; default not set ({path})")
        with self.subTest("no llmctl entry"):
            self.write_config(OWNER_TEXT)
            self.assertEqual(self.kimi_line(self.report()),
                             f"  Kimi: no llmctl entry yet (the next ./llmctl up writes it) ({path})")
            self.path.unlink()
            self.assertEqual(self.kimi_line(self.report()),
                             f"  Kimi: no llmctl entry yet (the next ./llmctl up writes it) ({path})")
        with self.subTest("the starting state"):
            self.write_config(self.start)
            lines = self.report()
            self.assertEqual(self.kimi_line(lines), "  Kimi: llmctl/gpt-oss-20b for gpt-oss-20b on vLLM, not the "
                                                    "running model (run ./llmctl up again); default llmctl/gpt-oss-20b "
                                                    f"({path})")
            self.assertFalse([line for line in lines if line.startswith("warning: Kimi")], lines)
        with self.subTest("no model running"):
            self.write_config(self.start)
            lines = self.report(None)
            self.assertEqual(self.kimi_line(lines), "  Kimi: llmctl/gpt-oss-20b for gpt-oss-20b on vLLM; default "
                                                    f"llmctl/gpt-oss-20b ({path})")
            self.assertIn("warning: Kimi's default llmctl/gpt-oss-20b points at llmctl, but no model is running",
                          lines)
            self.assertFalse([line for line in self.report(None, running_known=False)
                              if line.startswith("warning:")])
        with self.subTest("invalid file"):
            self.write_config(self.start + "[broken\n")
            line = self.kimi_line(self.report())
            self.assertTrue(line.startswith(f"  Kimi: cannot read {path}: "), line)
            self.assertFalse(line.endswith(f" ({path})"), line)
        with self.subTest("missing alias"):
            self.write_config(kimi_config.default_text(path, self.start, "llmctl/gone"))
            self.assertIn(f"warning: Kimi's default_model llmctl/gone names no model in {path}, so every kimi -p fails "
                          '(Model "llmctl/gone" is not configured in config.toml.); pick a model with /model in kimi, '
                          "or run ./llmctl up <model> --set-default", self.report())
        with self.subTest("without paths"):
            self.write_config(self.start)
            self.assertEqual(self.kimi_line(self.report(with_paths=False)),
                             "  Kimi: llmctl/gpt-oss-20b for gpt-oss-20b on vLLM, not the running model (run "
                             "./llmctl up again); default llmctl/gpt-oss-20b")
        with self.subTest("disabled"):
            self.host.env["LLMCTL_CLIENTS"] = "pi"
            self.assertEqual(self.kimi_line(self.report()), "  Kimi: disabled by LLMCTL_CLIENTS")
        with self.subTest("not installed"):
            self.host.env["LLMCTL_CLIENTS"] = "kimi"
            del self.host.installed["kimi"]
            self.assertEqual(self.kimi_line(self.report()), "  Kimi: not installed (no kimi on PATH)")
        self.assertEqual(self.host.calls, [])

    def down_lines(self):
        return clients.down_lines(self.host, self.catalog, "qwen3.6-35b-a3b")

    def test_down_lines_name_kimi(self):
        del self.host.env["LLMCTL_CLIENTS"]
        (self.fx.hermes_root / "active_profile").write_text("llmctl\n", encoding="utf-8")
        self.fx.write_opencode("llamacpp-docker/qwen3.6-35b-a3b")
        self.assertEqual(self.down_lines(), [
            "OpenCode, Hermes, Pi, dsh and Kimi still point at qwen3.6-35b-a3b and get no answer until the next "
            "./llmctl up",
            "  switch back: OpenCode /models; hermes profile use default; Pi /model; dsh: delete the "
            f"agent-default-model and acp lines in {self.fx.home_patch}; Kimi /model",
            "  Kimi retries a stopped server for about two and a half minutes before a prompt fails",
        ])
        self.host.env["LLMCTL_CLIENTS"] = "kimi"
        self.assertEqual(self.down_lines(), [
            "Kimi still points at qwen3.6-35b-a3b and gets no answer until the next ./llmctl up",
            "  switch back: Kimi /model",
            kimi_config.RETRY_NOTE,
        ])
        self.write_config(kimi_config.default_text(self.path, self.start, "other/m"))
        self.assertEqual(self.down_lines(), [])
        self.assertEqual(self.host.calls, [])


if __name__ == "__main__":
    unittest.main()
