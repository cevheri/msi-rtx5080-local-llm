"""A FakeHost whose HOME is a temporary folder holding the config roots of OpenCode, Hermes, Pi, dsh and Kimi."""

import hashlib
import json
import os
import re
import tomllib
from pathlib import Path

from launcher import dsh_config, hermes_config, kimi_config, opencode_config, pi_config
from launcher.tests.fakes import FakeHost, ok
from launcher.tests.test_dsh_config import dsh_fixture
from launcher.tests.test_opencode_config import doc_config_text

LIST_HEADER = "provider  model  context  max-out  thinking  images\n"


class ClientFixture:
    def __init__(self, host: FakeHost, root: Path):
        self.host = host
        self.root = Path(root)
        host.env["HOME"] = str(self.root / "home")
        host.env.pop("XDG_CONFIG_HOME", None)
        for folder in (".config/opencode", ".hermes", ".pi/agent", ".dsh/profiles", ".kimi-code"):
            (self.home / folder).mkdir(parents=True, exist_ok=True)

    @property
    def home(self) -> Path:
        return self.root / "home"

    @property
    def opencode_path(self) -> Path:
        return opencode_config.config_path(self.host.env)

    @property
    def hermes_root(self) -> Path:
        return hermes_config.hermes_root(self.host.env)

    @property
    def hermes_config(self) -> Path:
        return hermes_config.profile_config(self.host.env)

    @property
    def models_path(self) -> Path:
        return pi_config.models_path(self.host.env)

    @property
    def settings_path(self) -> Path:
        return pi_config.settings_path(self.host.env)

    @property
    def dsh_home(self) -> Path:
        return dsh_config.dsh_home(self.host.env)

    @property
    def home_patch(self) -> Path:
        return dsh_config.home_patch(self.host.env)

    @property
    def kimi_home(self) -> Path:
        return kimi_config.kimi_home(self.host.env)

    @property
    def kimi_file(self) -> Path:
        return kimi_config.config_path(self.host.env)

    def install(self, *names: str) -> None:
        for name in names:
            self.host.installed[name] = f"/usr/bin/{name}"

    def write_opencode(self, model: str) -> None:
        self.opencode_path.write_text(opencode_config.set_top_level_model(doc_config_text(), model), encoding="utf-8")

    def route_clients(self) -> None:
        self.host.on(["opencode", "reload"], ok())
        self.host.on(["hermes", "profile", "use", "llmctl"], self._hermes_use)
        self.host.on(["pi", "--offline", "--list-models", "llmctl"], self._pi_list)
        self.host.on(["dsh", "--version"], ok("0.2.2-rc.3\n"))
        self.host.on(["dsh", "--profile", "acp", "--dump-config"], self._dsh_dump)
        self.host.on(["kimi", "--version"], ok("2.5.3\n"))
        self.host.on(["kimi", "provider", "list", "--json"], self._kimi_list)

    def _hermes_use(self, argv):
        (self.hermes_root / "active_profile").write_text("llmctl\n", encoding="utf-8")
        return ok()

    def _pi_list(self, argv):
        directory = Path(self.host.env_extras[-1]["PI_CODING_AGENT_DIR"])
        doc = json.loads((directory / "models.json").read_text(encoding="utf-8"))
        models = doc.get("providers", {}).get("llmctl", {}).get("models", [])
        rows = "".join(f"llmctl    {model['id']}  131.1K   16.4K    yes       no\n" for model in models)
        return ok(LIST_HEADER + rows)

    def _dsh_dump(self, argv):
        return ok(dsh_fixture("s1-dump.txt", self.host.env_extras[-1]["DSH_HOME"] + "/cordis.patch.yml"))

    def _kimi_list(self, argv):
        """Kimi 2.5.3's reading of the private config.toml: each entry's keys in camelCase, empty maps when the
        file does not parse (E6)."""
        path = Path(self.host.env_extras[-1]["KIMI_CODE_HOME"]) / "config.toml"
        try:
            with open(path, "rb") as handle:
                data = tomllib.load(handle)
        except tomllib.TOMLDecodeError:
            return ok(json.dumps({"providers": {}, "models": {}}))

        def camel(entry):
            if not isinstance(entry, dict):
                return entry
            return {re.sub(r"_([a-z])", lambda match: match.group(1).upper(), key): value
                    for key, value in entry.items()}

        doc = {kind: {name: camel(entry) for name, entry in data.get(kind, {}).items()}
               for kind in ("providers", "models")}
        return ok(json.dumps(doc))

    def snapshot(self) -> dict[str, str]:
        result = {}
        for directory, dirs, files in os.walk(self.home):
            for name in dirs:
                result[os.path.relpath(os.path.join(directory, name), self.home)] = "dir"
            for name in files:
                path = os.path.join(directory, name)
                with open(path, "rb") as handle:
                    result[os.path.relpath(path, self.home)] = hashlib.sha256(handle.read()).hexdigest()
        return result
