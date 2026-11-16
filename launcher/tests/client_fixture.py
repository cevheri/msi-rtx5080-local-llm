"""A FakeHost whose HOME is a temporary folder holding the config roots of OpenCode, Hermes and Pi."""

import hashlib
import json
import os
from pathlib import Path

from launcher import hermes_config, opencode_config, pi_config
from launcher.tests.fakes import FakeHost, ok
from launcher.tests.test_opencode_config import doc_config_text

LIST_HEADER = "provider  model  context  max-out  thinking  images\n"


class ClientFixture:
    def __init__(self, host: FakeHost, root: Path):
        self.host = host
        self.root = Path(root)
        host.env["HOME"] = str(self.root / "home")
        host.env.pop("XDG_CONFIG_HOME", None)
        for folder in (".config/opencode", ".hermes", ".pi/agent"):
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

    def install(self, *names: str) -> None:
        for name in names:
            self.host.installed[name] = f"/usr/bin/{name}"

    def write_opencode(self, model: str) -> None:
        self.opencode_path.write_text(opencode_config.set_top_level_model(doc_config_text(), model), encoding="utf-8")

    def route_clients(self) -> None:
        self.host.on(["opencode", "reload"], ok())
        self.host.on(["hermes", "profile", "use", "llmctl"], self._hermes_use)
        self.host.on(["pi", "--offline", "--list-models", "llmctl"], self._pi_list)

    def _hermes_use(self, argv):
        (self.hermes_root / "active_profile").write_text("llmctl\n", encoding="utf-8")
        return ok()

    def _pi_list(self, argv):
        directory = Path(self.host.env_extras[-1]["PI_CODING_AGENT_DIR"])
        doc = json.loads((directory / "models.json").read_text(encoding="utf-8"))
        models = doc.get("providers", {}).get("llmctl", {}).get("models", [])
        rows = "".join(f"llmctl    {model['id']}  131.1K   16.4K    yes       no\n" for model in models)
        return ok(LIST_HEADER + rows)

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
