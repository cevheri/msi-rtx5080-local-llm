"""A FakeHost on which `./llmctl up qwen3.6` succeeds; tests break one thing at a time."""

import json
import tempfile
from pathlib import Path

from launcher.catalog import load_catalog
from launcher.hfcache import snapshot_dir
from launcher.tests.fakes import REPO, FakeHost, fail, http, ok
from launcher.tests.regen_golden import GOLDEN

GPU_QUERY = ("nvidia-smi", "--query-gpu=memory.total,memory.reserved,memory.used,memory.free",
             "--format=csv,noheader,nounits")
GPU_APPS = ("nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits")
HEALTH = "http://127.0.0.1:8080/health"


def managed_line(name, state, recipe, runtime="llamacpp", sha="ab" * 32, source="shipped", runtime_sha="cd" * 32):
    labels = (f"io.llmctl.managed=true,io.llmctl.runtime={runtime},io.llmctl.recipe={recipe},"
              f"io.llmctl.source={source},io.llmctl.sha256={sha},io.llmctl.evidence=measured,"
              f"io.llmctl.runtime-source=shipped,io.llmctl.runtime-sha256={runtime_sha}")
    return f"{name}|id-{name}|{state}|{labels}"


class OpsFixture:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.hf_home = root / "hf"
        self.runtime_dir = root / "run"
        self.runtime_dir.mkdir()
        self.host = FakeHost(env={"HOME": "/home/user", "HF_HOME": str(self.hf_home),
                                  "XDG_RUNTIME_DIR": str(self.runtime_dir)})
        self.catalog = load_catalog(REPO / "recipes", None)
        self.place_files("qwen3.6-35b-a3b")
        self.health = [http(503, b'{"error":"loading"}'), http(200, b'{"status":"ok"}')]
        self.managed_lines: list[str] = []
        self.foreign = None
        self.state = "running"
        h = self.host
        h.on(("docker", "info"), ok("29.9.3\n"))
        h.on(("docker", "image", "inspect"), ok("sha256:abc\n"))
        h.on(("docker", "container", "inspect"), self._inspect_by_name)
        h.on(("docker", "ps"), lambda argv: ok("\n".join(self.managed_lines) + ("\n" if self.managed_lines else "")))
        h.on(GPU_QUERY, ok("16303, 462, 112, 15731\n"))
        h.on(GPU_APPS, ok(""))
        h.on(("docker", "run"), ok("0123456789ab\n"))
        h.on(("docker", "inspect", "--format", "{{.State.Status}}"), lambda argv: ok(self.state + "\n"))
        h.on(("docker", "logs"), ok("line\n" * 50))
        h.on(("docker", "stop"), ok())
        h.on(("docker", "rm"), ok())
        h.on_http("GET", HEALTH, lambda *a: self.health.pop(0) if len(self.health) > 1 else self.health[0])

    def _inspect_by_name(self, argv):
        name = argv[-1]
        if self.foreign and self.foreign[0] == name:
            return ok(f"id-{name}|{self.foreign[1]}|{{}}\n")
        for line in self.managed_lines:
            parts = line.split("|")
            if parts[0] == name:
                labels = dict(item.split("=", 1) for item in parts[3].split(","))
                return ok(f"{parts[1]}|{parts[2]}|{json.dumps(labels)}\n")
        return fail(f"Error: No such container: {name}")

    def place_files(self, recipe_id, runtime="llamacpp", size=None):
        recipe = self.catalog.runtimes[runtime].recipes[recipe_id]
        snapshot = snapshot_dir(self.hf_home, recipe.model.repo, recipe.model.revision)
        snapshot.mkdir(parents=True, exist_ok=True)
        if runtime == "vllm":
            (snapshot / "config.json").write_text("{}")
            with open(snapshot / "model.safetensors", "wb") as handle:
                handle.truncate(recipe.model.size_bytes - 2)  # sparse; config.json holds the other 2 bytes
            return
        each = (size if size is not None else recipe.model.size_bytes) // len(recipe.model.files)
        for name in recipe.model.files:
            with open(snapshot / name, "wb") as handle:
                handle.truncate(each)  # sparse: the right size without the disk space

    def golden(self, runtime, recipe_id):
        argv = json.loads((GOLDEN / runtime / f"{recipe_id}.json").read_text())
        return [a.replace("/home/user/.cache/huggingface", str(self.hf_home)) for a in argv]

    def calls_starting(self, *prefix):
        return [c for c in self.host.calls if tuple(c[: len(prefix)]) == prefix]

    def cleanup(self):
        self.tmp.cleanup()
