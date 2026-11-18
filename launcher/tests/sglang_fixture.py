"""The SGLang adapter before its registration: fixture recipes under launcher/tests/fixtures/sglang/.

Their values stand for measurements and are not measurements; nothing copies them into a shipped recipe or a document.
"""

import json
import shutil
from pathlib import Path
from unittest import mock

from launcher.catalog import Catalog, load_catalog
from launcher.hfcache import snapshot_dir
from launcher.host import HttpResponse, Result
from launcher.runtimes import ADAPTERS, sglang
from launcher.tests.fakes import REPO, http, ok

FIXTURES = REPO / "launcher/tests/fixtures/sglang"
SGLANG = "http://127.0.0.1:30000"
# Qwen3.5-9B-NVFP4's generation_config.json as measurement A found it in the snapshot, before the overlay.
QWEN_GENERATION_CONFIG = (b'{"_from_model_config": true, "eos_token_id": 248044, "transformers_version": "5.21.0", '
                          b'"use_cache": true}\n')


def registered():
    """Register the adapter for a test: use it with `with`, or call start() and stop()."""
    return mock.patch.dict(ADAPTERS, {"sglang": sglang})


def recipes_with_sglang(root: Path) -> Path:
    """Copy the repository's recipes to root/recipes with the fixture recipes as its sglang/ folder."""
    recipes = Path(root) / "recipes"
    shutil.copytree(REPO / "recipes", recipes)
    if (recipes / "sglang").exists():
        shutil.rmtree(recipes / "sglang")
    shutil.copytree(FIXTURES, recipes / "sglang")
    return recipes


def fixture_catalog(root: Path) -> Catalog:
    """The catalog of the copied recipes; call it while registered()."""
    return load_catalog(recipes_with_sglang(root), None)


def place_snapshot(hf_home: Path, model, files: dict[str, bytes | None] | None = None) -> Path:
    """A complete snapshot of the model: config.json and Qwen's generation_config.json, each replaced by the entry
    of files with its name (None leaves it out), the other files, then a sparse model.safetensors so the snapshot
    adds up to model.size_bytes."""
    files = {"config.json": b"{}", "generation_config.json": QWEN_GENERATION_CONFIG, **(files or {})}
    files = {name: data for name, data in files.items() if data is not None}
    snapshot = snapshot_dir(Path(hf_home), model.repo, model.revision)
    snapshot.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (snapshot / name).parent.mkdir(parents=True, exist_ok=True)
        (snapshot / name).write_bytes(data)
    with open(snapshot / "model.safetensors", "wb") as handle:
        handle.truncate(model.size_bytes - sum(len(data) for data in files.values()))
    return snapshot


def server_info(capacity) -> HttpResponse:
    return http(200, json.dumps({"internal_states": [{"memory_usage": {"token_capacity": capacity}}]}).encode())


def gpu_line(free: int) -> Result:
    """nvidia-smi's memory line with free MiB free; gpu_line(15731) is the line the other tests use."""
    return ok(f"16303, 462, {15843 - free}, {free}\n")
