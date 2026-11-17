"""The SGLang adapter before its registration: fixture recipes under launcher/tests/fixtures/sglang/.

Their values stand for measurements and are not measurements; nothing copies them into a shipped recipe or a document.
"""

import shutil
from pathlib import Path
from unittest import mock

from launcher.catalog import Catalog, load_catalog
from launcher.runtimes import ADAPTERS, sglang
from launcher.tests.fakes import REPO

FIXTURES = REPO / "launcher/tests/fixtures/sglang"


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
