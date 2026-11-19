"""launcher/preview.py: the files the preview watches and the build it serves."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from launcher import preview
from launcher.site import data
from launcher.tests.site_fixture import SiteFixture

FACTS = data.Build("https://github.com/o/r", "a" * 40, 0)


class PreviewTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.fixture = SiteFixture(Path(self._tmp.name))

    def test_a_changed_source_changes_what_the_preview_watches(self):
        root = self.fixture.root
        before = preview.watched(root)
        recipe = sorted((root / "recipes").rglob("*.toml"))[0]
        os.utime(recipe, ns=(0, 0))
        self.assertNotEqual(preview.watched(root), before)

    def test_bytecode_and_superpowers_folders_are_not_watched(self):
        root = self.fixture.root
        before = preview.watched(root)
        for folder in (root / "launcher/__pycache__", root / "docs/superpowers"):
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "x").write_text("x", encoding="utf-8")
        self.assertEqual(preview.watched(root), before)

    def test_the_preview_builds_the_site_into_a_fresh_folder(self):
        out = preview.build_into(self.fixture.root, FACTS)
        self.addCleanup(shutil.rmtree, out)
        self.assertTrue((out / "index.html").is_file())
        self.assertTrue(any((out / "models").iterdir()))

    def test_a_failed_build_names_its_source(self):
        (self.fixture.root / "README.md").unlink()
        with self.assertRaisesRegex(data.SiteError, r"^README\.md"):
            preview.build_into(self.fixture.root, FACTS)


if __name__ == "__main__":
    unittest.main()
