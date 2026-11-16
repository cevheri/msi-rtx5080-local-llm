"""Environment paths, plain JSON, atomic writes, backups and the client errors."""

import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from launcher import files
from launcher.errors import ClientError, ClientWriteError, PreflightError, UsageError


def mode_of(path: Path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


class ErrorsTest(unittest.TestCase):
    def test_client_error_exit_codes(self):
        self.assertEqual(ClientError("x").exit_code, 3)
        self.assertIsInstance(ClientError("x"), PreflightError)
        self.assertEqual(ClientWriteError("x").exit_code, 6)


class EnvPathTest(unittest.TestCase):
    def test_home_dir_needs_home(self):
        for env in ({}, {"HOME": ""}):
            with self.subTest(env=env):
                with self.assertRaises(UsageError) as caught:
                    files.home_dir(env)
                self.assertIn("HOME", str(caught.exception))

    def test_env_path_uses_the_mapping(self):
        env = {"HOME": "/h", "X": "/x"}
        self.assertEqual(files.env_path(env, "~"), Path("/h"))
        self.assertEqual(files.env_path(env, "~/a"), Path("/h/a"))
        self.assertEqual(files.env_path(env, "$X/b"), Path("/x/b"))
        self.assertEqual(files.env_path(env, "${X}/b"), Path("/x/b"))
        self.assertEqual(files.env_path(env, "$NOPE/c"), Path("$NOPE/c"))
        with self.assertRaises(UsageError):
            files.env_path(env, "~bob/x")


class JsonTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_parse_json_plain_only(self):
        for text in ("NaN", "Infinity", "-Infinity", "1e400", '{"a": 1,}', "// c\n{}"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    files.parse_json(text)
        self.assertEqual(files.parse_json('{"a": [1, 2.5]}'), {"a": [1, 2.5]})

    def test_load_json_object(self):
        path = self.dir / "models.json"
        self.assertEqual(files.load_json_object(path, "Pi"), (None, {}))
        path.write_text('{"a": 1}')
        self.assertEqual(files.load_json_object(path, "Pi"), ('{"a": 1}', {"a": 1}))
        path.write_text("[1]")
        with self.assertRaises(ClientError) as caught:
            files.load_json_object(path, "Pi")
        self.assertIn(f"{path} is not a JSON object", str(caught.exception))
        path.write_text('{"a": 1,}')
        with self.assertRaises(ClientError) as caught:
            files.load_json_object(path, "Pi")
        self.assertIn(f"{path} is not plain JSON (", str(caught.exception))
        self.assertIn("leave Pi out with LLMCTL_CLIENTS", str(caught.exception))

    def test_load_json_object_refuses_lone_surrogate(self):
        path = self.dir / "settings.json"
        path.write_text('{"a": "\\ud800"}', encoding="utf-8")
        self.assertEqual(path.read_bytes(), b'{"a": "\\ud800"}')
        with self.assertRaises(ClientError) as caught:
            files.load_json_object(path, "Pi")
        self.assertEqual(str(caught.exception),
                         f"{path} holds an escaped lone surrogate (\\ud800 to \\udfff) that llmctl cannot write back "
                         "as UTF-8; fix it, or leave Pi out with LLMCTL_CLIENTS")

    def test_load_json_object_unreadable(self):
        path = self.dir / "dir.json"
        path.mkdir()
        with self.assertRaises(ClientError) as caught:
            files.load_json_object(path, "Pi")
        self.assertIn(f"cannot read {path}: ", str(caught.exception))


class WriteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(os.path.realpath(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def leftovers(self) -> list[str]:
        return sorted(p.name for p in self.dir.iterdir() if p.name.endswith(".llmctl-tmp"))

    def test_write_atomic_new_file(self):
        path = self.dir / "new.json"
        files.write_atomic(path, "{}\n")
        self.assertEqual(path.read_text(), "{}\n")
        self.assertEqual(mode_of(path), 0o600)
        self.assertEqual(self.leftovers(), [])

    def test_write_atomic_new_mode(self):
        path = self.dir / "new.yaml"
        files.write_atomic(path, "a: 1\n", new_mode=0o640)
        self.assertEqual(mode_of(path), 0o640)

    def test_write_atomic_keeps_mode(self):
        path = self.dir / "old.json"
        path.write_text("old")
        os.chmod(path, 0o640)
        files.write_atomic(path, "new")
        self.assertEqual(path.read_text(), "new")
        self.assertEqual(mode_of(path), 0o640)

    def test_write_atomic_follows_symlink(self):
        real = self.dir / "real.json"
        real.write_text("old")
        link = self.dir / "link"
        link.symlink_to("real.json")
        files.write_atomic(link, "new")
        self.assertTrue(link.is_symlink())
        self.assertEqual(real.read_text(), "new")
        self.assertEqual(self.leftovers(), [])

    def test_write_atomic_failure_keeps_original(self):
        path = self.dir / "old.json"
        path.write_text("old")
        with mock.patch("launcher.files.os.replace", side_effect=OSError("boom")) as replace:
            with self.assertRaises(OSError):
                files.write_atomic(path, "new")
        self.assertEqual(os.fspath(replace.call_args.args[0]), str(self.dir / "old.json.llmctl-tmp"))
        self.assertEqual(path.read_text(), "old")
        self.assertEqual(self.leftovers(), [])

    def test_write_atomic_temp_is_private(self):
        real_fsync = os.fsync
        for existing in (False, True):
            with self.subTest(existing=existing):
                path = self.dir / f"file-{existing}.json"
                if existing:
                    path.write_text("old")
                    os.chmod(path, 0o644)
                modes = []

                def spy(fd):
                    modes.append(stat.S_IMODE(os.fstat(fd).st_mode))
                    real_fsync(fd)

                with mock.patch("launcher.files.os.fsync", side_effect=spy):
                    files.write_atomic(path, "new")
                self.assertEqual(modes[0], 0o600)
                self.assertEqual(path.read_text(), "new")
                self.assertEqual(mode_of(path), 0o644 if existing else 0o600)

    def test_write_atomic_removes_a_leftover_temp(self):
        path = self.dir / "file.json"
        path.write_text("old")
        other = self.dir / "other.txt"
        other.write_text("other")
        (self.dir / "file.json.llmctl-tmp").symlink_to("other.txt")
        files.write_atomic(path, "new")
        self.assertEqual(path.read_text(), "new")
        self.assertEqual(other.read_text(), "other")
        self.assertEqual(self.leftovers(), [])

    def test_backup_path(self):
        self.assertEqual(files.backup_path(Path("/a/b.json")), Path("/a/b.json.llmctl-backup"))

    def test_backup_once(self):
        path = self.dir / "settings.json"
        path.write_text("first")
        os.chmod(path, 0o640)
        backup = files.backup_once(path, "first")
        self.assertEqual(backup, self.dir / "settings.json.llmctl-backup")
        self.assertEqual(backup.read_text(), "first")
        self.assertEqual(mode_of(backup), 0o640)
        self.assertIsNone(files.backup_once(path, "second"))
        self.assertEqual(backup.read_text(), "first")

    def test_backup_once_unencodable_text_leaves_no_backup(self):
        path = self.dir / "settings.json"
        path.write_text("{}\n")
        with self.assertRaises(UnicodeError):
            files.backup_once(path, "\ud800")
        self.assertFalse(os.path.lexists(files.backup_path(path)))

    def test_backup_beside_link(self):
        target_dir = self.dir / "dotfiles"
        target_dir.mkdir()
        real = target_dir / "models.json"
        real.write_text("{}")
        os.chmod(real, 0o604)
        link = self.dir / "models.json"
        link.symlink_to(real)
        backup = files.backup_once(link, "{}")
        self.assertEqual(backup, self.dir / "models.json.llmctl-backup")
        self.assertFalse(backup.is_symlink())
        self.assertEqual(mode_of(backup), 0o604)
        self.assertFalse((target_dir / "models.json.llmctl-backup").exists())

    def test_make_dirs(self):
        existing = self.dir / "existing"
        existing.mkdir()
        os.chmod(existing, 0o755)
        new = self.dir / "existing" / "a"
        newer = new / "b"
        files.make_dirs([existing, new, newer])
        self.assertEqual(mode_of(existing), 0o755)
        self.assertEqual(mode_of(new), 0o700)
        self.assertEqual(mode_of(newer), 0o700)

    def test_file_change_is_frozen(self):
        change = files.FileChange(Path("/a"), None, "x", "new file", "would write /a")
        self.assertEqual((change.mode, change.backup, change.dirs), (0o600, False, ()))
        with self.assertRaises(AttributeError):
            change.new_text = "y"


if __name__ == "__main__":
    unittest.main()
