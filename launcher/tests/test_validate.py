import hashlib
import tempfile
import unittest
from pathlib import Path

from launcher.errors import RecipeError
from launcher.validate import (
    check_keys,
    get_bool,
    get_float,
    get_int,
    get_str,
    get_str_list,
    parse_evidence,
    read_toml,
)


class CheckKeysTest(unittest.TestCase):
    def test_unknown_key_suggests(self):
        with self.assertRaises(RecipeError) as ctx:
            check_keys({"memmory_gib": 1}, {"memory_gib"}, "x.toml [container]")
        message = str(ctx.exception)
        self.assertIn("x.toml [container]", message)
        self.assertIn("unknown key 'memmory_gib'", message)
        self.assertIn("did you mean 'memory_gib'", message)

    def test_unknown_key_without_close_match(self):
        with self.assertRaises(RecipeError) as ctx:
            check_keys({"restart": "always"}, {"memory_gib", "context"}, "w")
        self.assertIn("unknown key 'restart'", str(ctx.exception))
        self.assertNotIn("did you mean", str(ctx.exception))

    def test_known_keys_pass(self):
        check_keys({"memory_gib": 1}, {"memory_gib", "context"}, "w")


class GettersTest(unittest.TestCase):
    def test_int_bounds(self):
        with self.assertRaises(RecipeError) as ctx:
            get_int({"a": 0}, "a", "w", lo=1, hi=5)
        self.assertIn("must be between 1 and 5", str(ctx.exception))
        self.assertEqual(get_int({"a": 5}, "a", "w", lo=1, hi=5), 5)

    def test_int_rejects_bool(self):
        with self.assertRaises(RecipeError) as ctx:
            get_int({"a": True}, "a", "w", lo=0, hi=5)
        self.assertIn("must be an integer", str(ctx.exception))

    def test_missing_required_names_key(self):
        with self.assertRaises(RecipeError) as ctx:
            get_str({}, "image", "llamacpp/_runtime.toml")
        self.assertIn("llamacpp/_runtime.toml: missing key 'image'", str(ctx.exception))
        self.assertIsNone(get_str({}, "image", "w", required=False))

    def test_bool_rejects_string(self):
        with self.assertRaises(RecipeError) as ctx:
            get_bool({"opencode": "true"}, "opencode", "w", default=False)
        self.assertIn("must be true or false", str(ctx.exception))
        self.assertTrue(get_bool({}, "chat_only", "w", default=True))

    def test_float_accepts_int_and_bounds(self):
        self.assertEqual(get_float({"a": 15}, "a", "w", lo=0, hi=64), 15.0)
        with self.assertRaises(RecipeError):
            get_float({"a": 65.5}, "a", "w", lo=0, hi=64)

    def test_str_list_rejects_mixed(self):
        with self.assertRaises(RecipeError) as ctx:
            get_str_list({"args": ["-t", 8]}, "args", "w")
        self.assertIn("must be a list of strings", str(ctx.exception))
        self.assertEqual(get_str_list({"args": ["-t", "8"]}, "args", "w"), ("-t", "8"))


class EvidenceTest(unittest.TestCase):
    def test_measured_evidence_needs_record(self):
        with self.assertRaises(RecipeError) as ctx:
            parse_evidence({"kind": "measured", "summary": "s", "date": "2026-11-16", "hardware": "h"}, "w")
        self.assertIn("record", str(ctx.exception))

    def test_estimate_needs_doc(self):
        with self.assertRaises(RecipeError) as ctx:
            parse_evidence({"kind": "estimate", "summary": "s"}, "w")
        self.assertIn("doc", str(ctx.exception))

    def test_unknown_kind(self):
        with self.assertRaises(RecipeError) as ctx:
            parse_evidence({"kind": "guess", "summary": "s"}, "w")
        self.assertIn("'measured' or 'estimate'", str(ctx.exception))

    def test_measured_evidence_parses(self):
        evidence = parse_evidence(
            {"kind": "measured", "summary": "s", "date": "2026-11-16", "hardware": "h", "record": "r/"}, "w"
        )
        self.assertEqual((evidence.kind, evidence.record, evidence.doc), ("measured", "r/", None))


class ReadTomlTest(unittest.TestCase):
    def test_toml_syntax_error_names_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.toml"
            path.write_text('id = "x"\nmemory_gib = = 3\n')
            with self.assertRaises(RecipeError) as ctx:
                read_toml(path, "override")
            self.assertIn(str(path), str(ctx.exception))
            self.assertIn("line 2", str(ctx.exception))

    def test_sha256_of_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.toml"
            raw = b'id = "qwen"\n# comment\n'
            path.write_bytes(raw)
            data, source = read_toml(path, "shipped")
            self.assertEqual(data, {"id": "qwen"})
            self.assertEqual(source.sha256, hashlib.sha256(raw).hexdigest())
            self.assertEqual((source.kind, source.path), ("shipped", path))


if __name__ == "__main__":
    unittest.main()
