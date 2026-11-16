import datetime
import hashlib
import tempfile
import unittest
from pathlib import Path

from launcher import model, validate
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


class DecodeSpeedTest(unittest.TestCase):
    base = {"kind": "measured", "summary": "s", "date": "2026-11-16", "hardware": "h", "record": "r/"}

    def test_speed_is_optional_and_bounded(self):
        self.assertIsNone(parse_evidence(dict(self.base), "w").decode_tok_s)
        self.assertEqual(parse_evidence(dict(self.base, decode_tok_s=122.5), "w").decode_tok_s, 122.5)
        for bad in ("fast", 0, 100000):
            with self.assertRaises(RecipeError):
                parse_evidence(dict(self.base, decode_tok_s=bad), "w")

    def test_estimate_may_not_state_a_speed(self):
        with self.assertRaises(RecipeError) as ctx:
            parse_evidence({"kind": "estimate", "summary": "s", "doc": "d", "decode_tok_s": 90}, "w")
        self.assertIn("decode_tok_s", str(ctx.exception))


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


class ClientsTest(unittest.TestCase):
    base = {"output": 16384, "thinking": "toggle", "default": "off"}

    def parse(self, data, context=131072):
        return validate.parse_clients(data, "w", context=context)

    def assertRefused(self, data, *fragments, context=131072):
        with self.assertRaises(RecipeError) as ctx:
            self.parse(data, context)
        message = str(ctx.exception)
        self.assertTrue(message.startswith("w: "), message)
        for fragment in fragments:
            self.assertIn(fragment, message)
        return message

    def test_toggle_parses(self):
        self.assertEqual(self.parse(dict(self.base)), model.ClientMeta(16384, "toggle", "off", (), ()))

    def test_effort_parses(self):
        meta = self.parse({"output": 16384, "thinking": "effort", "efforts": ["none", "low", "medium"]})
        self.assertEqual(meta.efforts, ("none", "low", "medium"))
        self.assertIsNone(meta.default)
        self.assertEqual(meta.medium_kwargs, ())

    def test_always_and_none_parse(self):
        for thinking in ("always", "none"):
            self.assertEqual(self.parse({"output": 16384, "thinking": thinking}),
                             model.ClientMeta(16384, thinking, None, (), ()))

    def test_medium_kwargs_keep_order(self):
        meta = self.parse(dict(self.base, medium_kwargs={"medium_effort": True, "b": 1}))
        self.assertEqual(meta.medium_kwargs, (("medium_effort", True), ("b", 1)))

    def test_clients_unknown_key_suggests(self):
        data = dict(self.base)
        data["outptu"] = data.pop("output")
        self.assertRefused(data, "unknown key 'outptu'", "did you mean 'output'")

    def test_output_required_and_bounded(self):
        data = dict(self.base)
        del data["output"]
        self.assertRefused(data, "missing key 'output'")
        for bad in (1023, 65537):
            self.assertRefused(dict(self.base, output=bad), "must be between 1024 and 65536")

    def test_output_bound_is_inclusive(self):
        for edge in (1024, 65536):
            self.assertEqual(self.parse(dict(self.base, output=edge)).output, edge)

    def test_thinking_must_be_known(self):
        self.assertRefused(dict(self.base, thinking="sometimes"),
                           "'thinking' must be one of toggle, effort, always, none, found 'sometimes'")

    def test_default_rules(self):
        data = dict(self.base)
        del data["default"]
        self.assertRefused(data, "missing key 'default'")
        self.assertRefused(dict(self.base, default="maybe"), "'default'")
        self.assertRefused({"output": 16384, "thinking": "effort", "efforts": ["medium"], "default": "off"},
                           "'default' applies only to thinking = \"toggle\"")
        self.assertEqual(self.parse(dict(self.base, default="on")).default, "on")

    def test_efforts_rules(self):
        effort = {"output": 16384, "thinking": "effort"}
        self.assertRefused(dict(effort), "missing key 'efforts'")
        for bad in ([], ["medium", "low"], ["turbo", "medium"], ["low", "high"], ["medium", "medium"], "medium"):
            with self.subTest(efforts=bad):
                self.assertRefused(dict(effort, efforts=bad), "'efforts'")
        self.assertRefused(dict(self.base, efforts=["medium"]), "'efforts' applies only to thinking = \"effort\"")

    def test_single_effort_is_valid(self):
        meta = self.parse({"output": 16384, "thinking": "effort", "efforts": ["medium"]})
        self.assertEqual(meta.efforts, ("medium",))

    def test_medium_kwargs_rules(self):
        self.assertRefused({"output": 16384, "thinking": "effort", "efforts": ["medium"],
                            "medium_kwargs": {"a": True}},
                           "'medium_kwargs' applies only to thinking = \"toggle\"")
        self.assertRefused(dict(self.base, medium_kwargs={"enable_thinking": True}),
                           "'medium_kwargs' may not set enable_thinking")
        for bad in ({"a": {"b": 1}}, {"a": [1]}):
            with self.subTest(medium_kwargs=bad):
                self.assertRefused(dict(self.base, medium_kwargs=bad), "'medium_kwargs.a' must be a string")
        self.assertRefused(dict(self.base, medium_kwargs="x"), "'medium_kwargs' must be a table")

    def test_medium_kwargs_must_be_json_scalars(self):
        for bad in (float("nan"), float("inf"), datetime.date(2026, 11, 16)):
            with self.subTest(value=bad):
                self.assertRefused(dict(self.base, medium_kwargs={"a": bad}),
                                   "'medium_kwargs.a' must be a string, boolean, integer or finite number")
        meta = self.parse(dict(self.base, medium_kwargs={"s": "x", "f": 0.5, "n": -1, "b": False}))
        self.assertEqual(meta.medium_kwargs, (("s", "x"), ("f", 0.5), ("n", -1), ("b", False)))


class FlagTest(unittest.TestCase):
    def test_flag_value_forms(self):
        self.assertEqual(validate.flag_value(("--reasoning", "off"), "--reasoning"), "off")
        self.assertEqual(validate.flag_value(("--reasoning=off",), "--reasoning"), "off")
        self.assertIsNone(validate.flag_value(("--reasoning-effort", "high"), "--reasoning"))
        self.assertEqual(validate.flag_value(("--x",), "--x"), "")
        self.assertIsNone(validate.flag_value((), "--x"))

    def test_flag_present_forms(self):
        self.assertTrue(validate.flag_present(("-t", "8", "--jinja"), "--jinja"))
        self.assertTrue(validate.flag_present(("--reasoning=off",), "--reasoning"))
        self.assertFalse(validate.flag_present(("--reasoning-effort", "high"), "--reasoning"))
        self.assertFalse(validate.flag_present(("x", "--reasoning"), "--jinja"))


if __name__ == "__main__":
    unittest.main()
