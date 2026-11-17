import json
import unittest

from launcher.catalog import load_catalog, resolve
from launcher.smoke import smoke
from launcher.tests.fakes import REPO, FakeHost, http

URL = "http://127.0.0.1:8080/v1/chat/completions"


def reply(message, finish="stop", **extra):
    return http(200, json.dumps({"choices": [{"message": message, "finish_reason": finish}], **extra}).encode())


def server(tool_message, tool_finish):
    def answer(method, url, headers, body):
        request = json.loads(body)
        assert request["model"] == "local", request["model"]
        if "tools" in request:
            return reply(tool_message, tool_finish)
        if "lighthouse" in request["messages"][0]["content"]:
            return reply({"content": "Once..."}, usage={"completion_tokens": 512},
                         timings={"predicted_per_second": 120.5})
        return reply({"content": "OK"})
    return answer


class SmokeTest(unittest.TestCase):
    def setUp(self):
        self.recipe, self.runtime = resolve(load_catalog(REPO / "recipes", None), "qwen3.6", None)

    def test_smoke_pass(self):
        host = FakeHost()
        call = {"function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}
        host.on_http("POST", URL, server({"content": None, "tool_calls": [call]}, "tool_calls"))
        self.assertEqual(smoke(host, self.recipe, self.runtime), 0)
        out = host.out.getvalue()
        self.assertIn("reply: ok", out)
        self.assertIn("tool call: ok (get_weather", out)
        self.assertIn("decode: 120.5 tok/s", out)

    def test_smoke_tool_call_missing(self):
        host = FakeHost()
        host.on_http("POST", URL, server({"content": "It is sunny in Paris."}, "stop"))
        self.assertEqual(smoke(host, self.recipe, self.runtime), 4)
        self.assertIn("the tool call did not arrive", host.out.getvalue())

    def test_reasoning_only_reply_counts(self):
        host = FakeHost()
        call = {"function": {"name": "get_weather", "arguments": "{}"}}

        def answer(method, url, headers, body):
            request = json.loads(body)
            if "tools" in request:
                return reply({"tool_calls": [call]}, "tool_calls")
            if "lighthouse" in request["messages"][0]["content"]:
                return reply({"content": "x"}, usage={"completion_tokens": 10}, timings={"predicted_per_second": 50.0})
            return reply({"content": "", "reasoning_content": "The user wants OK..."}, "length")

        host.on_http("POST", URL, answer)
        self.assertEqual(smoke(host, self.recipe, self.runtime), 0)
        self.assertIn("reply: ok (reasoning only", host.out.getvalue())

    def smoke_with_reply(self, content):
        host = FakeHost()
        call = {"function": {"name": "get_weather", "arguments": "{}"}}

        def answer(method, url, headers, body):
            request = json.loads(body)
            if "tools" in request:
                return reply({"tool_calls": [call]}, "tool_calls")
            if "lighthouse" in request["messages"][0]["content"]:
                return reply({"content": "x"}, usage={"completion_tokens": 10}, timings={"predicted_per_second": 50.0})
            return reply({"content": content})

        host.on_http("POST", URL, answer)
        return smoke(host, self.recipe, self.runtime), host.out.getvalue()

    def test_garbled_reply_fails(self):
        for content in ("\u010aOK", "NOT OK", "BOOK", "OKAY", "OK, but no"):
            with self.subTest(content=content):
                code, out = self.smoke_with_reply(content)
                self.assertEqual(code, 4)
                self.assertIn(f"reply: FAILED, expected OK, got {content!r}", out)

    def test_exact_ok_reply_passes(self):
        for content in ("OK", "ok.", "**OK**", " OK\n", '"OK"', "'OK'", "`OK`", "_OK_", "OK!", "OK,", '" OK"',
                        '"\u00a0OK"', "*\u2003OK\u3000*"):
            with self.subTest(content=content):
                code, out = self.smoke_with_reply(content)
                self.assertEqual(code, 0, out)
                self.assertIn("reply: ok", out)


if __name__ == "__main__":
    unittest.main()
