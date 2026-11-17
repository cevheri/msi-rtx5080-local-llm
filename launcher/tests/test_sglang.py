import copy
import dataclasses
import tempfile
import tomllib
import unittest
from pathlib import Path

from launcher import model
from launcher.errors import RecipeError
from launcher.hfcache import HfRepoModel
from launcher.model import GpuInfo, HostEnv, RuntimeConfig, Source
from launcher.runtimes import sglang
from launcher.tests.sglang_fixture import FIXTURES, fixture_catalog, registered
from launcher.validate import read_toml

ENV_KEYS = ["HF_HUB_OFFLINE", "HF_MODULES_CACHE", "MAX_JOBS", "FLASHINFER_NO_DOWNLOAD", "SYNC_TOKEN_IDS_ACROSS_TP",
            "NCCL_MAX_P2P_NCHANNELS", "TIKTOKEN_ENCODINGS_BASE"]
GPU = GpuInfo(total_mib=16303, reserved_mib=462, used_mib=0, free_mib=15841)
FRACTION_TEXT = "--mem-fraction-static is required, a number above 0 and at most 0.90 (sglang/README.md#memory-budget)"


def where(stem: str) -> str:
    return str(FIXTURES / f"{stem}.toml")


def load_runtime(edit=None):
    data, source = read_toml(FIXTURES / "_runtime.toml", "shipped")
    data = copy.deepcopy(data)
    if edit:
        edit(data)
    return sglang.parse_runtime(data, source)


def load(stem: str, edit=None):
    data, source = read_toml(FIXTURES / f"{stem}.toml", "shipped")
    data = copy.deepcopy(data)
    if edit:
        edit(data)
    return sglang.parse_recipe(data, source, stem)


def args_of(stem: str) -> list[str]:
    data, _ = read_toml(FIXTURES / f"{stem}.toml", "shipped")
    return list(data["server"]["args"])


def swap(args: list[str], flag: str, *new: str) -> list[str]:
    """Replace a flag and its value with new tokens (none drops both)."""
    index = args.index(flag)
    return args[:index] + list(new) + args[index + 2:]


def drop(args: list[str], flag: str) -> list[str]:
    """Remove a flag that takes no value."""
    return [arg for arg in args if arg != flag]


def with_args(args: list[str]):
    def edit(data):
        data["server"]["args"] = list(args)
    return edit


class SglangTestCase(unittest.TestCase):
    def refusal(self, stem: str, edit) -> str:
        with self.assertRaises(RecipeError) as ctx:
            load(stem, edit)
        return str(ctx.exception)

    def runtime_refusal(self, edit) -> str:
        with self.assertRaises(RecipeError) as ctx:
            load_runtime(edit)
        return str(ctx.exception)

    def assert_args_refused(self, stem: str, args: list[str], text: str, section: str = "[server]"):
        self.assertEqual(self.refusal(stem, with_args(args)), f"{where(stem)} {section}: {text}")


class SglangDataTest(SglangTestCase):
    def test_constants(self):
        self.assertEqual((sglang.NAME, sglang.LABEL, sglang.CONTAINER_PORT), ("sglang", "SGLang", 30000))
        self.assertEqual(sglang.CONTAINER_HF_HOME, "/root/.cache/huggingface")
        self.assertEqual(sglang.MODEL_MOUNT, "/llmctl/model")
        self.assertEqual(sglang.LAUNCH, ("python3", "-m", "sglang.launch_server"))
        self.assertEqual(sglang.HF_COMMAND, ("-m", "huggingface_hub.cli.hf"))
        self.assertEqual((sglang.MODELS_PATH, sglang.SERVER_INFO_PATH, sglang.FLUSH_PATH),
                         ("/v1/models", "/server_info", "/flush_cache"))
        self.assertEqual(sglang.SAMPLING_KEYS, ("temperature", "top_p", "top_k", "min_p", "repetition_penalty"))
        self.assertEqual(sglang.MAX_FRACTION, 0.90)
        self.assertEqual(sglang.OWNED_FLAGS, frozenset({"--model-path", "--model", "--served-model-name",
                                                        "--context-length", "--host", "--port", "--revision"}))
        self.assertEqual(sglang.SECRET_FLAGS, frozenset({"--api-key", "--admin-api-key", "--config"}))
        for name in ("SERVED_NAME", "_ALIAS", "_NAME", "_REPO", "_REVISION", "_VOLUME"):
            self.assertTrue(hasattr(sglang, name), name)

    def test_shared_data_defaults(self):
        self.assertIsNone(HostEnv(Path("/h"), 1, 1).cache_dir)
        runtime = RuntimeConfig("x", "img", "c", 1, "/", 10, (), (), "p", None, Source("shipped", Path("/x"), "0"))
        self.assertFalse(runtime.refuses_tools_without_parser)
        self.assertFalse(runtime.effort_enables_toggle_thinking)
        check = model.StartCheck(("x",), None)
        self.assertEqual((check.lines, check.refusal), (("x",), None))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            check.refusal = "y"

    def test_runtime_file_parses(self):
        runtime = load_runtime()
        self.assertIsInstance(runtime, sglang.SglangRuntimeConfig)
        self.assertEqual((runtime.name, runtime.image, runtime.container_name, runtime.host_port),
                         ("sglang", "lmsysorg/sglang:v0.5.23-runtime", "sglang", 30000))
        self.assertEqual((runtime.ready_path, runtime.start_timeout_s, runtime.opencode_provider),
                         ("/ready", 900, "sglang-docker"))
        self.assertEqual(runtime.volumes, ("sglang-cache:/root/.cache/sglang",))
        self.assertEqual([key for key, _ in runtime.env], ENV_KEYS)
        self.assertEqual(runtime.min_free_at_ready_mib, 640)
        self.assertIsNone(runtime.served_name)
        self.assertIs(runtime.effort_enables_toggle_thinking, True)
        self.assertIs(runtime.refuses_tools_without_parser, False)

    def test_runtime_refusals(self):
        file = str(FIXTURES / "_runtime.toml")
        cases = [
            (lambda d: d.update(served_name="x"), f"{file}: unknown key 'served_name'"),
            (lambda d: d.update(default_gpu_memory_utilization=0.9),
             f"{file}: unknown key 'default_gpu_memory_utilization'"),
            (lambda d: d.update(effort_enables_toggle_thinking=False),
             f"{file}: unknown key 'effort_enables_toggle_thinking'"),
            (lambda d: d.update(volumes=["/:/host"]),
             f"{file}: volume '/:/host' must be a named volume such as 'sglang-cache:/root/.cache/sglang'"),
            (lambda d: d.update(min_free_at_ready_mib=-1),
             f"{file}: 'min_free_at_ready_mib' must be between 0 and 65536, found -1"),
            (lambda d: d.update(min_free_at_ready_mib=65537),
             f"{file}: 'min_free_at_ready_mib' must be between 0 and 65536, found 65537"),
            (lambda d: d.pop("min_free_at_ready_mib"), f"{file}: missing key 'min_free_at_ready_mib'"),
            (lambda d: d.update(runtime="vllm"), f"{file}: 'runtime' must be 'sglang'"),
        ]
        for edit, text in cases:
            with self.subTest(text=text):
                self.assertEqual(self.runtime_refusal(edit), text)
        self.assertEqual(load_runtime(lambda d: d.update(min_free_at_ready_mib=0)).min_free_at_ready_mib, 0)


class SglangRecipeTest(SglangTestCase):
    def test_fixture_recipes_parse(self):
        recipes = {stem: load(stem) for stem in ("qwen3.5-9b", "k2-horizon-7b", "gpt-oss-20b",
                                                 "deepseek-r1-0528-qwen3-8b")}
        for recipe in recipes.values():
            self.assertIsInstance(recipe.model, sglang.SglangModel)
            self.assertIsInstance(recipe.model, HfRepoModel)
            self.assertEqual(recipe.runtime, "sglang")
        self.assertEqual(recipes["qwen3.5-9b"].model.sampling, (("temperature", 0.6), ("top_p", 0.95), ("top_k", 20)))
        self.assertEqual(recipes["k2-horizon-7b"].model.sampling, (("temperature", 1.0), ("top_p", 0.95)))
        self.assertEqual(recipes["gpt-oss-20b"].model.sampling, (("temperature", 1.0), ("top_p", 1.0)))
        self.assertEqual(recipes["deepseek-r1-0528-qwen3-8b"].model.sampling, (("temperature", 0.6), ("top_p", 0.95)))
        self.assertEqual({stem: (r.needs_gpu_mib, r.needs_ram_gib) for stem, r in recipes.items()}, {
            "qwen3.5-9b": (15500, 9.5), "k2-horizon-7b": (15500, 9.0), "gpt-oss-20b": (15200, 31.5),
            "deepseek-r1-0528-qwen3-8b": (15500, 9.0)})
        self.assertTrue(recipes["deepseek-r1-0528-qwen3-8b"].chat_only)
        self.assertFalse(recipes["qwen3.5-9b"].chat_only)
        self.assertEqual(recipes["gpt-oss-20b"].model.exclude, ("original/*", "metal/*"))

    def test_sampling_order_follows_sampling_keys(self):
        recipe = load("qwen3.5-9b", lambda d: d["server"].update(sampling={"top_k": 20, "temperature": 0.6}))
        self.assertEqual(recipe.model.sampling, (("temperature", 0.6), ("top_k", 20)))
        self.assertEqual(load("qwen3.5-9b", lambda d: d["server"].pop("sampling")).model.sampling, ())

    def test_owned_and_secret_flags(self):
        args = args_of("qwen3.5-9b")
        for flag in sorted(sglang.OWNED_FLAGS):
            with self.subTest(flag=flag):
                self.assert_args_refused("qwen3.5-9b", args + [flag, "x"],
                                         f"'{flag}' is set by llmctl and may not appear in 'args'")
        self.assert_args_refused("qwen3.5-9b", args + ["--port=1"],
                                 "'--port' is set by llmctl and may not appear in 'args'")
        for flag in ("--api-key", "--admin-api-key"):
            with self.subTest(flag=flag):
                self.assert_args_refused("qwen3.5-9b", args + [flag, "x"],
                                         f"'{flag}' would put a secret on the command line and may not appear in 'args'")
        self.assert_args_refused("qwen3.5-9b", args + ["--config", "x.yaml"],
                                 "'--config' could carry a key in its YAML file and may not appear in 'args'")

    def test_abbreviations_of_owned_and_secret_flags(self):
        args = args_of("qwen3.5-9b")
        self.assert_args_refused("qwen3.5-9b", args + ["--context-len", "4096"],
                                 "'--context-len' is set by llmctl and may not appear in 'args' "
                                 "(it abbreviates --context-length)")
        self.assert_args_refused("qwen3.5-9b", args + ["--mod", "x"],
                                 "'--mod' is set by llmctl and may not appear in 'args' "
                                 "(it abbreviates --model or --model-path)")
        self.assert_args_refused("qwen3.5-9b", args + ["--admin-api=x"],
                                 "'--admin-api' would put a secret on the command line and may not appear in 'args' "
                                 "(it abbreviates --admin-api-key)")
        self.assert_args_refused("qwen3.5-9b", args + ["--conf", "x.yaml"],
                                 "'--conf' could carry a key in its YAML file and may not appear in 'args' "
                                 "(it abbreviates --config)")

    def test_checked_flags_count_in_every_form(self):
        args = args_of("qwen3.5-9b")
        for short, full in (("--mem-fraction", "--mem-fraction-static"), ("--sampling-def", "--sampling-defaults"),
                            ("--max-running-req", "--max-running-requests")):
            with self.subTest(flag=short):
                value = "openai" if short == "--sampling-def" else "0.97" if short == "--mem-fraction" else "2"
                self.assert_args_refused("qwen3.5-9b", args + [short, value],
                                         f"'{short}' abbreviates {full}; write the flag in full")
        self.assert_args_refused("qwen3.5-9b", args + ["--disable-"],
                                 "'--disable-' abbreviates --disable-cuda-graph or --disable-prefill-cuda-graph; "
                                 "write the flag in full")
        self.assert_args_refused("qwen3.5-9b", args + ["--mem-fraction-static", "0.97"],
                                 "'--mem-fraction-static' appears more than once in 'args'; "
                                 "SGLang uses the last one, so write it once")
        self.assert_args_refused("qwen3.5-9b", swap(args, "--tool-call-parser", "--tool-call-parser=auto"),
                                 "--tool-call-parser needs an explicit parser name, not auto", "[clients]")
        self.assertEqual(load("qwen3.5-9b", with_args(args + ["--disable-cuda-graph-padding"])).id, "qwen3.5-9b")
        self.assertEqual(load("qwen3.5-9b", with_args(args + ["--enable-metrics"])).id, "qwen3.5-9b")

    def test_fraction_rule(self):
        args = args_of("qwen3.5-9b")
        for case in (swap(args, "--mem-fraction-static"), swap(args, "--mem-fraction-static", "--mem-fraction-static", "0"),
                     swap(args, "--mem-fraction-static", "--mem-fraction-static", "0.91"),
                     swap(args, "--mem-fraction-static", "--mem-fraction-static", "abc")):
            with self.subTest(args=case):
                self.assert_args_refused("qwen3.5-9b", case, FRACTION_TEXT)
        for case in (swap(args, "--mem-fraction-static", "--mem-fraction-static", "0.88"),
                     swap(args, "--mem-fraction-static", "--mem-fraction-static", "0.90"),
                     swap(args, "--mem-fraction-static", "--mem-fraction-static=0.5")):
            with self.subTest(args=case):
                self.assertEqual(load("qwen3.5-9b", with_args(case)).id, "qwen3.5-9b")

    def test_running_requests_rule(self):
        args = args_of("qwen3.5-9b")
        text = "SGLang recipes run one request at a time: --max-running-requests 1 is required"
        self.assert_args_refused("qwen3.5-9b", swap(args, "--max-running-requests"), text)
        self.assert_args_refused("qwen3.5-9b", swap(args, "--max-running-requests", "--max-running-requests", "2"),
                                 text)

    def test_prefill_rule(self):
        args = drop(args_of("qwen3.5-9b"), "--disable-prefill-cuda-graph")
        self.assertEqual(load("qwen3.5-9b", with_args(args)).id, "qwen3.5-9b")

        def estimate(data):
            data["server"]["args"] = list(args)
            data["evidence"] = {"kind": "estimate", "summary": "x", "doc": "sglang/models.md#qwen35-9b-nvfp4"}
        self.assertEqual(self.refusal("qwen3.5-9b", estimate),
                         f"{where('qwen3.5-9b')} [server]: keep --disable-prefill-cuda-graph until the prefill CUDA "
                         "graph is measured with this recipe ([evidence] kind = \"measured\")")

    def test_deprecated_and_sampling_flags(self):
        args = args_of("qwen3.5-9b")
        self.assert_args_refused("qwen3.5-9b", args + ["--disable-cuda-graph"],
                                 "--disable-cuda-graph is deprecated in SGLang v0.5.23 and also turns the decode "
                                 "graph off; use --disable-prefill-cuda-graph")
        self.assert_args_refused("qwen3.5-9b", args + ["--sampling-defaults", "openai"],
                                 "--sampling-defaults must stay model, or the overlay's sampling is ignored")
        self.assert_args_refused("qwen3.5-9b", args + ["--preferred-sampling-params", "{}"],
                                 "--preferred-sampling-params does not reach chat requests in SGLang v0.5.23; "
                                 "pin sampling in [server] sampling")
        self.assertEqual(load("qwen3.5-9b", with_args(args + ["--sampling-defaults", "model"])).id, "qwen3.5-9b")

    def test_parsers_need_names(self):
        args = args_of("qwen3.5-9b")
        tool = "--tool-call-parser needs an explicit parser name, not auto"
        reasoning = "--reasoning-parser needs an explicit parser name, not auto"
        self.assert_args_refused("qwen3.5-9b", swap(args, "--tool-call-parser", "--tool-call-parser", "auto"), tool,
                                 "[clients]")
        self.assert_args_refused("qwen3.5-9b", swap(args, "--reasoning-parser", "--reasoning-parser=auto"), reasoning,
                                 "[clients]")
        self.assert_args_refused("qwen3.5-9b", swap(args, "--tool-call-parser") + ["--tool-call-parser"], tool,
                                 "[clients]")
        for flag, text in (("--tool-call-parser", tool), ("--reasoning-parser", reasoning)):
            with self.subTest(flag=flag):  # a missing value in the middle: the next flag is not a parser name
                self.assert_args_refused("qwen3.5-9b", swap(args, flag, flag, "--enable-metrics"), text, "[clients]")

    def test_paged_experts_rules(self):
        args = args_of("gpt-oss-20b")
        cases = [
            (swap(args, "--paged-experts-num-resident"), "--enable-paged-experts needs --paged-experts-num-resident <K>"),
            (swap(args, "--paged-experts-num-resident", "--paged-experts-num-resident", "0"),
             "--paged-experts-num-resident needs a positive integer, found '0'"),
            (swap(args, "--paged-experts-num-resident", "--paged-experts-num-resident", "x"),
             "--paged-experts-num-resident needs a positive integer, found 'x'"),
            (swap(args, "--moe-runner-backend"), "--enable-paged-experts needs --moe-runner-backend marlin"),
            (swap(args, "--moe-runner-backend", "--moe-runner-backend", "triton"),
             "--enable-paged-experts needs --moe-runner-backend marlin"),
            (drop(args, "--enable-paged-experts"),
             "--paged-experts-num-resident works only with --enable-paged-experts"),
        ]
        for case, text in cases:
            with self.subTest(text=text):
                self.assert_args_refused("gpt-oss-20b", case, text)
        self.assertEqual(load("gpt-oss-20b").args, tuple(args))

    def test_client_rules(self):
        qwen = args_of("qwen3.5-9b")
        self.assertEqual(self.refusal("qwen3.5-9b", lambda d: d.update(clients={"output": 16384, "thinking": "none"})),
                         f"{where('qwen3.5-9b')} [clients]: thinking = \"none\" may not set --reasoning-parser")
        self.assert_args_refused("qwen3.5-9b", swap(qwen, "--reasoning-parser"),
                                 "thinking = \"toggle\" needs --reasoning-parser", "[clients]")
        self.assert_args_refused("qwen3.5-9b", swap(qwen, "--default-chat-template-kwargs",
                                                    "--default-chat-template-kwargs", "[1]"),
                                 "'--default-chat-template-kwargs' must be followed by a JSON object, found '[1]'",
                                 "[clients]")
        self.assert_args_refused("qwen3.5-9b", swap(qwen, "--default-chat-template-kwargs",
                                                    "--default-chat-template-kwargs", '{"enable_thinking":true}'),
                                 "default = \"off\" but --default-chat-template-kwargs sets enable_thinking to true",
                                 "[clients]")
        self.assert_args_refused("qwen3.5-9b", swap(qwen, "--tool-call-parser"),
                                 "a recipe with tools (chat_only false) needs --tool-call-parser", "[clients]")
        self.assert_args_refused("deepseek-r1-0528-qwen3-8b",
                                 args_of("deepseek-r1-0528-qwen3-8b") + ["--tool-call-parser", "qwen3_coder"],
                                 "a chat-only recipe may not set --tool-call-parser", "[clients]")

    def test_sampling_table_bounds(self):
        file = where("qwen3.5-9b")

        def sampling(table):
            return lambda d: d["server"].update(sampling=table)
        self.assertEqual(self.refusal("qwen3.5-9b", sampling({"temprature": 0.6})),
                         f"{file} [server] sampling: unknown key 'temprature' (did you mean 'temperature'?)")
        bounds = {
            "temperature": "a number from 0 to 2",
            "top_p": "a number above 0 and at most 1",
            "top_k": "-1 or an integer from 1 to 1000000",
            "min_p": "a number from 0 to 1",
            "repetition_penalty": "a number above 0 and at most 2",
        }
        refused = [("top_k", 0, "0"), ("top_k", 1.5, "1.5"), ("top_k", -2, "-2"), ("top_k", True, "true"),
                   ("top_p", 0, "0"), ("temperature", 2.5, "2.5"), ("temperature", -0.1, "-0.1"),
                   ("temperature", True, "true"), ("temperature", float("nan"), "NaN"), ("min_p", 1.5, "1.5"),
                   ("repetition_penalty", 0, "0"), ("repetition_penalty", 2.5, "2.5")]
        for key, value, shown in refused:
            with self.subTest(key=key, value=value):
                self.assertEqual(self.refusal("qwen3.5-9b", sampling({key: value})),
                                 f"{file} [server] sampling: '{key}' must be {bounds[key]}, found {shown}")
        for text, shown in (("temperature = 2026-11-17", '"2026-11-17"'), ("temperature = 07:32:00", '"07:32:00"')):
            with self.subTest(toml=text):
                with self.assertRaises(RecipeError) as ctx:
                    load("qwen3.5-9b", sampling(tomllib.loads(text)))
                self.assertEqual(ctx.exception.exit_code, 2)
                self.assertEqual(str(ctx.exception), f"{file} [server] sampling: 'temperature' must be "
                                                     f"{bounds['temperature']}, found {shown}")
        for key, value in (("top_p", 1), ("temperature", 0), ("temperature", 2), ("top_k", -1)):
            with self.subTest(key=key, value=value):
                self.assertEqual(load("qwen3.5-9b", sampling({key: value})).model.sampling, ((key, value),))

    def test_needs_table(self):
        file = where("qwen3.5-9b")
        cases = [
            (lambda d: d.pop("needs"), f"{file}: missing key 'needs'"),
            (lambda d: d["needs"].pop("gpu_mib"), f"{file} [needs]: missing key 'gpu_mib'"),
            (lambda d: d["needs"].pop("ram_gib"), f"{file} [needs]: missing key 'ram_gib'"),
            (lambda d: d["needs"].update(gpu_mib=100), f"{file} [needs]: 'gpu_mib' must be between 256 and 262144, "
                                                       "found 100"),
            (lambda d: d["needs"].update(ram_gib=33), f"{file} [needs]: ram_gib 33 exceeds [container] memory_gib 32, "
                                                      "so the container would be killed at its measured peak"),
        ]
        for edit, text in cases:
            with self.subTest(text=text):
                self.assertEqual(self.refusal("qwen3.5-9b", edit), text)

    def test_members(self):
        runtime = load_runtime()
        qwen, gpt_oss = load("qwen3.5-9b"), load("gpt-oss-20b")
        self.assertEqual(sglang.served_name(qwen, runtime), "qwen3.5-9b")
        self.assertEqual(sglang.gpu_need_mib(qwen, runtime, GPU), 15500)
        self.assertEqual(sglang.gpu_need_mib(qwen, runtime, GpuInfo(8000, 0, 0, 8000)), 15500)
        self.assertEqual(sglang.gpu_need_text(qwen, runtime), "15500 MiB")
        self.assertEqual(sglang.gpu_advice(gpt_oss),
                         "write an override with fewer resident experts (--paged-experts-num-resident) and the "
                         "gpu_mib you measure with it (sglang/README.md#paged-experts)")
        self.assertEqual(sglang.gpu_advice(qwen),
                         "write an override with a smaller --mem-fraction-static or context and the gpu_mib you "
                         "measure with it (sglang/README.md#memory-budget)")

    def test_fixture_catalog(self):
        with tempfile.TemporaryDirectory() as tmp, registered():
            catalog = fixture_catalog(Path(tmp))
            self.assertEqual(sorted(catalog.runtimes["sglang"].recipes),
                             ["deepseek-r1-0528-qwen3-8b", "gpt-oss-20b", "k2-horizon-7b", "qwen3.5-9b"])
            for name, entry in catalog.runtimes.items():
                self.assertEqual([p.message for p in entry.problems], [], name)
        from launcher.runtimes import ADAPTERS
        self.assertNotIn("sglang", ADAPTERS)


if __name__ == "__main__":
    unittest.main()
