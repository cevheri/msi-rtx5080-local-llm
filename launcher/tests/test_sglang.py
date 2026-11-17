import copy
import dataclasses
import http.client
import io
import json
import os
import stat
import tempfile
import tomllib
import unittest
from pathlib import Path

from launcher import gpu, model
from launcher.command import docker_run_argv
from launcher.errors import DownloadError, PreflightError, RecipeError, UsageError
from launcher.hfcache import HfAuth, HfRepoModel, snapshot_dir
from launcher.model import GpuInfo, HostEnv, RuntimeConfig, Source, StartCheck
from launcher.runtimes import sglang
from launcher.tests.fakes import FakeHost, fail, http as response, ok
from launcher.tests.sglang_fixture import (
    FIXTURES,
    QWEN_GENERATION_CONFIG,
    SGLANG,
    fixture_catalog,
    gpu_line,
    place_snapshot,
    registered,
    server_info,
)
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


QWEN_REV = "a4df85a20fb7e54740f564f15af0c2f5d9c6616c"
QWEN_SNAPSHOT = f"/root/.cache/huggingface/hub/models--RedHatAI--Qwen3.5-9B-NVFP4/snapshots/{QWEN_REV}"
QWEN_SAMPLING = "and generation_config.json with temperature 0.6, top_p 0.95, top_k 20"
OWN_FILE = "so the checkpoint's own generation_config.json applies"
SECRET = "hf_secret123456789"


def without_sampling(recipe):
    return dataclasses.replace(recipe, model=dataclasses.replace(recipe.model, sampling=()))


def links_of(folder: Path) -> dict[str, str]:
    """Every symbolic link under folder, by its relative path, with its target."""
    found = {}
    for path in sorted(folder.rglob("*")):
        if path.is_symlink():
            found[path.relative_to(folder).as_posix()] = os.readlink(path)
    return found


class OverlayTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.hf = self.root / "hf"
        self.cache = self.root / "cache" / "llmctl"
        self.env = HostEnv(self.hf, 1000, 1000, self.cache)
        self.runtime = load_runtime()
        self.qwen = load("qwen3.5-9b")
        self.folder = self.cache / "sglang" / "overlays" / "qwen3.5-9b"

    def tearDown(self):
        self.tmp.cleanup()

    def snapshot(self, recipe, files: dict[str, bytes]) -> Path:
        """A snapshot written directly, file by file, without place_snapshot's model.safetensors."""
        snapshot = snapshot_dir(self.hf, recipe.model.repo, recipe.model.revision)
        for name, data in files.items():
            (snapshot / name).parent.mkdir(parents=True, exist_ok=True)
            (snapshot / name).write_bytes(data)
        return snapshot

    def prepare(self, recipe=None, dry_run=False):
        return sglang.prepare_start(recipe or self.qwen, self.runtime, self.env, dry_run=dry_run)


class SglangContainerTest(OverlayTestCase):
    def test_container_spec(self):
        spec = sglang.container_spec(self.qwen, self.runtime, self.env)
        self.assertIsNone(spec.user)
        self.assertEqual(spec.env, self.runtime.env)
        self.assertEqual(spec.mounts, (f"{self.hf}:/root/.cache/huggingface:ro",
                                       f"{self.folder}:/llmctl/model:ro", "sglang-cache:/root/.cache/sglang"))
        self.assertEqual(spec.extra_flags, ("--ipc=host",))
        self.assertEqual(spec.image, "lmsysorg/sglang:v0.5.23-runtime")
        prefix = ("python3", "-m", "sglang.launch_server", "--model-path", "/llmctl/model", "--served-model-name",
                  "qwen3.5-9b", "--context-length", "131072", "--host", "0.0.0.0", "--port", "30000")
        self.assertEqual(spec.args, prefix + self.qwen.args)
        self.assertEqual(spec.container_port, 30000)
        self.assertEqual(docker_run_argv(self.qwen, self.runtime, spec), [
            "docker", "run", "-d", "--name", "sglang", "--label", "io.llmctl.managed=true",
            "--label", "io.llmctl.runtime=sglang", "--label", "io.llmctl.recipe=qwen3.5-9b",
            "--label", "io.llmctl.source=shipped", "--label", f"io.llmctl.sha256={self.qwen.source.sha256}",
            "--label", "io.llmctl.evidence=measured", "--label", "io.llmctl.runtime-source=shipped",
            "--label", f"io.llmctl.runtime-sha256={self.runtime.source.sha256}",
            "--device", "nvidia.com/gpu=all", "--memory", "32g", "--memory-swap", "32g",
            "-p", "127.0.0.1:30000:30000", "-e", "HF_HUB_OFFLINE=1", "-e", "HF_MODULES_CACHE=/tmp/hf_modules",
            "-e", "MAX_JOBS=2", "-e", "FLASHINFER_NO_DOWNLOAD=1", "-e", "SYNC_TOKEN_IDS_ACROSS_TP=1",
            "-e", "NCCL_MAX_P2P_NCHANNELS=1",
            "-e", "TIKTOKEN_ENCODINGS_BASE=/opt/sglang/lib/python3.12/site-packages/tml_renderers/data",
            "-v", f"{self.hf}:/root/.cache/huggingface:ro", "-v", f"{self.folder}:/llmctl/model:ro",
            "-v", "sglang-cache:/root/.cache/sglang", "--ipc=host", "lmsysorg/sglang:v0.5.23-runtime",
            "python3", "-m", "sglang.launch_server", "--model-path", "/llmctl/model",
            "--served-model-name", "qwen3.5-9b", "--context-length", "131072", "--host", "0.0.0.0",
            "--port", "30000", "--mem-fraction-static", "0.88", "--max-running-requests", "1",
            "--max-mamba-cache-size", "8", "--disable-prefill-cuda-graph", "--tool-call-parser", "qwen3_coder",
            "--reasoning-parser", "qwen3", "--default-chat-template-kwargs", '{"enable_thinking":false}'])

    def test_container_spec_needs_an_absolute_cache_dir(self):
        with self.assertRaises(ValueError):
            sglang.container_spec(self.qwen, self.runtime, HostEnv(self.hf, 1000, 1000))
        for cache in (Path("cache"), Path("~/.cache/llmctl")):
            with self.subTest(cache=str(cache)), self.assertRaises(UsageError) as ctx:
                sglang.container_spec(self.qwen, self.runtime, HostEnv(self.hf, 1000, 1000, cache))
            self.assertEqual(ctx.exception.exit_code, 1)
            self.assertEqual(str(ctx.exception), f"llmctl's cache folder {cache} is not an absolute path; "
                                                 "set XDG_CACHE_HOME (or HOME) to one")


class SglangOverlayTest(OverlayTestCase):
    def test_overlay_links_every_snapshot_file(self):
        self.snapshot(self.qwen, {"config.json": b"{}", "tokenizer.json": b"{}", ".gitattributes": b"",
                                  "generation_config.json": QWEN_GENERATION_CONFIG, "assets/a.png": b"png"})
        lines = self.prepare()
        self.assertEqual(links_of(self.folder), {name: f"{QWEN_SNAPSHOT}/{name}" for name in
                                                 (".gitattributes", "assets/a.png", "config.json", "tokenizer.json")})
        self.assertTrue((self.folder / "assets").is_dir())
        self.assertFalse((self.folder / "assets").is_symlink())
        self.assertFalse((self.folder / "generation_config.json").is_symlink())
        self.assertTrue((self.folder / "generation_config.json").is_file())
        self.assertEqual(lines, (f"wrote {self.folder}: model overlay with 4 links into snapshot a4df85a20fb7 of "
                                 f"RedHatAI/Qwen3.5-9B-NVFP4, {QWEN_SAMPLING}",))

    def test_generation_config_is_merged(self):
        place_snapshot(self.hf, self.qwen.model)
        self.prepare()
        written = self.folder / "generation_config.json"
        after = {"_from_model_config": True, "eos_token_id": 248044, "transformers_version": "5.21.0",
                 "use_cache": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
        self.assertEqual(written.read_text(), json.dumps(after, indent=2) + "\n")
        self.assertEqual(stat.S_IMODE(written.stat().st_mode), 0o644)

        place_snapshot(self.hf, self.qwen.model, {"generation_config.json": b'{"a": 1, "temperature": 0.7, "b": 2}'})
        self.prepare()
        self.assertEqual(list(json.loads(written.read_text()).items()),
                         [("a", 1), ("temperature", 0.6), ("b", 2), ("top_p", 0.95), ("top_k", 20)])

    def test_without_sampling_the_link_stays(self):
        place_snapshot(self.hf, self.qwen.model)
        lines = self.prepare(without_sampling(self.qwen))
        self.assertEqual(links_of(self.folder)["generation_config.json"], f"{QWEN_SNAPSHOT}/generation_config.json")
        self.assertTrue(lines[0].endswith(f"RedHatAI/Qwen3.5-9B-NVFP4, {OWN_FILE}"), lines[0])

        place_snapshot(self.hf, self.qwen.model, {"generation_config.json": None})
        (snapshot_dir(self.hf, self.qwen.model.repo, QWEN_REV) / "generation_config.json").unlink()
        self.prepare()
        written = self.folder / "generation_config.json"
        self.assertFalse(written.is_symlink())
        self.assertEqual(json.loads(written.read_text()), {"temperature": 0.6, "top_p": 0.95, "top_k": 20})

        self.prepare(without_sampling(self.qwen))
        self.assertFalse(os.path.lexists(written))

    def test_rebuild_follows_the_snapshot(self):
        snapshot = self.snapshot(self.qwen, {"config.json": b"{}", "tokenizer.json": b"{}",
                                             "generation_config.json": QWEN_GENERATION_CONFIG})
        self.prepare()
        (snapshot / "extra.json").write_bytes(b"{}")
        (snapshot / "tokenizer.json").unlink()
        (self.folder / "junk").write_bytes(b"x")
        self.prepare()
        self.assertEqual(links_of(self.folder), {name: f"{QWEN_SNAPSHOT}/{name}"
                                                 for name in ("config.json", "extra.json")})
        self.assertFalse(os.path.lexists(self.folder / "junk"))

        revision = "b" * 40
        moved = dataclasses.replace(self.qwen, model=dataclasses.replace(self.qwen.model, revision=revision))
        self.snapshot(moved, {"config.json": b"{}", "tokenizer.json": b"{}", "sub/x.bin": b"x"})
        lines = self.prepare(moved)
        root = f"/root/.cache/huggingface/hub/models--RedHatAI--Qwen3.5-9B-NVFP4/snapshots/{revision}"
        self.assertEqual(links_of(self.folder), {name: f"{root}/{name}"
                                                 for name in ("config.json", "sub/x.bin", "tokenizer.json")})
        self.assertIn(f"into snapshot {revision[:12]} of", lines[0])

    def test_leftovers_are_removed(self):
        place_snapshot(self.hf, self.qwen.model)
        overlays = self.folder.parent
        (overlays / ".qwen3.5-9b.llmctl-tmp").mkdir(parents=True)
        (overlays / ".qwen3.5-9b.llmctl-tmp" / "x").write_bytes(b"x")
        (overlays / ".qwen3.5-9b.llmctl-old").mkdir()
        (overlays / ".qwen3.5-9b.llmctl-old" / "y").write_bytes(b"y")
        self.prepare()
        self.assertEqual(sorted(path.name for path in overlays.iterdir()), ["qwen3.5-9b"])

    def test_dry_run_writes_nothing(self):
        lines = self.prepare(dry_run=True)
        self.assertEqual(lines, (f"would write {self.folder}: model overlay with links into snapshot a4df85a20fb7 of "
                                 f"RedHatAI/Qwen3.5-9B-NVFP4, {QWEN_SAMPLING}",))
        self.assertFalse(self.cache.exists())
        lines = self.prepare(without_sampling(self.qwen), dry_run=True)
        self.assertTrue(lines[0].endswith(OWN_FILE))
        self.assertFalse(self.cache.exists())

    def test_overlay_failures_are_exit_3(self):
        with self.assertRaises(PreflightError) as ctx:
            self.prepare()
        self.assertEqual(str(ctx.exception), f"cannot write the model overlay {self.folder}: missing snapshot "
                                             f"{QWEN_REV} of RedHatAI/Qwen3.5-9B-NVFP4")
        self.assertEqual(ctx.exception.exit_code, 3)

        place_snapshot(self.hf, self.qwen.model, {"generation_config.json": b"[1]"})
        with self.assertRaises(PreflightError) as ctx:
            self.prepare()
        self.assertEqual(str(ctx.exception), f"cannot write the model overlay {self.folder}: generation_config.json "
                                             "in snapshot a4df85a20fb7 is not a JSON object (its top level is not an "
                                             "object), so the overlay cannot add the recipe's sampling; delete that "
                                             "file and its blob from the Hugging Face cache, then pull again")
        self.assertFalse(os.path.lexists(self.folder))
        self.assertFalse(os.path.lexists(self.folder.parent / ".qwen3.5-9b.llmctl-tmp"))

        if os.geteuid() == 0:
            self.skipTest("root writes into a read-only folder")
        place_snapshot(self.hf, self.qwen.model)
        overlays = self.folder.parent
        overlays.mkdir(parents=True, exist_ok=True)
        overlays.chmod(0o500)
        try:
            with self.assertRaises(PreflightError) as ctx:
                self.prepare()
        finally:
            overlays.chmod(0o700)
        text = str(ctx.exception)
        self.assertTrue(text.startswith(f"cannot write the model overlay {self.folder}: "), text)
        self.assertTrue(text.endswith(f"; check the free space and permissions of {self.cache}"), text)
        self.assertEqual(ctx.exception.exit_code, 3)

    def test_file_problems(self):
        place_snapshot(self.hf, self.qwen.model)
        self.assertEqual(sglang.file_problems(self.qwen, self.hf), [])
        problem_start = "generation_config.json in snapshot a4df85a20fb7 is not a JSON object ("
        problem_end = ("), so the overlay cannot add the recipe's sampling; delete that file and its blob from the "
                       "Hugging Face cache, then pull again")
        for data in (b"not json", b"[1]"):
            with self.subTest(data=data):
                place_snapshot(self.hf, self.qwen.model, {"generation_config.json": data})
                problems = sglang.file_problems(self.qwen, self.hf)
                self.assertEqual(len(problems), 1, problems)
                self.assertTrue(problems[0].startswith(problem_start), problems[0])
                self.assertTrue(problems[0].endswith(problem_end), problems[0])
                self.assertEqual(sglang.file_problems(without_sampling(self.qwen), self.hf), [])
        snapshot = place_snapshot(self.hf, self.qwen.model)
        with open(snapshot / "model.safetensors", "r+b") as handle:
            handle.truncate(10)
        problems = sglang.file_problems(self.qwen, self.hf)
        self.assertEqual(len(problems), 1, problems)
        self.assertTrue(problems[0].startswith("the snapshot holds "), problems[0])
        self.assertTrue(problems[0].endswith(", the recipe expects 9381214599; the download is incomplete"))


def models_body(*names: str) -> bytes:
    return json.dumps({"data": [{"id": name} for name in names]}).encode()


class SglangReadyTest(unittest.TestCase):
    def setUp(self):
        self.runtime = load_runtime()
        self.qwen = load("qwen3.5-9b")

    def test_ready(self):
        host = FakeHost()
        host.on_http("GET", f"{SGLANG}/ready", response(503))
        host.on_http("GET", f"{SGLANG}/v1/models", response(200, models_body("qwen3.5-9b")))
        self.assertFalse(sglang.ready(host, self.runtime, self.qwen))
        host.on_http("GET", f"{SGLANG}/ready", response(200))
        host.on_http("GET", f"{SGLANG}/v1/models", response(200, models_body("other")))
        self.assertFalse(sglang.ready(host, self.runtime, self.qwen))
        host.on_http("GET", f"{SGLANG}/v1/models", response(200, models_body("other", "qwen3.5-9b")))
        host.http_calls.clear()
        self.assertTrue(sglang.ready(host, self.runtime, self.qwen))
        self.assertEqual([url for _, url, _ in host.http_calls], [f"{SGLANG}/ready", f"{SGLANG}/v1/models"])
        host.on_http("GET", f"{SGLANG}/v1/models", response(200, b"<html>"))
        self.assertFalse(sglang.ready(host, self.runtime, self.qwen))
        errors = (ConnectionResetError(), http.client.RemoteDisconnected("closed"), http.client.IncompleteRead(b""))
        for error in errors:
            with self.subTest(error=type(error).__name__):
                host = FakeHost()
                host.on_http("GET", f"{SGLANG}/ready", error)
                self.assertFalse(sglang.ready(host, self.runtime, self.qwen))


SERVER_INFO_START = f"could not read internal_states[0].memory_usage.token_capacity from {SGLANG}/server_info: "
SERVER_INFO_END = " (sglang/README.md#readiness-and-the-memory-check)"
POOL_TEXT = ("the KV pool holds 120000 tokens, fewer than the context 131072, so a session that fills the context "
             "would fail; write an override with a smaller context, FP8 KV (--kv-cache-dtype fp8_e4m3) or fewer Mamba "
             "slots (sglang/README.md#memory-budget)")


class SglangCheckTest(unittest.TestCase):
    def setUp(self):
        self.runtime = load_runtime()
        self.qwen = load("qwen3.5-9b")

    def host(self, capacity=140976, flush=None, free=1013, base=SGLANG):
        host = FakeHost()
        host.on_http("GET", f"{base}/server_info", server_info(capacity) if isinstance(capacity, int) else capacity)
        host.on_http("POST", f"{base}/flush_cache", response(200) if flush is None else flush)
        host.on(gpu.QUERY, gpu_line(free) if isinstance(free, int) else free)
        return host

    def free_text(self, free: int, need: int = 640) -> str:
        return (f"only {free} MiB of GPU memory is free after the start, below the {need} MiB that long prompts and "
                f"grammar requests need (min_free_at_ready_mib in {self.runtime.source.path}); close other GPU "
                "programs, or write an override with a --mem-fraction-static 0.02 lower, which frees about 300 MiB "
                "(sglang/README.md#memory-budget)")

    def test_check_passes(self):
        host = self.host()
        seen = []
        host.on(gpu.QUERY, lambda argv: seen.append(len(host.http_calls)) or gpu_line(1013))
        check = sglang.check_started(host, self.runtime, self.qwen)
        self.assertEqual(check, StartCheck(("memory: KV pool 140976 tokens for context 131072; 1013 MiB of GPU "
                                            "memory free after the start, at least 640 MiB needed",), None))
        self.assertEqual([(method, url) for method, url, _ in host.http_calls],
                         [("GET", f"{SGLANG}/server_info"), ("POST", f"{SGLANG}/flush_cache")])
        self.assertEqual(host.calls, [gpu.QUERY])
        self.assertEqual(seen, [2])

    def test_short_pool_and_low_memory(self):
        line = ("memory: KV pool 120000 tokens for context 131072; 1013 MiB of GPU memory free after the start, "
                "at least 640 MiB needed")
        self.assertEqual(sglang.check_started(self.host(capacity=120000), self.runtime, self.qwen),
                         StartCheck((line,), POOL_TEXT))
        check = sglang.check_started(self.host(free=600), self.runtime, self.qwen)
        self.assertTrue(check.refusal.startswith("only 600 MiB of GPU memory is free after the start, below the 640 "
                                                 "MiB"), check.refusal)
        self.assertIn(str(self.runtime.source.path), check.refusal)
        self.assertEqual(check.refusal, self.free_text(600))
        check = sglang.check_started(self.host(capacity=120000, free=600), self.runtime, self.qwen)
        self.assertEqual(check.refusal, f"{POOL_TEXT}; {self.free_text(600)}")
        self.assertEqual(len(check.lines), 1)

    def test_zero_threshold_keeps_the_pool_check(self):
        runtime = dataclasses.replace(self.runtime, min_free_at_ready_mib=0)
        check = sglang.check_started(self.host(free=1), runtime, self.qwen)
        self.assertIsNone(check.refusal)
        self.assertEqual(check.lines, ("memory: KV pool 140976 tokens for context 131072; 1 MiB of GPU memory free "
                                       "after the start, at least 0 MiB needed",))
        check = sglang.check_started(self.host(capacity=120000, free=1), runtime, self.qwen)
        self.assertEqual(check.refusal, POOL_TEXT)

    def test_server_info_failures(self):
        def body(value):
            return response(200, json.dumps(value).encode())

        def capacity(value):
            return body({"internal_states": [{"memory_usage": {"token_capacity": value}}]})

        cases = [response(500), response(200, b"<html>"), body({}), body({"internal_states": []}),
                 capacity("140976"), capacity(True), OSError("connection refused")]
        for case in cases:
            with self.subTest(case=case):
                host = self.host(capacity=case)
                check = sglang.check_started(host, self.runtime, self.qwen)
                self.assertEqual(check.lines, ())
                self.assertTrue(check.refusal.startswith(SERVER_INFO_START), check.refusal)
                self.assertTrue(check.refusal.endswith(SERVER_INFO_END), check.refusal)
                self.assertEqual([method for method, _, _ in host.http_calls], ["GET"])
                self.assertEqual(host.calls, [])

    def test_flush_and_nvidia_smi_failures(self):
        host = self.host(flush=response(500))
        check = sglang.check_started(host, self.runtime, self.qwen)
        self.assertEqual(check, StartCheck((), f"POST {SGLANG}/flush_cache answered 500, so llmctl could not read "
                                               "the free memory after the start"))
        self.assertEqual(host.calls, [])
        check = sglang.check_started(self.host(flush=OSError("timed out")), self.runtime, self.qwen)
        self.assertEqual(check.refusal, f"POST {SGLANG}/flush_cache failed (timed out), so llmctl could not read the "
                                        "free memory after the start")
        check = sglang.check_started(self.host(capacity=120000, flush=response(500)), self.runtime, self.qwen)
        self.assertEqual(check.refusal, f"{POOL_TEXT}; POST {SGLANG}/flush_cache answered 500, so llmctl could not "
                                        "read the free memory after the start")
        check = sglang.check_started(self.host(free=fail("NVML error")), self.runtime, self.qwen)
        self.assertEqual(check, StartCheck((), "nvidia-smi failed: NVML error; is the NVIDIA driver loaded?"))

    def test_check_uses_the_runtime_port(self):
        runtime = dataclasses.replace(self.runtime, host_port=30001)
        base = "http://127.0.0.1:30001"
        check = sglang.check_started(self.host(base=base), runtime, self.qwen)
        self.assertIsNone(check.refusal)
        check = sglang.check_started(self.host(capacity=response(500), base=base), runtime, self.qwen)
        self.assertTrue(check.refusal.startswith(f"could not read internal_states[0].memory_usage.token_capacity "
                                                 f"from {base}/server_info: "), check.refusal)
        check = sglang.check_started(self.host(flush=response(500), base=base), runtime, self.qwen)
        self.assertTrue(check.refusal.startswith(f"POST {base}/flush_cache answered 500"), check.refusal)


class SglangPullTest(OverlayTestCase):
    def test_pull_runs_the_python_module(self):
        host = FakeHost()
        host.on(("docker", "run"), lambda argv: place_snapshot(self.hf, self.qwen.model) and ok())
        out = io.StringIO()
        sglang.pull(host, self.qwen, self.runtime, self.hf, HfAuth(SECRET, None), out)
        argv = host.calls[0]
        self.assertEqual(argv[argv.index("--entrypoint"):], [
            "--entrypoint", "python3", "lmsysorg/sglang:v0.5.23-runtime", "-m", "huggingface_hub.cli.hf",
            "download", "RedHatAI/Qwen3.5-9B-NVFP4", "--revision", QWEN_REV])
        self.assertFalse(any(SECRET in item for item in argv))
        self.assertNotIn(SECRET, out.getvalue())

        host = FakeHost()
        host.on(("docker", "run"),
                lambda argv: place_snapshot(self.hf, self.qwen.model, {"generation_config.json": b"[1]"}) and ok())
        with self.assertRaises(DownloadError) as ctx:
            sglang.pull(host, self.qwen, self.runtime, self.hf, HfAuth(None, None), out)
        self.assertTrue(str(ctx.exception).startswith("qwen3.5-9b: generation_config.json in snapshot"),
                        str(ctx.exception))
        self.assertTrue(str(ctx.exception).endswith("; delete that file and its blob from the Hugging Face cache, "
                                                    "then pull again"), str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
