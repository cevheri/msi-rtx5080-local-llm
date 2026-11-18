import copy
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from launcher.errors import DownloadError, PreflightError, RecipeError, UsageError
from launcher.hfcache import HfAuth, snapshot_dir
from launcher.model import GpuInfo, HostEnv
from launcher.runtimes import vllm
from launcher.tests.fakes import REPO, FakeHost, http, ok
from launcher.validate import read_toml

RECIPES = REPO / "recipes" / "vllm"
GPT_OSS = RECIPES / "gpt-oss-20b.toml"
REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
ENV = HostEnv(Path("/home/user/.cache/huggingface"), 1000, 1000)
MODELS_URL = "http://127.0.0.1:8000/v1/models"


def load_runtime():
    data, source = read_toml(RECIPES / "_runtime.toml", "shipped")
    return vllm.parse_runtime(data, source)


def gpt_oss(edit=None):
    data, source = read_toml(GPT_OSS, "shipped")
    data = copy.deepcopy(data)
    if edit:
        edit(data)
    return vllm.parse_recipe(data, source, "gpt-oss-20b")


class VllmRecipeTest(unittest.TestCase):
    def test_runtime_file_parses(self):
        runtime = load_runtime()
        self.assertEqual((runtime.image, runtime.container_name, runtime.host_port),
                         ("vllm/vllm-openai:v0.33.0", "vllm", 8000))
        self.assertIsNone(runtime.served_name)
        self.assertEqual(runtime.env, (("HF_HUB_OFFLINE", "1"), ("VLLM_NO_USAGE_STATS", "1"), ("MAX_JOBS", "2"),
                                       ("HF_MODULES_CACHE", "/tmp/hf_modules")))
        self.assertEqual(runtime.volumes, ("vllm-cache:/root/.cache/vllm", "flashinfer-cache:/root/.cache/flashinfer"))
        self.assertEqual(runtime.default_gpu_memory_utilization, 0.92)

    def test_runtime_facts(self):
        # vLLM refuses a request with tools on a recipe without --tool-call-parser (HTTP 400); the effort fact is
        # SGLang's only (design D11).
        runtime = load_runtime()
        self.assertEqual((runtime.refuses_tools_without_parser, runtime.effort_enables_toggle_thinking), (True, False))

    def test_runtime_env_secret_refused(self):
        data, source = read_toml(RECIPES / "_runtime.toml", "shipped")
        data["env"]["HF_TOKEN"] = "x"
        with self.assertRaises(RecipeError) as ctx:
            vllm.parse_runtime(data, source)
        self.assertIn("[env]: 'HF_TOKEN' would put a secret into the docker run command line", str(ctx.exception))
        load_runtime()

    def test_runtime_rejects_host_path_volume(self):
        data, source = read_toml(RECIPES / "_runtime.toml", "shipped")
        data["volumes"] = ["/:/host"]
        with self.assertRaises(RecipeError) as ctx:
            vllm.parse_runtime(data, source)
        self.assertIn("named volume", str(ctx.exception))

    def test_gpt_oss_recipe_parses(self):
        recipe = gpt_oss()
        self.assertEqual((recipe.id, recipe.context, recipe.memory_gib), ("gpt-oss-20b", 65536, 32))
        self.assertEqual(recipe.model.revision, REVISION)
        self.assertEqual(recipe.model.exclude, ("original/*", "metal/*"))
        self.assertEqual(recipe.evidence.kind, "measured")
        self.assertTrue(recipe.opencode)

    def test_gpt_oss_container_spec(self):
        spec = vllm.container_spec(gpt_oss(), load_runtime(), ENV)
        self.assertIsNone(spec.user)
        self.assertEqual(spec.env, (("HF_HUB_OFFLINE", "1"), ("VLLM_NO_USAGE_STATS", "1"), ("MAX_JOBS", "2"),
                                    ("HF_MODULES_CACHE", "/tmp/hf_modules")))
        self.assertEqual(spec.mounts, (
            "/home/user/.cache/huggingface:/root/.cache/huggingface:ro",
            "vllm-cache:/root/.cache/vllm",
            "flashinfer-cache:/root/.cache/flashinfer",
        ))
        self.assertEqual(spec.extra_flags, ("--ipc=host",))
        self.assertEqual((spec.image, spec.container_port), ("vllm/vllm-openai:v0.33.0", 8000))
        self.assertEqual(list(spec.args), [
            f"/root/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/{REVISION}",
            "--served-model-name", "gpt-oss-20b", "--max-model-len", "65536",
            "--gpu-memory-utilization", "0.965", "--max-num-seqs", "1", "--max-num-batched-tokens", "256",
            "--enforce-eager", "--kv-cache-dtype", "fp8", "--attention-backend", "TRITON_ATTN",
            "--enable-auto-tool-choice", "--tool-call-parser", "openai", "--reasoning-parser", "openai_gptoss",
            "--reasoning-effort-map", '{"none":"low","minimal":"low","xhigh":"high","max":"high"}',
            "--override-generation-config", '{"temperature":1.0,"top_p":1.0}',
        ])

    def test_gpu_need_from_utilization(self):
        gpu = GpuInfo(total_mib=16303, reserved_mib=462, used_mib=112, free_mib=15731)
        self.assertEqual(vllm.gpu_need_mib(gpt_oss(), load_runtime(), gpu), 15287)

    def test_utilization_in_equals_form(self):
        def edit(data):
            args = data["server"]["args"]
            index = args.index("--gpu-memory-utilization")
            data["server"]["args"] = args[:index] + ["--gpu-memory-utilization=0.5"] + args[index + 2:]
        gpu = GpuInfo(total_mib=16303, reserved_mib=462, used_mib=112, free_mib=15731)
        self.assertEqual(vllm.gpu_need_mib(gpt_oss(edit), load_runtime(), gpu), 7921)

    def test_utilization_defaults_to_runtime(self):
        def edit(data):
            args = data["server"]["args"]
            index = args.index("--gpu-memory-utilization")
            data["server"]["args"] = args[:index] + args[index + 2:]
        gpu = GpuInfo(total_mib=16303, reserved_mib=462, used_mib=112, free_mib=15731)
        self.assertEqual(vllm.gpu_need_mib(gpt_oss(edit), load_runtime(), gpu), 14574)

    def test_bad_utilization_rejected(self):
        def edit(data):
            args = data["server"]["args"]
            index = args.index("--gpu-memory-utilization")
            args[index + 1] = "1.5"
        with self.assertRaises(RecipeError) as ctx:
            gpt_oss(edit)
        self.assertIn("--gpu-memory-utilization", str(ctx.exception))

    def test_utilization_without_value_rejected(self):
        def edit(data):
            args = data["server"]["args"]
            index = args.index("--gpu-memory-utilization")
            data["server"]["args"] = args[:index] + args[index + 2:] + ["--gpu-memory-utilization"]
        with self.assertRaises(RecipeError):
            gpt_oss(edit)

    def test_revision_flag_owned(self):
        for flag in ("--revision", "--served-model-name", "--api-key"):
            with self.assertRaises(RecipeError) as ctx:
                gpt_oss(lambda data, f=flag: data["server"]["args"].extend([f, "x"]))
            self.assertIn(flag, str(ctx.exception))

    def test_served_name_is_recipe_value(self):
        self.assertEqual(vllm.served_name(gpt_oss(), load_runtime()), "gpt-oss-20b")


class VllmFilesTest(unittest.TestCase):
    def test_snapshot_and_config_are_required(self):
        recipe = gpt_oss()
        with tempfile.TemporaryDirectory() as tmp:
            hf_home = Path(tmp)
            snapshot = hf_home / "hub/models--openai--gpt-oss-20b/snapshots" / REVISION
            self.assertEqual(vllm.file_problems(recipe, hf_home), [f"missing snapshot {REVISION} of openai/gpt-oss-20b"])
            snapshot.mkdir(parents=True)
            self.assertEqual(vllm.file_problems(recipe, hf_home), [f"missing config.json in snapshot {REVISION}"])
            (snapshot / "config.json").write_text("{}")
            with open(snapshot / "model.safetensors", "wb") as handle:
                handle.truncate(recipe.model.size_bytes - 2)
            self.assertEqual(vllm.file_problems(recipe, hf_home), [])


class VllmReadyTest(unittest.TestCase):
    def setUp(self):
        self.recipe = gpt_oss()
        self.runtime = load_runtime()

    def test_ready_needs_served_name(self):
        host = FakeHost()
        host.on_http("GET", MODELS_URL, http(200, json.dumps({"data": [{"id": "other"}]}).encode()))
        self.assertFalse(vllm.ready(host, self.runtime, self.recipe))
        host.on_http("GET", MODELS_URL, http(200, json.dumps({"data": [{"id": "gpt-oss-20b"}]}).encode()))
        self.assertTrue(vllm.ready(host, self.runtime, self.recipe))

    def test_not_ready_when_refused_or_garbled(self):
        self.assertFalse(vllm.ready(FakeHost(), self.runtime, self.recipe))
        host = FakeHost()
        host.on_http("GET", MODELS_URL, http(200, b"<html>"))
        self.assertFalse(vllm.ready(host, self.runtime, self.recipe))


def recipe_from(stem, edit=None):
    data, source = read_toml(RECIPES / f"{stem}.toml", "shipped")
    data = copy.deepcopy(data)
    if edit:
        edit(data)
    return vllm.parse_recipe(data, source, stem)


def drop_args(data, *flags):
    """Remove each flag together with its value (the next token unless that is another flag)."""
    args = data["server"]["args"]
    for flag in flags:
        index = args.index(flag)
        end = index + 2 if index + 1 < len(args) and not args[index + 1].startswith("--") else index + 1
        args = args[:index] + args[end:]
    data["server"]["args"] = args


class VllmClientRulesTest(unittest.TestCase):
    def assertRefused(self, stem, edit, *fragments):
        with self.assertRaises(RecipeError) as ctx:
            recipe_from(stem, edit)
        message = str(ctx.exception)
        self.assertIn(str(RECIPES / f"{stem}.toml"), message)
        for fragment in fragments:
            self.assertIn(fragment, message)

    def test_label(self):
        self.assertEqual(vllm.LABEL, "vLLM")

    def test_clients_table_required(self):
        self.assertRefused("gpt-oss-20b", lambda data: data.pop("clients"), "missing key 'clients'")

    def test_clients_parsed_into_recipe(self):
        clients = gpt_oss().clients
        self.assertEqual((clients.thinking, clients.efforts), ("effort", ("low", "medium", "high")))

    def test_reasoning_parser_if_and_only_if_thinking(self):
        def no_thinking(data):
            data["clients"]["thinking"] = "none"
            del data["clients"]["efforts"]
        self.assertRefused("gpt-oss-20b", no_thinking, "[clients]", "--reasoning-parser")
        self.assertRefused("groot-19b-a3b", lambda data: drop_args(data, "--reasoning-parser"),
                           "[clients]", "--reasoning-parser")

    def test_toggle_default_must_match_template_kwargs(self):
        self.assertRefused("groot-19b-a3b", lambda data: data["clients"].update(default="on"),
                           "[clients]", "--default-chat-template-kwargs")
        # Without enable_thinking in the server args, vLLM cannot tell, so either default loads.
        self.assertEqual(recipe_from("mellum2.1-12b").clients.default, "on")
        self.assertEqual(recipe_from("mellum2.1-12b", lambda data: data["clients"].update(default="off")).clients.default,
                         "off")

    def test_tools_need_auto_choice_and_parser(self):
        self.assertRefused("gpt-oss-20b", lambda data: drop_args(data, "--enable-auto-tool-choice"),
                           "--enable-auto-tool-choice")
        self.assertRefused("gpt-oss-20b", lambda data: drop_args(data, "--tool-call-parser"),
                           "--tool-call-parser")

    def test_chat_only_has_no_tool_flags(self):
        self.assertRefused("deepseek-r1-0528-qwen3-8b",
                           lambda data: data["server"]["args"].append("--enable-auto-tool-choice"),
                           "--enable-auto-tool-choice")

    def test_served_name_pattern(self):
        self.assertRefused("gpt-oss-20b", lambda data: data["model"].update(served_name="gpt:oss"), "served_name")
        self.assertTrue(vllm.SERVED_NAME.fullmatch("mellum2.1"))


DEEPSEEK = RECIPES / "deepseek-r1-0528-qwen3-8b.toml"
DS_REVISION = "6e8885a6ff5c1dc5201574c8fd700323f23c25fa"
DS_REPO = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
TENV = HostEnv(Path("/home/user/.cache/huggingface"), 1000, 1000, Path("/home/user/.cache/llmctl"))
DS_SNAPSHOT = f"/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528-Qwen3-8B/snapshots/{DS_REVISION}"
DS_CONFIG = {"add_bos_token": False, "bos_token": "<｜begin▁of▁sentence｜>",
             "tokenizer_class": "LlamaTokenizerFast",
             "chat_template": "{% for m in messages %}{{ m['content'] }}{% endfor %}"}
MISSING_TEXT = ("tokenizer_config.json in snapshot 6e8885a6ff5c is missing, so the tokenizer overlay cannot set "
                "tokenizer_class; remove tokenizer_class from the recipe, or make the download include that file, "
                "then pull again")
NOT_OBJECT_START = "tokenizer_config.json in snapshot 6e8885a6ff5c is not a JSON object ("
NOT_OBJECT_END = ("), so the tokenizer overlay cannot set tokenizer_class; delete that file and its blob from the "
                  "Hugging Face cache, then pull again")


def deepseek(edit=None):
    """The shipped DeepSeek-R1 recipe with [model] tokenizer_class set, then edit applied."""
    data, source = read_toml(DEEPSEEK, "shipped")
    data = copy.deepcopy(data)
    data["model"]["tokenizer_class"] = "PreTrainedTokenizerFast"
    if edit:
        edit(data)
    return vllm.parse_recipe(data, source, "deepseek-r1-0528-qwen3-8b")


def write_snapshot(hf_home: Path, model, files: dict[str, bytes], fill=True) -> Path:
    """A snapshot of files; with fill, a sparse model.safetensors brings it to model.size_bytes."""
    snapshot = snapshot_dir(hf_home, model.repo, model.revision)
    snapshot.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (snapshot / name).parent.mkdir(parents=True, exist_ok=True)
        (snapshot / name).write_bytes(data)
    if fill:
        with open(snapshot / "model.safetensors", "wb") as handle:
            handle.truncate(model.size_bytes - sum(len(data) for data in files.values()))
    return snapshot


def config_bytes(config) -> bytes:
    return json.dumps(config, indent=2, ensure_ascii=False).encode("utf-8")


class VllmTokenizerOverlayTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.hf = root / "hf"
        self.cache = root / "cache" / "llmctl"
        self.env = HostEnv(self.hf, 1000, 1000, self.cache)
        self.runtime = load_runtime()
        self.folder = self.cache / "vllm" / "tokenizers" / "deepseek-r1-0528-qwen3-8b"

    def prepare(self, recipe=None, dry_run=False, env=None):
        return vllm.prepare_start(recipe or deepseek(), self.runtime, env or self.env, dry_run=dry_run)

    def assertRefused(self, edit, fragment, recipe=deepseek):
        with self.assertRaises(RecipeError) as ctx:
            recipe(edit)
        self.assertIn(fragment, str(ctx.exception))

    def test_tokenizer_class_parses(self):
        model = deepseek().model
        self.assertIsInstance(model, vllm.VllmModel)
        self.assertEqual(model.tokenizer_class, "PreTrainedTokenizerFast")
        self.assertIsNone(gpt_oss().model.tokenizer_class)
        qwen2 = deepseek(lambda d: d["model"].update(tokenizer_class="Qwen2TokenizerFast"))
        self.assertEqual(qwen2.model.tokenizer_class, "Qwen2TokenizerFast")
        for value in ("Pre Trained", "x.y", "../t", "9Tok"):
            with self.subTest(value=value):
                self.assertRefused(lambda d, v=value: d["model"].update(tokenizer_class=v),
                                   "[model]: 'tokenizer_class' must be a Python class name such as "
                                   f"'PreTrainedTokenizerFast', found '{value}'")
        for value in ("", 1):
            with self.subTest(value=value):
                self.assertRefused(lambda d, v=value: d["model"].update(tokenizer_class=v),
                                   "'tokenizer_class' must be a non-empty string")

    def test_tokenizer_flag_conflicts_with_the_key(self):
        text = ("[server]: '--tokenizer' is set by llmctl when [model] tokenizer_class is set and may not appear "
                "in 'args'")
        self.assertRefused(lambda d: d["server"]["args"].extend(["--tokenizer", "/x"]), text)
        self.assertRefused(lambda d: d["server"]["args"].append("--tokenizer=/x"), text)
        self.assertEqual(deepseek(lambda d: d["server"]["args"].extend(["--tokenizer-mode", "auto"])).args[-2:],
                         ("--tokenizer-mode", "auto"))
        recipe = gpt_oss(lambda d: d["server"]["args"].extend(["--tokenizer", "/x"]))
        self.assertEqual(recipe.args[-2:], ("--tokenizer", "/x"))
        spec = vllm.container_spec(recipe, self.runtime, TENV)
        self.assertFalse(any(mount.split(":")[1] == "/llmctl/tokenizer" for mount in spec.mounts))

    def test_tokenizer_container_spec(self):
        recipe = deepseek()
        spec = vllm.container_spec(recipe, self.runtime, TENV)
        self.assertEqual(spec.mounts, (
            "/home/user/.cache/huggingface:/root/.cache/huggingface:ro",
            "/home/user/.cache/llmctl/vllm/tokenizers/deepseek-r1-0528-qwen3-8b:/llmctl/tokenizer:ro",
            "vllm-cache:/root/.cache/vllm",
            "flashinfer-cache:/root/.cache/flashinfer",
        ))
        self.assertEqual(spec.args[:7], (DS_SNAPSHOT, "--served-model-name", "deepseek-r1-8b", "--max-model-len",
                                         str(recipe.context), "--tokenizer", "/llmctl/tokenizer"))
        self.assertEqual(spec.args[7:], recipe.args)
        self.assertEqual(vllm.container_spec(gpt_oss(), self.runtime, TENV),
                         vllm.container_spec(gpt_oss(), self.runtime, ENV))

    def test_tokenizer_overlay_needs_an_absolute_cache_dir(self):
        with self.assertRaises(ValueError):
            vllm.container_spec(deepseek(), self.runtime, ENV)
        for cache in (Path("cache"), Path("~/.cache/llmctl")):
            with self.subTest(cache=cache):
                env = HostEnv(ENV.hf_home, 1000, 1000, cache)
                with self.assertRaises(UsageError) as ctx:
                    vllm.container_spec(deepseek(), self.runtime, env)
                self.assertEqual(ctx.exception.exit_code, 1)
                self.assertEqual(str(ctx.exception), f"llmctl's cache folder {cache} is not an absolute path; "
                                                     "set XDG_CACHE_HOME (or HOME) to one")
        self.assertEqual(vllm.container_spec(gpt_oss(), self.runtime, HostEnv(ENV.hf_home, 1000, 1000, Path("cache"))),
                         vllm.container_spec(gpt_oss(), self.runtime, ENV))

    def test_without_the_key_nothing_is_written(self):
        absent = self.cache / "absent"
        for cache in (absent, None):
            for dry_run in (False, True):
                with self.subTest(cache=cache, dry_run=dry_run):
                    self.assertEqual(self.prepare(gpt_oss(), dry_run, HostEnv(self.hf, 1000, 1000, cache)), ())
        self.assertFalse(os.path.lexists(absent))

    def ds_snapshot(self, config=None, revision=DS_REVISION):
        files = {"config.json": b"{}", "tokenizer.json": b"{}", "LICENSE": b"MIT", "figures/benchmark.png": b"png",
                 "tokenizer_config.json": config_bytes(DS_CONFIG if config is None else config)}
        model = deepseek(lambda d: d["model"].update(revision=revision)).model
        return write_snapshot(self.hf, model, files, fill=False)

    def test_tokenizer_overlay_links_and_rewrites_the_config(self):
        self.ds_snapshot()
        self.assertEqual(self.prepare(), (
            f"wrote {self.folder}: tokenizer overlay with 4 links into snapshot 6e8885a6ff5c of "
            "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B, and tokenizer_config.json with tokenizer_class "
            "PreTrainedTokenizerFast",))
        for name in ("config.json", "tokenizer.json", "LICENSE", "figures/benchmark.png"):
            with self.subTest(name=name):
                self.assertEqual(os.readlink(self.folder / name), f"{DS_SNAPSHOT}/{name}")
        self.assertTrue((self.folder / "figures").is_dir())
        self.assertFalse((self.folder / "figures").is_symlink())
        written = self.folder / "tokenizer_config.json"
        self.assertFalse(written.is_symlink())
        self.assertEqual(written.stat().st_mode & 0o777, 0o644)
        expected = dict(DS_CONFIG, tokenizer_class="PreTrainedTokenizerFast")
        self.assertEqual(written.read_text(encoding="utf-8"),
                         json.dumps(expected, indent=2, ensure_ascii=False) + "\n")
        self.assertEqual(list(json.loads(written.read_text(encoding="utf-8"))), list(DS_CONFIG))

        without = {key: value for key, value in DS_CONFIG.items() if key != "tokenizer_class"}
        self.ds_snapshot(without)
        self.prepare()
        config = json.loads(written.read_text(encoding="utf-8"))
        self.assertEqual(list(config)[-1], "tokenizer_class")
        self.assertEqual(config["tokenizer_class"], "PreTrainedTokenizerFast")

    def test_tokenizer_overlay_follows_the_snapshot(self):
        self.ds_snapshot()
        self.prepare()
        self.ds_snapshot(dict(DS_CONFIG, model_max_length=1))
        self.prepare()
        config = json.loads((self.folder / "tokenizer_config.json").read_text(encoding="utf-8"))
        self.assertEqual((config["model_max_length"], config["tokenizer_class"]), (1, "PreTrainedTokenizerFast"))

        other = "0123456789abcdef0123456789abcdef01234567"
        self.ds_snapshot(revision=other)
        self.prepare(deepseek(lambda d: d["model"].update(revision=other)))
        targets = [os.readlink(Path(dirpath) / name) for dirpath, _, names in os.walk(self.folder) for name in names
                   if (Path(dirpath) / name).is_symlink()]
        self.assertEqual(len(targets), 4)
        self.assertTrue(all(f"/snapshots/{other}/" in target for target in targets), targets)

    def test_tokenizer_dry_run_writes_nothing(self):
        self.assertEqual(self.prepare(dry_run=True), (
            f"would write {self.folder}: tokenizer overlay with links into snapshot 6e8885a6ff5c of "
            "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B, and tokenizer_config.json with tokenizer_class "
            "PreTrainedTokenizerFast",))
        self.assertFalse(os.path.lexists(self.cache))

    def test_tokenizer_overlay_failures_are_exit_3(self):
        with self.assertRaises(PreflightError) as ctx:
            self.prepare()
        self.assertEqual(str(ctx.exception), f"cannot write the tokenizer overlay {self.folder}: missing snapshot "
                                             f"{DS_REVISION} of {DS_REPO}")
        self.assertEqual(ctx.exception.exit_code, 3)

        snapshot = self.ds_snapshot()
        (snapshot / "tokenizer_config.json").unlink()
        with self.assertRaises(PreflightError) as ctx:
            self.prepare()
        self.assertEqual(str(ctx.exception), f"cannot write the tokenizer overlay {self.folder}: {MISSING_TEXT}")
        (snapshot / "tokenizer_config.json").write_bytes(b"[1]")
        with self.assertRaises(PreflightError) as ctx:
            self.prepare()
        text = str(ctx.exception)
        self.assertTrue(text.startswith(f"cannot write the tokenizer overlay {self.folder}: {NOT_OBJECT_START}"), text)
        self.assertTrue(text.endswith(NOT_OBJECT_END), text)
        self.assertEqual(ctx.exception.exit_code, 3)
        self.assertFalse(os.path.lexists(self.folder))
        self.assertFalse(os.path.lexists(self.folder.parent / ".deepseek-r1-0528-qwen3-8b.llmctl-tmp"))

        if os.geteuid() == 0:
            self.skipTest("root writes into a read-only folder")
        self.ds_snapshot()
        tokenizers = self.folder.parent
        tokenizers.mkdir(parents=True, exist_ok=True)
        tokenizers.chmod(0o500)
        self.addCleanup(tokenizers.chmod, 0o700)
        with self.assertRaises(PreflightError) as ctx:
            self.prepare()
        text = str(ctx.exception)
        self.assertTrue(text.startswith(f"cannot write the tokenizer overlay {self.folder}: "), text)
        self.assertTrue(text.endswith(f"; check the free space and permissions of {self.cache}"), text)
        self.assertEqual(ctx.exception.exit_code, 3)

    def test_tokenizer_file_problems(self):
        recipe = deepseek()
        base = {"config.json": b"{}", "tokenizer.json": b"{}"}
        write_snapshot(self.hf, recipe.model, {**base, "tokenizer_config.json": config_bytes(DS_CONFIG)})
        self.assertEqual(vllm.file_problems(recipe, self.hf), [])
        snapshot = snapshot_dir(self.hf, recipe.model.repo, recipe.model.revision)
        (snapshot / "tokenizer_config.json").unlink()
        write_snapshot(self.hf, recipe.model, base)
        self.assertEqual(vllm.file_problems(recipe, self.hf), [MISSING_TEXT])
        for data in (b"[1]", b"not json"):
            with self.subTest(data=data):
                write_snapshot(self.hf, recipe.model, {**base, "tokenizer_config.json": data})
                [problem] = vllm.file_problems(recipe, self.hf)
                self.assertTrue(problem.startswith(NOT_OBJECT_START), problem)
                self.assertTrue(problem.endswith(NOT_OBJECT_END), problem)
        with open(snapshot / "model.safetensors", "wb") as handle:
            handle.truncate(10)
        problems = vllm.file_problems(recipe, self.hf)
        self.assertEqual(len(problems), 1)
        self.assertTrue(problems[0].startswith("the snapshot holds "), problems)
        self.assertTrue(problems[0].endswith("; the download is incomplete"), problems)

        oss = gpt_oss()
        write_snapshot(self.hf, oss.model, {"config.json": b"{}"})
        self.assertEqual(vllm.file_problems(oss, self.hf), [])

    def test_pull_checks_the_tokenizer_config(self):
        recipe = deepseek()
        base = {"config.json": b"{}", "tokenizer.json": b"{}"}
        host = FakeHost()
        host.on(("docker", "run"), lambda argv: write_snapshot(self.hf, recipe.model, base) and ok())
        with self.assertRaises(DownloadError) as ctx:
            vllm.pull(host, recipe, self.runtime, self.hf, HfAuth(None, None), io.StringIO())
        self.assertTrue(str(ctx.exception).startswith(
            "deepseek-r1-0528-qwen3-8b: tokenizer_config.json in snapshot 6e8885a6ff5c is missing"), str(ctx.exception))
        host = FakeHost()
        host.on(("docker", "run"), lambda argv: write_snapshot(
            self.hf, recipe.model, {**base, "tokenizer_config.json": config_bytes(DS_CONFIG)}) and ok())
        vllm.pull(host, recipe, self.runtime, self.hf, HfAuth(None, None), io.StringIO())

    def test_shipped_deepseek_uses_the_overlay(self):
        data, source = read_toml(DEEPSEEK, "shipped")
        recipe = vllm.parse_recipe(data, source, "deepseek-r1-0528-qwen3-8b")
        self.assertEqual(recipe.model.tokenizer_class, "PreTrainedTokenizerFast")
        self.assertEqual(recipe.evidence.kind, "measured")
        self.assertFalse(recipe.opencode)  # OpenCode sends tool_choice auto, which vLLM refuses without a parser
        self.assertNotIn("--tokenizer", recipe.args)
        self.assertTrue((REPO / recipe.evidence.record).is_dir(), recipe.evidence.record)
        others = sorted(path for path in RECIPES.glob("*.toml") if not path.name.startswith("_") and path != DEEPSEEK)
        self.assertEqual(len(others), 7)
        for path in others:
            with self.subTest(recipe=path.stem):
                self.assertIsNone(vllm.parse_recipe(*read_toml(path, "shipped"), path.stem).model.tokenizer_class)

    def test_optional_members(self):
        self.assertEqual(vllm.TOKENIZER_MOUNT, "/llmctl/tokenizer")
        self.assertTrue(callable(vllm.prepare_start))
        self.assertIsNone(getattr(vllm, "check_started", None))


if __name__ == "__main__":
    unittest.main()
