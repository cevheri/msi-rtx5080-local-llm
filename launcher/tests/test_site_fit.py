"""The local LLM page's machine rules (spec 5): verdicts, the hero's pick, the change plans and the orders, run on a
pinned snapshot of the 29 setups of 2026-11-19 (spec 5.10, P10) and on the fixture repository."""

import collections
import dataclasses
import re
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from launcher import clients
from launcher.site import data, fit
from launcher.tests.site_fixture import SiteFixture, cell_of, model_of

# (model, engine, rank, gpu need, RAM peak, memory cap, download bytes, tok/s, chat only, offload, agents that passed)
SNAPSHOT = (
    ("deepseek-r1-0528-qwen3-8b", "llamacpp", 10, "13174 MiB", "3.0", 16, 8709519872, "70.6", True, False, "OHPDK"),
    ("deepseek-r1-0528-qwen3-8b", "vllm", 10, "92%", None, 32, 16388927770, "72.5", True, False, "HPDK"),
    ("gemma-4-12b", "llamacpp", None, "9660 MiB", "12.9", 20, 6975879296, "77.1", False, False, "OHPDK"),
    ("gemma-4-12b", "vllm", None, "92%", None, 32, 10296461612, "69.5", False, False, "OHPK"),
    ("gemma-4-12b", "sglang", None, "14891 MiB", "6.0", 32, 10296461612, "70.4", False, False, "OHPDK"),
    ("gemma-4-26b-a4b", "llamacpp", 5, "14952 MiB", "10.6", 20, 14439363584, "148.2", False, True, "OHPK"),
    ("gemma-4-e4b", "llamacpp", None, "5310 MiB", "7.9", 20, 5154941280, "151.3", False, False, "OHPDK"),
    ("gemma-4-e4b", "vllm", None, "92%", None, 32, 11546108983, "45.2", False, False, "OHPK"),
    ("gemma-4-e4b", "sglang", None, "14327 MiB", "6.0", 32, 11546108983, "141.3", False, False, "OHPK"),
    ("gpt-oss-20b", "llamacpp", 9, "13296 MiB", None, 16, 12109566624, "166.2", False, False, "OHPDK"),
    ("gpt-oss-20b", "vllm", 9, "96.5%", None, 32, 13789264674, "83.5", False, False, "OPDK"),
    ("gpt-oss-20b", "sglang", 9, "14724 MiB", "16.0", 20, 13789264674, "124.6", False, True, "HPDK"),
    ("groot-19b-a3b", "llamacpp", 8, "13710 MiB", "4.4", 16, 11657618560, "169.8", False, False, "OHPDK"),
    ("groot-19b-a3b", "vllm", 8, "92%", None, 32, 13562080504, "193.4", False, False, "OHPDK"),
    ("k2-horizon-7b", "llamacpp", 4, "14896 MiB", "5.5", 16, 9573973280, "66.9", False, False, "OHPDK"),
    ("k2-horizon-7b", "vllm", 4, "94%", None, 32, 8145038688, "94.8", False, False, "OK"),
    ("k2-horizon-7b", "sglang", 4, "14908 MiB", "5.0", 32, 8145038688, "89.6", False, False, "OPDK"),
    ("k2-horizon-mova-36b", "llamacpp", 3, "14736 MiB", "19.7", 26, 22701885664, "37.2", False, True, "OHPDK"),
    ("mellum2.1-12b", "llamacpp", None, "12562 MiB", "13.2", 16, 10879834400, "188.2", False, False, "OHPD"),
    ("mellum2.1-12b", "vllm", None, "92%", None, 32, 7686428495, "257.2", False, False, "OHPDK"),
    ("nemotron-3.5-bolt", "llamacpp", 7, "14952 MiB", "17.8", 26, 22018326528, "73.5", False, True, "OHD"),
    ("nemotron-3.5-lightning", "llamacpp", None, "14994 MiB", "10.3", 18, 18898091584, "143.6", False, True, "OHPDK"),
    ("qwen3-coder-next", "llamacpp", 6, "14964 MiB", "31.9", 35, 38429272064, "71.5", False, True, "OPDK"),
    ("qwen3.5-9b", "llamacpp", None, "12670 MiB", "15.0", 16, 9527502048, "66.0", False, False, "OHPDK"),
    ("qwen3.5-9b", "vllm", None, "92%", None, 32, 9381214599, "90.8", False, False, "OHPDK"),
    ("qwen3.5-9b", "sglang", None, "15193 MiB", "6.0", 32, 9381214599, "93.7", False, False, "OHPDK"),
    ("qwen3.6-35b-a3b", "llamacpp", 2, "15078 MiB", "14.9", 20, 20542732576, "122.5", False, True, "OPDK"),
    ("qwen3.6-35b-a3b", "sglang", 2, "15266 MiB", "23.0", 27, 23462477857, "151.3", False, True, "OPDK"),
    ("qwen3.8-27b", "llamacpp", 1, "14678 MiB", None, 16, 13146393504, "36.4", False, False, "OHPDK"),
)
LETTER = {"O": "opencode", "H": "hermes", "P": "pi", "D": "dsh", "K": "kimi"}
NAMES = {"deepseek-r1-0528-qwen3-8b": "DeepSeek-R1-0528-Qwen3-8B", "gemma-4-12b": "Gemma 4 12B",
         "gemma-4-26b-a4b": "Gemma 4 26B-A4B", "gemma-4-e4b": "Gemma 4 E4B", "gpt-oss-20b": "gpt-oss-20b",
         "groot-19b-a3b": "Groot-19B-A3B-261023", "k2-horizon-7b": "K2 Horizon 7B",
         "k2-horizon-mova-36b": "K2 Horizon MoVA 36B A4B", "mellum2.1-12b": "Mellum2.1 12B-A2.5B Thinking",
         "nemotron-3.5-bolt": "NVIDIA Nemotron 3.5 Bolt 74B-A9B",
         "nemotron-3.5-lightning": "NVIDIA Nemotron 3.5 Lightning 30B-A3B", "qwen3-coder-next": "Qwen3-Coder-Next",
         "qwen3.5-9b": "Qwen3.5-9B", "qwen3.6-35b-a3b": "Qwen3.6-35B-A3B", "qwen3.8-27b": "Qwen3.8-27B"}

# Spec 5.10 without its goal column (spec 5.6): ((gpu, ram, agent), setups ok, models ok, (model, runtime, step), more)
VECTORS = (
    ((16, 64, "opencode"), 29, 15, ("qwen3.6-35b-a3b", "llamacpp", "default"), ()),
    ((16, 64, "hermes"), 29, 15, ("qwen3.8-27b", "llamacpp", "default-failed"), ()),
    ((16, 32, "opencode"), 13, 12, ("qwen3.6-35b-a3b", "llamacpp", "default"), (fit.More("ram", 64, 3),)),
    ((16, 32, "none"), 13, 12, ("qwen3.6-35b-a3b", "llamacpp", "default"), (fit.More("ram", 64, 3),)),
    ((12, 32, "opencode"), 2, 2, ("gemma-4-12b", "llamacpp", "unranked"), (fit.More("gpu", 16, 10),)),
    ((12, 64, "dsh"), 2, 2, ("gemma-4-12b", "llamacpp", "unranked"), (fit.More("gpu", 16, 13),)),
    ((8, 32, "opencode"), 1, 1, ("gemma-4-e4b", "llamacpp", "unranked"), (fit.More("gpu", 12, 1),)),
    ((8, 16, "opencode"), 0, 0, ("gemma-4-e4b", "llamacpp", "change"), (fit.More("ram", 32, 1),)),
    ((16, 16, "opencode"), 0, 0, ("k2-horizon-7b", "llamacpp", "change"), (fit.More("ram", 32, 12),)),
    ((24, 64, "pi"), 21, 15, ("qwen3.6-35b-a3b", "llamacpp", "default"), ()),
    ((24, 32, "kimi"), 13, 12, ("qwen3.6-35b-a3b", "llamacpp", "default"), (fit.More("ram", 64, 3),)),
)


def snapshot_bench() -> fit.Bench:
    setups = []
    for model, runtime, rank, gpu, peak, cap, size, decode, chat, offload, passed in SNAPSHOT:
        mib, share = fit.gpu_need(gpu, f"recipes/{runtime}/{model}.toml")
        setups.append(fit.Setup(model, runtime, rank, runtime == "llamacpp", mib, share,
                                None if peak is None else Fraction(peak), cap, size, Fraction(decode), chat, offload,
                                frozenset(LETTER[c] for c in passed)))
    return fit.Bench(tuple(setups), ("qwen3.6-35b-a3b", "llamacpp"), 10, 16, fit.Machine(16, 64),
                     ("opencode", "hermes", "pi", "dsh", "kimi"))


def with_cell(matrix, model_id: str, runtime: str, **changes):
    """The matrix with one cell's fields replaced."""
    model, cell = model_of(matrix, model_id), cell_of(matrix, model_id, runtime)
    new = dataclasses.replace(cell, **changes)
    return dataclasses.replace(matrix, models=tuple(
        dataclasses.replace(m, cells=tuple(new if c is cell else c for c in m.cells)) if m is model else m
        for m in matrix.models))


def with_class(matrix, model_id: str, runtime: str, client: str, cls: str):
    """The matrix with one cell's class in one agent replaced."""
    cell = cell_of(matrix, model_id, runtime)
    return with_cell(matrix, model_id, runtime, clients=tuple(
        dataclasses.replace(r, cls=cls) if r.client == client else r for r in cell.clients))


def hand_setup(**changes) -> fit.Setup:
    """A hand-made llama.cpp setup of a ranked model that passed OpenCode, with `changes` applied."""
    base = fit.Setup("m", "llamacpp", 1, True, 8000, None, None, 16, 1, Fraction(50), False, False,
                     frozenset({"opencode"}))
    return dataclasses.replace(base, **changes)


class SiteFitTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_machines_and_estimates(self):
        self.assertEqual(len(fit.machines()), 15)
        self.assertEqual(fit.machines()[:4], (fit.Machine(8, 16), fit.Machine(8, 32), fit.Machine(8, 64), fit.Machine(12, 16)))
        self.assertEqual([fit.free_ram(r) for r in fit.RAMS], [Fraction(21, 2), Fraction(21), Fraction(42)])
        self.assertEqual([fit.usable_mib(g) for g in fit.GPUS], [7168, 11264, 15360, 23552, 31744])

    def test_gpu_need_reads_mib_and_share(self):
        self.assertEqual(fit.gpu_need("15078 MiB", "r.toml"), (15078, None))
        self.assertEqual(fit.gpu_need("96.5%", "r.toml"), (None, Fraction(193, 200)))
        with self.assertRaisesRegex(data.SiteError, r"^r\.toml: the GPU need '15 GB' is neither <N> MiB nor <N>%$"):
            fit.gpu_need("15 GB", "r.toml")

    def test_verdicts_follow_r2(self):
        b = snapshot_bench()
        s = {x.key: x for x in b.setups}
        v = lambda key, g, r: fit.verdict(s[key], fit.Machine(g, r), 16)
        self.assertEqual(v(("qwen3.6-35b-a3b", "llamacpp"), 16, 64).key, "ok")
        got = v(("qwen3.6-35b-a3b", "llamacpp"), 12, 32)
        self.assertEqual((got.key, got.short, got.ram, got.usable_mib), ("change", Fraction(1907, 512), "ok", 11264))
        got = v(("qwen3.8-27b", "llamacpp"), 8, 16)
        self.assertEqual((got.key, got.short, got.ram), ("change", Fraction(3755, 512), "cap-unmeasured"))
        got = v(("gemma-4-e4b", "llamacpp"), 8, 16)
        self.assertEqual((got.key, got.short, got.ram, got.free), ("change", None, "cap", Fraction(21, 2)))
        got = v(("qwen3-coder-next", "llamacpp"), 16, 32)
        self.assertEqual((got.key, got.short, got.ram), ("ram", None, "too-big"))
        got = v(("gemma-4-12b", "vllm"), 12, 64)
        self.assertEqual((got.key, got.short, got.ram), ("only16", None, None))
        self.assertEqual(v(("gpt-oss-20b", "llamacpp"), 16, 16).ram, "cap-unmeasured")

    def test_direction_vectors(self):
        b = snapshot_bench()
        for (g, r, agent), setups, models, pick, more in VECTORS:
            m = fit.Machine(g, r)
            with self.subTest(machine=m, agent=agent):
                self.assertEqual(len(fit.runs_as_measured(b, m)), setups)
                self.assertEqual(len(fit.models_as_measured(b, m)), models)
                got = fit.hero_pick(b, m, agent)
                self.assertEqual((got.setup.model, got.setup.runtime, got.step), pick)
                self.assertEqual(fit.more_memory(b, m), more)

    def test_direction_notes(self):
        b = snapshot_bench()
        s = {x.key: x for x in b.setups}
        self.assertEqual(fit.verdict(s[("qwen3-coder-next", "llamacpp")], fit.Machine(16, 32), 16).key, "ram")
        vllm = [x for x in b.setups if x.runtime == "vllm"]
        self.assertEqual(len(vllm), 8)
        self.assertTrue(all(fit.verdict(x, fit.Machine(g, 64), 16).key == "only16" for x in vllm for g in (24, 32)))

    def assert_picks_agree_with_rows(self, b: fit.Bench) -> collections.Counter:
        """Spec 5.5's agreement in every state; returns how often each (model, runtime, step) was picked."""
        seen = collections.Counter()
        for m in fit.machines():
            for agent in (*b.agents, fit.NONE):
                with self.subTest(machine=m, agent=agent):
                    pick = fit.hero_pick(b, m, agent)
                    row = fit.row_setup(b, pick.setup.model, m)
                    row_key = fit.verdict(row, m, b.test_gpu).key
                    if row == pick.setup:
                        self.assertEqual(row_key, pick.verdict.key)
                    if pick.verdict.key == "ok":
                        self.assertEqual(row_key, "ok")
                    seen[(pick.setup.model, pick.setup.runtime, pick.step)] += 1
        return seen

    def test_hero_pick_agrees_with_its_row_in_every_state(self):
        seen = self.assert_picks_agree_with_rows(snapshot_bench())
        self.assertEqual(seen, collections.Counter({("qwen3.6-35b-a3b", "llamacpp", "default"): 30,
                                                    ("k2-horizon-7b", "llamacpp", "change"): 18,
                                                    ("gemma-4-e4b", "llamacpp", "change"): 12,
                                                    ("gemma-4-e4b", "llamacpp", "unranked"): 12,
                                                    ("gemma-4-12b", "llamacpp", "unranked"): 12,
                                                    ("qwen3.8-27b", "llamacpp", "default-failed"): 6}))

    def test_change_plans(self):
        b = snapshot_bench()
        s = {x.key: x for x in b.setups}
        cases = {(("qwen3.8-27b", "llamacpp"), 8, 16): fit.Change("layers", "cap-unmeasured"),
                 (("k2-horizon-7b", "llamacpp"), 12, 32): fit.Change("context", None),
                 (("qwen3.6-35b-a3b", "llamacpp"), 12, 32): fit.Change("experts", None),
                 (("gpt-oss-20b", "sglang"), 12, 64): fit.Change("resident", None),
                 (("gemma-4-e4b", "sglang"), 8, 64): fit.Change("impossible", None),
                 (("gemma-4-e4b", "llamacpp"), 8, 16): fit.Change(None, "cap"),
                 (("k2-horizon-7b", "llamacpp"), 16, 16): fit.Change(None, "cap"),
                 (("gpt-oss-20b", "llamacpp"), 16, 16): fit.Change(None, "cap-unmeasured"),
                 (("qwen3.6-35b-a3b", "llamacpp"), 16, 64): None,
                 (("qwen3-coder-next", "llamacpp"), 16, 32): None}
        for (key, g, r), expected in cases.items():
            with self.subTest(key=key, gpu=g, ram=r):
                m = fit.Machine(g, r)
                self.assertEqual(fit.change(s[key], fit.verdict(s[key], m, 16)), expected)

    def test_orders(self):
        b = snapshot_bench()
        self.assertEqual(fit.model_order(b), (
            "qwen3.8-27b", "qwen3.6-35b-a3b", "k2-horizon-mova-36b", "k2-horizon-7b", "gemma-4-26b-a4b",
            "qwen3-coder-next", "nemotron-3.5-bolt", "groot-19b-a3b", "gpt-oss-20b", "deepseek-r1-0528-qwen3-8b",
            "nemotron-3.5-lightning", "qwen3.5-9b", "mellum2.1-12b", "gemma-4-12b", "gemma-4-e4b"))
        home = fit.Machine(16, 64)
        self.assertEqual(fit.sort_order(b, home, "speed"), (
            "groot-19b-a3b", "gpt-oss-20b", "gemma-4-26b-a4b", "qwen3.6-35b-a3b", "nemotron-3.5-bolt", "qwen3-coder-next",
            "deepseek-r1-0528-qwen3-8b", "k2-horizon-7b", "k2-horizon-mova-36b", "qwen3.8-27b",
            "mellum2.1-12b", "gemma-4-e4b", "nemotron-3.5-lightning", "gemma-4-12b", "qwen3.5-9b"))
        self.assertEqual(fit.sort_order(b, home, "size"), (
            "deepseek-r1-0528-qwen3-8b", "k2-horizon-7b", "groot-19b-a3b", "gpt-oss-20b", "qwen3.8-27b", "gemma-4-26b-a4b",
            "qwen3.6-35b-a3b", "nemotron-3.5-bolt", "k2-horizon-mova-36b", "qwen3-coder-next",
            "gemma-4-e4b", "gemma-4-12b", "qwen3.5-9b", "mellum2.1-12b", "nemotron-3.5-lightning"))
        self.assertEqual(fit.models_as_measured(b, fit.Machine(12, 32)), ("gemma-4-12b", "gemma-4-e4b"))

    def test_bench_reads_the_matrix_and_checks_the_estimates(self):
        fixture = SiteFixture(Path(self._tmp.name))
        matrix = fixture.load()
        b = fit.bench(matrix)
        self.assertEqual((len(b.setups), b.default, b.top, b.test_gpu, b.home, b.agents),
                         (10, ("qwen3.6-35b-a3b", "llamacpp"), 3, 16, fit.Machine(16, 64), tuple(clients.NAMES)))
        q = next(x for x in b.setups if x.key == ("qwen3.6-35b-a3b", "llamacpp"))
        self.assertEqual(q, fit.Setup("qwen3.6-35b-a3b", "llamacpp", 1, True, 15078, None, Fraction("14.9"), 20,
                                      20542732576, Fraction("122.5"), False, True, frozenset(clients.NAMES)))
        for change, message in (
                ({"ram_free_gib": 39}, "README.md#machine: the RAM row says 61 GiB usable, about 39 GiB free; "
                                       "launcher/site/fit.py estimates free RAM from 40 of 61 GiB: change one to match the other"),
                ({"gpu_gb": 20}, "recipes: the test GPU has 20 GB, not one of the GPU sizes 8, 12, 16, 24 and 32")):
            with self.subTest(change=change), self.assertRaisesRegex(data.SiteError, "^" + re.escape(message) + "$"):
                fit.bench(dataclasses.replace(matrix, laptop=dataclasses.replace(matrix.laptop, **change)))
        # R2 says "failed the agent test", so our default may not be untested or unsupported in any agent (P25)
        for cls in ("not-tested", "not-supported"):
            with self.subTest(cls=cls), self.assertRaisesRegex(data.SiteError, "^" + re.escape(
                    f"launcher/site/fit.py: our default, qwen3.6-35b-a3b on llamacpp, is {cls} in hermes, and the page "
                    "only says that a default failed the agent test there") + "$"):
                fit.bench(with_class(matrix, "qwen3.6-35b-a3b", "llamacpp", "hermes", cls))

    def test_our_default_may_fail_an_agent_test(self):
        matrix = SiteFixture(Path(self._tmp.name)).load()
        for cls in ("failed", "known-issue"):
            with self.subTest(cls=cls):
                b = fit.bench(with_class(matrix, "qwen3.6-35b-a3b", "llamacpp", "hermes", cls))
                default = next(x for x in b.setups if x.key == b.default)
                self.assertEqual(default.passed, frozenset(clients.NAMES) - {"hermes"})

    def test_passed_holds_only_the_agents_whose_class_is_passed(self):
        matrix = SiteFixture(Path(self._tmp.name)).load()
        setups = {x.key: x for x in fit.bench(matrix).setups}
        # the fixture's DeepSeek on vLLM is not-supported in OpenCode
        self.assertEqual(setups[("deepseek-r1-0528-qwen3-8b", "vllm")].passed, frozenset(clients.NAMES) - {"opencode"})
        for cls in ("failed", "known-issue", "not-tested", "not-supported"):
            with self.subTest(cls=cls):
                b = fit.bench(with_class(matrix, "gpt-oss-20b", "llamacpp", "hermes", cls))
                got = next(x for x in b.setups if x.key == ("gpt-oss-20b", "llamacpp"))
                self.assertEqual(got.passed, frozenset(clients.NAMES) - {"hermes"})

    def test_a_chat_only_default_stops_the_build(self):
        # candidates() never holds a chat-only setup, so R2 would say our default failed the agent test everywhere
        matrix = SiteFixture(Path(self._tmp.name)).load()
        recipe = dataclasses.replace(cell_of(matrix, "qwen3.6-35b-a3b", "llamacpp").recipe, chat_only=True)
        with self.assertRaisesRegex(data.SiteError, "^" + re.escape(
                "launcher/site/fit.py: our default, qwen3.6-35b-a3b on llamacpp, is chat only, and the page only says "
                "that a default failed the agent test") + "$"):
            fit.bench(with_cell(matrix, "qwen3.6-35b-a3b", "llamacpp", recipe=recipe))

    def test_a_row_falls_back_to_the_fastest_engine_that_runs_as_measured(self):
        b = snapshot_bench()
        b = dataclasses.replace(b, setups=tuple(dataclasses.replace(x, gpu_mib=16000)
                                                if x.key == ("k2-horizon-7b", "llamacpp") else x for x in b.setups))
        # vLLM (94.8 tok/s) and SGLang (89.6 tok/s) run as measured at 16 GB; neither does at 12 GB
        self.assertEqual(fit.row_setup(b, "k2-horizon-7b", fit.Machine(16, 64)).key, ("k2-horizon-7b", "vllm"))
        self.assertEqual(fit.row_setup(b, "k2-horizon-7b", fit.Machine(12, 64)).key, ("k2-horizon-7b", "llamacpp"))
        self.assert_picks_agree_with_rows(b)

    def test_r2_limits_are_inclusive(self):
        m = fit.Machine(16, 32)
        self.assertEqual(fit.free_ram(32), 21)
        edge = hand_setup(gpu_mib=fit.usable_mib(16), memory_cap=21)
        self.assertEqual(fit.verdict(edge, m, 16), fit.Verdict("ok", None, "ok", 15360, Fraction(21)))
        capped = hand_setup(gpu_mib=fit.usable_mib(16), memory_cap=22, ram_peak=Fraction("21.0"))
        self.assertEqual(fit.verdict(capped, m, 16), fit.Verdict("change", None, "cap", 15360, Fraction(21)))

    def test_key_ram_keeps_the_gpu_short(self):
        s = {x.key: x for x in snapshot_bench().setups}
        got = fit.verdict(s[("qwen3.6-35b-a3b", "llamacpp")], fit.Machine(8, 16), 16)
        self.assertEqual((got.key, got.short, got.ram), ("ram", Fraction(3955, 512), "too-big"))

    def test_change_compares_the_download_in_gib(self):
        # 8 GB leaves 7168 MiB, 7 GiB: 7.5e9 bytes are 6.98 GiB, 7.6e9 bytes are 7.08 GiB
        m = fit.Machine(8, 64)
        for size, plan in ((7_500_000_000, "context"), (7_600_000_000, "layers")):
            with self.subTest(size=size):
                s = hand_setup(size_bytes=size)
                self.assertEqual(fit.change(s, fit.verdict(s, m, 16)), fit.Change(plan, None))

    def test_the_faster_of_two_other_engines_is_recommended(self):
        default = hand_setup(gpu_mib=16000)
        vllm = hand_setup(runtime="vllm", default_engine=False, gpu_mib=None, gpu_share=Fraction(92, 100),
                          decode=Fraction(60))
        sglang = hand_setup(runtime="sglang", default_engine=False, gpu_mib=14000, decode=Fraction(70))
        b = fit.Bench((default, vllm, sglang), ("m", "llamacpp"), 10, 16, fit.Machine(16, 64), ("opencode",))
        pick = fit.hero_pick(b, fit.Machine(16, 64), "opencode")
        self.assertEqual((pick.setup, pick.step), (sglang, "ranked"))

    def test_sort_order_refuses_an_unknown_sort(self):
        with self.assertRaisesRegex(ValueError, r"^launcher/site/fit\.py: unknown sort 'context'; "
                                                r"expected rank, speed or size$"):
            fit.sort_order(snapshot_bench(), fit.Machine(16, 64), "context")

    def test_no_candidate_stops_the_build(self):
        b = snapshot_bench()
        chat = dataclasses.replace(b, setups=tuple(dataclasses.replace(x, chat_only=True) for x in b.setups))
        with self.assertRaisesRegex(data.SiteError, r"^launcher/site/fit\.py: no setup to recommend for 8 GB of GPU memory, "
                                                    r"16 GB of RAM and opencode$"):
            fit.hero_pick(chat, fit.Machine(8, 16), "opencode")


if __name__ == "__main__":
    unittest.main()
