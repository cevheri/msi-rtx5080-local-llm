"""The local LLM page's sentences (spec 7): every text the pages show, checked against the spec's exact copy on the
pinned snapshot of test_site_fit.py and on the fixture repository."""

import dataclasses
import os
import re
import tempfile
import time
import unittest
from unittest import mock
from fractions import Fraction
from pathlib import Path

from launcher.site import data, fit, words
from launcher.tests.site_fixture import SiteFixture, cell_of, model_of, py311_fraction_format
from launcher.tests.test_site_fit import NAMES, snapshot_bench

LABELS = words.Labels(models=NAMES, engines={"llamacpp": "llama.cpp", "vllm": "vLLM", "sglang": "SGLang"},
                      agents={"opencode": "OpenCode", "hermes": "Hermes", "pi": "Pi", "dsh": "dsh", "kimi": "Kimi",
                              "none": "None yet"},
                      first_engine="llamacpp", test_gpu=16, top=10, total=15, default=("qwen3.6-35b-a3b", "llamacpp"))
AGENTS = ("opencode", "hermes", "pi", "dsh", "kimi")
# U+2013, U+2014, U+00B7, arrows, check marks and other dingbats, emoji and the emoji variation selector.
FORBIDDEN = re.compile("[\u2013\u2014\u00b7\u2190-\u21ff\u2600-\u27bf\u2b00-\u2bff\U0001f000-\U0001faff\ufe0f]")


def result(client: str, cls: str, passed: int = 5, runs: int = 5) -> data.ClientResult:
    return data.ClientResult(client, cls, passed, runs, "2026-11-18", None)


def setups() -> dict[tuple[str, str], fit.Setup]:
    return {x.key: x for x in snapshot_bench().setups}


def texts(value) -> list[str]:
    """Every text in a constant: a str, the text parts of a rich value, and the strings inside tuples and mappings."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for key, item in value.items() for text in (*texts(key), *texts(item))]
    if isinstance(value, (tuple, list)):
        return [text for item in value for text in texts(item)]
    return []


class SiteWordsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._matrix = None

    def matrix(self) -> data.Matrix:
        if self._matrix is None:
            self._matrix = SiteFixture(Path(self._tmp.name)).load()
        return self._matrix

    def facts(self, model: str, runtime: str) -> data.RecipeFacts:
        return cell_of(self.matrix(), model, runtime).recipe

    def test_rich_parts(self):
        self.assertEqual([words.code("x"), words.bold("y"), words.link("z", "/")], [("code", "x"), ("b", "y"), ("a", "z", "/")])
        self.assertEqual(words.plain(("a ", words.code("b"), " c ", words.link("d", "docs/x.md"), words.bold("!"))),
                         "a b c d!")
        self.assertEqual(words.plain("text"), "text")

    def test_labels_come_from_the_matrix(self):
        matrix = self.matrix()
        labels = words.labels(matrix)
        self.assertEqual(labels.models, {m.id: m.name for m in matrix.models})
        self.assertEqual(labels.engines, {"llamacpp": "llama.cpp", "vllm": "vLLM", "sglang": "SGLang"})
        self.assertEqual(list(labels.agents.items()), list(LABELS.agents.items()))
        self.assertEqual((labels.first_engine, labels.test_gpu, labels.top, labels.total, labels.default),
                         ("llamacpp", 16, 3, 6, ("qwen3.6-35b-a3b", "llamacpp")))

    def test_units(self):
        self.assertEqual([words.gb_mib(15078), words.gb_bytes(20542732576), words.context_k(131072), words.speed(None),
                          words.speed(66.0), words.half_up(Fraction(21, 2), 0), words.gb(Fraction(1907, 512))],
                         ["14.7", "20.5", "128k", "not measured", "66.0", "11", "3.7"])
        self.assertEqual([words.day("2026-11-16"), words.day_short("2026-11-18"), words.engine_version("SGLang v0.5.23"),
                          words.engine_version("llama.cpp b12407"), words.version_detail("vLLM v0.33.0")],
                         ["16 November 2026", "18 Nov 2026", "SGLang 0.5.23", "llama.cpp b12407", "0.33.0"])
        self.assertEqual([words.count_word(2), words.count_word(10), words.joined(["a", "b", "c"]), words.joined_or(["a", "b"])],
                         ["two", "10", "a, b and c", "a or b"])
        self.assertEqual([words.gpu_phrase(8), words.gpu_phrase(16), words.gpu_phrase(32), words.ram_phrase(32),
                          words.ram_phrase(64), words.gpu_chip(32), words.ram_chip(16)],
                         ["an 8 GB GPU", "a 16 GB GPU", "a GPU of 32 GB or more", "32 GB of RAM", "64 GB of RAM or more",
                          "32 GB or more", "16 GB"])

    def test_half_up_rounds_half_up_with_integer_digits(self):
        self.assertEqual([words.half_up(Fraction(1, 20)), words.half_up(Fraction(149, 100)), words.half_up(Fraction(0)),
                          words.half_up(Fraction(5, 2), 0), words.half_up(Fraction(12345, 1000), 2), words.half_up(7, 0)],
                         ["0.1", "1.5", "0.0", "3", "12.35", "7"])
        self.assertEqual([words.joined(["a"]), words.joined_or(["a", "b", "c"]), words.count_word(1), words.count_word(9)],
                         ["a", "a, b or c", "one", "nine"])
        self.assertEqual(words.title(LABELS, "qwen3.6-35b-a3b", "sglang", False), "Qwen3.6-35B-A3B on SGLang")
        self.assertEqual(words.title(LABELS, "qwen3.6-35b-a3b", "llamacpp", True), "Qwen3.6-35B-A3B")

    def test_hero_constants(self):
        self.assertEqual(words.H1, "Will it run on your machine?")
        self.assertEqual(words.LEAD, "llmctl starts a tested open coding model in Docker and connects it to your coding "
                         "agent. For Linux with an NVIDIA GPU: tell us your GPU and RAM, and get the commands.")
        self.assertEqual(words.GPU_HELP, ("Not sure? ", ("code", "nvidia-smi"), " shows it."))
        self.assertEqual(words.RAM_HELP, (("code", "free -h"), " shows it."))
        self.assertEqual(words.agent_help(5), "llmctl sets up all five for you.")
        self.assertEqual([words.STATE_UNTOUCHED, words.STATE_BAD_MACHINE, words.STATE_BAD_AGENT, words.RESET],
                         ["Set to our test laptop. Change it to yours.",
                          "Showing our laptop: the link held a value this page doesn't know.",
                          "Showing OpenCode: the link named an agent this page doesn't know.", "Back to our test laptop"])
        self.assertEqual(words.OS_BODY[-1], ("a", "Try them remotely",
                                             "docs/opencode.md#trying-a-model-remotely-first-with-openrouter"))
        self.assertEqual(words.pick_heading("Gemma 4 12B"), "Start with Gemma 4 12B")

    def test_verdict_lines(self):
        self.assertEqual([words.verdict_line(True, 15, 15), words.verdict_line(False, 2, 15),
                          words.verdict_line(False, 1, 15), words.verdict_line(False, 0, 15),
                          words.verdict_line(False, 12, 15)],
                         ["On our test laptop: all 15 models run as we measured them.",
                          "On your machine: two of 15 models run as we measured them.",
                          "On your machine: one of 15 models runs as we measured it.",
                          "On your machine: none of the 15 models runs as we measured it.",
                          "On your machine: 12 of 15 models run as we measured them."])

    def test_reasons_in_todays_states(self):
        b, labels = snapshot_bench(), LABELS
        def reason(g, r, agent):
            m = fit.Machine(g, r)
            pick = fit.hero_pick(b, m, agent)
            names = [labels.models[i] for i in fit.models_as_measured(b, m)]
            return words.reason(labels, pick, fit.default_verdict(b, m), fit.change(pick.setup, pick.verdict), names,
                                fit.more_memory(b, m), agent)
        self.assertEqual(reason(12, 32, "opencode"), "Our pick, Qwen3.6-35B-A3B, is 3.7 GB short of GPU memory there, so "
                         "start with Gemma 4 12B. Gemma 4 12B and Gemma 4 E4B run as we measured them. With a 16 GB GPU, 10 more would.")
        self.assertEqual(reason(8, 16, "opencode"), "Our pick, Qwen3.6-35B-A3B, is 7.7 GB short of GPU memory there, so start "
                         "with Gemma 4 E4B, the closest fit. It fits your GPU but needs a lower RAM cap (it used 7.9 GB at "
                         "peak); its commands show how. With 32 GB of RAM, one more would.")
        self.assertEqual(reason(16, 16, "opencode"), "Our pick, Qwen3.6-35B-A3B, is too big for the RAM there, so start with "
                         "K2 Horizon 7B, the closest fit. It fits your GPU but needs a lower RAM cap (it used 5.5 GB at peak); "
                         "its commands show how. With 32 GB of RAM, 12 more would.")
        self.assertEqual(reason(16, 32, "opencode"), "Our default, and the model our OpenCode setup starts with. With 64 GB of "
                         "RAM or more, three more would.")
        self.assertEqual(reason(8, 32, "opencode"), "Our pick, Qwen3.6-35B-A3B, is 7.7 GB short of GPU memory there, so start "
                         "with Gemma 4 E4B. With a 12 GB GPU, one more would.")
        self.assertEqual(reason(16, 64, "hermes"), "Our default, Qwen3.6-35B-A3B, failed the agent test in Hermes, so this is "
                         "the highest-ranked model that passed there.")
        self.assertEqual(reason(16, 64, "opencode"), "Our default, and the model our OpenCode setup starts with.")

    def test_a_change_pick_of_our_default_has_no_switch(self):
        s = setups()
        default = s["qwen3.6-35b-a3b", "llamacpp"]
        judged = fit.verdict(default, fit.Machine(12, 32), 16)
        pick = fit.Pick(default, judged, "change")
        self.assertEqual(words.reason(LABELS, pick, judged, fit.change(default, judged), [], (), "opencode"),
                         "The closest fit, with a change we haven't measured. Your card is 3.7 GB short of what it used; "
                         "its commands show the first change to try, not measured.")

    def test_switch_reasons(self):
        cases = [(fit.Verdict("ram", None, "too-big", 15360, Fraction(21)),
                  "Our pick, Qwen3.6-35B-A3B, is too big for the RAM there, so start with K2 Horizon 7B."),
                 (fit.Verdict("only16", None, None, 23552, Fraction(42)),
                  "Our pick, Qwen3.6-35B-A3B, was measured on 16 GB cards only, so start with K2 Horizon 7B."),
                 (fit.Verdict("change", None, "cap", 15360, Fraction(21, 2)),
                  "Our pick, Qwen3.6-35B-A3B, needs more free RAM than there is, so start with K2 Horizon 7B.")]
        for verdict, expected in cases:
            with self.subTest(verdict=verdict):
                self.assertEqual(words.switch_sentence(LABELS, verdict, "K2 Horizon 7B"), expected)

    def test_change_sentences(self):
        s = setups()
        def sentence(key, g, r, change):
            setup = s[key]
            return words.change_sentence(LABELS, setup, fit.verdict(setup, fit.Machine(g, r), 16), change)
        self.assertEqual(sentence(("qwen3.8-27b", "llamacpp"), 8, 16, fit.Change("layers", "cap-unmeasured")),
                         "Its files alone are 13.1 GB, more than your GPU holds; its commands show what to try, much slower "
                         "and not measured.")
        self.assertEqual(sentence(("k2-horizon-7b", "llamacpp"), 12, 32, fit.Change("context", None)),
                         "Your card is 3.5 GB short of what it used; its commands show the first change to try, not measured.")
        self.assertEqual(fit.verdict(s["k2-horizon-7b", "llamacpp"], fit.Machine(12, 32), 16).short, Fraction(227, 64))
        self.assertEqual(sentence(("gemma-4-e4b", "llamacpp"), 8, 16, fit.Change(None, "cap")),
                         "It fits your GPU but needs a lower RAM cap (it used 7.9 GB at peak); its commands show how.")
        self.assertEqual(sentence(("gpt-oss-20b", "llamacpp"), 16, 16, fit.Change(None, "cap-unmeasured")),
                         "It fits your GPU but needs a lower RAM cap (its RAM use wasn't measured); its commands show how.")

    def test_agents_lines(self):
        everyone = [result(a, "passed") for a in AGENTS]
        self.assertEqual(words.agents_line(everyone, LABELS, False, "llamacpp", "none"), "Works in all five agents we tested.")
        hermes = [result(a, "known-issue", 0) if a == "hermes" else result(a, "passed") for a in AGENTS]
        self.assertEqual(words.agents_line(hermes, LABELS, False, "llamacpp", "opencode"),
                         "Passed the agent test in OpenCode (5 of 5 runs). Also works in Pi, dsh and Kimi. Hermes: known issue.")
        self.assertEqual(words.agents_line(hermes, LABELS, False, "llamacpp", "none"),
                         "Works in OpenCode, Pi, dsh and Kimi. Hermes: known issue.")
        self.assertEqual(words.agents_line(everyone, LABELS, False, "llamacpp", "pi"),
                         "Passed the agent test in Pi (5 of 5 runs). Also works in OpenCode, Hermes, dsh and Kimi.")
        mixed = [result("opencode", "not-supported", 0, 0), result("hermes", "failed", 0), result("pi", "not-tested", 0, 0),
                 result("dsh", "known-issue", 1), result("kimi", "failed", 0)]
        self.assertEqual(words.agents_line(mixed, LABELS, False, "vllm", "none"),
                         "dsh: known issue. Hermes and Kimi: failed the latest test. Pi: not tested yet. OpenCode: not set "
                         "up by llmctl.")
        sentence = lambda r, chat=False, runtime="llamacpp": words.result_sentence(r, LABELS, chat, runtime)
        self.assertEqual([sentence(result("opencode", "passed")), sentence(result("pi", "passed", 1, 1), True),
                          sentence(result("hermes", "known-issue", 0)), sentence(result("hermes", "known-issue", 0, 1)),
                          sentence(result("dsh", "failed", 0)), sentence(result("kimi", "not-tested", 0, 0)),
                          sentence(result("opencode", "not-supported", 0, 0), True, "vllm"),
                          sentence(result("opencode", "passed", 1, 1))],
                         ["Passed the agent test in OpenCode (5 of 5 runs).",
                          "Passed our chat test in Pi (1 of 1 run), without tools.",
                          "Known issue in Hermes: the latest test failed (0 of 5 runs passed).",
                          "Known issue in Hermes: the latest test failed (0 of 1 run passed).",
                          "Failed the latest test in dsh.", "Not tested in Kimi yet.",
                          "llmctl doesn't set up OpenCode for this model on vLLM.",
                          "Passed the agent test in OpenCode (1 of 1 run)."])

    def test_shelf_and_compare_agent_words(self):
        hermes = [result(a, "known-issue", 0) if a == "hermes" else result(a, "passed") for a in AGENTS]
        everyone = [result(a, "passed", 1, 1) for a in AGENTS]
        self.assertEqual([words.agents_cell(everyone, LABELS, False, "none"), words.agents_cell(hermes, LABELS, False, "none"),
                          words.agents_cell(hermes, LABELS, False, "opencode"), words.agents_cell(everyone, LABELS, False, "pi"),
                          words.agents_cell(everyone, LABELS, True, "pi"), words.agents_cell(hermes, LABELS, False, "hermes")],
                         ["All 5 agents", "4 of 5 agents", "Passed (5 of 5 runs)", "Passed (1 of 1 run)",
                          "Chat test passed (1 of 1), no tools", "Known issue (0 of 5 runs passed)"])
        self.assertEqual([words.result_short(result("dsh", "failed", 0), False), words.result_short(result("dsh", "not-tested"), False),
                          words.result_short(result("dsh", "not-supported"), False)],
                         ["Failed the latest test", "Not tested yet", "Not set up by llmctl"])
        self.assertEqual([words.compare_agents(everyone, LABELS, False, "sglang", "none"),
                          words.compare_agents(hermes, LABELS, False, "sglang", "none"),
                          words.compare_agents(everyone, LABELS, False, "sglang", "opencode")],
                         ["All 5 agents", "4 of 5 agents: Hermes known issue",
                          "Passed the agent test in OpenCode (1 of 1 run)."])
        self.assertEqual([words.agents_head("none", LABELS), words.agents_head("kimi", LABELS)], ["Coding agents", "In Kimi"])

    def test_map_texts(self):
        s = setups()
        def text(key, g, r, laptop=False, bold=False):
            m = fit.Machine(g, r)
            return words.map_text(LABELS, s[key], fit.verdict(s[key], m, 16), m, laptop, bold)
        got = text(("gemma-4-12b", "llamacpp"), 12, 64, bold=True)
        self.assertEqual(got, words.MapText(
            gpu_left=(("b", "9.4 GB"), " used of your 12 GB"), gpu_right="about 1 GB for your desktop (estimate)",
            gpu_short=False, ram_left=(("b", "12.9 GB"), " used at peak. llmctl asks for 20 GB free to start it."),
            ram_right="about 22 GB in use by your desktop and apps (estimate)", ram_short=False,
            alt="Gemma 4 12B used 9.4 GB of GPU memory and 12.9 GB of RAM at its peak on our laptop. llmctl asks for 20 GB "
                "of free RAM before it starts it. Your machine: a 12 GB GPU and about 42 GB of free RAM (estimate). Runs as "
                "measured."))
        self.assertEqual(text(("qwen3.6-35b-a3b", "llamacpp"), 16, 64, laptop=True), words.MapText(
            gpu_left=("14.7 GB used of our 16 GB",), gpu_right="about 1 GB for the desktop (estimate)", gpu_short=False,
            ram_left=("14.9 GB used at peak. llmctl asks for 20 GB free to start it.",),
            ram_right="about 21 GB in use by the desktop and apps (estimate)", ram_short=False,
            alt="Qwen3.6-35B-A3B used 14.7 GB of GPU memory and 14.9 GB of RAM at its peak on our laptop. llmctl asks for "
                "20 GB of free RAM before it starts it. Our laptop: a 16 GB GPU and about 40 GB of free RAM (estimate). Runs "
                "as measured."))
        e4b = text(("gemma-4-e4b", "llamacpp"), 8, 16)
        self.assertEqual((e4b.ram_right, e4b.ram_short), ("about 10 GB free on yours (estimate)", True))
        self.assertTrue(e4b.alt.endswith("Your machine: an 8 GB GPU and about 10 GB of free RAM (estimate). Needs a change "
                                         "we haven't measured."))
        coder = text(("qwen3-coder-next", "llamacpp"), 16, 32)
        self.assertEqual((coder.ram_right, coder.ram_short), ("10.9 GB short", True))
        self.assertTrue(coder.alt.endswith("Too big for your RAM."))
        short = text(("qwen3.6-35b-a3b", "llamacpp"), 12, 32)
        self.assertEqual((short.gpu_left, short.gpu_right, short.gpu_short), (("14.7 GB used of your 12 GB",), "3.7 GB short", True))
        share = text(("gemma-4-12b", "vllm"), 12, 64)
        self.assertEqual((share.gpu_left, share.gpu_right, share.gpu_short),
                         (("Measured on 16 GB cards only.",), "outline: 92% of your card, not measured", False))
        self.assertEqual(share.ram_left, ("RAM use wasn't measured. llmctl asks for 32 GB free to start it.",))
        self.assertEqual(share.alt, "On our laptop, vLLM took 92% of the 16 GB card for Gemma 4 12B; its RAM use wasn't "
                         "measured. llmctl asks for 32 GB of free RAM before it starts it. Your machine: a 12 GB GPU and about "
                         "42 GB of free RAM (estimate). Measured on 16 GB cards only.")
        self.assertEqual(text(("gemma-4-12b", "vllm"), 16, 64, bold=True).gpu_left,
                         ("vLLM takes ", ("b", "92%"), " of the card up front."))
        self.assertEqual(text(("qwen3.8-27b", "llamacpp"), 32, 64).gpu_left, ("14.3 GB used of your 32 GB or more",))
        self.assertEqual(text(("qwen3.8-27b", "llamacpp"), 16, 64).alt.split(". ")[0],
                         "Qwen3.8-27B used 14.3 GB of GPU memory on our laptop; its RAM use wasn't measured")

    def test_verdict_and_map_words(self):
        self.assertEqual([words.verdict_sentence(k, LABELS) for k in ("ok", "change", "ram", "only16")],
                         ["Runs as measured.", "Needs a change we haven't measured.", "Too big for your RAM.",
                          "Measured on 16 GB cards only."])
        self.assertEqual([words.verdict_word(k, LABELS) for k in ("ok", "only16")], ["Runs as measured", "Measured on 16 GB cards only"])
        self.assertEqual(words.CAPTION, ("Solid: measured on our laptop.", "Outline: what llmctl asks for.", "Hatched: estimate."))
        s = setups()
        qwen = s["qwen3.6-35b-a3b", "llamacpp"]
        self.assertEqual(words.moe_question(qwen), "Why does a 20.5 GB download fit in 14.7 GB of GPU memory?")
        self.assertEqual(words.moe_answer("llamacpp"), (
            "It's a mixture-of-experts model, so most of its experts wait in RAM and the busiest stay on the GPU. ",
            ("a", "How that works", "llamacpp/README.md#moe-expert-offload")))
        self.assertEqual(words.moe_answer("sglang")[-1], ("a", "How that works", "sglang/README.md#paged-experts"))
        self.assertEqual(words.moe_paragraph(s["qwen3.6-35b-a3b", "sglang"]), (
            "Why a 23.5 GB download fits in 14.9 GB of GPU memory: it's a mixture-of-experts model, so most of its "
            "experts wait in RAM and the busiest stay on the GPU. ", ("a", "How that works", "sglang/README.md#paged-experts")))
        eighteen = dataclasses.replace(qwen, size_bytes=18_900_000_000)
        self.assertEqual(words.moe_question(eighteen), "Why does an 18.9 GB download fit in 14.7 GB of GPU memory?")
        self.assertTrue(words.moe_paragraph(eighteen)[0].startswith("Why an 18.9 GB download fits in 14.7 GB"))
        eight = dataclasses.replace(qwen, size_bytes=8_700_000_000, gpu_mib=7000)
        self.assertEqual(words.moe_question(eight), "Why does an 8.7 GB download fit in 6.8 GB of GPU memory?")
        eleven = dataclasses.replace(qwen, size_bytes=11_200_000_000, gpu_mib=7000)
        self.assertEqual(words.moe_question(eleven), "Why does an 11.2 GB download fit in 6.8 GB of GPU memory?")
        for key in (("gemma-4-12b", "llamacpp"), ("gemma-4-26b-a4b", "llamacpp")):
            with self.subTest(key=key):   # not offloaded; offloaded, but its 14.4 GB download is under its 14.6 GB need
                self.assertIsNone(words.moe_question(s[key]))
                self.assertIsNone(words.moe_paragraph(s[key]))

    def test_shelf_words(self):
        s = setups()
        self.assertEqual(words.top_heading(LABELS), "Our Top 10")
        self.assertEqual(words.KEY, ("Each block is 1 GB.", "Used on our laptop", "The desktop (estimate)"))
        self.assertEqual(words.SORTS, (("rank", "Our ranking"), ("speed", "Fastest first"), ("size", "Smallest download first")))
        self.assertEqual([words.strip_head(fit.Machine(16, 64), LABELS, True), words.strip_head(fit.Machine(12, 64), LABELS, False),
                          words.strip_head(fit.Machine(32, 16), LABELS, False)],
                         ["Our 16 GB GPU", "Your 12 GB GPU", "A 32 GB GPU"])
        self.assertEqual([words.machine_head(True), words.machine_head(False)], ["On our laptop", "On your machine"])
        self.assertEqual([words.row_engine(LABELS, s["gemma-4-12b", "vllm"]), words.row_engine(LABELS, s["gemma-4-12b", "llamacpp"])],
                         ["on vLLM", None])
        self.assertEqual([words.uses(LABELS, s["qwen3.6-35b-a3b", "llamacpp"]), words.uses(LABELS, s["gemma-4-12b", "vllm"])],
                         ["Uses 14.7 GB of GPU memory.", "Uses 92% of a 16 GB card."])

    def test_laptop_wording_says_our(self):
        b = snapshot_bench()
        said = []
        for x in b.setups:
            for bold in (False, True):
                got = words.map_text(LABELS, x, fit.verdict(x, b.home, 16), b.home, True, bold)
                said += [words.plain(got.gpu_left), got.gpu_right, words.plain(got.ram_left), got.ram_right, got.alt]
        said += [words.verdict_line(True, 15, 15), words.panel_heading(True),
                 words.panel_label("Gemma 4 12B on llama.cpp", True), words.machine_head(True),
                 words.strip_head(fit.Machine(16, 64), LABELS, True), words.GS_LEAD, *words.KEY]
        for text in said:
            with self.subTest(text=text):
                self.assertIsNone(re.search(r"\byour\b", text, re.I))
        self.assertEqual([words.panel_heading(False), words.panel_label("Gemma 4 12B on llama.cpp", False)],
                         ["Your machine", "Gemma 4 12B on llama.cpp, on your machine:"])

    def test_override_lines(self):
        s = setups()
        def lines(key, g, r):
            setup = s[key]
            verdict = fit.verdict(setup, fit.Machine(g, r), 16)
            return words.override_lines(setup, verdict, fit.change(setup, verdict), "llamacpp")
        show = lambda model, rt="llamacpp", flag="": (
            f"$ ./llmctl show {model}{flag} --as-override > ~/.config/llmctl/recipes/{rt}/{model}.toml")
        both = "# Your machine has less GPU memory and free RAM than this setup needs. Make your own copy of it:"
        gpu = "# Your GPU has less memory than this setup used. Make your own copy of it:"
        ram = "# Your machine has less free RAM than llmctl asks for. Make your own copy of this setup:"
        mkdir = "$ mkdir -p ~/.config/llmctl/recipes/llamacpp"
        cases = {
            (("qwen3.8-27b", "llamacpp"), 8, 16): [
                both, mkdir, show("qwen3.8-27b"),
                "# Its files alone are 13.1 GB, more than your GPU holds: lower -ngl to keep some layers in RAM; much "
                "slower, and not measured",
                "# Then set memory_gib in that file below the RAM you have free (free -g); we didn't measure how much it uses"],
            (("k2-horizon-7b", "llamacpp"), 12, 32): [gpu, mkdir, show("k2-horizon-7b"),
                                                     "# Then give it a smaller context; not measured"],
            (("qwen3.6-35b-a3b", "llamacpp"), 12, 32): [
                gpu, mkdir, show("qwen3.6-35b-a3b"),
                "# Then lower --moe-cache-mib, then raise --n-cpu-moe; slower, and not measured"],
            (("gpt-oss-20b", "sglang"), 12, 64): [
                gpu, "$ mkdir -p ~/.config/llmctl/recipes/sglang", show("gpt-oss-20b", "sglang", " --runtime sglang"),
                "# Then give it fewer resident experts or a smaller --mem-fraction-static; not measured"],
            (("gemma-4-e4b", "sglang"), 8, 64): [
                gpu, "$ mkdir -p ~/.config/llmctl/recipes/sglang", show("gemma-4-e4b", "sglang", " --runtime sglang"),
                "# Its files alone are 11.5 GB, more than your GPU holds, so this may not be possible on your card"],
            (("gemma-4-e4b", "llamacpp"), 8, 16): [
                ram, mkdir, show("gemma-4-e4b"),
                "# Then set memory_gib in that file below the RAM you have free (free -g) and above the 7.9 GB it used at peak"],
            (("k2-horizon-7b", "llamacpp"), 16, 16): [
                ram, mkdir, show("k2-horizon-7b"),
                "# Then set memory_gib in that file below the RAM you have free (free -g) and above the 5.5 GB it used at peak"],
            (("gpt-oss-20b", "llamacpp"), 16, 16): [
                ram, mkdir, show("gpt-oss-20b"),
                "# Then set memory_gib in that file below the RAM you have free (free -g); we didn't measure how much it uses"],
            (("qwen3.6-35b-a3b", "llamacpp"), 16, 64): [],
            (("qwen3-coder-next", "llamacpp"), 16, 32): [],
        }
        for (key, g, r), expected in cases.items():
            with self.subTest(key=key, gpu=g, ram=r):
                self.assertEqual(lines(key, g, r), expected)

    def test_terminal_and_notes(self):
        facts = self.facts("qwen3.6-35b-a3b", "llamacpp")
        self.assertEqual(words.head_lines("https://github.com/o/r"),
                         ["# Get llmctl (once)", "$ git clone https://github.com/o/r", "$ cd r"])
        self.assertEqual(words.pull_lines(facts, "llama.cpp", "7.05"),
                         ["# Download 20.5 GB of model files; the llama.cpp engine (7.05 GB on disk) comes too, the first time",
                          "$ ./llmctl pull qwen3.6-35b-a3b"])
        self.assertEqual(words.up_lines(facts), ["# Start it: llmctl checks your machine first, then points your agents at it",
                                                 "$ ./llmctl up qwen3.6-35b-a3b"])
        self.assertEqual(words.agent_lines(facts, LABELS, "llamacpp", "opencode"),
                         ["# Open OpenCode and pick llamacpp-docker/qwen3.6-35b-a3b with /models", "$ opencode"])
        self.assertEqual(words.agent_lines(facts, LABELS, "llamacpp", "none"),
                         words.agent_lines(facts, LABELS, "llamacpp", "opencode"))
        self.assertEqual(words.agent_lines(facts, LABELS, "llamacpp", "kimi"),
                         ["# Open Kimi on the model llmctl added", "$ kimi -m llmctl/local"])
        self.assertEqual(words.agent_lines(facts, LABELS, "llamacpp", "hermes"),
                         ["# Open Hermes with the llmctl profile that llmctl wrote", "$ hermes -p llmctl"])
        self.assertEqual(words.agent_lines(facts, LABELS, "llamacpp", "dsh"),
                         ["# Open dsh in your browser, then pick qwen3.6-35b-a3b on llama.cpp (llmctl) in its model picker",
                          "$ dsh web"])
        self.assertEqual(words.tail_lines(facts),
                         ["# Check: a reply, a tool call and your speed (ours: 122.5 tok/s)", "$ ./llmctl test",
                          "# Stop it when you're done, so your GPU can rest", "$ ./llmctl down"])
        self.assertEqual(words.tail_lines(dataclasses.replace(facts, decode=None))[0],
                         "# Check: a reply, a tool call and your speed")
        # ./llmctl test skips the tool call for a chat-only recipe (launcher/smoke.py)
        self.assertEqual(words.tail_lines(dataclasses.replace(facts, chat_only=True))[0],
                         "# Check: a reply and your speed (ours: 122.5 tok/s)")
        self.assertEqual(words.tail_lines(dataclasses.replace(facts, chat_only=True, decode=None))[0],
                         "# Check: a reply and your speed")
        self.assertEqual(words.agent_lines(self.facts("deepseek-r1-0528-qwen3-8b", "vllm"), LABELS, "vllm", "opencode"),
                         ["# llmctl doesn't set up OpenCode for this model on vLLM; pick another agent above"])
        chat = self.facts("deepseek-r1-0528-qwen3-8b", "llamacpp")
        self.assertEqual(words.agent_lines(chat, LABELS, "llamacpp", "kimi"),
                         ["# Kimi always sends its tools, so make this chat-only model the default first; "
                          "--set-default does that for every agent llmctl set up",
                          "$ ./llmctl up deepseek-r1-0528-qwen3-8b --set-default",
                          '$ kimi -p "your prompt" --agent-file launcher/kimi-chat-only.md'])
        self.assertEqual(words.agent_lines(chat, LABELS, "llamacpp", "pi"),
                         ["# Open Pi on the model llmctl added", "$ pi --model llmctl/local --no-tools"])
        no_agent = ("No coding agent yet? Install ", ("a", "OpenCode", "https://opencode.ai"),
                    " first; it's the one this project is built around.")
        config = ("OpenCode needs our provider config once: copy it from ",
                  ("a", "the setup guide", "docs/opencode.md#the-validated-config"), ".")
        replace = ("Another model already running? Add ", ("code", "--replace"), " to the up line and llmctl stops it first.")
        thinking = "always on, at the effort the server sets"
        self.assertEqual(words.terminal_notes(chat, LABELS, "none"),
                         (no_agent, config, replace, (f"Thinking in OpenCode: {thinking}.",)))
        self.assertEqual(words.terminal_notes(chat, LABELS, "opencode"), (config, replace, (f"Thinking in OpenCode: {thinking}.",)))
        self.assertEqual(words.terminal_notes(chat, LABELS, "pi"), (
            replace, (f"Thinking in Pi: {thinking}.",),
            ("Pi with a chat-only model: read ", ("a", "chat-only recipes", "docs/clients.md#chat-only-recipes"), " first.")))
        self.assertEqual(words.terminal_notes(self.facts("deepseek-r1-0528-qwen3-8b", "vllm"), LABELS, "opencode"),
                         (replace, (f"Thinking in OpenCode: {thinking}.",)))
        self.assertEqual(words.plain(words.endpoint(8080, "local")), "Other tools: http://127.0.0.1:8080/v1 with the model name local.")
        sglang = self.facts("qwen3.6-35b-a3b", "sglang")
        self.assertEqual(words.agent_lines(sglang, LABELS, "sglang", "pi"),
                         ["# Open Pi on the model llmctl added", "$ pi --model llmctl/qwen3.6-35b-a3b"])
        self.assertEqual(words.agent_lines(sglang, LABELS, "sglang", "kimi"),
                         ["# Open Kimi on the model llmctl added", "$ kimi -m llmctl/qwen3.6-35b-a3b"])
        self.assertEqual(words.plain(words.endpoint(30000, sglang.served)),
                         "Other tools: http://127.0.0.1:30000/v1 with the model name qwen3.6-35b-a3b.")

    def test_get_started_words(self):
        self.assertEqual(words.GS_LEAD, "The short path, for the model we picked at the top of the page. Every model's page "
                         "has its own commands.")
        self.assertEqual([what for what, _ in words.CHECKS], ["An NVIDIA GPU and its driver.", "Docker Engine.",
                                                              "The NVIDIA Container Toolkit.", "Python 3.11 or newer."])
        self.assertEqual(words.plain(words.CHECKS[2][1]), "Check: nvidia-ctk cdi list lists nvidia.com/gpu=all. If it doesn't, "
                         "follow the setup guide.")
        self.assertEqual(words.disk_check(20542732576, "7.05", "llama.cpp"),
                         ("Disk space: 20.5 GB for this model, plus the engine image the first time (7.05 GB for llama.cpp).",
                          ("Check: ", ("code", "df -h ~"))))
        self.assertEqual([words.terminal_title("Gemma 4 12B", "vLLM"), words.commands_label("Gemma 4 12B"),
                          words.copy_label("./llmctl test"), words.COMMANDS_LABEL],
                         ["Commands for Gemma 4 12B on vLLM", "Commands for Gemma 4 12B", "Copy: ./llmctl test",
                          "Commands for the model we picked"])
        self.assertEqual([message for message, _ in words.REFUSALS],
                         ["... may use up to N GiB of RAM", "... needs N MiB of free GPU memory", "no CDI spec for nvidia.com/gpu",
                          "Docker refused the connection (permission denied", "another model server is running"])
        self.assertEqual([part[2] for rich in words.REFUSE_LINKS for part in rich if isinstance(part, tuple)],
                         ["docs/llmctl.md#when-up-refuses", "docs/llmctl.md#overrides-for-another-machine"])

    def test_compare_and_static_words(self):
        matrix = self.matrix()
        self.assertEqual(words.compare_lead(LABELS, 10, 18), "Every model on every engine we tried. 10 of the 18 combinations "
                         "run on our 16 GB laptop GPU; the others don't fit (our arithmetic) or didn't start (we tried).")
        self.assertEqual([words.engine_head(r, i == 0) for i, r in enumerate(matrix.runtimes)],
                         [("llama.cpp", "b12407, the default engine"), ("vLLM", "0.33.0"), ("SGLang", "0.5.23")])
        self.assertEqual([words.not_top(LABELS), words.ranked(4), words.does_not_fit(LABELS), words.did_not_start("2026-11-17")],
                         ["Not in our Top 10", "Ranked 4", ("Doesn't fit on 16 GB", "(our arithmetic)"),
                          ("Didn't start", "tried 17 Nov 2026")])
        qwen = model_of(matrix, "qwen3.6-35b-a3b")
        labels = words.labels(matrix)
        self.assertEqual([words.lead_words(qwen, c, labels) for c in qwen.cells if c.status == "runs"],
                         [None, "Fastest for this model, 24% ahead"])
        tied = dataclasses.replace(qwen, cells=tuple(dataclasses.replace(c, fastest="Tied fastest") if c.status == "runs" else c
                                                     for c in qwen.cells))
        self.assertEqual(words.lead_words(tied, tied.cells[0], labels), "Tied fastest for this model with SGLang")
        self.assertEqual(words.flow(LABELS)[0], ("Your coding agent", "OpenCode, Hermes, Pi, dsh or Kimi"))
        self.assertEqual(len(words.RULES), 5)
        questions = words.questions(LABELS, [65536, 131072])
        self.assertEqual([q for q, _ in questions], [
            "Does it work on Windows or a Mac?", "Will I get the same speed?", "What does “passed the agent test” mean?",
            "How were the models ranked?", "Why not just use Ollama?", "What do tok/s, context and thinking mean?"])
        self.assertIn("It shows the model can call the agent's tools; it doesn't grade how well the model codes.",
                      words.plain(questions[2][1]))
        self.assertIn("one swap keeps the DeepSeek family in the Top 10.", words.plain(questions[3][1]))
        self.assertIn("with 64k to 128k of context, tool calls tested in five agents,", words.plain(questions[4][1]))
        self.assertIn("OpenCode needs at least 64k.", words.plain(questions[5][1]))
        tested = [words.plain(p) for p in words.tested(matrix, matrix.runs[0])]
        self.assertEqual(tested[0], "Every number on this page comes from one laptop: an MSI Raider 18 Max HX A2WI with an RTX "
                         "5080 Laptop GPU (16 GB) and 61 GB of RAM, measured between 16 and 18 November 2026. It ran Ubuntu "
                         "26.04.1, with the GPU at its 140 W power limit.")
        self.assertEqual(tested[2], "The agent test: llmctl starts each setup, then each agent writes hello.py and reads it back. "
                         "In the latest run, on 18 November 2026, 106 of 106 steps passed across 10 setups and five agents. "
                         "Chat-only models get a plain reply test instead, without tools.")
        self.assertEqual(tested[3], "Solid marks are measured. Hatched marks are estimates: arithmetic, not a test run. To fit a "
                         "setup to your machine, the page assumes about 1 GB of GPU memory goes to your desktop and that the "
                         "same share of your RAM is free as on our laptop (40 of 61 GB).")
        self.assertEqual(tested[4], "Engines: llama.cpp b12407, vLLM 0.33.0 and SGLang 0.5.23. Agents: OpenCode 2.3.0, "
                         "Hermes 0.23.1, Pi 1.5.2, dsh 0.2.2-rc.3 and Kimi 2.5.3.")
        later = dataclasses.replace(matrix, laptop=dataclasses.replace(matrix.laptop, last="2026-12-02"))
        self.assertIn("measured between 16 November 2026 and 2 December 2026.", words.plain(words.tested(later, matrix.runs[0])[0]))
        self.assertEqual([words.plain(x) for x in words.tested_links("https://github.com/o/r")],
                         ["How we ranked the models", "Measurement records", "llmctl's guide", "The knowledge base on GitHub"])
        rich, licenses, privacy = words.footer("https://github.com/o/r", "a" * 40, 0, ("CC BY 4.0", "MIT"))
        self.assertEqual(rich, ("llmctl and this page come from ", ("a", "r", "https://github.com/o/r"), " on GitHub. ",
                                ("a", "Updated 1 January 1970", "https://github.com/o/r/commit/" + "a" * 40), "."))
        self.assertEqual((licenses, privacy), ("Docs and measurements: CC BY 4.0. Code: MIT.",
                                               "This page sets no cookies, has no analytics and loads nothing from other sites."))

    def test_model_page_texts(self):
        matrix = self.matrix()
        qwen = model_of(matrix, "qwen3.6-35b-a3b")
        self.assertEqual(words.role_line(dataclasses.replace(qwen, rank=2), LABELS), "Ranked 2 of 10 in our review, and our default.")
        self.assertEqual(words.role_line(model_of(matrix, "gpt-oss-20b"), LABELS), "Ranked 2 of 10 in our review.")
        self.assertEqual(words.role_line(model_of(matrix, "mellum2.1-12b"), LABELS), "An honorable mention in our review.")
        self.assertEqual(words.role_line(model_of(matrix, "k2-horizon-7b"), LABELS), "Not in our Top 10.")
        self.assertEqual(words.role_line(model_of(matrix, "deepseek-r1-0528-qwen3-8b"), LABELS),
                         "Ranked 3 of 10 in our review. Chat only: it can't call tools, so it can't drive a coding agent.")
        groot = dataclasses.replace(qwen, maker=data.Maker("GrootTeam (anonymous)", "About 19B / 3B active MoE",
                                                           "Apache-2.0 in metadata, no LICENSE file"))
        self.assertEqual(words.maker_line(groot), "GrootTeam (anonymous), Apache-2.0 in metadata, no LICENSE file, about 19B / 3B "
                         "active MoE.")
        self.assertIsNone(words.maker_line(model_of(matrix, "k2-horizon-7b")))
        self.assertEqual([words.build_sentence("35B / 3B active MoE", "Qwen3.6-35B-A3B"),
                          words.build_sentence("About 19B / 3B active MoE", "Groot-19B-A3B-261023"),
                          words.build_sentence("37.44B / about 4B active MoE", "K2 Horizon MoVA 36B A4B"),
                          words.build_sentence("27.8B dense", "Qwen3.8-27B"),
                          words.build_sentence(None, "Mellum2.1 12B-A2.5B Thinking"),
                          words.build_sentence(None, "Gemma 4 E4B")],
                         ["A mixture-of-experts model: 35B parameters, 3B of them active for each token.",
                          "A mixture-of-experts model: about 19B parameters, 3B of them active for each token.",
                          "A mixture-of-experts model: 37.44B parameters, about 4B of them active for each token.",
                          "A dense model: all 27.8B of its parameters work on every token.",
                          "A mixture-of-experts model: 12B parameters, 2.5B of them active for each token.", None])
        self.assertEqual(words.build_sentence("120B / 12B active hybrid MoE", "x"),
                         "A mixture-of-experts model: 120B parameters, 12B of them active for each token.")
        facts = self.facts("qwen3.6-35b-a3b", "llamacpp")
        thinking = [dataclasses.replace(facts, thinking="toggle", default="on"),
                    dataclasses.replace(facts, thinking="toggle", default="off"),
                    dataclasses.replace(facts, thinking="effort", efforts=("none", "low", "medium", "high")),
                    dataclasses.replace(facts, thinking="effort", efforts=("low", "medium", "high")),
                    dataclasses.replace(facts, thinking="always"), dataclasses.replace(facts, thinking="none")]
        self.assertEqual([words.thinking_sentence(f) for f in thinking], [
            "Thinking is on by default; you can turn it off.", "Thinking is off by default; you can turn it on per session.",
            "It thinks at medium effort by default; you can switch to low or high, or turn thinking off.",
            "It always thinks, at low, medium or high effort.",
            "It always thinks before it answers, which costs time and tokens.", "It answers without a thinking step."])
        self.assertEqual(words.thinking_line("off, as the server sets it"), "Thinking: off, as the server sets it")
        labels = words.labels(matrix)
        image_gb = {r.id: r.image_gb for r in matrix.runtimes}
        cells = {c.runtime: c for c in qwen.cells}
        self.assertEqual(words.faster_sentence(qwen, cells["llamacpp"], labels, image_gb),
                         "24% faster on SGLang (151.3 tok/s), but llmctl asks for 27 GB of free RAM and SGLang's engine is a "
                         "49.7 GB image.")
        self.assertIsNone(words.faster_sentence(qwen, cells["sglang"], labels, image_gb))
        deepseek = model_of(matrix, "deepseek-r1-0528-qwen3-8b")
        self.assertIsNone(words.faster_sentence(deepseek, deepseek.cells[0], labels, image_gb))   # 3% is under 5%
        runtimes = {r.id: r for r in matrix.runtimes}
        self.assertEqual(words.not_run_sentence(cells["vllm"], runtimes["vllm"], labels),
                         "Doesn't fit on 16 GB with vLLM (our arithmetic, not a test run).")
        self.assertEqual(words.not_run_sentence(deepseek.cells[2], runtimes["sglang"], labels),
                         "Didn't start on SGLang 0.5.23 (tried 17 November 2026).")
        self.assertEqual(words.why_clause("too large for the budget, measured 2026-11-17 ([record](https://example.org/r))"),
                         "Too large for the budget.")
        self.assertEqual(words.why_clause("too large for the budget (estimate), from [vllm/models.md](../vllm/models.md)"),
                         "Too large for the budget (estimate).")
        self.assertEqual(words.why_clause("out of GPU memory in the `lm_head`;"), "Out of GPU memory in the `lm_head`.")
        self.assertEqual([words.why_label("https://github.com/o/r/blob/x/research/2026-11-17/a/README.md"),
                          words.why_label("https://github.com/o/r/blob/x/vllm/models.md")],
                         ["Measurement record", "Engine notes"])
        self.assertEqual(words.summary(qwen, cells["llamacpp"], labels),
                         "A mixture-of-experts model: 35B parameters, 3B of them active for each token. On our 16 GB laptop "
                         "GPU it writes 122.5 tokens a second and keeps 128k of context in view. It uses 14.7 GB of GPU memory "
                         "and keeps part of itself in RAM, 14.9 GB at peak. Works in all five agents we tested. "
                         + words.thinking_sentence(facts))
        vllm = deepseek.cells[1]
        self.assertTrue(words.summary(deepseek, vllm, labels).endswith(
            "vLLM takes 92% of a 16 GB card up front. Chats in Hermes, Pi, dsh and Kimi, without tools. OpenCode: not set up "
            "by llmctl. " + words.thinking_sentence(vllm.recipe)))
        self.assertEqual([words.figures_label(labels, "llamacpp", True), words.figures_label(labels, "sglang", False)],
                         ["On llama.cpp, the default engine", "On SGLang"])
        self.assertEqual(words.engine_facts(labels, setups()["qwen3.6-35b-a3b", "llamacpp"], 131072), (
            (("b", "122.5"), " tok/s"), (("b", "128k"), " context"), (("b", "14.7 GB"), " of GPU memory"),
            ("asks for ", ("b", "20 GB"), " of free RAM"), (("b", "20.5 GB"), " download")))
        self.assertEqual(words.engine_facts(labels, setups()["gemma-4-12b", "vllm"], 65536)[2], (("b", "92%"), " of a 16 GB card"))
        self.assertEqual(words.use_engine_label("SGLang"), "Use this engine: SGLang")
        self.assertEqual(words.on_engine("SGLang"), "On SGLang")
        self.assertEqual([words.model_title("Gemma 4 12B"), words.model_description("Gemma 4 12B", LABELS),
                          words.home_description(LABELS)],
                         ["Gemma 4 12B, llmctl", "Gemma 4 12B on your own GPU: what it needs, how fast it ran on our 16 GB "
                          "laptop GPU, and the commands that start it.",
                          "15 open coding models measured on a 16 GB laptop GPU and tested in five coding agents: what each "
                          "needs, how fast it was, and the command that starts it."])

    def test_model_page_notes_and_commands(self):
        s = setups()
        coder = s["qwen3-coder-next", "llamacpp"]
        m = fit.Machine(16, 32)
        self.assertEqual(words.fit_note(LABELS, coder, fit.verdict(coder, m, 16), "llamacpp"), (
            ("b", "Too big for your RAM:"), " it used 31.9 GB at peak; you have about 21 GB free (estimate). ",
            ("code", "up"), " checks free RAM first, so it will stop before it starts anything."))
        share = s["gemma-4-12b", "vllm"]
        self.assertEqual(words.fit_note(LABELS, share, fit.verdict(share, fit.Machine(12, 64), 16), "vllm"), (
            ("b", "Measured on 16 GB cards only:"),
            " vLLM takes a share of the whole card, so we only know how it behaves on 16 GB."))
        self.assertIsNone(words.fit_note(LABELS, coder, fit.verdict(coder, fit.Machine(16, 64), 16), "llamacpp"))
        b = snapshot_bench()
        pick = fit.hero_pick(b, fit.Machine(16, 64), "hermes")
        self.assertEqual(words.agent_pick(LABELS, pick, "hermes", False), (
            "In Hermes, ", ("a", "Qwen3.8-27B", "/models/qwen3.8-27b/"),
            " passed the agent test and runs on your machine as measured."))
        self.assertEqual(words.agent_pick(LABELS, pick, "hermes", True)[-1],
                         " passed the agent test and runs on our laptop as measured.")
        change = fit.hero_pick(b, fit.Machine(16, 16), "none")
        self.assertEqual(words.agent_pick(LABELS, change, "none", False), (
            "In OpenCode, the closest fit on your machine is ", ("a", "K2 Horizon 7B", "/models/k2-horizon-7b/"),
            ", with a change we haven't measured."))
        vllm = fit.Pick(s["gemma-4-12b", "vllm"], fit.verdict(s["gemma-4-12b", "vllm"], fit.Machine(16, 64), 16), "unranked")
        self.assertEqual(words.agent_pick(LABELS, vllm, "kimi", False), (
            "In Kimi, ", ("a", "Gemma 4 12B", "/models/gemma-4-12b/"),
            " on vLLM passed the agent test and runs on your machine as measured."))
        known = result("hermes", "known-issue", 0, 1)
        self.assertEqual(words.agent_note(known, LABELS, False, "llamacpp", words.agent_pick(LABELS, pick, "hermes", False)), (
            ("b", "Known issue in Hermes: the latest test failed (0 of 1 run passed)."), " ", "In Hermes, ",
            ("a", "Qwen3.8-27B", "/models/qwen3.8-27b/"), " passed the agent test and runs on your machine as measured."))
        self.assertEqual(words.agent_note(known, LABELS, False, "llamacpp", None),
                         (("b", "Known issue in Hermes: the latest test failed (0 of 1 run passed)."),))
        self.assertIsNone(words.agent_note(result("hermes", "passed"), LABELS, False, "llamacpp", None))
        facts = self.facts("qwen3.6-35b-a3b", "sglang")
        self.assertEqual([words.agent_command(facts, a) for a in AGENTS], [
            (("code", "opencode"), ", then pick ", ("code", "sglang-docker/qwen3.6-35b-a3b"), " with /models"),
            (("code", "hermes -p llmctl"),), (("code", "pi --model llmctl/qwen3.6-35b-a3b"),),
            (("code", "dsh web"), ", then pick qwen3.6-35b-a3b on SGLang (llmctl) in its model picker"),
            (("code", "kimi -m llmctl/qwen3.6-35b-a3b"),)])
        chat = self.facts("deepseek-r1-0528-qwen3-8b", "vllm")
        self.assertIsNone(words.agent_command(chat, "opencode"))
        self.assertEqual(words.agent_command(chat, "pi"), (("code", "pi --model llmctl/deepseek-r1-8b --no-tools"),))
        self.assertEqual(words.agent_command(chat, "kimi"), (
            ("code", 'kimi -p "your prompt" --agent-file launcher/kimi-chat-only.md'), ", after ",
            ("code", "./llmctl up deepseek-r1-0528-qwen3-8b --runtime vllm --set-default"),
            ", which makes it the default in every agent llmctl set up"))
        qwen = model_of(self.matrix(), "qwen3.6-35b-a3b")
        good = words.good_to_know(qwen, qwen.cells[2], LABELS, 30000, "faster")
        self.assertEqual(good[0], ("faster",))
        self.assertEqual([words.plain(x) for x in good[1:]], [
            f"Answers up to {facts.output:,} tokens long.",
            "Other tools can use it like OpenAI's API: http://127.0.0.1:30000/v1 with the model name qwen3.6-35b-a3b.",
            "Measured on 18 November 2026."])
        self.assertEqual(len(words.good_to_know(qwen, qwen.cells[2], LABELS, 30000, None)), 3)
        self.assertEqual(words.evidence_links(facts, "research/2026-11-18/run/summary.md"), (
            (("a", "Setup notes", facts.doc),), (("a", "Measurement record", facts.record),),
            (("a", "Agent test summary", "research/2026-11-18/run/summary.md"),)))
        self.assertEqual(len(words.evidence_links(facts, None)), 2)

    def test_a_cap_on_our_laptop_says_our_laptop(self):
        coder = dataclasses.replace(setups()["qwen3-coder-next", "llamacpp"], memory_cap=48)
        m = fit.Machine(16, 64)
        got = words.map_text(LABELS, coder, fit.verdict(coder, m, 16), m, True, False)
        self.assertEqual((got.ram_right, got.ram_short), ("about 40 GB free on our laptop (estimate)", True))

    def test_the_laptop_wording_gives_our_laptops_own_ram(self):
        # README.md#machine: 61 GiB usable, about 40 GiB free; the laptop machine is the 64 GB chip, whose estimate is 42
        m = fit.Machine(16, 64)
        for setup in setups().values():
            verdict = fit.verdict(setup, m, 16)
            if verdict.key == "ram" or (setup.gpu_mib is None and setup.ram_peak is not None):
                continue
            with self.subTest(setup=setup.key):
                got = words.map_text(LABELS, setup, verdict, m, True, False)
                self.assertIn("Our laptop: a 16 GB GPU and about 40 GB of free RAM (estimate).", got.alt)
                if verdict.ram == "ok":
                    self.assertEqual(got.ram_right, "about 21 GB in use by the desktop and apps (estimate)")

    def test_the_free_ram_beside_a_peak_never_reads_above_a_peak_it_rejects(self):
        # 16 GB of RAM leaves about 10.5 GB free; a peak of 10.6 GB must not sit beside "about 11 GB free"
        for key, setup in setups().items():
            for g in fit.GPUS:
                for r in fit.RAMS:
                    m = fit.Machine(g, r)
                    verdict = fit.verdict(setup, m, 16)
                    if verdict.key != "ram":
                        continue
                    with self.subTest(key=key, gpu=g, ram=r):
                        peak = Fraction(words.gb(setup.ram_peak))
                        note = "".join(part if isinstance(part, str) else part[1]
                                       for part in words.fit_note(LABELS, setup, verdict, key[1]))
                        free = int(re.search(r"you have about (\d+) GB free", note).group(1))
                        self.assertGreater(peak, free)
                        alt = words.map_text(LABELS, setup, verdict, m, False, False).alt
                        self.assertGreater(peak, int(re.search(r"about (\d+) GB of free RAM", alt).group(1)))

    def test_the_footer_date_is_the_commit_day_in_utc(self):
        # the test runs in a zone 14 hours east of UTC, so a footer in local time fails here even where CI runs at UTC
        old = os.environ.get("TZ")
        os.environ["TZ"] = "Pacific/Kiritimati"
        time.tzset()
        self.addCleanup(time.tzset)
        self.addCleanup(lambda: os.environ.__setitem__("TZ", old) if old is not None else os.environ.pop("TZ", None))
        rich, _, _ = words.footer("https://github.com/o/r", "a" * 40, 1763510399, ("CC BY 4.0", "MIT"))
        self.assertEqual(rich[3][1], "Updated 18 November 2025")   # 23:59:59 UTC; 19 November east of Greenwich

    def test_why_clause_cuts_at_a_comma_before_measured_only(self):
        self.assertEqual(words.why_clause("x measured 2026-11-17"), "X measured 2026-11-17.")

    def test_an_impossible_change_reads_r4(self):
        e4b = setups()["gemma-4-e4b", "sglang"]
        self.assertEqual(words.change_sentence(LABELS, e4b, fit.verdict(e4b, fit.Machine(8, 64), 16),
                                               fit.Change("impossible", None)),
                         "Its files alone are 11.5 GB, more than your GPU holds; its commands show what to try, much slower "
                         "and not measured.")

    def test_compare_agents_joins_the_classes_with_semicolons(self):
        mixed = [result("opencode", "passed"), result("hermes", "known-issue", 0), result("pi", "failed", 0),
                 result("dsh", "passed"), result("kimi", "passed")]
        self.assertEqual(words.compare_agents(mixed, LABELS, False, "sglang", "none"),
                         "3 of 5 agents: Hermes known issue; Pi failed the latest test")

    def test_another_engine_five_percent_faster_is_named(self):
        matrix = self.matrix()
        qwen = model_of(matrix, "qwen3.6-35b-a3b")
        decode = {"llamacpp": 100.0, "sglang": 105.0}
        cells = tuple(dataclasses.replace(c, recipe=dataclasses.replace(c.recipe, decode=decode[c.runtime]))
                      if c.runtime in decode else c for c in qwen.cells)
        five = dataclasses.replace(qwen, cells=cells)
        image_gb = {r.id: r.image_gb for r in matrix.runtimes}
        got = words.faster_sentence(five, cells[0], words.labels(matrix), image_gb)
        self.assertIsNotNone(got)
        self.assertEqual(got[:31], "5% faster on SGLang (105.0 tok/")

    def test_the_desktop_gb_follows_fit(self):
        g = setups()["gemma-4-12b", "llamacpp"]
        m = fit.Machine(12, 64)
        with mock.patch.object(fit, "DESKTOP_GPU_MIB", 2048):
            self.assertEqual(words.map_text(LABELS, g, fit.verdict(g, m, 16), m, False, False).gpu_right,
                             "about 2 GB for your desktop (estimate)")
            self.assertIn("the page assumes about 2 GB of GPU memory goes to your desktop",
                          words.plain(words.tested(self.matrix(), self.matrix().runs[0])[3]))

    def test_states_without_copy_stop_the_build(self):
        s = setups()
        share = dataclasses.replace(s["gemma-4-12b", "vllm"], ram_peak=Fraction(19, 2))
        m = fit.Machine(16, 64)
        with self.assertRaisesRegex(ValueError, "^launcher/site/words.py: .*a share of the card and a RAM peak"):
            words.map_text(LABELS, share, fit.verdict(share, m, 16), m, False, False)
        big = dataclasses.replace(s["gemma-4-12b", "llamacpp"], ram_peak=Fraction(45), memory_cap=48)
        judged = fit.verdict(big, m, 16)
        self.assertEqual(judged.key, "ram")
        words.map_text(LABELS, big, judged, m, False, False)
        with self.assertRaisesRegex(ValueError, "^launcher/site/words.py: .*too big for the RAM of our laptop"):
            words.map_text(LABELS, big, judged, m, True, False)
        matrix = self.matrix()
        qwen = model_of(matrix, "qwen3.6-35b-a3b")
        cell = dataclasses.replace(qwen.cells[0], recipe=dataclasses.replace(qwen.cells[0].recipe, decode=None))
        with self.assertRaisesRegex(ValueError, "^launcher/site/words.py: .*no measured speed"):
            words.summary(qwen, cell, words.labels(matrix))

    def test_code_errors_start_with_words_source(self):
        s = setups()
        default = s["qwen3.6-35b-a3b", "llamacpp"]
        ok = fit.verdict(default, fit.Machine(16, 64), 16)
        facts = self.facts("qwen3.6-35b-a3b", "llamacpp")
        calls = [lambda: words.reason(LABELS, fit.Pick(default, ok, "change"), ok, None, [], (), "opencode"),
                 lambda: words.reason(LABELS, fit.Pick(s["gemma-4-12b", "llamacpp"], ok, "ranked"), ok, None, [], (),
                                      "opencode"),
                 lambda: words.joined([]), lambda: words.joined_or([]),
                 lambda: words.result_sentence(result("pi", "odd"), LABELS, False, "llamacpp"),
                 lambda: words.result_short(result("pi", "odd"), False),
                 lambda: words.agents_cell([result("pi", "passed")], LABELS, False, "kimi"),
                 lambda: words.moe_answer("vllm"),
                 lambda: words.override_lines(default, ok, fit.Change("odd", None), "llamacpp"),
                 lambda: words.override_lines(s["gpt-oss-20b", "llamacpp"], ok, fit.Change(None, "cap"), "llamacpp"),
                 lambda: words.agent_lines(facts, LABELS, "llamacpp", "odd"),
                 lambda: words.agent_command(facts, "odd"),
                 lambda: words.not_run_sentence(model_of(self.matrix(), "qwen3.6-35b-a3b").cells[0],
                                                self.matrix().runtimes[0], LABELS)]
        for i, call in enumerate(calls):
            with self.subTest(call=i):
                with self.assertRaisesRegex(ValueError, "^launcher/site/words.py: "):
                    call()

    def test_copy_constants_follow_spec_7(self):
        """Spec 7.3 to 7.12, constant by constant, as the spec writes them."""
        self.assertEqual((words.BRAND, words.BRAND_LABEL, words.NAV_LABEL, words.NAV, words.SKIP_TO_MODELS,
                          words.SKIP_TO_MODEL),
                         ("llmctl", "llmctl, home", "Site",
                          {"models": "Models", "get-started": "Get started", "how-we-tested": "How we tested",
                           "github": "GitHub"}, "Skip to the models", "Skip to the model"))
        self.assertEqual((words.GPU_LEGEND, words.RAM_LEGEND, words.AGENT_LEGEND), ("GPU memory", "RAM", "Coding agent"))
        self.assertEqual(words.NOJS_NOTE, "Turn on JavaScript to check your own machine. Without it, this is our laptop's "
                         "answer, and the table below shows every model on it.")
        self.assertEqual(words.OS_SUMMARY, "No NVIDIA GPU, or on Windows or a Mac?")
        self.assertEqual(words.OS_BODY, (
            "llmctl needs Linux and an NVIDIA GPU, so these numbers won't apply to your machine. You can still try several "
            "of these models remotely through OpenRouter before you buy hardware; your prompts then leave your machine. ",
            ("a", "Try them remotely", "docs/opencode.md#trying-a-model-remotely-first-with-openrouter")))
        self.assertEqual((words.ANSWER_LABEL, words.FACT_LABELS, words.SPEED_UNIT, words.GB_UNIT, words.CTA,
                          words.VERDICT_PREFIX),
                         ("Your answer", ("writing speed on our laptop", "context", "download"), "tok/s", "GB",
                          "Get the commands", ("On our test laptop: ", "On your machine: ")))
        self.assertEqual((words.GPU_TRACK, words.RAM_TRACK), ("GPU memory", "RAM"))
        self.assertEqual((words.MORE_HEADING, words.SORT_LABEL, words.COLUMN_HEADS, words.ROW_HIDDEN, words.TAGS),
                         ("More models we measured", "Sort", ("", "Model", "Writing speed", "Context", "Download"),
                          {"rank": "Rank ", "context": " of context", "download": " download"},
                          {"default": "Our default", "chat": "Chat only", "pick": "Start here"}))
        self.assertEqual((words.GS_HEADING, words.BEFORE_HEADING, words.BEFORE_LEAD),
                         ("Get started", "Before you start",
                          "llmctl runs on Linux with an NVIDIA GPU. Windows, including WSL, and macOS aren't covered."))
        self.assertEqual(words.CHECKS, (
            ("An NVIDIA GPU and its driver.", ("Check: ", ("code", "nvidia-smi"), " shows your card and its memory.")),
            ("Docker Engine.", ("Check: ", ("code", "docker info"), " answers without sudo once you're in the docker group.")),
            ("The NVIDIA Container Toolkit.",
             ("Check: ", ("code", "nvidia-ctk cdi list"), " lists nvidia.com/gpu=all. If it doesn't, follow ",
              ("a", "the setup guide", "docs/host-setup.md#install-the-nvidia-container-toolkit"), ".")),
            ("Python 3.11 or newer.", ("Check: ", ("code", "python3 --version"),
                                       ". llmctl uses only Python's standard library, so there's nothing else to install."))))
        self.assertEqual(words.SUDO_LINE, ("Every setup step that needs sudo is listed in one place, with its reason: ",
                                           ("a", "commands that need sudo", "docs/host-setup.md#commands-that-need-sudo"),
                                           "."))
        self.assertEqual((words.COMMANDS_HEADING, words.COPY_HINTS, words.UI),
                         ("The commands", ("Click a line to copy it", "Tap a line to copy it"),
                          {"copy": "Copy", "copied": "Copied", "selected": "Selected", "copiedLive": "Copied: ",
                           "copyFailed": "Couldn't copy it for you. The command is selected; press Ctrl+C to copy it."}))
        self.assertEqual((words.REFUSE_HEADING, words.REFUSE_LEAD),
                         ("If llmctl refuses", (("code", "./llmctl up"), " checks your machine before it starts anything, "
                                                "and says what it found. The messages you're most likely to see:")))
        self.assertEqual([fix for _, fix in words.REFUSALS], [
            ("Close programs and run ", ("code", "./llmctl up"), " again. Still short? Make your own copy of the setup "
                                                                 "with a lower memory_gib."),
            ("Close programs that use the GPU, or make your own copy of the setup that uses less GPU memory.",),
            ("Install the NVIDIA Container Toolkit and generate the spec.",),
            ("Add yourself to the docker group, then log out and back in.",),
            ("Run ", ("code", "./llmctl down"), ", or add ", ("code", "--replace"), " to the up line.")])
        self.assertEqual([words.plain(x) for x in words.REFUSE_LINKS],
                         ["Every message and its fix", "Making your own copy of a setup"])
        self.assertEqual((words.COMPARE_HEADING, words.MODEL_HEAD, words.engine_cell_label("vLLM")),
                         ("Compare engines", "Model", "vLLM: "))
        self.assertEqual((words.HOW_HEADING, words.FLOW_LABEL, words.RULES_INTRO),
                         ("What llmctl does on your machine", "Your coding agent talks to llmctl's container on 127.0.0.1, "
                          "which uses your GPU memory and RAM", "It keeps a few rules you can't switch off:"))
        self.assertEqual(words.flow(LABELS)[1:], (("llmctl's container", "Engine and model, on 127.0.0.1 only"),
                                                  ("Your GPU memory and RAM", "")))
        self.assertEqual(words.HOW_PROSE, (
            ("code", "./llmctl up"), " checks your GPU memory, RAM and Docker before it starts anything, runs the model in a "
            "container, waits until it answers, and points your coding agents at it. ", ("code", "./llmctl down"),
            " stops it."))
        self.assertEqual(words.RULES, (
            "The model listens on 127.0.0.1 only, so other machines on your network can't reach it.",
            "The container gets a memory cap, so a model can't swap your machine to a crawl.",
            "One model runs at a time, and nothing restarts on its own, so your GPU can power down.",
            "It asks before it downloads, and never downloads during up.",
            "Before it first changes an agent's config file that holds your settings, it saves a backup next to it."))
        self.assertEqual(words.HOW_LINK, (("a", "Everything llmctl does, command by command", "docs/llmctl.md"),))
        self.assertEqual((words.QUESTIONS_HEADING, words.TESTED_HEADING), ("Questions", "How we tested"))
        self.assertEqual((words.BACK, words.FIGURES_ARIA, words.RUN_HEADING,
                          words.AGENT_SELECT, words.ENGINES_HEADING, words.USE_ENGINE, words.WHY, words.AGENTS_HEADING,
                          words.YOUR_AGENT, words.GOOD_HEADING, words.EVIDENCE_HEADING, words.PAGER, words.HOME_TITLE),
                         ("All models", "Key numbers", "Run it", "Your agent", "Engines",
                          "Use this engine", "Why", "In your coding agent", "Your agent", "Good to know", "Evidence",
                          {"label": "More models", "previous": "Previous", "next": "Next"},
                          "Open coding models for your own GPU, llmctl"))

    def test_words_hold_with_python_3_11_fractions(self):
        b = snapshot_bench()
        with py311_fraction_format():
            with self.assertRaises(TypeError):
                f"{Fraction(1, 2):.1f}"
            for m in fit.machines():
                names = [LABELS.models[i] for i in fit.models_as_measured(b, m)]
                self.assertEqual(words.verdict_line(m == b.home, len(names), LABELS.total)[:3], "On ")
                for agent in (*b.agents, fit.NONE):
                    with self.subTest(machine=m, agent=agent):
                        pick = fit.hero_pick(b, m, agent)
                        change = fit.change(pick.setup, pick.verdict)
                        reason = words.reason(LABELS, pick, fit.default_verdict(b, m), change, names, fit.more_memory(b, m),
                                              agent)
                        self.assertIsNone(FORBIDDEN.search(reason))
                        words.override_lines(pick.setup, pick.verdict, change, LABELS.first_engine)
                for x in b.setups:
                    got = words.map_text(LABELS, x, fit.verdict(x, m, 16), m, m == b.home, True)
                    self.assertTrue(got.alt)
            self.assertEqual(words.gb(Fraction(1907, 512)), "3.7")
            self.assertEqual(words.half_up(Fraction(21, 2), 0), "11")

    def test_no_text_breaks_the_copy_rules(self):
        constants = {name: value for name, value in vars(words).items() if name.isupper()}
        self.assertGreater(len(constants), 40)
        for name, value in constants.items():
            for text in texts(value):
                with self.subTest(constant=name, text=text):
                    self.assertIsNone(FORBIDDEN.search(text))
                    self.assertNotIn("N/A", text)


if __name__ == "__main__":
    unittest.main()
