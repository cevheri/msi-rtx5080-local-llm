"""The local LLM page's sentences (spec 7): every text a reader sees, hears or copies, as a constant or a pure function
over the repository's facts and the machine rules of fit.py.

No HTML and no URL lives here: a sentence that holds a link, a code span or bold text is rich text (Rich), a tuple of
parts that render.py resolves and escapes. No Fraction meets a format spec (Python 3.11 refuses it), so every number is
written through half_up's integer arithmetic or data.speed_text over a float.
"""

import datetime
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from launcher import clients
from launcher.site import data, fit

SOURCE = "launcher/site/words.py"
Part = str | tuple[str, ...]
Rich = tuple[Part, ...]

MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December")
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
CLASS_WORDS = (("known-issue", "known issue"), ("failed", "failed the latest test"),
               ("not-tested", "not tested yet"), ("not-supported", "not set up by llmctl"))
NONE_LABEL = "None yet"

_PARAMS_MOE = re.compile(r"^(about )?([\d.]+)B / (about )?([\d.]+)B active (?:hybrid )?MoE$", re.I)
_PARAMS_DENSE = re.compile(r"^([\d.]+)B dense$", re.I)
_NAME_MOE = re.compile(r"(\d+(?:\.\d+)?)B[- ]A(\d+(?:\.\d+)?)B")
_WHY_CUT = re.compile(r", measured \d{4}-\d{2}-\d{2}|,? from \[")
_AGENT_VERSION = re.compile(r"v?(\d+\.\d+\.\d+(?:-rc\.\d+)?)")


# --- rich text (spec 7.1) ------------------------------------------------------------------------------------------

def code(text: str) -> tuple[str, str]:
    return ("code", text)


def bold(text: str) -> tuple[str, str]:
    return ("b", text)


def link(text: str, target: str) -> tuple[str, str, str]:
    """A link part; the target is https://..., a page of this site (/...) or a repository path with an anchor."""
    return ("a", text, target)


def plain(value: str | Rich) -> str:
    """The text of a plain or rich value, without its markup."""
    if isinstance(value, str):
        return value
    return "".join(part if isinstance(part, str) else part[1] for part in value)


def _rich(*parts: Part) -> Rich:
    """A rich value with adjacent text parts joined and empty ones left out."""
    out: list[Part] = []
    for part in parts:
        if isinstance(part, str):
            if not part:
                continue
            if out and isinstance(out[-1], str):
                out[-1] = out[-1] + part
                continue
        out.append(part)
    return tuple(out)


# --- the names every sentence needs ------------------------------------------------------------------------------

@dataclass(frozen=True)
class Labels:
    models: Mapping[str, str]
    engines: Mapping[str, str]
    agents: Mapping[str, str]
    first_engine: str
    test_gpu: int
    top: int
    total: int
    default: tuple[str, str]


def labels(matrix: data.Matrix) -> Labels:
    """The display names of the models, engines and agents (none reads "None yet"), and the counts the copy uses."""
    agents = {client.id: client.label for client in matrix.clients}
    agents[fit.NONE] = NONE_LABEL
    return Labels(models={model.id: model.name for model in matrix.models},
                  engines={runtime.id: runtime.label for runtime in matrix.runtimes}, agents=agents,
                  first_engine=matrix.runtimes[0].id, test_gpu=matrix.laptop.gpu_gb, top=matrix.top,
                  total=len(matrix.models), default=matrix.default)


def _agent_count(labels: Labels) -> int:
    return len([agent for agent in labels.agents if agent != fit.NONE])


def _agent_label(labels: Labels, agent: str) -> str:
    """The agent's label; "none" reads as OpenCode, the agent this project is built around."""
    return labels.agents[fit.DEFAULT_AGENT if agent == fit.NONE else agent]


# --- units and small words (spec 7.2, R1) --------------------------------------------------------------------------

def half_up(value: Fraction, places: int = 1) -> str:
    """`value` rounded half up to `places` decimals, its digits built from an integer (no format spec)."""
    n = math.floor(Fraction(value) * 10 ** places + Fraction(1, 2))
    if places == 0:
        return str(n)
    digits = str(abs(n)).rjust(places + 1, "0")
    return ("-" if n < 0 else "") + digits[:-places] + "." + digits[-places:]


def gb_mib(mib: int) -> str:
    """GPU memory and RAM in GB as the system counts them (1 GB = 1,024 MiB), one decimal."""
    return half_up(Fraction(mib, 1024))


def gb_bytes(size: int) -> str:
    """A download in GB as llmctl pull prints it (10^9 bytes), one decimal."""
    return half_up(Fraction(size, 10 ** 9))


def gb(value: Fraction) -> str:
    """GiB already counted in GB, one decimal."""
    return half_up(value)


def context_k(context: int) -> str:
    return f"{context // 1024}k"


def speed(decode: float | None) -> str:
    """RecipeFacts.decode as the page writes it, or "not measured" (P17); a fit.Setup caller passes float(decode)."""
    return "not measured" if decode is None else data.speed_text(decode)


def _speed_of(setup: fit.Setup) -> str:
    return speed(None if setup.decode is None else float(setup.decode))


def _pct(share: Fraction) -> str:
    return half_up(share * 100, 0)


def day(iso: str) -> str:
    year, month, date = iso.split("-")
    return f"{int(date)} {MONTHS[int(month) - 1]} {year}"


def day_short(iso: str) -> str:
    year, month, date = iso.split("-")
    return f"{int(date)} {MONTHS[int(month) - 1][:3]} {year}"


def engine_version(heading: str) -> str:
    """"SGLang v0.5.23" reads "SGLang 0.5.23"; "llama.cpp b12407" stays."""
    return re.sub(r" v(\d)", r" \1", heading)


def version_detail(heading: str) -> str:
    """The version alone: "b12407", "0.33.0"."""
    return engine_version(heading).split(" ", 1)[1]


def count_word(n: int) -> str:
    """Counts below ten in running prose are words."""
    return NUMBER_WORDS.get(n, str(n))


def joined(items: Sequence[str]) -> str:
    items = list(items)
    if not items:
        raise ValueError(f"{SOURCE}: nothing to join")
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def joined_or(items: Sequence[str]) -> str:
    items = list(items)
    if not items:
        raise ValueError(f"{SOURCE}: nothing to join")
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " or " + items[-1]


def gpu_chip(gpu: int) -> str:
    return f"{gpu} GB" + (" or more" if gpu == fit.GPUS[-1] else "")


def ram_chip(ram: int) -> str:
    return f"{ram} GB" + (" or more" if ram == fit.RAMS[-1] else "")


def gpu_phrase(gpu: int) -> str:
    if gpu == fit.GPUS[-1]:
        return f"a GPU of {gpu} GB or more"
    article = "an" if str(gpu).startswith("8") or gpu in (11, 18) else "a"
    return f"{article} {gpu} GB GPU"


def ram_phrase(ram: int) -> str:
    return f"{ram} GB of RAM" + (" or more" if ram == fit.RAMS[-1] else "")


def title(labels: Labels, model: str, runtime: str, default_engine: bool) -> str:
    """The model's name, plus " on {engine}" off its default engine."""
    return labels.models[model] + ("" if default_engine else f" on {labels.engines[runtime]}")


def _title_of(labels: Labels, setup: fit.Setup) -> str:
    return title(labels, setup.model, setup.runtime, setup.default_engine)


def _desktop_gb() -> str:
    return half_up(Fraction(fit.DESKTOP_GPU_MIB, 1024), 0)


# --- header and hero (spec 7.3) ------------------------------------------------------------------------------------

BRAND = "llmctl"
BRAND_LABEL = "llmctl, home"
NAV_LABEL = "Site"
NAV = {"models": "Models", "get-started": "Get started", "how-we-tested": "How we tested", "github": "GitHub"}
SKIP_TO_MODELS = "Skip to the models"
SKIP_TO_MODEL = "Skip to the model"
H1 = "Will it run on your machine?"
LEAD = ("llmctl starts a tested open coding model in Docker and connects it to your coding agent. For Linux with an "
        "NVIDIA GPU: tell us your GPU and RAM, and get the commands.")
GPU_LEGEND = "GPU memory"
RAM_LEGEND = "RAM"
AGENT_LEGEND = "Coding agent"
GPU_HELP: Rich = ("Not sure? ", code("nvidia-smi"), " shows it.")
RAM_HELP: Rich = (code("free -h"), " shows it.")
STATE_UNTOUCHED = "Set to our test laptop. Change it to yours."
STATE_BAD_MACHINE = "Showing our laptop: the link held a value this page doesn't know."
STATE_BAD_AGENT = f"Showing {clients.LABELS[fit.DEFAULT_AGENT]}: the link named an agent this page doesn't know."
RESET = "Back to our test laptop"
NOJS_NOTE = ("Turn on JavaScript to check your own machine. Without it, this is our laptop's answer, and the table "
             "below shows every model on it.")
OS_SUMMARY = "No NVIDIA GPU, or on Windows or a Mac?"
_REMOTE = "docs/opencode.md#trying-a-model-remotely-first-with-openrouter"
OS_BODY: Rich = ("llmctl needs Linux and an NVIDIA GPU, so these numbers won't apply to your machine. You can still "
                 "try several of these models remotely through OpenRouter before you buy hardware; your prompts then "
                 "leave your machine. ", link("Try them remotely", _REMOTE))
ANSWER_LABEL = "Your answer"
CTA = "Get the commands"
FACT_LABELS = ("writing speed on our laptop", "context", "download")
SPEED_UNIT = "tok/s"
GB_UNIT = "GB"
VERDICT_PREFIX = ("On our test laptop: ", "On your machine: ")  # laptop wording, then the described machine


def agent_help(n: int) -> str:
    return f"llmctl sets up all {count_word(n)} for you."


def verdict_line(laptop: bool, n: int, total: int) -> str:
    """H7: the prefix, then how many models run as measured."""
    if n == 0:
        count = f"none of the {total} models runs as we measured it."
    elif n == total:
        count = f"all {total} models run as we measured them."
    elif n == 1:
        count = f"one of {total} models runs as we measured it."
    else:
        count = f"{count_word(n)} of {total} models run as we measured them."
    return VERDICT_PREFIX[0 if laptop else 1] + count


def pick_heading(title: str) -> str:
    """H8."""
    return f"Start with {title}"


def switch_sentence(labels: Labels, default: fit.Verdict, title: str) -> str:
    """R3: why our default is not the answer here, and what to start with instead."""
    if default.short is not None:
        why = f"is {gb(default.short)} GB short of GPU memory there"
    elif default.key == "ram":
        why = "is too big for the RAM there"
    elif default.key == "only16":
        why = f"was measured on {labels.test_gpu} GB cards only"
    else:
        why = "needs more free RAM than there is"
    return f"Our pick, {labels.models[labels.default[0]]}, {why}, so start with {title}."


def _peak_words(setup: fit.Setup) -> str:
    return "its RAM use wasn't measured" if setup.ram_peak is None else f"it used {gb(setup.ram_peak)} GB at peak"


def change_sentence(labels: Labels, setup: fit.Setup, verdict: fit.Verdict, change: fit.Change) -> str:
    """R4 to R6: the change the pick needs, in one sentence; never a smaller context when the files alone are larger
    than the usable card (fix 5)."""
    if change.gpu in ("layers", "impossible"):
        return (f"Its files alone are {gb_bytes(setup.size_bytes)} GB, more than your GPU holds; its commands show "
                "what to try, much slower and not measured.")
    if change.gpu is not None:
        return (f"Your card is {gb(verdict.short)} GB short of what it used; its commands show the first change to "
                "try, not measured.")
    return f"It fits your GPU but needs a lower RAM cap ({_peak_words(setup)}); its commands show how."


def _more_sentence(more: fit.More) -> str:
    """R8."""
    phrase = gpu_phrase(more.size) if more.kind == "gpu" else ram_phrase(more.size)
    return f"With {phrase}, {count_word(more.count)} more would."


def reason(labels: Labels, pick: fit.Pick, default: fit.Verdict, change: fit.Change | None, names: Sequence[str],
           more: Sequence[fit.More], agent: str) -> str:
    """The reason under "Start with {title}" (spec 7.3): the switch or the step, the change, the names, the more-memory
    sentences."""
    switch = None
    if pick.setup.key != labels.default and default.key != "ok":
        switch = switch_sentence(labels, default, _title_of(labels, pick.setup))
    if pick.step == "change":
        if change is None:
            raise ValueError(f"{SOURCE}: a change pick of {pick.setup.key} without a change plan")
        first = switch[:-1] + ", the closest fit." if switch else "The closest fit, with a change we haven't measured."
        parts = [first, change_sentence(labels, pick.setup, pick.verdict, change)]
    else:
        if switch:
            parts = [switch]
        elif pick.step == "default":
            parts = [f"Our default, and the model our {labels.agents[fit.DEFAULT_AGENT]} setup starts with."]
        elif pick.step == "default-failed":
            parts = [f"Our default, {labels.models[labels.default[0]]}, failed the agent test in "
                     f"{_agent_label(labels, agent)}, so this is the highest-ranked model that passed there."]
        else:
            raise ValueError(f"{SOURCE}: no reason for step {pick.step} while our default runs as measured")
        if 1 < len(names) <= 4:
            parts.append(f"{joined(names)} run as we measured them.")
    parts += [_more_sentence(item) for item in more]
    return " ".join(parts)


def _runs_word(n: int) -> str:
    return "run" if n == 1 else "runs"


def result_sentence(result: data.ClientResult, labels: Labels, chat_only: bool, runtime: str) -> str:
    """A3: one agent's result."""
    label = labels.agents[result.client]
    counted = f"{result.passed_runs} of {result.runs} {_runs_word(result.runs)}"
    if result.cls == "passed":
        if chat_only:
            return f"Passed our chat test in {label} ({counted}), without tools."
        return f"Passed the agent test in {label} ({counted})."
    if result.cls == "known-issue":
        return f"Known issue in {label}: the latest test failed ({counted} passed)."
    if result.cls == "failed":
        return f"Failed the latest test in {label}."
    if result.cls == "not-tested":
        return f"Not tested in {label} yet."
    if result.cls == "not-supported":
        return f"llmctl doesn't set up {label} for this model on {labels.engines[runtime]}."
    raise ValueError(f"{SOURCE}: unknown agent result class {result.cls}")


def _result_of(results: Sequence[data.ClientResult], agent: str) -> data.ClientResult:
    client = fit.DEFAULT_AGENT if agent == fit.NONE else agent
    found = [result for result in results if result.client == client]
    if len(found) != 1:
        raise ValueError(f"{SOURCE}: {len(found)} results for the agent {client}")
    return found[0]


def _class_sentences(results: Sequence[data.ClientResult], labels: Labels) -> list[str]:
    out = []
    for cls, word in CLASS_WORDS:
        names = [labels.agents[result.client] for result in results if result.cls == cls]
        if names:
            out.append(f"{joined(names)}: {word}.")
    return out


def _agents_summary(results: Sequence[data.ClientResult], labels: Labels, skip: str | None) -> list[str]:
    """A1 over every agent, or over every agent but `skip` with "Also works in"."""
    others = [result for result in results if result.client != skip]
    passed = [labels.agents[result.client] for result in others if result.cls == "passed"]
    if skip is None and len(passed) == len(others):
        return [f"Works in all {count_word(len(others))} agents we tested."]
    head = [("Also works in " if skip else "Works in ") + joined(passed) + "."] if passed else []
    return head + _class_sentences(others, labels)


def agents_line(results: Sequence[data.ClientResult], labels: Labels, chat_only: bool, runtime: str, agent: str) -> str:
    """The agents line under the hero's map: A1 for none, else the agent's A3 and A2's summary of the others."""
    if agent == fit.NONE:
        return " ".join(_agents_summary(results, labels, None))
    mine = result_sentence(_result_of(results, agent), labels, chat_only, runtime)
    return " ".join([mine, *_agents_summary(results, labels, agent)])


# --- the memory map and the verdict words (spec 7.4) ---------------------------------------------------------------

VERDICT_SENTENCES = {"ok": "Runs as measured.", "change": "Needs a change we haven't measured.",
                     "ram": "Too big for your RAM.", "only16": "Measured on {test_gpu} GB cards only."}
CAPTION = ("Solid: measured on our laptop.", "Outline: what llmctl asks for.", "Hatched: estimate.")
GPU_TRACK = "GPU memory"
RAM_TRACK = "RAM"
_MOE_ANSWER = "It's a mixture-of-experts model, so most of its experts wait in RAM and the busiest stay on the GPU."
_MOE_LINKS = {"llamacpp": "llamacpp/README.md#moe-expert-offload", "sglang": "sglang/README.md#paged-experts"}


@dataclass(frozen=True)
class MapText:
    gpu_left: Rich
    gpu_right: str
    gpu_short: bool
    ram_left: Rich
    ram_right: str
    ram_short: bool
    alt: str


def verdict_sentence(key: str, labels: Labels) -> str:
    """V1."""
    return VERDICT_SENTENCES[key].format(test_gpu=labels.test_gpu)


def verdict_word(key: str, labels: Labels) -> str:
    """V1 without its period, for the model list."""
    return verdict_sentence(key, labels)[:-1]


def _setup_alt(labels: Labels, setup: fit.Setup) -> str:
    name = labels.models[setup.model]
    asks = f"llmctl asks for {setup.memory_cap} GB of free RAM before it starts it."
    if setup.gpu_mib is None:
        return (f"On our laptop, {labels.engines[setup.runtime]} took {_pct(setup.gpu_share)}% of the "
                f"{labels.test_gpu} GB card for {name}; its RAM use wasn't measured. {asks}")
    if setup.ram_peak is None:
        return (f"{name} used {gb_mib(setup.gpu_mib)} GB of GPU memory on our laptop; its RAM use wasn't measured. "
                f"{asks}")
    return (f"{name} used {gb_mib(setup.gpu_mib)} GB of GPU memory and {gb(setup.ram_peak)} GB of RAM at its peak on "
            f"our laptop. {asks}")


def map_text(labels: Labels, setup: fit.Setup, verdict: fit.Verdict, machine: fit.Machine, laptop: bool,
             bold: bool) -> MapText:
    """M2 to M5 and M7, one rule for the hero and the model pages (P12); `laptop` speaks of ours, `bold` marks the
    leading numbers as the model pages do. A share setup with a RAM peak and a setup too big for our laptop's RAM
    have no copy (M4 and M7 disagree, V1 says "your"), so they stop the build."""
    if setup.gpu_mib is None and setup.ram_peak is not None:
        raise ValueError(f"{SOURCE}: {setup.key} has a share of the card and a RAM peak, which M7 has no sentence for")
    if laptop and verdict.key == "ram":
        raise ValueError(f"{SOURCE}: {setup.key} is too big for the RAM of our laptop, which V1 has no laptop wording for")
    mark = (lambda text: ("b", text)) if bold else (lambda text: text)
    whose = "our" if laptop else "your"
    if setup.gpu_mib is not None:
        gpu_left = _rich(mark(f"{gb_mib(setup.gpu_mib)} GB"), f" used of {whose} {gpu_chip(machine.gpu)}")
    elif machine.gpu == labels.test_gpu:
        gpu_left = _rich(f"{labels.engines[setup.runtime]} takes ", mark(f"{_pct(setup.gpu_share)}%"),
                         " of the card up front.")
    else:
        gpu_left = (f"Measured on {labels.test_gpu} GB cards only.",)
    if verdict.short is not None:
        gpu_right, gpu_short = f"{gb(verdict.short)} GB short", True
    elif setup.gpu_mib is None and machine.gpu != labels.test_gpu:
        gpu_right, gpu_short = f"outline: {_pct(setup.gpu_share)}% of your card, not measured", False
    else:
        gpu_right, gpu_short = f"about {_desktop_gb()} GB for {'the' if laptop else 'your'} desktop (estimate)", False
    asks = f"llmctl asks for {setup.memory_cap} GB free to start it."
    if setup.ram_peak is None:
        ram_left = (f"RAM use wasn't measured. {asks}",)
    else:
        ram_left = _rich(mark(f"{gb(setup.ram_peak)} GB"), f" used at peak. {asks}")
    if verdict.key == "ram":
        ram_right, ram_short = f"{gb(setup.ram_peak - verdict.free)} GB short", True
    elif verdict.ram in ("cap", "cap-unmeasured"):
        where = "on our laptop" if laptop else "on yours"
        ram_right, ram_short = f"about {half_up(verdict.free, 0)} GB free {where} (estimate)", True
    else:
        by = "the" if laptop else "your"
        ram_right = f"about {half_up(machine.ram - verdict.free, 0)} GB in use by {by} desktop and apps (estimate)"
        ram_short = False
    owner = "Our laptop" if laptop else "Your machine"
    machine_alt = f"{owner}: {gpu_phrase(machine.gpu)} and about {half_up(verdict.free, 0)} GB of free RAM (estimate)."
    alt = " ".join([_setup_alt(labels, setup), machine_alt, verdict_sentence(verdict.key, labels)])
    return MapText(gpu_left, gpu_right, gpu_short, ram_left, ram_right, ram_short, alt)


def _moe_fits(setup: fit.Setup) -> bool:
    return setup.offload and setup.gpu_mib is not None and \
        Fraction(setup.size_bytes, 10 ** 9) > Fraction(setup.gpu_mib, 1024)


def moe_question(setup: fit.Setup) -> str | None:
    """M8's summary, when an offloading setup's download is larger than its GPU need."""
    if not _moe_fits(setup):
        return None
    return f"Why does a {gb_bytes(setup.size_bytes)} GB download fit in {gb_mib(setup.gpu_mib)} GB of GPU memory?"


def _moe_link(runtime: str) -> tuple[str, str, str]:
    if runtime not in _MOE_LINKS:
        raise ValueError(f"{SOURCE}: no expert offload notes for the runtime {runtime}")
    return link("How that works", _MOE_LINKS[runtime])


def moe_answer(runtime: str) -> Rich:
    """M8's body and its link."""
    return (_MOE_ANSWER + " ", _moe_link(runtime))


def moe_paragraph(setup: fit.Setup) -> Rich | None:
    """M9, the model page's form of M8."""
    if not _moe_fits(setup):
        return None
    answer = _MOE_ANSWER[0].lower() + _MOE_ANSWER[1:]
    return (f"Why a {gb_bytes(setup.size_bytes)} GB download fits in {gb_mib(setup.gpu_mib)} GB of GPU memory: "
            f"{answer} ", _moe_link(setup.runtime))


# --- the shelf (spec 7.5) ------------------------------------------------------------------------------------------

MORE_HEADING = "More models we measured"
KEY = ("Each block is 1 GB.", "Used on our laptop", "The desktop (estimate)")
SORT_LABEL = "Sort"
SORTS = (("rank", "Our ranking"), ("speed", "Fastest first"), ("size", "Smallest download first"))
COLUMN_HEADS = ("", "Model", "Writing speed", "Context", "Download")  # the strip, agents and machine heads are computed
ROW_HIDDEN = {"rank": "Rank ", "context": " of context", "download": " download"}
TAGS = {"default": "Our default", "chat": "Chat only", "pick": "Start here"}


def top_heading(labels: Labels) -> str:
    return f"Our Top {labels.top}"


def strip_head(machine: fit.Machine, labels: Labels, laptop: bool) -> str:
    if laptop:
        return f"Our {labels.test_gpu} GB GPU"
    if machine.gpu == fit.GPUS[-1]:
        return f"A {machine.gpu} GB GPU"
    return f"Your {machine.gpu} GB GPU"


def agents_head(agent: str, labels: Labels) -> str:
    return "Coding agents" if agent == fit.NONE else f"In {labels.agents[agent]}"


def machine_head(laptop: bool) -> str:
    return "On our laptop" if laptop else "On your machine"


def row_engine(labels: Labels, setup: fit.Setup) -> str | None:
    """The engine beside a row's name, off the model's default engine."""
    return None if setup.default_engine else f"on {labels.engines[setup.runtime]}"


def gpu_used(labels: Labels, setup: fit.Setup) -> str:
    if setup.gpu_mib is not None:
        return f"{gb_mib(setup.gpu_mib)} GB of GPU memory"
    return f"{_pct(setup.gpu_share)}% of a {labels.test_gpu} GB card"


def uses(labels: Labels, setup: fit.Setup) -> str:
    return f"Uses {gpu_used(labels, setup)}."


def result_short(result: data.ClientResult, chat_only: bool) -> str:
    """S5 for one agent, under a head that names it."""
    counted = f"{result.passed_runs} of {result.runs}"
    if result.cls == "passed":
        if chat_only:
            return f"Chat test passed ({counted}), no tools"
        return f"Passed ({counted} {_runs_word(result.runs)})"
    if result.cls == "known-issue":
        return f"Known issue ({counted} {_runs_word(result.runs)} passed)"
    words = {"failed": "Failed the latest test", "not-tested": "Not tested yet",
             "not-supported": "Not set up by llmctl"}
    if result.cls not in words:
        raise ValueError(f"{SOURCE}: unknown agent result class {result.cls}")
    return words[result.cls]


def _agents_count(results: Sequence[data.ClientResult]) -> str:
    passed = len([result for result in results if result.cls == "passed"])
    return f"All {passed} agents" if passed == len(results) else f"{passed} of {len(results)} agents"


def agents_cell(results: Sequence[data.ClientResult], labels: Labels, chat_only: bool, agent: str) -> str:
    """S5: the row's agents cell."""
    if agent == fit.NONE:
        return _agents_count(results)
    return result_short(_result_of(results, agent), chat_only)


# --- Get started (spec 7.6) ----------------------------------------------------------------------------------------

GS_HEADING = "Get started"
GS_LEAD = "The short path, for the model we picked at the top of the page. Every model's page has its own commands."
BEFORE_HEADING = "Before you start"
BEFORE_LEAD = "llmctl runs on Linux with an NVIDIA GPU. Windows, including WSL, and macOS aren't covered."
CHECKS: tuple[tuple[str, Rich], ...] = (
    ("An NVIDIA GPU and its driver.", ("Check: ", code("nvidia-smi"), " shows your card and its memory.")),
    ("Docker Engine.", ("Check: ", code("docker info"), " answers without sudo once you're in the docker group.")),
    ("The NVIDIA Container Toolkit.",
     ("Check: ", code("nvidia-ctk cdi list"), " lists nvidia.com/gpu=all. If it doesn't, follow ",
      link("the setup guide", "docs/host-setup.md#install-the-nvidia-container-toolkit"), ".")),
    ("Python 3.11 or newer.",
     ("Check: ", code("python3 --version"), ". llmctl uses only Python's standard library, so there's nothing else to "
                                            "install.")),
)
SUDO_LINE: Rich = ("Every setup step that needs sudo is listed in one place, with its reason: ",
                   link("commands that need sudo", "docs/host-setup.md#commands-that-need-sudo"), ".")
COMMANDS_HEADING = "The commands"
COPY_HINTS = ("Click a line to copy it", "Tap a line to copy it")
COMMANDS_LABEL = "Commands for the model we picked"
UI = {"copy": "Copy", "copied": "Copied", "selected": "Selected", "copiedLive": "Copied: ",
      "copyFailed": "Couldn't copy it for you. The command is selected; press Ctrl+C to copy it."}
REFUSE_HEADING = "If llmctl refuses"
REFUSE_LEAD = ("up checks your machine before it starts anything, and says what it found. The messages you're most "
               "likely to see:")
REFUSALS: tuple[tuple[str, str], ...] = (
    ("... may use up to N GiB of RAM",
     "Close programs and run up again. Still short? Make your own copy of the setup with a lower memory_gib."),
    ("... needs N MiB of free GPU memory",
     "Close programs that use the GPU, or make your own copy of the setup that uses less GPU memory."),
    ("no CDI spec for nvidia.com/gpu", "Install the NVIDIA Container Toolkit and generate the spec."),
    ("Docker refused the connection (permission denied", "Add yourself to the docker group, then log out and back in."),
    ("another model server is running", "Run ./llmctl down, or add --replace to up."),
)
REFUSE_LINKS: tuple[Rich, ...] = (
    (link("Every message and its fix", "docs/llmctl.md#when-up-refuses"),),
    (link("Making your own copy of a setup", "docs/llmctl.md#overrides-for-another-machine"),))
_NO_AGENT_NOTE: Rich = ("No coding agent yet? Install ", link("OpenCode", "https://opencode.ai"),
                        " first; it's the one this project is built around.")
_CONFIG_NOTE: Rich = ("OpenCode needs our provider config once: copy it from ",
                      link("the setup guide", data.OPENCODE_CONFIG), ".")
_REPLACE_NOTE: Rich = ("Another model already running? Add ", code("--replace"),
                       " to the up line and llmctl stops it first.")


def disk_check(download: int, image_gb: str, engine: str) -> tuple[str, Rich]:
    """G3's disk check: `download` in bytes, `image_gb` as docs/host-setup.md writes it, `engine` the label."""
    return (f"Disk space: {gb_bytes(download)} GB for this model, plus the engine image the first time ({image_gb} GB "
            f"for {engine}).", ("Check: ", code("df -h ~")))


def terminal_title(name: str, engine: str) -> str:
    return f"Commands for {name} on {engine}"


def commands_label(name: str) -> str:
    return f"Commands for {name}"


def copy_label(command: str) -> str:
    return f"Copy: {command}"


def head_lines(repository: str) -> list[str]:
    return ["# Get llmctl (once)", f"$ git clone {repository}", f"$ cd {repository.rsplit('/', 1)[1]}"]


def pull_lines(facts: data.RecipeFacts, engine: str, image_gb: str) -> list[str]:
    return [f"# Download {gb_bytes(facts.size_bytes)} GB of model files; the {engine} engine ({image_gb} GB on disk) "
            "comes too, the first time", f"$ {facts.pull}"]


def override_lines(setup: fit.Setup, verdict: fit.Verdict, change: fit.Change | None, first_engine: str) -> list[str]:
    """T1 to T9: the lines between pull and up when the setup needs a change on the machine; none without one."""
    if change is None:
        return []
    if change.gpu is not None and change.ram is not None:
        intro = "# Your machine has less GPU memory and free RAM than this setup needs. Make your own copy of it:"
    elif change.gpu is not None:
        intro = "# Your GPU has less memory than this setup used. Make your own copy of it:"
    else:
        intro = "# Your machine has less free RAM than llmctl asks for. Make your own copy of this setup:"
    runtime, model = setup.runtime, setup.model
    flag = "" if runtime == first_engine else f" --runtime {runtime}"
    lines = [intro, f"$ mkdir -p ~/.config/llmctl/recipes/{runtime}",
             f"$ ./llmctl show {model}{flag} --as-override > ~/.config/llmctl/recipes/{runtime}/{model}.toml"]
    files = f"# Its files alone are {gb_bytes(setup.size_bytes)} GB, more than your GPU holds"
    gpu = {None: None,
           "experts": "# Then lower --moe-cache-mib, then raise --n-cpu-moe; slower, and not measured",
           "resident": "# Then give it fewer resident experts or a smaller --mem-fraction-static; not measured",
           "layers": f"{files}: lower -ngl to keep some layers in RAM; much slower, and not measured",
           "impossible": f"{files}, so this may not be possible on your card",
           "context": "# Then give it a smaller context; not measured"}
    ram = {None: None,
           "cap": "# Then set memory_gib in that file below the RAM you have free (free -g) and above the "
                  f"{'' if setup.ram_peak is None else gb(setup.ram_peak)} GB it used at peak",
           "cap-unmeasured": "# Then set memory_gib in that file below the RAM you have free (free -g); we didn't "
                             "measure how much it uses"}
    if change.gpu not in gpu or change.ram not in ram:
        raise ValueError(f"{SOURCE}: unknown change plan {change}")
    if change.ram == "cap" and setup.ram_peak is None:
        raise ValueError(f"{SOURCE}: {setup.key}: a cap plan needs a measured RAM peak")
    return lines + [line for line in (gpu[change.gpu], ram[change.ram]) if line]


def up_lines(facts: data.RecipeFacts) -> list[str]:
    return ["# Start it: llmctl checks your machine first, then points your agents at it", f"$ {facts.up}"]


def agent_lines(facts: data.RecipeFacts, labels: Labels, runtime: str, agent: str) -> list[str]:
    """The terminal's agent lines (spec 7.6), every name from RecipeFacts.launch."""
    launch = facts.launch
    client = fit.DEFAULT_AGENT if agent == fit.NONE else agent
    if client not in labels.agents:
        raise ValueError(f"{SOURCE}: unknown agent {agent}")
    label = labels.agents[client]
    if client == "opencode":
        if launch.opencode is None:
            return [f"# llmctl doesn't set up {label} for this model on {labels.engines[runtime]}; pick another agent "
                    "above"]
        return [f"# Open {label} and pick {launch.opencode} with /models", "$ opencode"]
    if client == "hermes":
        return [f"# Open {label} with the {launch.hermes} profile that llmctl wrote", f"$ hermes -p {launch.hermes}"]
    if client == "pi":
        return [f"# Open {label} on the model llmctl added",
                f"$ pi --model {launch.pi}" + (" --no-tools" if facts.chat_only else "")]
    if client == "dsh":
        return [f"# Open {label} in your browser, then pick {launch.dsh} in its model picker", "$ dsh web"]
    if client == "kimi":
        if facts.chat_only:
            return [f"# {label} always sends its tools, so make this chat-only model {label}'s default first",
                    f"$ {facts.up} --set-default", f'$ kimi -p "your prompt" --agent-file {launch.kimi_file}']
        return [f"# Open {label} on the model llmctl added", f"$ kimi -m {launch.kimi}"]
    raise ValueError(f"{SOURCE}: unknown agent {agent}")


def tail_lines(facts: data.RecipeFacts) -> list[str]:
    ours = "" if facts.decode is None else f" (ours: {speed(facts.decode)} tok/s)"
    return [f"# Check: a reply, a tool call and your speed{ours}", "$ ./llmctl test",
            "# Stop it when you're done, so your GPU can rest", "$ ./llmctl down"]


def terminal_notes(facts: data.RecipeFacts, labels: Labels, agent: str) -> tuple[Rich, ...]:
    """The notes under the terminal, in spec order."""
    client = fit.DEFAULT_AGENT if agent == fit.NONE else agent
    notes: list[Rich] = []
    if agent == fit.NONE:
        notes.append(_NO_AGENT_NOTE)
    if client == "opencode" and facts.launch.opencode is not None:
        notes.append(_CONFIG_NOTE)
    notes.append(_REPLACE_NOTE)
    notes.append((f"Thinking in {labels.agents[client]}: {dict(facts.thinking_texts)[client]}.",))
    if facts.chat_only and client in ("dsh", "pi"):
        notes.append((f"{labels.agents[client]} with a chat-only model: read ",
                      link("chat-only recipes", "docs/clients.md#chat-only-recipes"), " first."))
    return tuple(notes)


def endpoint(port: int, served: str) -> Rich:
    """G6."""
    return ("Other tools: ", code(f"http://127.0.0.1:{port}/v1"), " with the model name ", code(served), ".")


# --- Compare engines (spec 7.7) ------------------------------------------------------------------------------------

COMPARE_HEADING = "Compare engines"
MODEL_HEAD = "Model"


def compare_lead(labels: Labels, runs: int, combinations: int) -> str:
    return (f"Every model on every engine we tried. {runs} of the {combinations} combinations run on our "
            f"{labels.test_gpu} GB laptop GPU; the others don't fit (our arithmetic) or didn't start (we tried).")


def engine_head(runtime: data.Runtime, first: bool) -> tuple[str, str]:
    """C2: the label, then the version detail with ", the default engine" for the first engine."""
    return runtime.label, version_detail(runtime.heading) + (", the default engine" if first else "")


def engine_cell_label(engine: str) -> str:
    """C4's phone label."""
    return f"{engine}: "


def not_top(labels: Labels) -> str:
    return f"Not in our Top {labels.top}"


def ranked(rank: int) -> str:
    return f"Ranked {rank}"


def compare_agents(results: Sequence[data.ClientResult], labels: Labels, chat_only: bool, runtime: str,
                   agent: str) -> str:
    """C4's agents words: the count and the classes for none, else the agent's A3 sentence."""
    if agent != fit.NONE:
        return result_sentence(_result_of(results, agent), labels, chat_only, runtime)
    rest = []
    for cls, word in CLASS_WORDS:
        names = [labels.agents[result.client] for result in results if result.cls == cls]
        if names:
            rest.append(f"{joined(names)} {word}")
    count = _agents_count(results)
    return f"{count}: {'; '.join(rest)}" if rest else count


def lead_words(model: data.Model, cell: data.Cell, labels: Labels) -> str | None:
    """C4's lead from the cell's fastest tag."""
    found = data.lead(model, cell, labels.engines)
    if found is None:
        return None
    visible, _ = found
    if visible == "tied":
        others = [labels.engines[other.runtime] for other in model.cells
                  if other.fastest == "Tied fastest" and other is not cell]
        return f"Tied fastest for this model with {joined(others)}"
    return f"Fastest for this model, {visible} ahead"


def does_not_fit(labels: Labels) -> tuple[str, str]:
    return f"Doesn't fit on {labels.test_gpu} GB", "(our arithmetic)"


def did_not_start(measured: str) -> tuple[str, str]:
    return "Didn't start", f"tried {day_short(measured)}"


# --- what llmctl does, questions, how we tested, footer (spec 7.8 to 7.11) -----------------------------------------

HOW_HEADING = "What llmctl does on your machine"
FLOW_LABEL = "Your coding agent talks to llmctl's container on 127.0.0.1, which uses your GPU memory and RAM"
HOW_PROSE: Rich = (code("./llmctl up"), " checks your GPU memory, RAM and Docker before it starts anything, runs the "
                   "model in a container, waits until it answers, and points your coding agents at it. ",
                   code("./llmctl down"), " stops it.")
RULES_INTRO = "It keeps a few rules you can't switch off:"
RULES = (
    "The model listens on 127.0.0.1 only, so other machines on your network can't reach it.",
    "The container gets a memory cap, so a model can't swap your machine to a crawl.",
    "One model runs at a time, and nothing restarts on its own, so your GPU can power down.",
    "It asks before it downloads, and never downloads during up.",
    "Before it first changes an agent's config file that holds your settings, it saves a backup next to it.",
)
HOW_LINK: Rich = (link("Everything llmctl does, command by command", "docs/llmctl.md"),)
QUESTIONS_HEADING = "Questions"
TESTED_HEADING = "How we tested"
_PRIVACY = "This page sets no cookies, has no analytics and loads nothing from other sites."


def flow(labels: Labels) -> tuple[tuple[str, str], ...]:
    """The three nodes of the diagram; the last one's second line is the mini map, drawn by render.py."""
    agents = [label for agent, label in labels.agents.items() if agent != fit.NONE]
    return (("Your coding agent", joined_or(agents)), ("llmctl's container", "Engine and model, on 127.0.0.1 only"),
            ("Your GPU memory and RAM", ""))


def questions(labels: Labels, contexts: Sequence[int]) -> tuple[tuple[str, Rich], ...]:
    """The six questions and their answers (spec 7.9); `contexts` are the setups' contexts."""
    smallest, largest = min(contexts), max(contexts)
    return (
        ("Does it work on Windows or a Mac?",
         ("No. llmctl needs Linux with an NVIDIA GPU, Docker and the NVIDIA Container Toolkit. To try some of these "
          "models before you have the hardware, the setup guide shows how to run them remotely through OpenRouter; "
          "your prompts then leave your machine. ", link("Try them remotely", _REMOTE))),
        ("Will I get the same speed?",
         ("Only on similar hardware. Speed depends on your GPU and, on a laptop, on its power mode. ",
          code("./llmctl test"), " prints your speed, so you can compare it with ours.")),
        ("What does “passed the agent test” mean?",
         ("llmctl starts the model, then the agent writes a small file, hello.py, and reads it back. It shows the "
          "agent can use the model's tools; it doesn't grade how well the model codes.",)),
        ("How were the models ranked?",
         ("Three reviews, one for quality, one for practicality and one for coverage, voted on each model; one swap "
          f"keeps the DeepSeek family in the Top {labels.top}. Some ranks rest on scores the vendors published "
          "themselves. ", link("Read how we ranked them", f"{data.SELECTION}#decision-summary"))),
        ("Why not just use Ollama?",
         ("Ollama is a quick way to chat with a model. llmctl starts setups tuned and measured for coding agents, with "
          f"{context_k(smallest)} to {context_k(largest)} of context, tool calls tested in "
          f"{count_word(_agent_count(labels))} agents, and your agents connected for you. If you use Ollama with "
          "OpenCode, read our notes first. ", link("Our Ollama notes", "docs/opencode.md#ollama-notes"))),
        ("What do tok/s, context and thinking mean?",
         ("tok/s: tokens per second, how fast the model writes; a token is a short piece of a word. Context: how much "
          "text the model keeps in view at once, counted in tokens; OpenCode needs at least "
          f"{data.OPENCODE_MINIMUM // 1024}k. Thinking: some models reason before they answer, which costs time and "
          "tokens; most let you switch it per session.",)),
    )


def _agent_version(client: data.Client) -> str:
    for version, _ in client.versions:
        found = _AGENT_VERSION.search(version)
        if found:
            return f"{client.label} {found.group(1)}"
    return client.label


def tested(matrix: data.Matrix, run: data.Run) -> tuple[Rich, ...]:
    """The five paragraphs of How we tested; `run` is the latest run, by date and then folder."""
    lap = matrix.laptop
    if lap.first[:7] == lap.last[:7]:
        span = f"{int(lap.first[8:])} and {day(lap.last)}"
    else:
        span = f"{day(lap.first)} and {day(lap.last)}"
    engines = joined([engine_version(runtime.heading) for runtime in matrix.runtimes])
    agents = joined([_agent_version(client) for client in matrix.clients])
    return (
        (f"Every number on this page comes from one laptop: an {lap.name} with an {lap.gpu_name} ({lap.gpu_gb} GB) "
         f"and {lap.ram_gib} GB of RAM, measured between {span}. It ran {lap.os}, with the GPU at its {lap.power_w} W "
         "power limit.",),
        ("Writing speed is the decode speed of one reply for one user. GPU memory and RAM are the peaks the server "
         "used; llmctl's RAM figure is the free RAM it asks for before it starts.",),
        ("The agent test: llmctl starts each setup, then each agent writes hello.py and reads it back. In the latest "
         f"run, on {day(run.date)}, {run.passed_rows} of {run.rows} steps passed across {run.recipes} setups and "
         f"{count_word(len(run.clients))} agents. Chat-only models get a plain reply test instead, without tools.",),
        ("Solid marks are measured. Hatched marks are estimates: arithmetic, not a test run. To fit a setup to your "
         f"machine, the page assumes about {_desktop_gb()} GB of GPU memory goes to your desktop and that the same "
         f"share of your RAM is free as on our laptop ({fit.FREE_RAM_GIB} of {fit.USABLE_RAM_GIB} GB).",),
        (f"Engines: {engines}. Agents: {agents}.",),
    )


def tested_links(repository: str) -> tuple[Rich, ...]:
    return ((link("How we ranked the models", f"{data.SELECTION}#decision-summary"),),
            (link("Measurement records", "research"),), (link("llmctl's guide", "docs/llmctl.md"),),
            (link("The knowledge base on GitHub", repository),))


def footer(repository: str, commit: str, commit_time: int, licenses: tuple[str, str]) -> tuple[Rich, str, str]:
    """The footer's three paragraphs; the date is the commit's, in UTC."""
    date = datetime.datetime.fromtimestamp(commit_time, datetime.timezone.utc).date().isoformat()
    source = ("llmctl and this page come from ", link(repository.rsplit("/", 1)[1], repository), " on GitHub. ",
              link(f"Updated {day(date)}", f"{repository}/commit/{commit}"), ".")
    return source, f"Docs and measurements: {licenses[0]}. Code: {licenses[1]}.", _PRIVACY


# --- the model page (spec 7.12) ------------------------------------------------------------------------------------

BACK = "All models"
FIGURES_ARIA = "Key numbers"
GPU_CHIPS_LEGEND = "Your GPU"
RAM_CHIPS_LEGEND = "Your RAM"
RUN_HEADING = "Run it"
AGENT_SELECT = "Your agent"
ENGINES_HEADING = "Engines"
USE_ENGINE = "Use this engine"
WHY = "Why"
AGENTS_HEADING = "In your coding agent"
YOUR_AGENT = "Your agent"
GOOD_HEADING = "Good to know"
EVIDENCE_HEADING = "Evidence"
PAGER = {"label": "More models", "previous": "Previous", "next": "Next"}
HOME_TITLE = "Open coding models for your own GPU, llmctl"


def role_line(model: data.Model, labels: Labels) -> str:
    """M10's role line."""
    if model.rank is not None:
        line = f"Ranked {model.rank} of {labels.top} in our review" + (
            ", and our default." if model.id == labels.default[0] else ".")
    elif model.honorable:
        line = "An honorable mention in our review."
    else:
        line = f"Not in our Top {labels.top}."
    if model.chat_only:
        line += " Chat only: it can't call tools, so it can't drive a coding agent."
    return line


def _lower_about(text: str) -> str:
    return "about" + text[5:] if text[:5].lower() == "about" else text


def maker_line(model: data.Model) -> str | None:
    """M10's maker line, for a ranked model."""
    if model.maker is None:
        return None
    maker = model.maker
    return f"{maker.vendor}, {maker.license}, {_lower_about(maker.params)}."


def build_sentence(params: str | None, name: str) -> str | None:
    """M11's first sentence: from the Params column, else a total-active pattern in the name, else none."""
    moe = _PARAMS_MOE.match(params or "")
    if moe:
        total = ("about " if moe.group(1) else "") + moe.group(2)
        active = ("about " if moe.group(3) else "") + moe.group(4)
        return f"A mixture-of-experts model: {total}B parameters, {active}B of them active for each token."
    dense = _PARAMS_DENSE.match(params or "")
    if dense:
        return f"A dense model: all {dense.group(1)}B of its parameters work on every token."
    named = _NAME_MOE.search(name)
    if named:
        return (f"A mixture-of-experts model: {named.group(1)}B parameters, {named.group(2)}B of them active for each "
                "token.")
    return None


def thinking_sentence(facts: data.RecipeFacts) -> str:
    """M11's thinking sentence."""
    if facts.thinking == "toggle":
        return ("Thinking is on by default; you can turn it off." if facts.default == "on"
                else "Thinking is off by default; you can turn it on per session.")
    if facts.thinking == "effort":
        if "none" in facts.efforts:
            others = [effort for effort in facts.efforts if effort not in ("none", "medium")]
            return (f"It thinks at medium effort by default; you can switch to {joined_or(others)}, or turn thinking "
                    "off.")
        return f"It always thinks, at {joined_or(facts.efforts)} effort."
    if facts.thinking == "always":
        return "It always thinks before it answers, which costs time and tokens."
    if facts.thinking == "none":
        return "It answers without a thinking step."
    raise ValueError(f"{facts.file}: unknown thinking type {facts.thinking}")


def thinking_line(text: str) -> str:
    return f"Thinking: {text}"


def _summary_agents(results: Sequence[data.ClientResult], labels: Labels, chat_only: bool) -> str:
    passed = [labels.agents[result.client] for result in results if result.cls == "passed"]
    if len(passed) == len(results):
        verb = "Chats" if chat_only else "Works"
        return f"{verb} in all {count_word(len(results))} agents we tested" + (", without tools." if chat_only else ".")
    if passed:
        head = f"Chats in {joined(passed)}, without tools." if chat_only else f"Works in {joined(passed)}."
    else:
        head = "Didn't pass the agent test in any agent."
    return " ".join([head, *_class_sentences(results, labels)])


def summary(model: data.Model, cell: data.Cell, labels: Labels) -> str:
    """M11 from the model's default-engine setup `cell`: build, speed, GPU, agents and thinking; a setup without a
    measured speed has no speed sentence in M11, so it stops the build."""
    facts = cell.recipe
    if facts.decode is None:
        raise ValueError(f"{SOURCE}: {cell.model} on {cell.runtime} has no measured speed, which M11 has no sentence for")
    sentences = []
    built = build_sentence(None if model.maker is None else model.maker.params, model.name)
    if built:
        sentences.append(built)
    sentences.append(f"On our {labels.test_gpu} GB laptop GPU it writes {speed(facts.decode)} tokens a second and "
                     f"keeps {context_k(facts.context)} of context in view.")
    mib, share = fit.gpu_need(facts.gpu, facts.file)
    if mib is not None:
        tail = ""
        if facts.offload:
            tail = " and keeps part of itself in RAM" + (
                "" if facts.ram_peak is None else f", {gb(Fraction(str(facts.ram_peak)))} GB at peak")
        sentences.append(f"It uses {gb_mib(mib)} GB of GPU memory{tail}.")
    else:
        sentences.append(f"{labels.engines[cell.runtime]} takes {_pct(share)}% of a {labels.test_gpu} GB card up "
                         "front.")
    sentences.append(_summary_agents(cell.clients, labels, facts.chat_only))
    sentences.append(thinking_sentence(facts))
    return " ".join(sentences)


def figures_label(labels: Labels, runtime: str, default_engine: bool) -> str:
    return f"On {labels.engines[runtime]}" + (", the default engine" if default_engine else "")


def panel_heading(laptop: bool) -> str:
    return "Our test laptop" if laptop else "Your machine"


def panel_label(title: str, laptop: bool) -> str:
    """M12's verdict label; `title` is "{name} on {engine}"."""
    return f"{title}, on {'our test laptop' if laptop else 'your machine'}:"


def fit_note(labels: Labels, setup: fit.Setup, verdict: fit.Verdict, runtime: str) -> Rich | None:
    """M13's fit note above the terminal, for a setup too big for the RAM or measured on the test card only."""
    if verdict.key == "ram":
        return (bold("Too big for your RAM:"), f" it used {gb(setup.ram_peak)} GB at peak; you have about "
                f"{half_up(verdict.free, 0)} GB free (estimate). ", code("up"),
                " checks free RAM first, so it will stop before it starts anything.")
    if verdict.key == "only16":
        return (bold(f"Measured on {labels.test_gpu} GB cards only:"), f" {labels.engines[runtime]} takes a share of "
                f"the whole card, so we only know how it behaves on {labels.test_gpu} GB.")
    return None


def agent_pick(labels: Labels, pick: fit.Pick, agent: str, laptop: bool) -> Rich:
    """M13: the model to try in `agent` instead, a link to its page."""
    setup = pick.setup
    name = link(labels.models[setup.model], f"/models/{setup.model}/")
    on = "" if setup.default_engine else f" on {labels.engines[setup.runtime]}"
    label = _agent_label(labels, agent)
    where = "on our laptop" if laptop else "on your machine"
    if pick.step == "change":
        return _rich(f"In {label}, the closest fit {where} is ", name, f"{on}, with a change we haven't measured.")
    return _rich(f"In {label}, ", name, f"{on} passed the agent test and runs {where} as measured.")


def agent_note(result: data.ClientResult, labels: Labels, chat_only: bool, runtime: str,
               pick: Rich | None) -> Rich | None:
    """M13's agent note, when the chosen agent did not pass: its A3 sentence in bold, then the pick."""
    if result.cls == "passed":
        return None
    note: Rich = (bold(result_sentence(result, labels, chat_only, runtime)),)
    return note if pick is None else (*note, " ", *pick)


def engine_facts(labels: Labels, setup: fit.Setup, context: int) -> tuple[Rich, ...]:
    """M15's facts of one engine row, each leading number bold; `context` is the recipe's context."""
    speed_text = _speed_of(setup)
    if setup.gpu_mib is not None:
        gpu = (bold(f"{gb_mib(setup.gpu_mib)} GB"), " of GPU memory")
    else:
        gpu = (bold(f"{_pct(setup.gpu_share)}%"), f" of a {labels.test_gpu} GB card")
    return ((bold(speed_text),) if setup.decode is None else (bold(speed_text), " tok/s"),
            (bold(context_k(context)), " context"), gpu,
            ("asks for ", bold(f"{setup.memory_cap} GB"), " of free RAM"),
            (bold(f"{gb_bytes(setup.size_bytes)} GB"), " download"))


def use_engine_label(label: str) -> str:
    return f"{USE_ENGINE}: {label}"


def not_run_sentence(cell: data.Cell, runtime: data.Runtime, labels: Labels) -> str:
    """M15: a combination that does not run."""
    if cell.status == "does-not-fit":
        return f"Doesn't fit on {labels.test_gpu} GB with {runtime.label} (our arithmetic, not a test run)."
    if cell.status == "failed":
        return f"Didn't start on {engine_version(runtime.heading)} (tried {day(cell.measured)})."
    raise ValueError(f"{SOURCE}: {cell.model} on {cell.runtime}: the cell runs")


def why_clause(note: str) -> str:
    """M15's Why: the note's first clause, upper-cased and closed with a period, still Markdown for render.inline."""
    cut = _WHY_CUT.search(note)
    text = (note[:cut.start()] if cut else note).strip().rstrip(",;")
    if text[:1].isalpha():
        text = text[0].upper() + text[1:]
    return text + "."


def why_label(url: str) -> str:
    return "Measurement record" if "/research/" in url else "Engine notes"


def on_engine(label: str) -> str:
    return f"On {label}"


def agent_command(facts: data.RecipeFacts, agent: str) -> Rich | None:
    """M16: how to open `agent` on this setup, every name from RecipeFacts.launch; None for OpenCode without an
    entry."""
    launch = facts.launch
    client = fit.DEFAULT_AGENT if agent == fit.NONE else agent
    if client == "opencode":
        if launch.opencode is None:
            return None
        return (code("opencode"), ", then pick ", code(launch.opencode), " with /models")
    if client == "hermes":
        return (code(f"hermes -p {launch.hermes}"),)
    if client == "pi":
        return (code(f"pi --model {launch.pi}" + (" --no-tools" if facts.chat_only else "")),)
    if client == "dsh":
        return (code("dsh web"), f", then pick {launch.dsh} in its model picker")
    if client == "kimi":
        if facts.chat_only:
            return (code(f'kimi -p "your prompt" --agent-file {launch.kimi_file}'), ", after ",
                    code(f"{facts.up} --set-default"))
        return (code(f"kimi -m {launch.kimi}"),)
    raise ValueError(f"{SOURCE}: unknown agent {agent}")


def faster_sentence(model: data.Model, cell: data.Cell, labels: Labels, image_gb: Mapping[str, str]) -> str | None:
    """M17's first item: another engine at least 5% faster, with what it costs."""
    runs = [other for other in model.cells if other.status == "runs" and other.recipe.decode is not None]
    if cell.recipe.decode is None or not runs:
        return None
    best = max(runs, key=lambda other: Fraction(str(other.recipe.decode)))
    if best is cell:
        return None
    ratio = Fraction(str(best.recipe.decode)) / Fraction(str(cell.recipe.decode))
    pct = math.floor(100 * (ratio - 1) + Fraction(1, 2))
    if pct < 5:
        return None
    engine = labels.engines[best.runtime]
    return (f"{pct}% faster on {engine} ({speed(best.recipe.decode)} tok/s), but llmctl asks for "
            f"{best.recipe.memory_cap} GB of free RAM and {engine}'s engine is a {image_gb[best.runtime]} GB image.")


def good_to_know(model: data.Model, cell: data.Cell, labels: Labels, port: int, faster: str | None) -> tuple[Rich, ...]:
    """M17's items for one engine."""
    facts = cell.recipe
    items: list[Rich] = [] if faster is None else [(faster,)]
    items += [(f"Answers up to {facts.output:,} tokens long.",),
              ("Other tools can use it like OpenAI's API: ", code(f"http://127.0.0.1:{port}/v1"),
               " with the model name ", code(facts.served), "."),
              (f"Measured on {day(facts.date)}.",)]
    return tuple(items)


def evidence_links(facts: data.RecipeFacts, summary: str | None) -> tuple[Rich, ...]:
    """M18; `summary` is the latest run's summary.md, None when the setup has no history."""
    links: list[Rich] = [(link("Setup notes", facts.doc),), (link("Measurement record", facts.record),)]
    if summary is not None:
        links.append((link("Agent test summary", summary),))
    return tuple(links)


def home_description(labels: Labels) -> str:
    return (f"{labels.total} open coding models measured on a {labels.test_gpu} GB laptop GPU and tested in "
            f"{count_word(_agent_count(labels))} coding agents: what each needs, how fast it was, and the command that "
            "starts it.")


def model_title(name: str) -> str:
    return f"{name}, llmctl"


def model_description(name: str, labels: Labels) -> str:
    return (f"{name} on your own GPU: what it needs, how fast it ran on our {labels.test_gpu} GB laptop GPU, and the "
            "commands that start it.")
