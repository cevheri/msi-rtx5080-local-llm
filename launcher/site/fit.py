"""The local LLM page's machine rules (spec 5): which setup runs on which machine, what the hero recommends, the change
a setup needs, the more-memory counts and the orders of the model list.

Pure functions over data.Matrix; no I/O; every comparison of measured numbers is a Fraction.
"""

import math
import re
from dataclasses import dataclass
from fractions import Fraction

from launcher.runtimes import ADAPTERS
from launcher.site import data

SOURCE = "launcher/site/fit.py"
GPUS = (8, 12, 16, 24, 32)  # the GPU chips; the last reads "32 GB or more"
RAMS = (16, 32, 64)  # the RAM chips; the last reads "64 GB or more"
# Estimate: the GPU memory the desktop and driver hold; our desktop held about 0.11 GiB (docs/model-selection.md,
# Per-model notes), so 1 GiB is cautious.
DESKTOP_GPU_MIB = 1024
# Estimate: README.md#machine, RAM row, "61 GiB usable, about 40 GiB free with the desktop running".
USABLE_RAM_GIB = 61
FREE_RAM_GIB = 40
NONE = "none"  # the "None yet" agent, which reads as OpenCode
DEFAULT_AGENT = "opencode"  # the agent this project is built around

_MIB = re.compile(r"^(\d+) MiB$")
_SHARE = re.compile(r"^(\d+(?:\.\d+)?)%$")
# The classes in which our default did not take the agent test, so R2's "failed the agent test" would be untrue (P25).
_UNTESTED_CLASSES = ("not-tested", "not-supported")


@dataclass(frozen=True, order=True)
class Machine:
    gpu: int
    ram: int


@dataclass(frozen=True)
class Setup:
    """One Runs cell as the rules read it (spec 5.2)."""
    model: str
    runtime: str
    rank: int | None
    default_engine: bool
    gpu_mib: int | None
    gpu_share: Fraction | None
    ram_peak: Fraction | None
    memory_cap: int
    size_bytes: int
    decode: Fraction | None
    chat_only: bool
    offload: bool
    passed: frozenset[str]

    @property
    def key(self) -> tuple[str, str]:
        return (self.model, self.runtime)


@dataclass(frozen=True)
class Bench:
    """The setups and the facts every rule needs: our default, the Top 10 size, the test GPU, the laptop machine and
    the agents."""
    setups: tuple[Setup, ...]
    default: tuple[str, str]
    top: int
    test_gpu: int
    home: Machine
    agents: tuple[str, ...]


@dataclass(frozen=True)
class Verdict:
    """R2 (spec 5.3): key ok, change, ram or only16; the GPU memory short in GiB; the RAM judgement."""
    key: str
    short: Fraction | None
    ram: str | None
    usable_mib: int
    free: Fraction


@dataclass(frozen=True)
class Pick:
    setup: Setup
    verdict: Verdict
    step: str


@dataclass(frozen=True)
class More:
    """How many more models run as measured with the next GPU chip (kind gpu) or RAM chip (kind ram) up."""
    kind: str
    size: int
    count: int


@dataclass(frozen=True)
class Change:
    """The change a setup needs on a machine (spec 5.7): the GPU plan and the RAM plan, None where nothing changes."""
    gpu: str | None
    ram: str | None


def machines() -> tuple[Machine, ...]:
    """The 15 machines of the chips, GPU-major."""
    return tuple(Machine(gpu, ram) for gpu in GPUS for ram in RAMS)


def usable_mib(gpu: int) -> int:
    """The GPU memory a model may use on a card of `gpu` GB, after the desktop's estimate."""
    return gpu * 1024 - DESKTOP_GPU_MIB


def free_ram(ram: int) -> Fraction:
    """The free RAM estimate for `ram` GB, scaled from our laptop's and rounded half up to one decimal."""
    return Fraction(math.floor(Fraction(ram * FREE_RAM_GIB, USABLE_RAM_GIB) * 10 + Fraction(1, 2)), 10)


def gpu_need(text: str, source: str) -> tuple[int | None, Fraction | None]:
    """A recipe's GPU need as (MiB, None) or (None, share of the card)."""
    if match := _MIB.match(text):
        return int(match.group(1)), None
    if match := _SHARE.match(text):
        return None, Fraction(match.group(1)) / 100
    raise data.SiteError(f"{source}: the GPU need {text!r} is neither <N> MiB nor <N>%")


def _fraction(value: float | None) -> Fraction | None:
    return None if value is None else Fraction(str(value))


def _sizes(values: tuple[int, ...]) -> str:
    return ", ".join(str(value) for value in values[:-1]) + f" and {values[-1]}"


def bench(matrix: data.Matrix) -> Bench:
    """The rules' view of the matrix; raises SiteError when the estimates or the test GPU disagree with
    README.md#machine and the recipes, or when our default is untested or unsupported in an agent (P25) or chat only."""
    laptop = matrix.laptop
    if (laptop.ram_gib, laptop.ram_free_gib) != (USABLE_RAM_GIB, FREE_RAM_GIB):
        raise data.SiteError(f"README.md#machine: the RAM row says {laptop.ram_gib} GiB usable, about "
                             f"{laptop.ram_free_gib} GiB free; {SOURCE} estimates free RAM from {FREE_RAM_GIB} of "
                             f"{USABLE_RAM_GIB} GiB: change one to match the other")
    if laptop.gpu_gb not in GPUS:
        raise data.SiteError(f"recipes: the test GPU has {laptop.gpu_gb} GB, not one of the GPU sizes {_sizes(GPUS)}")
    agents = tuple(client.id for client in matrix.clients)
    default = next(cell for model in matrix.models for cell in model.cells
                   if (model.id, cell.runtime) == matrix.default)
    for result in default.clients:
        if result.cls in _UNTESTED_CLASSES:
            raise data.SiteError(f"{SOURCE}: our default, {default.model} on {default.runtime}, is {result.cls} in "
                                 f"{result.client}, and the page only says that a default failed the agent test there")
    if default.recipe.chat_only:
        raise data.SiteError(f"{SOURCE}: our default, {default.model} on {default.runtime}, is chat only, and the page "
                             "only says that a default failed the agent test")
    setups = []
    for model in sorted(matrix.models, key=lambda m: m.id):
        for cell in sorted(model.cells, key=lambda c: _engine_place(c.runtime)):
            if cell.status != "runs":
                continue
            facts = cell.recipe
            mib, share = gpu_need(facts.gpu, facts.file)
            setups.append(Setup(model=model.id, runtime=cell.runtime, rank=model.rank,
                                default_engine=cell.runtime == facts.plain_runtime, gpu_mib=mib, gpu_share=share,
                                ram_peak=_fraction(facts.ram_peak), memory_cap=facts.memory_cap,
                                size_bytes=facts.size_bytes, decode=_fraction(facts.decode), chat_only=facts.chat_only,
                                offload=facts.offload,
                                passed=frozenset(r.client for r in cell.clients if r.cls == "passed")))
    home = Machine(laptop.gpu_gb, next((ram for ram in RAMS if ram >= laptop.ram_gib), RAMS[-1]))
    return Bench(setups=tuple(setups), default=matrix.default, top=matrix.top, test_gpu=laptop.gpu_gb, home=home,
                 agents=agents)


def verdict(setup: Setup, machine: Machine, test_gpu: int) -> Verdict:
    """R2: the verdict of one setup on one machine (spec 5.3)."""
    usable = usable_mib(machine.gpu)
    free = free_ram(machine.ram)
    if setup.gpu_mib is None and machine.gpu != test_gpu:
        return Verdict("only16", None, None, usable, free)
    gpu_fits = setup.gpu_mib is None or setup.gpu_mib <= usable
    short = None if gpu_fits else Fraction(setup.gpu_mib - usable, 1024)
    if setup.memory_cap <= free:
        ram = "ok"
    elif setup.ram_peak is not None and setup.ram_peak <= free:
        ram = "cap"
    elif setup.ram_peak is None:
        ram = "cap-unmeasured"
    else:
        return Verdict("ram", short, "too-big", usable, free)
    return Verdict("ok" if gpu_fits and ram == "ok" else "change", short, ram, usable, free)


def _engine_place(runtime: str) -> int:
    return list(ADAPTERS).index(runtime)


def _decode(setup: Setup) -> Fraction:
    return Fraction(0) if setup.decode is None else setup.decode


def _need(setup: Setup) -> int:
    """The GPU need that orders unranked models; a share counts 0."""
    return setup.gpu_mib or 0


def _candidate_order(setup: Setup) -> tuple:
    """Spec 5.5: ranked first by rank, unranked by the larger GPU need, the default engine first, the faster first."""
    return (setup.rank is None, setup.rank or 0, -_need(setup) if setup.rank is None else 0, not setup.default_engine,
            -_decode(setup), setup.model, _engine_place(setup.runtime))


def candidates(bench: Bench, agent: str) -> list[Setup]:
    """The setups that may be recommended for `agent`: not chat only and passed there (none reads as OpenCode)."""
    client = DEFAULT_AGENT if agent == NONE else agent
    return [setup for setup in bench.setups if not setup.chat_only and client in setup.passed]


def _default_setup(bench: Bench) -> Setup:
    return next(setup for setup in bench.setups if setup.key == bench.default)


def recommend(bench: Bench, machine: Machine, agent: str) -> Pick | None:
    """R3 (spec 5.5): our default when it runs as measured, else the best that does, else the closest change."""
    found = [(setup, verdict(setup, machine, bench.test_gpu)) for setup in candidates(bench, agent)]
    for setup, judged in found:
        if setup.key == bench.default and judged.key == "ok":
            return Pick(setup, judged, "default")
    measured = [item for item in found if item[1].key == "ok"]
    if measured:
        setup, judged = min(measured, key=lambda item: _candidate_order(item[0]))
        return Pick(setup, judged, "unranked" if setup.rank is None else "ranked")
    change = [item for item in found if item[1].key == "change"]
    for group in ([item for item in change if item[1].short is None and item[1].ram == "cap"],
                  [item for item in change if item[1].short is None and item[1].ram == "cap-unmeasured"],
                  [item for item in change if item[1].short is not None]):
        if group:
            setup, judged = min(group, key=lambda item: _candidate_order(item[0]))
            return Pick(setup, judged, "change")
    return None


def hero_pick(bench: Bench, machine: Machine, agent: str) -> Pick:
    """The hero's pick: R3, with step default-failed when our default runs as measured here but failed the agent."""
    pick = recommend(bench, machine, agent)
    if pick is None:
        raise data.SiteError(f"{SOURCE}: no setup to recommend for {machine.gpu} GB of GPU memory, {machine.ram} GB of "
                             f"RAM and {agent}")
    if pick.step in ("ranked", "unranked") and default_verdict(bench, machine).key == "ok" and \
            bench.default not in {setup.key for setup in candidates(bench, agent)}:
        return Pick(pick.setup, pick.verdict, "default-failed")
    return pick


def default_verdict(bench: Bench, machine: Machine) -> Verdict:
    """Our default's verdict on `machine`, for the switch (spec 5.8)."""
    return verdict(_default_setup(bench), machine, bench.test_gpu)


def _model_setups(bench: Bench, model: str) -> list[Setup]:
    return [setup for setup in bench.setups if setup.model == model]


def row_setup(bench: Bench, model: str, machine: Machine) -> Setup:
    """A model's row (spec 5.4): its default engine when that runs as measured, else its fastest other setup that does,
    else its default engine."""
    setups = _model_setups(bench, model)
    default = next(setup for setup in setups if setup.default_engine)
    if verdict(default, machine, bench.test_gpu).key == "ok":
        return default
    others = [setup for setup in setups
              if setup is not default and verdict(setup, machine, bench.test_gpu).key == "ok"]
    if not others:
        return default
    return min(others, key=lambda setup: (-_decode(setup), _engine_place(setup.runtime)))


def runs_as_measured(bench: Bench, machine: Machine) -> frozenset[tuple[str, str]]:
    """The (model, runtime) pairs that run as measured on `machine`."""
    return frozenset(setup.key for setup in bench.setups if verdict(setup, machine, bench.test_gpu).key == "ok")


def model_order(bench: Bench) -> tuple[str, ...]:
    """Rank order (spec 5.9): ranked by rank, then unranked by the larger GPU need of the default engine, then ID."""
    defaults = [setup for setup in bench.setups if setup.default_engine]
    return tuple(setup.model for setup in sorted(
        defaults, key=lambda s: (s.rank is None, s.rank or 0, -_need(s) if s.rank is None else 0, s.model)))


def models_as_measured(bench: Bench, machine: Machine) -> tuple[str, ...]:
    """The models with at least one setup that runs as measured on `machine`, in rank order."""
    ok = {model for model, _ in runs_as_measured(bench, machine)}
    return tuple(model for model in model_order(bench) if model in ok)


def more_memory(bench: Bench, machine: Machine) -> tuple[More, ...]:
    """How many more models run as measured with the next GPU chip up, then the next RAM chip up (spec 5.4)."""
    here = set(models_as_measured(bench, machine))
    out = []
    for kind, there in (("gpu", [Machine(gpu, machine.ram) for gpu in GPUS if gpu > machine.gpu][:1]),
                        ("ram", [Machine(machine.gpu, ram) for ram in RAMS if ram > machine.ram][:1])):
        for other in there:
            count = len(set(models_as_measured(bench, other)) - here)
            if count:
                out.append(More(kind, other.gpu if kind == "gpu" else other.ram, count))
    return tuple(out)


def change(setup: Setup, verdict: Verdict) -> Change | None:
    """The change plan of fix 5 (spec 5.7); None unless the verdict is change. A smaller context is never the plan
    when the model files alone are larger than the usable card."""
    if verdict.key != "change":
        return None
    if verdict.short is None:
        gpu = None
    elif setup.offload:
        gpu = "experts" if setup.runtime == "llamacpp" else "resident"
    elif Fraction(setup.size_bytes, 2 ** 30) > Fraction(verdict.usable_mib, 1024):
        gpu = "layers" if setup.runtime == "llamacpp" else "impossible"
    else:
        gpu = "context"
    return Change(gpu, None if verdict.ram == "ok" else verdict.ram)


def sort_order(bench: Bench, machine: Machine, key: str) -> tuple[str, ...]:
    """The model list's order (spec 5.9) by rank, speed or size; speed and size read each model's row setup."""
    if key == "rank":
        return model_order(bench)
    rows = {model: row_setup(bench, model, machine) for model in model_order(bench)}
    if key == "speed":
        return tuple(sorted(rows, key=lambda model: (rows[model].rank is None, -_decode(rows[model]), model)))
    if key == "size":
        return tuple(sorted(rows, key=lambda model: (rows[model].rank is None, rows[model].size_bytes, model)))
    raise ValueError(f"{SOURCE}: unknown sort {key!r}; expected rank, speed or size")
