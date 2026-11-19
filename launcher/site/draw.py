"""The local LLM page's memory drawings (spec 8.3) as descriptors: the GPU track of the maps, the strip of the model
list and the RAM bar. The static markup and page.js render the same descriptor, so no geometry rule lives in the
script; every geometry is a class, never a style attribute (spec 8.4).

Pure functions over fit.Setup and fit.Machine; Fractions throughout, rounded with integer arithmetic.
"""

import math
from collections.abc import Mapping
from fractions import Fraction

from launcher.site import fit

# The track's alphabet (v1's hero grammar): used blocks, tenths of a used block, free blocks, outline blocks, the
# desktop's hatched block, the card's end, and blocks past the end, whole or in tenths.
TRACK_CODES: dict[str, str] = {"#": "hb on", ".": "hb off", "o": "hb line", "/": "hb desk", "|": "hend",
                               "+": "hb over",
                               **{str(n): f"hb on p{n}" for n in range(1, 10)},
                               **{chr(ord("a") + n - 1): f"hb over g{n}" for n in range(1, 10)}}
# The strip's alphabet (v1's B grammar): blocks of the card with 0 to 10 tenths, the card's last block (the desktop's,
# hatched under its fill), blocks past the card, the tick before them and the empty slots that keep the scale; a
# leading "~" draws the fill as an outline and is not an element.
STRIP_CODES: dict[str, str] = {".": "b", "#": "b t10", "/": "b rs", "@": "b rs t10", "+": "b o t10", "|": "tick",
                               "_": "b x",
                               **{str(n): f"b t{n}" for n in range(1, 10)},
                               **{chr(ord("A") + n - 1): f"b rs t{n}" for n in range(1, 10)},
                               **{chr(ord("a") + n - 1): f"b o t{n}" for n in range(1, 10)}}
OUTLINE = "~"
DENSE = 20  # a strip with more slots than this draws its blocks closer
# The RAM bar's rows per class, (y, height, rx): `map` is the 28 px bar of the memory maps (v1's ram_svg), `mini` the
# 10 px bar of the diagram, scaled from it.
RAM_GEOMETRY: Mapping[str, Mapping[str, tuple[int, int, int]]] = {
    "map": {"base": (26, 2, 0), "tk": (20, 6, 0), "desk": (0, 26, 0), "peak": (0, 26, 2), "cap": (1, 24, 2),
            "end": (-4, 34, 0)},
    "mini": {"base": (9, 1, 0), "tk": (7, 2, 0), "desk": (0, 9, 0), "peak": (0, 9, 1), "cap": (1, 7, 1),
             "end": (-2, 14, 0)},
}
RAM_HEIGHT = {"map": 28, "mini": 10}
MARK_WIDTH = {"tk": "1", "end": "2"}  # the hairline and the end mark are drawn in pixels, not percentages


def fill(setup: fit.Setup, machine: fit.Machine, test_gpu: int) -> tuple[Fraction, bool]:
    """The GPU memory the setup fills on the machine in GB, and whether it is drawn as an outline: a measured need, or
    its share of the card, an outline off the test card."""
    if setup.gpu_mib is not None:
        return Fraction(setup.gpu_mib, 1024), False
    return setup.gpu_share * machine.gpu, machine.gpu != test_gpu


def tenths(value: Fraction) -> int:
    """`value` in tenths, rounded half up."""
    return math.floor(10 * value + Fraction(1, 2))


def _usable_blocks(machine: fit.Machine) -> int:
    return machine.gpu - fit.DESKTOP_GPU_MIB // 1024


def track(setup: fit.Setup, machine: fit.Machine, test_gpu: int) -> str:
    """The map's GPU track: the usable blocks, the desktop's block, then what does not fit past the card's end."""
    value, outline = fill(setup, machine, test_gpu)
    usable = _usable_blocks(machine)
    if outline:
        return "o" * usable + "/"
    need = tenths(value)
    out = []
    for block in range(usable):
        part = min(max(need - 10 * block, 0), 10)
        out.append("#" if part == 10 else "." if part == 0 else str(part))
    out.append("/")
    over = need - 10 * usable
    if over > 0:
        whole, part = divmod(over, 10)
        out.append("|" + "+" * whole + (chr(ord("a") + part - 1) if part else ""))
    return "".join(out)


def _blocks(setup: fit.Setup, machine: fit.Machine, test_gpu: int) -> int:
    return max(machine.gpu, -(-tenths(fill(setup, machine, test_gpu)[0]) // 10))


def scale(bench: fit.Bench, machine: fit.Machine) -> int:
    """The slots of every strip on the machine's GPU size: the most blocks any setup needs there."""
    return max(_blocks(setup, machine, bench.test_gpu) for setup in bench.setups)


def strip(setup: fit.Setup, machine: fit.Machine, test_gpu: int, scale: int) -> str:
    """The model list's strip: one block per GB of the card, more past it after a tick, then empty slots up to
    `scale`, so every strip of a list shares one block size."""
    value, outline = fill(setup, machine, test_gpu)
    need, blocks = tenths(value), _blocks(setup, machine, test_gpu)
    if blocks > scale:
        raise ValueError(f"launcher/site/draw.py: {setup.key} needs {blocks} blocks on {machine.gpu} GB, more than the "
                         f"scale of {scale}")
    out = [OUTLINE] if outline else []
    for block in range(blocks):
        part = min(max(need - 10 * block, 0), 10)
        if block < machine.gpu - 1:
            out.append("#" if part == 10 else "." if part == 0 else str(part))
        elif block == machine.gpu - 1:
            out.append("@" if part == 10 else "/" if part == 0 else chr(ord("A") + part - 1))
        else:
            if block == machine.gpu:
                out.append("|")
            out.append("+" if part == 10 else chr(ord("a") + part - 1))
    out.append("_" * (scale - blocks))
    return "".join(out)


def pct(value: Fraction) -> str:
    """A share as a percentage with two decimals, rounded half up with integer arithmetic (no format spec)."""
    n = math.floor(Fraction(value) * 10000 + Fraction(1, 2))
    return f"{n // 100}.{n % 100:02d}%"


def ram_bar(setup: fit.Setup, machine: fit.Machine) -> tuple[tuple[str, str, str | None], ...]:
    """The RAM bar as (class, x, width) rows, percentages of the largest of the RAM, the cap and the peak: the
    machine's RAM, a hairline every 4 GB, the desktop's share (hatched), the measured peak, what llmctl asks for above
    it, and the end of the RAM when the need passes it."""
    ram = machine.ram
    free = fit.free_ram(ram)
    peak = setup.ram_peak
    extent = max(ram, setup.memory_cap, peak or 0)
    x = lambda value: pct(Fraction(value) / extent)
    rows: list[tuple[str, str, str | None]] = [("base", x(0), x(ram))]
    rows += [("tk", x(mark), None) for mark in range(4, ram, 4)]
    rows.append(("desk", x(free), x(ram - free)))
    if peak is not None:
        rows.append(("peak", x(0), x(peak)))
    start = peak or 0
    if setup.memory_cap > start:
        rows.append(("cap", x(start), x(setup.memory_cap - start)))
    if extent > ram:
        rows.append(("end", x(ram), None))
    return tuple(rows)


def track_html(code: str) -> str:
    """The track's elements, one per character."""
    return "".join(f'<i class="{TRACK_CODES[char]}"></i>' for char in code)


def strip_html(code: str) -> str:
    """The strip's elements in their span; a block with a fill gets the fill as a child."""
    outline = code.startswith(OUTLINE)
    body = code[1:] if outline else code
    slots = len([char for char in body if char != "|"])
    klass = "strip" + (" k" if outline else "") + (" dense" if slots > DENSE else "")
    parts = []
    for char in body:
        cls = STRIP_CODES[char]
        inner = '<i class="f"></i>' if any(token.startswith("t") for token in cls.split()[1:]) else ""
        parts.append(f'<i class="{cls}">{inner}</i>')
    return f'<span class="{klass}">{"".join(parts)}</span>'


def ram_svg(bar: tuple[tuple[str, str, str | None], ...], size: str) -> str:
    """The RAM bar as an inline SVG of the size `map` or `mini`; x and width are attributes, which the page's policy
    does not govern."""
    geometry = RAM_GEOMETRY[size]
    rects = []
    for cls, x, width in bar:
        y, height, rx = geometry[cls]
        attrs = f'class="{cls}" x="{x}" y="{y}" width="{MARK_WIDTH[cls] if width is None else width}" height="{height}"'
        corner = f' rx="{rx}"' if rx else ""
        rects.append(f"<rect {attrs}{corner}></rect>")
    return (f'<svg class="ram-bar" width="100%" height="{RAM_HEIGHT[size]}" aria-hidden="true" focusable="false">'
            f'{"".join(rects)}</svg>')
