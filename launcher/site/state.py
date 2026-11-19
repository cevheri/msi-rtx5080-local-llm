"""The state each page's script reads (spec 9.2): every text, drawing and rule outcome of every machine and agent
state, precomputed from fit, words and draw, so page.js only looks states up.

Texts go into one string table `S` that keeps each value once (a rich value keyed by its JSON, its links resolved
for the page); every other key refers to it by index, -1 for no text. Drawings go into the `track`, `strip` and
`bar` tables the same way.
"""

import json
from collections.abc import Sequence

from launcher.site import data, draw, fit, render, words

VERDICT_KEYS = ("ok", "change", "ram", "only16")  # the order of every verdict index


class _Pack:
    """The tables of one page: the strings and the three drawings, each value kept once."""

    def __init__(self, links: render.Links, here: str):
        self.links = links
        self.here = here
        self.tables: dict[str, list] = {"S": [], "track": [], "strip": [], "bar": []}
        self._seen: dict[str, dict[str, int]] = {name: {} for name in self.tables}

    def _add(self, table: str, value) -> int:
        key = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        seen = self._seen[table]
        if key not in seen:
            seen[key] = len(self.tables[table])
            self.tables[table].append(value)
        return seen[key]

    def text(self, value: str | words.Rich | None) -> int:
        """The index of a plain or rich text; a rich value without markup is kept as a plain string."""
        if value is None or value == "" or value == ():
            return -1
        if isinstance(value, str):
            return self._add("S", value)
        parts: list = []
        for part in value:
            if isinstance(part, str):
                parts.append(part)
            elif len(part) == 2 and part[0] in ("b", "code"):
                parts.append([part[0], part[1]])
            elif len(part) == 3 and part[0] == "a":
                parts.append(["a", part[1], render._href(part[2], self.links, self.here)])
            else:
                raise ValueError(f"launcher/site/state.py: not a rich text part: {part!r}")
        if all(isinstance(part, str) for part in parts):
            return self._add("S", "".join(parts))
        return self._add("S", parts)

    def texts(self, values: Sequence) -> list[int]:
        return [self.text(value) for value in values]

    def track(self, code: str) -> int:
        return self._add("track", code)

    def strip(self, code: str) -> int:
        return self._add("strip", code)

    def bar(self, bar: tuple[tuple[str, str, str | None], ...]) -> int:
        return self._add("bar", [list(row) for row in bar])


def _common(pack: _Pack, bench: fit.Bench, labels: words.Labels, links: render.Links) -> dict:
    """The keys every page shares (spec 9.2), the tables last."""
    machines = fit.machines()
    ui = {**words.UI, "copyLabel": words.copy_label(""), "untouched": words.STATE_UNTOUCHED,
          "badMachine": words.STATE_BAD_MACHINE, "badAgent": words.STATE_BAD_AGENT}
    return {"machines": [[m.gpu, m.ram] for m in machines], "home": machines.index(bench.home),
            "agents": [*bench.agents, fit.NONE], "defaultAgent": fit.DEFAULT_AGENT,
            "ui": {key: pack.text(value) for key, value in ui.items()},
            "vshort": pack.texts([words.verdict_word(key, labels) for key in VERDICT_KEYS]),
            "head": pack.texts(words.head_lines(links.build.repository))}


def _cell(matrix: data.Matrix, key: tuple[str, str]) -> data.Cell:
    return next(cell for model in matrix.models if model.id == key[0] for cell in model.cells if cell.runtime == key[1])


def _runtime(matrix: data.Matrix, runtime: str) -> data.Runtime:
    return next(item for item in matrix.runtimes if item.id == runtime)


def _terminal(pack: _Pack, matrix: data.Matrix, labels: words.Labels, setup: fit.Setup, agents: Sequence[str]) -> dict:
    """A setup's terminal parts and notes, the agent lines and notes per agent (spec 7.6)."""
    facts = _cell(matrix, setup.key).recipe
    runtime = _runtime(matrix, setup.runtime)
    engine = labels.engines[setup.runtime]
    return {"hd": pack.text(words.terminal_title(labels.models[setup.model], engine)),
            "pull": pack.texts(words.pull_lines(facts, engine, runtime.image_gb)),
            "up": pack.texts(words.up_lines(facts)),
            "agent": [pack.texts(words.agent_lines(facts, labels, setup.runtime, agent)) for agent in agents],
            "tail": pack.texts(words.tail_lines(facts)),
            "notes": [pack.texts(words.terminal_notes(facts, labels, agent)) for agent in agents]}


def _over(pack: _Pack, setup: fit.Setup, machine: fit.Machine, bench: fit.Bench, labels: words.Labels) -> list[int]:
    """The override lines between pull and up on the machine (T1 to T9)."""
    verdict = fit.verdict(setup, machine, bench.test_gpu)
    return pack.texts(words.override_lines(setup, verdict, fit.change(setup, verdict), labels.first_engine))


def _facts(pack: _Pack, matrix: data.Matrix, setup: fit.Setup) -> dict:
    facts = _cell(matrix, setup.key).recipe
    unit = None if facts.decode is None else words.SPEED_UNIT          # "not measured" goes without one (P17)
    return {"sp": pack.text(words.speed(facts.decode)), "spu": pack.text(unit),
            "cx": pack.text(words.context_k(facts.context)), "dl": pack.text(words.gb_bytes(facts.size_bytes))}


def home_state(matrix: data.Matrix, bench: fit.Bench, labels: words.Labels, links: render.Links) -> dict:
    """The home's state (spec 9.2): the hero of every machine and agent and of the laptop in its wording, the rows,
    the orders, the heads and every setup's cells, strips and, for a setup some state picks, its commands."""
    pack = _Pack(links, render.HOME)
    out = _common(pack, bench, labels, links)
    machines, agents = fit.machines(), out["agents"]
    index = {setup.key: n for n, setup in enumerate(bench.setups)}
    ids = [model.id for model in matrix.models]

    def hero(machine: fit.Machine, agent: str, laptop: bool) -> list:
        pick = fit.hero_pick(bench, machine, agent)
        setup, verdict = pick.setup, pick.verdict
        names = [labels.models[model] for model in fit.models_as_measured(bench, machine)]
        reason = words.reason(labels, pick, fit.default_verdict(bench, machine), fit.change(setup, verdict), names,
                              fit.more_memory(bench, machine), agent)
        text = words.map_text(labels, setup, verdict, machine, laptop, False)
        cell = _cell(matrix, setup.key)
        agents_line = words.agents_line(cell.clients, labels, setup.chat_only, setup.runtime, agent)
        return [pack.text(words.verdict_line(laptop, len(names), labels.total)), index[setup.key], pack.text(reason),
                pack.text(text.gpu_left), pack.text(text.gpu_right), int(text.gpu_short), pack.text(text.ram_left),
                pack.text(text.ram_right), int(text.ram_short), pack.text(text.alt), pack.text(agents_line),
                pack.track(draw.track(setup, machine, bench.test_gpu)), pack.bar(draw.ram_bar(setup, machine))]

    out["hero"] = [[hero(machine, agent, False) for agent in agents] for machine in machines]
    out["heroLap"] = [hero(bench.home, agent, True) for agent in agents]
    out["prefix"] = pack.texts(words.VERDICT_PREFIX)
    out["ids"] = ids
    out["rows"] = []
    for model in ids:
        row = []
        for machine in machines:
            setup = fit.row_setup(bench, model, machine)
            row.append([index[setup.key], VERDICT_KEYS.index(fit.verdict(setup, machine, bench.test_gpu).key)])
        out["rows"].append(row)
    out["order"] = {sort: [[ids.index(model) for model in fit.sort_order(bench, machine, sort)] for machine in machines]
                    for sort in ("speed", "size")}
    out["heads"] = {"strip": pack.texts([words.strip_head(machine, labels, False) for machine in machines]),
                    "stripLap": pack.text(words.strip_head(bench.home, labels, True)),
                    "ag": pack.texts([words.agents_head(agent, labels) for agent in agents]),
                    "vd": pack.texts([words.machine_head(True), words.machine_head(False)])}
    picks = {out["hero"][mi][ai][1] for mi in range(len(machines)) for ai in range(len(agents))}
    picks |= {record[1] for record in out["heroLap"]}
    scales = [draw.scale(bench, machine) for machine in machines]
    setups = []
    for n, setup in enumerate(bench.setups):
        cell = _cell(matrix, setup.key)
        item = {"m": ids.index(setup.model), "rt": setup.runtime, "on": pack.text(words.row_engine(labels, setup)),
                **_facts(pack, matrix, setup), "galt": pack.text(words.uses(labels, setup)),
                "ag": pack.texts([words.agents_cell(cell.clients, labels, setup.chat_only, agent) for agent in agents]),
                "cag": pack.texts([words.compare_agents(cell.clients, labels, setup.chat_only, setup.runtime, agent)
                                   for agent in agents]),
                "strip": [pack.strip(draw.strip(setup, machine, bench.test_gpu, scale))
                          for machine, scale in zip(machines, scales)]}
        if n in picks:
            facts = cell.recipe
            runtime = _runtime(matrix, setup.runtime)
            question = words.moe_question(setup)
            item.update({
                "tt": pack.text(words.pick_heading(words.title(labels, setup.model, setup.runtime,
                                                               setup.default_engine))),
                "mq": pack.text(question),
                "ml": -1 if question is None else pack.text(words.moe_answer(setup.runtime)),
                "disk": pack.text(words.disk_check(facts.size_bytes, runtime.image_gb,
                                                   labels.engines[setup.runtime])[0]),
                "ep": pack.text(words.endpoint(runtime.port, facts.served)),
                **_terminal(pack, matrix, labels, setup, agents),
                "over": [_over(pack, setup, machine, bench, labels) for machine in machines]})
        setups.append(item)
    out["setups"] = setups
    out.update(pack.tables)
    return out


def model_state(matrix: data.Matrix, bench: fit.Bench, labels: words.Labels, model: data.Model,
                links: render.Links) -> dict:
    """A model page's state (spec 9.2): per running engine its panel, map, notes and commands on every machine and
    agent, the laptop's wording, the setup each machine shows first and the agent picks."""
    pack = _Pack(links, render._model_path(model.id))
    out = _common(pack, bench, labels, links)
    machines, agents = fit.machines(), out["agents"]
    running = [setup for cell in model.cells if cell.status == "runs"
               for setup in bench.setups if setup.key == (model.id, cell.runtime)]
    scales = [draw.scale(bench, machine) for machine in machines]
    setups = []
    for setup in running:
        cell = _cell(matrix, setup.key)
        title = words.title(labels, model.id, setup.runtime, False)      # M12 names the engine, default or not
        records, alts = [], []
        for machine, scale in zip(machines, scales):
            verdict = fit.verdict(setup, machine, bench.test_gpu)
            text = words.map_text(labels, setup, verdict, machine, False, True)
            records.append([VERDICT_KEYS.index(verdict.key), pack.track(draw.track(setup, machine, bench.test_gpu)),
                            pack.bar(draw.ram_bar(setup, machine)), pack.text(text.gpu_left),
                            pack.text(text.gpu_right), int(text.gpu_short), pack.text(text.ram_right),
                            int(text.ram_short), pack.text(words.fit_note(labels, setup, verdict, setup.runtime)),
                            _over(pack, setup, machine, bench, labels),
                            pack.strip(draw.strip(setup, machine, bench.test_gpu, scale))])
            alts.append(pack.text(text.alt))
        home = bench.home
        lap = words.map_text(labels, setup, fit.verdict(setup, home, bench.test_gpu), home, True, True)
        results = [next(item for item in cell.clients if item.client == (fit.DEFAULT_AGENT if agent == fit.NONE
                                                                           else agent)) for agent in agents]
        setups.append({
            "rt": setup.runtime, "en": pack.text(words.figures_label(labels, setup.runtime, setup.default_engine)),
            "label": pack.texts([words.panel_label(title, True), words.panel_label(title, False)]),
            **_facts(pack, matrix, setup), **_terminal(pack, matrix, labels, setup, agents),
            "anote": [pack.text(words.agent_note(result, labels, setup.chat_only, setup.runtime, None))
                      for result in results],
            "moe": pack.text(words.moe_paragraph(setup)), "rl": pack.text(lap.ram_left), "fit": records,
            "fitLap": pack.texts([lap.gpu_left, lap.gpu_right, lap.ram_right, lap.alt]), "alt": alts})
    out["setups"] = setups
    keys = [setup.key for setup in running]
    out["rows"] = [keys.index(fit.row_setup(bench, model.id, machine).key) for machine in machines]

    def pick(machine: fit.Machine, agent: str, laptop: bool) -> int:
        found = fit.recommend(bench, machine, agent)
        return -1 if found is None else pack.text(words.agent_pick(labels, found, agent, laptop))

    out["pick"] = [[pick(machine, agent, False) for agent in agents] for machine in machines]
    out["pickLap"] = [pick(bench.home, agent, True) for agent in agents]
    out["verdict"] = pack.texts([words.verdict_sentence(key, labels) for key in VERDICT_KEYS])
    out["panel"] = pack.texts([words.panel_heading(True), words.panel_heading(False)])
    out.update(pack.tables)
    return out


def dumps(value: dict) -> str:
    """The state as JSON for its inline script (spec 8.1): every < escaped, so no text can end the element."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
