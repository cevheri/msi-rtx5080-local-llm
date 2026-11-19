"""The per-page state the script reads (spec 9.2) and the script's own rules (spec 9.1): every state follows fit's
rules and words' copy, the JSON cannot end its script element, and page.js holds no English, no rule and no unsafe
call."""

import ast
import json
import re
import tempfile
import unittest
from pathlib import Path

from launcher.site import data, draw, fit, render, state, words
from launcher.tests.fakes import REPO
from launcher.tests.site_fixture import SiteFixture, cell_of, model_of, parse
from launcher.tests.test_site_fit import with_class

PAGE_JS = REPO / "launcher/site/page.js"
BUILD = data.Build("https://github.com/o/r", "a" * 40, 0)
HERO_TEXTS = (0, 2, 3, 4, 6, 7, 9, 10)  # the text positions of a hero record (spec 9.2)


def text_of(value) -> str:
    """A plain string, or the joined texts of a rich value's parts, the state's lists and words' tuples alike."""
    if isinstance(value, str):
        return value
    return "".join(part if isinstance(part, str) else part[1] for part in value)


_COMPOUND = re.compile(r"^([a-z0-9]*)((?:[.#][\w-]+|\[[\w-]+\]|:not\([^()]+\))*)$")


def _matches(node, compound: str) -> bool:
    """Whether an element matches one compound selector of the kinds page.js uses: tag, .class, #id, [attr], :not()."""
    found = _COMPOUND.match(compound)
    if found is None:
        raise ValueError(f"not a selector this test reads: {compound!r}")
    tag, rest = found.groups()
    classes = (node.attrs.get("class") or "").split()
    if tag and node.tag != tag:
        return False
    for part in re.findall(r"[.#][\w-]+|\[[\w-]+\]|:not\([^()]+\)", rest):
        if part.startswith(":not("):
            ok = not _matches(node, part[5:-1])
        elif part.startswith("."):
            ok = part[1:] in classes
        elif part.startswith("#"):
            ok = node.attrs.get("id") == part[1:]
        else:
            ok = part[1:-1] in node.attrs
        if not ok:
            return False
    return True


def select(root, selector: str) -> list:
    """The descendants of root that match a selector of compounds joined by spaces, as querySelectorAll finds them."""
    first, *rest = selector.split()
    found = [node for node in root.iter() if _matches(node, first)]
    if not rest:
        return found
    out = []
    for node in found:
        out += [match for match in select(node, " ".join(rest)) if match not in out]
    return out


class SiteStateTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.fixture = SiteFixture(Path(self._tmp.name))

    def pages(self, matrix=None) -> dict[str, str]:
        return render.site(matrix or self.fixture.load(), BUILD, self.fixture.root)

    def test_home_state_follows_the_rules_in_every_state(self):
        matrix = self.fixture.load()
        b, labels = fit.bench(matrix), words.labels(matrix)
        st = state.home_state(matrix, b, labels, render.Links(self.fixture.root, BUILD))
        S = st["S"]
        machines = fit.machines()
        self.assertEqual(st["machines"], [[m.gpu, m.ram] for m in machines])
        self.assertEqual(st["agents"], ["opencode", "hermes", "pi", "dsh", "kimi", "none"])
        for mi, m in enumerate(machines):
            for ai, agent in enumerate(st["agents"]):
                with self.subTest(machine=m, agent=agent):
                    record = st["hero"][mi][ai]
                    pick = fit.hero_pick(b, m, agent)
                    setup = st["setups"][record[1]]
                    self.assertEqual((matrix.models[setup["m"]].id, setup["rt"]), pick.setup.key)
                    names = [labels.models[i] for i in fit.models_as_measured(b, m)]
                    self.assertEqual(S[record[0]], words.verdict_line(False, len(names), labels.total))
                    self.assertEqual(S[record[2]], words.reason(labels, pick, fit.default_verdict(b, m),
                                                                fit.change(pick.setup, pick.verdict), names,
                                                                fit.more_memory(b, m), agent))
        lap = st["heroLap"][0]
        self.assertEqual(S[lap[0]], "On our test laptop: all 6 models run as we measured them.")
        laptop_texts = [text_of(S[record[i]]) for record in st["heroLap"] for i in HERO_TEXTS if record[i] >= 0]
        self.assertFalse([t for t in laptop_texts if re.search(r"\byour\b", t, re.I)])

    def test_home_rows_orders_heads_and_cells_follow_the_rules_in_every_state(self):
        matrix = self.fixture.load()
        b, labels = fit.bench(matrix), words.labels(matrix)
        st = state.home_state(matrix, b, labels, render.Links(self.fixture.root, BUILD))
        S, ids, keys = st["S"], [m.id for m in matrix.models], state.VERDICT_KEYS
        setup_of = lambda i: (ids[st["setups"][i]["m"]], st["setups"][i]["rt"])
        index = {setup_of(i): i for i in range(len(st["setups"]))}
        for mi, m in enumerate(fit.machines()):
            with self.subTest(machine=m):
                for k, model in enumerate(ids):
                    row = fit.row_setup(b, model, m)
                    self.assertEqual(st["rows"][k][mi], [index[row.key], keys.index(fit.verdict(row, m, b.test_gpu).key)])
                for sort in ("speed", "size"):
                    self.assertEqual(st["order"][sort][mi], [ids.index(x) for x in fit.sort_order(b, m, sort)])
                self.assertEqual(text_of(S[st["heads"]["strip"][mi]]), text_of(words.strip_head(m, labels, False)))
                scale = draw.scale(b, m)
                for x in b.setups:
                    s = st["setups"][index[x.key]]
                    self.assertEqual(st["strip"][s["strip"][mi]], draw.strip(x, m, b.test_gpu, scale))
                for ai, agent in enumerate(st["agents"]):
                    record, pick = st["hero"][mi][ai], fit.hero_pick(b, m, agent)
                    setup, vi = st["rows"][ids.index(pick.setup.model)][mi]   # L6: the pick agrees with its row
                    if setup == record[1]:
                        self.assertEqual(keys[vi], pick.verdict.key)
                    if pick.verdict.key == "ok":
                        self.assertEqual(keys[vi], "ok")
        cell = lambda model, rt: next(c for c in model_of(matrix, model).cells if c.runtime == rt)
        for ai, agent in enumerate(st["agents"]):
            with self.subTest(agent=agent):
                self.assertEqual(text_of(S[st["heads"]["ag"][ai]]), text_of(words.agents_head(agent, labels)))
                for x in b.setups:
                    c, s = cell(*x.key), st["setups"][index[x.key]]
                    self.assertEqual(text_of(S[s["ag"][ai]]), text_of(words.agents_cell(c.clients, labels, x.chat_only, agent)))
                    self.assertEqual(text_of(S[s["cag"][ai]]),
                                     text_of(words.compare_agents(c.clients, labels, x.chat_only, x.runtime, agent)))
        self.assertEqual([text_of(S[i]) for i in st["heads"]["vd"]], [words.machine_head(True), words.machine_head(False)])
        self.assertEqual(text_of(S[st["heads"]["stripLap"]]), text_of(words.strip_head(b.home, labels, True)))

    def test_model_state_covers_every_running_engine(self):
        matrix = self.fixture.load()
        b, labels = fit.bench(matrix), words.labels(matrix)
        model = next(m for m in matrix.models if m.id == "qwen3.6-35b-a3b")
        st = state.model_state(matrix, b, labels, model, render.Links(self.fixture.root, BUILD))
        self.assertEqual([s["rt"] for s in st["setups"]], ["llamacpp", "sglang"])
        self.assertEqual([len(s["fit"]) for s in st["setups"]], [15, 15])
        self.assertEqual(st["rows"][st["home"]], 0)
        self.assertEqual(st["setups"][0]["anote"][st["agents"].index("hermes")], -1)
        self.assertEqual([st["S"][i] for i in st["panel"]], ["Our test laptop", "Your machine"])
        pick = st["S"][st["pick"][st["machines"].index([12, 32])][0]]
        self.assertIn(["a", "Qwen3.6-35B-A3B", "../qwen3.6-35b-a3b/"], pick)

    def test_model_state_follows_the_rules_in_every_state(self):
        matrix = self.fixture.load()
        b, labels = fit.bench(matrix), words.labels(matrix)
        model = next(m for m in matrix.models if m.id == "qwen3.6-35b-a3b")
        st = state.model_state(matrix, b, labels, model, render.Links(self.fixture.root, BUILD))
        S, keys = st["S"], state.VERDICT_KEYS
        scale = {m: draw.scale(b, m) for m in fit.machines()}
        for n, s in enumerate(st["setups"]):
            setup = next(x for x in b.setups if x.key == (model.id, s["rt"]))
            for mi, m in enumerate(fit.machines()):
                with self.subTest(engine=s["rt"], machine=m):
                    verdict = fit.verdict(setup, m, b.test_gpu)
                    text = words.map_text(labels, setup, verdict, m, False, True)
                    record = s["fit"][mi]
                    self.assertEqual(keys[record[0]], verdict.key)
                    self.assertEqual(st["track"][record[1]], draw.track(setup, m, b.test_gpu))
                    self.assertEqual(st["bar"][record[2]], [list(r) for r in draw.ram_bar(setup, m)])
                    self.assertEqual(text_of(S[record[3]]), text_of(text.gpu_left))
                    self.assertEqual((S[record[4]], record[5]), (text.gpu_right, int(text.gpu_short)))
                    self.assertEqual((S[record[6]], record[7]), (text.ram_right, int(text.ram_short)))
                    self.assertEqual(st["strip"][record[10]], draw.strip(setup, m, b.test_gpu, scale[m]))
                    self.assertEqual(S[s["alt"][mi]], text.alt)
                    if mi == st["home"]:
                        lap = words.map_text(labels, setup, verdict, m, True, True)
                        self.assertEqual([text_of(S[i]) for i in s["fitLap"]],
                                         [text_of(lap.gpu_left), lap.gpu_right, lap.ram_right, lap.alt])
            self.assertEqual([text_of(S[i]) for i in s["label"]],
                             [words.panel_label(words.title(labels, model.id, s["rt"], False), laptop)
                              for laptop in (True, False)])
        self.assertEqual([S[i] for i in st["verdict"]], [words.verdict_sentence(k, labels) for k in keys])
        laptop = [text_of(S[i]) for s in st["setups"] for i in s["fitLap"]] + [text_of(S[i]) for i in st["pickLap"] if i >= 0]
        self.assertFalse([t for t in laptop if re.search(r"\byour\b", t, re.I)])

    def assert_terminal(self, st, s, setup, matrix, labels, here):
        """A setup's terminal parts, per agent where they depend on it, are words' for its own engine and agent."""
        pack = state._Pack(render.Links(self.fixture.root, BUILD), here)
        want = lambda value: state._Pack.text(pack, value)
        got = lambda i: None if i < 0 else st["S"][i]
        exp = lambda value: None if want(value) < 0 else pack.tables["S"][want(value)]
        facts = cell_of(matrix, *setup.key).recipe
        runtime = next(r for r in matrix.runtimes if r.id == setup.runtime)
        engine = labels.engines[setup.runtime]
        self.assertEqual(got(s["hd"]), exp(words.terminal_title(labels.models[setup.model], engine)))
        self.assertEqual([got(i) for i in s["pull"]], [exp(x) for x in words.pull_lines(facts, engine, runtime.image_gb)])
        self.assertEqual([got(i) for i in s["up"]], [exp(x) for x in words.up_lines(facts)])
        self.assertEqual([got(i) for i in s["tail"]], [exp(x) for x in words.tail_lines(facts)])
        for ai, agent in enumerate(st["agents"]):
            self.assertEqual([got(i) for i in s["agent"][ai]],
                             [exp(x) for x in words.agent_lines(facts, labels, setup.runtime, agent)])
            self.assertEqual([got(i) for i in s["notes"][ai]],
                             [exp(x) for x in words.terminal_notes(facts, labels, agent)])
        if "ep" in s:
            self.assertEqual(got(s["ep"]), exp(words.endpoint(runtime.port, facts.served)))
            self.assertEqual(got(s["disk"]), exp(words.disk_check(facts.size_bytes, runtime.image_gb, engine)[0]))
        return exp

    def test_home_commands_and_change_plans_follow_their_setup_machine_and_agent(self):
        # the pick's terminal parts and the override lines of fix 5 on every machine (spec 7.6, T1 to T9)
        # our default needs 50 GiB of RAM on llama.cpp, so the hero picks it on SGLang, whose port and title differ
        self.fixture.set_value("llamacpp", "qwen3.6-35b-a3b", "memory_gib", "50")
        matrix = self.fixture.load()
        b, labels = fit.bench(matrix), words.labels(matrix)
        st = state.home_state(matrix, b, labels, render.Links(self.fixture.root, BUILD))
        ids = [m.id for m in matrix.models]
        picked = {r[1] for row in st["hero"] for r in row} | {r[1] for r in st["heroLap"]}
        self.assertIn("sglang", {st["setups"][i]["rt"] for i in picked})
        overs = 0
        for i in sorted(picked):
            s = st["setups"][i]
            setup = next(x for x in b.setups if x.key == (ids[s["m"]], s["rt"]))
            with self.subTest(setup=setup.key):
                exp = self.assert_terminal(st, s, setup, matrix, labels, render.HOME)
                for mi, m in enumerate(fit.machines()):
                    verdict = fit.verdict(setup, m, b.test_gpu)
                    lines = words.override_lines(setup, verdict, fit.change(setup, verdict), labels.first_engine)
                    self.assertEqual([st["S"][j] for j in s["over"][mi]], [exp(x) for x in lines])
                    overs += len(lines)
        self.assertGreater(overs, 0)

    def test_model_commands_notes_and_picks_follow_their_setup_machine_and_agent(self):
        matrix = self.fixture.load()
        b, labels = fit.bench(matrix), words.labels(matrix)
        links = render.Links(self.fixture.root, BUILD)
        seen, overs, notes = set(), 0, 0
        for model in matrix.models:
            st = state.model_state(matrix, b, labels, model, links)
            here = render._model_path(model.id)
            for s in st["setups"]:
                setup = next(x for x in b.setups if x.key == (model.id, s["rt"]))
                seen.add(s["rt"])
                with self.subTest(setup=setup.key):
                    exp = self.assert_terminal(st, s, setup, matrix, labels, here)
                    for mi, m in enumerate(fit.machines()):
                        verdict = fit.verdict(setup, m, b.test_gpu)
                        lines = words.override_lines(setup, verdict, fit.change(setup, verdict), labels.first_engine)
                        self.assertEqual([st["S"][j] for j in s["fit"][mi][9]], [exp(x) for x in lines])
                        note = words.fit_note(labels, setup, verdict, setup.runtime)
                        self.assertEqual(None if s["fit"][mi][8] < 0 else st["S"][s["fit"][mi][8]], exp(note))
                        overs, notes = overs + len(lines), notes + (note is not None)
            for mi, m in enumerate(fit.machines()):
                for ai, agent in enumerate(st["agents"]):
                    with self.subTest(model=model.id, machine=m, agent=agent):
                        found = fit.recommend(b, m, agent)
                        expected = -1 if found is None else exp(words.agent_pick(labels, found, agent, False))
                        self.assertEqual(-1 if st["pick"][mi][ai] < 0 else st["S"][st["pick"][mi][ai]], expected)
            for ai, agent in enumerate(st["agents"]):
                found = fit.recommend(b, b.home, agent)
                expected = -1 if found is None else exp(words.agent_pick(labels, found, agent, True))
                self.assertEqual(-1 if st["pickLap"][ai] < 0 else st["S"][st["pickLap"][ai]], expected)
        self.assertEqual(seen, {"llamacpp", "vllm", "sglang"})
        self.assertGreater(overs, 0)
        self.assertGreater(notes, 0)

    def test_pages_embed_the_state_the_script_reads(self):
        matrix = self.fixture.load()
        pages = self.pages(matrix)
        b, labels = fit.bench(matrix), words.labels(matrix)
        links = render.Links(self.fixture.root, BUILD)
        found = re.search(r'<script type="application/json" id="state">(.*?)</script>', pages["index.html"], re.S)
        self.assertEqual(found.group(1), state.dumps(state.home_state(matrix, b, labels, links)))
        home = json.loads(found.group(1))
        cells = parse(pages["index.html"]).by_id("compare").find_all("a", "cell")
        for a in cells:                                    # a compare cell names its setup in the state
            s = home["setups"][int(a.attrs["data-s"])]
            self.assertEqual(a.attrs["href"], f"models/{home['ids'][s['m']]}/#engine-{s['rt']}")
        self.assertEqual(len(cells), len(home["setups"]))
        for model in matrix.models:
            page = pages[f"models/{model.id}/index.html"]
            found = re.search(r'<script type="application/json" id="state">(.*?)</script>', page, re.S)
            with self.subTest(model=model.id):
                self.assertEqual(found.group(1), state.dumps(state.model_state(matrix, b, labels, model, links)))

    def test_state_json_cannot_end_its_script(self):
        self.assertEqual(state.dumps({"S": ["</script><b>&"]}), '{"S":["\\u003c/script>\\u003cb>&"]}')
        self.fixture.set_value("llamacpp", "k2-horizon-7b", "name", '"K2 </script> Horizon Q8_0"')
        for path, page in self.pages().items():
            with self.subTest(page=path):
                self.assertEqual(page.count("</script>"), 3)   # head, state, page script

    def test_states_follow_the_rules_where_our_default_failed_and_a_row_moves(self):
        # Hermes failed on our default (R2, so the reason depends on the agent) and gpt-oss-20b needs 50 GiB of RAM on
        # llama.cpp, so its row moves to SGLang on the laptop: the reason, a model page's rows and its agent notes
        # must follow the chosen agent and machine, not OpenCode or the default engine
        self.fixture.set_value("llamacpp", "gpt-oss-20b", "memory_gib", "50")
        matrix = with_class(self.fixture.load(), "qwen3.6-35b-a3b", "llamacpp", "hermes", "failed")
        b, labels = fit.bench(matrix), words.labels(matrix)
        links = render.Links(self.fixture.root, BUILD)
        st = state.home_state(matrix, b, labels, links)
        machines = fit.machines()
        reasons = set()
        for mi, m in enumerate(machines):
            for ai, agent in enumerate(st["agents"]):
                with self.subTest(machine=m, agent=agent):
                    pick = fit.hero_pick(b, m, agent)
                    names = [labels.models[i] for i in fit.models_as_measured(b, m)]
                    reason = words.reason(labels, pick, fit.default_verdict(b, m), fit.change(pick.setup, pick.verdict),
                                          names, fit.more_memory(b, m), agent)
                    self.assertEqual(text_of(st["S"][st["hero"][mi][ai][2]]), text_of(reason))
                    reasons.add((mi, pick.step))
        self.assertIn((st["home"], "default-failed"), reasons)
        moved = 0
        for model in matrix.models:
            page = state.model_state(matrix, b, labels, model, links)
            keys = [(model.id, s["rt"]) for s in page["setups"]]
            with self.subTest(model=model.id):
                rows = [keys.index(fit.row_setup(b, model.id, m).key) for m in machines]
                self.assertEqual(page["rows"], rows)
                moved += rows[page["home"]] != 0
                for s in page["setups"]:
                    setup = next(x for x in b.setups if x.key == (model.id, s["rt"]))
                    cell = cell_of(matrix, model.id, s["rt"])
                    for ai, agent in enumerate(page["agents"]):
                        client = fit.DEFAULT_AGENT if agent == fit.NONE else agent
                        result = next(r for r in cell.clients if r.client == client)
                        note = words.agent_note(result, labels, setup.chat_only, setup.runtime, None)
                        self.assertEqual(-1 if s["anote"][ai] < 0 else text_of(page["S"][s["anote"][ai]]),
                                         -1 if not note else text_of(note))
        self.assertEqual(moved, 1)

    def test_a_setup_without_a_speed_shows_its_unit_only_when_measured(self):
        # spec H9 and P17: "not measured" goes without a unit, and the script switches to a setup with or without one
        self.fixture.set_value("sglang", "qwen3.6-35b-a3b", "decode_tok_s", None)
        matrix = self.fixture.load()
        b, labels = fit.bench(matrix), words.labels(matrix)
        links = render.Links(self.fixture.root, BUILD)
        model = model_of(matrix, "qwen3.6-35b-a3b")
        home, page = state.home_state(matrix, b, labels, links), state.model_state(matrix, b, labels, model, links)
        for st in (home, page):
            for s in st["setups"]:
                key = (st["ids"][s["m"]] if "m" in s else model.id, s["rt"])
                with self.subTest(setup=key, model_page="m" not in s):
                    decode = next(x for x in b.setups if x.key == key).decode
                    measured = decode is not None
                    self.assertEqual(st["S"][s["sp"]], words.speed(None if decode is None else float(decode)))
                    self.assertEqual(-1 if s["spu"] < 0 else st["S"][s["spu"]], words.SPEED_UNIT if measured else -1)
        self.assertEqual([s["spu"] < 0 for s in page["setups"]], [False, True])
        js = (REPO / "launcher/site/page.js").read_text(encoding="utf-8")
        self.assertEqual(len(re.findall(r'\.spu"\)\.hidden = \w+\.spu < 0;', js)), 3)   # the hero, a row, a model page
        pages = self.pages(matrix)
        for path, html in pages.items():
            tree = parse(html)
            boxes = [n for n in tree.iter() if any("f-sp" in (c.attrs.get("class") or "").split()
                                                   for c in n.children if not isinstance(c, str))]
            boxes += tree.find_all("span", "sp")
            with self.subTest(page=path):
                self.assertTrue(boxes)
                for box in boxes:                         # the unit is always there for the script to show or hide
                    units = [c for c in box.children if not isinstance(c, str) and "spu" in c.attrs.get("class", "")]
                    self.assertEqual(len(units), 1, box.text())
                    self.assertEqual(units[0].text().strip(), words.SPEED_UNIT)
                    self.assertEqual("hidden" in units[0].attrs, words.speed(None) in box.text())

    def test_every_selector_the_script_reads_matches_the_markup(self):
        # a renamed class would throw in page.js and stop the script at load; a selector read on each element of a
        # `for (const row of document.querySelectorAll(...))` loop must match inside every such element
        js = (REPO / "launcher/site/page.js").read_text(encoding="utf-8")
        pages = self.pages()
        bodies = {name: re.search(rf"\n  function {name}\(.*?\n  }}\n", js, re.S).group(0)
                  for name in ("renderHome", "renderModel", "map", "terminal")}
        for name, paths in (("renderHome", ["index.html"]),
                            ("renderModel", [p for p in pages if p != "index.html"])):
            body = bodies[name] + bodies["map"] + bodies["terminal"]
            loops = [(m.start(), m.group(1), m.group(2)) for m in
                     re.finditer(r'for \(const (\w+) of document\.querySelectorAll\("([^"]+)"\)\)', body)]
            calls = []
            for m in re.finditer(r'(\w+)\.(querySelector(?:All)?|getElementById)\("([^"]+)"\)', body):
                loop = [sel for at, var, sel in loops if at < m.start() and var == m.group(1)]   # the nearest loop
                calls.append((loop[-1] if loop else None, m.group(2), m.group(3)))
            self.assertGreater(len(calls), 30)
            for path in paths:
                root = parse(pages[path])
                for owner, method, selector in calls:
                    if method == "querySelectorAll":
                        continue                      # an empty list throws nothing (button.x is the script's own)
                    if method == "getElementById":
                        found = root.by_id(selector) is not None
                    else:
                        found = bool(select(root, selector))
                    with self.subTest(page=path, selector=selector):
                        self.assertTrue(found)
                        if owner is not None:
                            for element in select(root, owner):
                                self.assertTrue(select(element, selector), f"{owner} has no {selector}")

    def test_site_modules_do_not_import_each_other_at_module_level(self):
        # Python 3.11 to 3.13 evaluate annotations when a def runs, so a module-level cycle (render imports state,
        # state imports render and names render.Links) stops the import there; Python 3.14 defers them and hides it
        site = REPO / "launcher/site"
        imports: dict[str, set[str]] = {}
        for path in sorted(site.glob("*.py")):
            names = set()
            for node in ast.parse(path.read_text(encoding="utf-8")).body:
                if isinstance(node, ast.ImportFrom) and node.module == "launcher.site":
                    names |= {alias.name for alias in node.names}
                elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("launcher.site."):
                    names.add(node.module.split(".")[2])
                elif isinstance(node, ast.Import):
                    names |= {a.name.split(".")[2] for a in node.names if a.name.startswith("launcher.site.")}
            imports[path.stem] = names
        cycles = sorted((a, b) for a, names in imports.items() for b in names if a < b and a in imports.get(b, set()))
        self.assertEqual(cycles, [])

    def test_script_hygiene(self):
        js = (REPO / "launcher/site/page.js").read_text(encoding="utf-8")
        for word in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", ".style", "cssText", ".animate(",
                     "localStorage", "sessionStorage", "indexedDB", "document.cookie", "fetch(", "XMLHttpRequest", "sendBeacon",
                     "import("):
            self.assertNotIn(word, js)
        self.assertIsNone(re.search(r"\beval\s*\(|\bFunction\s*\(|set(?:Timeout|Interval)\(\s*[\"'`]", js))
        calls = re.findall(r"querySelector(?:All)?\(", js)
        literal = re.findall(r"querySelector(?:All)?\(\s*(?:\"[^\"\\]*\"|'[^'\\]*')\s*\)", js)
        self.assertEqual(len(calls), len(literal))
        for text in (*words.UI.values(), words.STATE_UNTOUCHED, words.STATE_BAD_MACHINE, words.STATE_BAD_AGENT):
            self.assertNotIn(text, js)

    def test_script_reads_only_listed_values(self):
        js = (REPO / "launcher/site/page.js").read_text(encoding="utf-8")
        self.assertIn("URLSearchParams", js)
        self.assertRegex(js, r"\.includes\(")          # a value is used only when the state's own list holds it
        for text in ("a[data-scroll]", "hashchange", "popstate", "preventDefault"):
            self.assertIn(text, js)                    # spec 4.4: in-page jumps keep the engine; acceptance A13 proves it

    def test_script_handles_the_reviewed_edge_cases(self):
        # Task 8's review, each proven in Chrome by a probe: any change clears the bad-value line (spec 4.1, the sort
        # too), a reflow that an SVG element has (it has no offsetWidth), a copy that reaches its fallback without
        # navigator.clipboard, and focus kept on a control after the reset button hides itself
        js = PAGE_JS.read_text(encoding="utf-8")
        handler = re.search(r'document\.addEventListener\("change", .*?\n  }\);', js, re.S).group(0)
        self.assertRegex(handler, r'const key = input\.dataset\.k;\n    state\.bad = "";')
        self.assertNotIn("offsetWidth", js)
        self.assertIn("node.getBoundingClientRect();", js)
        self.assertIn("Promise.resolve().then(() => navigator.clipboard.writeText(cmd))", js)
        reset = re.search(r'reset\.addEventListener\("click", .*?\n    }\);', js, re.S).group(0)
        self.assertRegex(reset, r'render\(true\);\n      '
                                r'document\.querySelector\("\[data-k=gpu\]:checked"\)\.focus\(\);')

    def test_script_pins_the_address_rules(self):
        # spec 4.1 to 4.4, Review Focus 2: only a value the state lists is used, a repeated parameter is a bad value, only
        # a fragment that names a running engine changes the engine, links to other pages carry the shown state, and
        # "Use this engine" writes the shown state, not the address it arrived with
        js = PAGE_JS.read_text(encoding="utf-8")
        body = lambda name: re.search(r"\n  function " + name + r"\(.*?\n  }\n", js, re.S).group(0)
        read = body("readURL")
        self.assertIn('return all.length === 0 ? null : all.length === 1 ? all[0] : "";', read)
        self.assertIn("if ((gpu === null || gpus.includes(gpu)) && (ram === null || rams.includes(ram))) {", read)
        self.assertIn("if (agent !== null && D.agents.includes(agent)) {", read)
        self.assertRegex(read, r'} else {\n      state\.mi = D\.home;\n      state\.described = false;\n'
                               r'      state\.bad = "machine";')
        self.assertIn("if (engine !== null) state.engine = engine;", body("followHash"))
        self.assertIn('if (a.origin === location.origin && !a.getAttribute("href").startsWith("#")) a.search = qs;',
                      body("carry"))
        self.assertIn("const qs = query();", body("carry"))
        handler = re.search(r'document\.addEventListener\("change", .*?\n  }\);', js, re.S).group(0)
        engine = re.search(r'if \(key === "engine"\) \{.*?\n    }', handler, re.S).group(0)
        self.assertIn('history.replaceState(history.state, "", location.pathname + query() + "#engine-" + input.value);',
                      engine)
        self.assertNotIn("location.search", engine)

    def test_script_rewrites_the_live_verdict_label_only_when_it_changes(self):
        # p.verdict is a live region: an unchanged label must keep its text node, or a screen reader may read it again
        js = PAGE_JS.read_text(encoding="utf-8")
        self.assertNotIn('querySelector(".verdict .on").textContent =', js)
        self.assertIn("if (on.textContent !== label) on.textContent = label;", js)

    def test_script_constants_match_draw(self):
        js = (REPO / "launcher/site/page.js").read_text(encoding="utf-8")
        for name, value in (("TRACK", draw.TRACK_CODES), ("STRIP", draw.STRIP_CODES), ("RAM", draw.RAM_GEOMETRY),
                            ("MARK_WIDTH", draw.MARK_WIDTH), ("OUTLINE", draw.OUTLINE), ("DENSE", draw.DENSE)):
            self.assertIn(f"const {name} = " + json.dumps(value, separators=(",", ":"), sort_keys=True) + ";", js)

    def test_script_names_the_map_on_its_image_and_makes_the_engines_a_radio_group(self):
        # the lead's rulings: the map's name goes on the role=img element, never on the figure (Task 6), and the
        # Engines list becomes a radio group only when the script reveals its radios (Task 7)
        js = (REPO / "launcher/site/page.js").read_text(encoding="utf-8")
        self.assertRegex(js, r'querySelector\("\.hdraw"\)\.setAttribute\("aria-label", ')
        self.assertEqual(len(re.findall(r'setAttribute\("aria-label", ', js)), 2)   # the map, and a copy button
        self.assertRegex(js, r'setAttribute\("role", "radiogroup"\)')


if __name__ == "__main__":
    unittest.main()
