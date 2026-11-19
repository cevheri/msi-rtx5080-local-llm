(() => {
  "use strict";
  // The model guide's script (spec 4 and 9). Every text, drawing and rule outcome of every state comes precomputed
  // from launcher/site/state.py; this script reads the address, looks the state up, draws draw.py's descriptors with
  // the alphabets below and keeps the address. It holds no text and no rule of its own.
  const TRACK = {"#":"hb on","+":"hb over",".":"hb off","/":"hb desk","1":"hb on p1","2":"hb on p2","3":"hb on p3","4":"hb on p4","5":"hb on p5","6":"hb on p6","7":"hb on p7","8":"hb on p8","9":"hb on p9","a":"hb over g1","b":"hb over g2","c":"hb over g3","d":"hb over g4","e":"hb over g5","f":"hb over g6","g":"hb over g7","h":"hb over g8","i":"hb over g9","o":"hb line","|":"hend"};
  const STRIP = {"#":"b t10","+":"b o t10",".":"b","/":"b rs","1":"b t1","2":"b t2","3":"b t3","4":"b t4","5":"b t5","6":"b t6","7":"b t7","8":"b t8","9":"b t9","@":"b rs t10","A":"b rs t1","B":"b rs t2","C":"b rs t3","D":"b rs t4","E":"b rs t5","F":"b rs t6","G":"b rs t7","H":"b rs t8","I":"b rs t9","_":"b x","a":"b o t1","b":"b o t2","c":"b o t3","d":"b o t4","e":"b o t5","f":"b o t6","g":"b o t7","h":"b o t8","i":"b o t9","|":"tick"};
  const RAM = {"map":{"base":[26,2,0],"cap":[1,24,2],"desk":[0,26,0],"end":[-4,34,0],"peak":[0,26,2],"tk":[20,6,0]},"mini":{"base":[9,1,0],"cap":[1,7,1],"desk":[0,9,0],"end":[-2,14,0],"peak":[0,9,1],"tk":[7,2,0]}};
  const MARK_WIDTH = {"end":"2","tk":"1"};
  const OUTLINE = "~";
  const DENSE = 20;
  const SVG = "http://www.w3.org/2000/svg";

  const D = JSON.parse(document.getElementById("state").textContent);
  const S = D.S;
  const model = "panel" in D;
  const live = document.getElementById("live");
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  const home = D.machines[D.home];
  const defaultAgent = D.agents.indexOf(D.defaultAgent);
  const state = { mi: D.home, ai: defaultAgent, described: false, bad: "", sort: "rank", engine: null };

  const text = (i) => (i < 0 ? "" : typeof S[i] === "string" ? S[i] : S[i].map((p) => (typeof p === "string" ? p : p[1])).join(""));

  // --- the address (spec 4.1 to 4.4) ---------------------------------------------------------------------------

  function machineIndex(gpu, ram) {
    return D.machines.findIndex((m) => m[0] === gpu && m[1] === ram);
  }

  function readURL() {
    const q = new URLSearchParams(location.search);
    // a parameter given more than once is a value this page doesn't know either
    const one = (name) => {
      const all = q.getAll(name);
      return all.length === 0 ? null : all.length === 1 ? all[0] : "";
    };
    const gpus = D.machines.map((m) => String(m[0]));
    const rams = D.machines.map((m) => String(m[1]));
    const gpu = one("gpu");
    const ram = one("ram");
    const agent = one("agent");
    state.bad = "";
    if ((gpu === null || gpus.includes(gpu)) && (ram === null || rams.includes(ram))) {
      state.mi = machineIndex(gpu === null ? home[0] : +gpu, ram === null ? home[1] : +ram);
      state.described = state.described || gpu !== null || ram !== null;
    } else {
      state.mi = D.home;
      state.described = false;
      state.bad = "machine";
    }
    if (agent !== null && D.agents.includes(agent)) {
      state.ai = D.agents.indexOf(agent);
    } else {
      state.ai = defaultAgent;
      if (agent !== null && !state.bad) state.bad = "agent";
    }
  }

  function query() {
    const q = new URLSearchParams();
    const [gpu, ram] = D.machines[state.mi];
    if (gpu !== home[0]) q.set("gpu", gpu);
    if (ram !== home[1]) q.set("ram", ram);
    if (state.ai !== defaultAgent) q.set("agent", D.agents[state.ai]);
    const qs = q.toString();
    return qs ? "?" + qs : "";
  }

  function writeURL() {
    history.replaceState(history.state, "", location.pathname + query() + location.hash);
  }

  // every link to another page carries the state; a jump inside this page keeps the address as it is
  function carry() {
    const qs = query();
    for (const a of document.querySelectorAll("a[href]")) {
      if (a.origin === location.origin && !a.getAttribute("href").startsWith("#")) a.search = qs;
    }
  }

  function engineFromHash() {
    const found = /^#engine-(.+)$/.exec(location.hash);
    if (!found) return null;
    const i = D.setups.findIndex((x) => x.rt === found[1]);
    return i < 0 ? null : i;
  }

  // --- building ------------------------------------------------------------------------------------------------

  function el(tag, cls, content) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (content !== undefined) node.textContent = content;
    return node;
  }

  function nodes(i) {
    if (i < 0) return [];
    const value = S[i];
    if (typeof value === "string") return [value];
    return value.map((p) => {
      if (typeof p === "string") return p;
      const node = el(p[0] === "a" ? "a" : p[0], "", p[1]);
      if (p[0] === "a") node.setAttribute("href", p[2]);
      return node;
    });
  }

  function rich(node, i) {
    node.replaceChildren(...nodes(i));
  }

  function restart(node, cls, animate) {
    if (!animate || reduced.matches) return;
    node.classList.remove(cls);
    node.getBoundingClientRect();   // a reflow, so the animation starts again; it works on an SVG element too
    node.classList.add(cls);
  }

  function say(node, i, animate) {
    const value = text(i);
    if (node.textContent === value) return;
    node.textContent = value;
    restart(node, "fade", animate);
  }

  function track(node, code, animate) {
    node.replaceChildren(...[...code].map((c) => el("i", TRACK[c])));
    restart(node, "wipe", animate);
  }

  function strip(box, code, animate) {
    const outline = code.startsWith(OUTLINE);
    const body = [...(outline ? code.slice(1) : code)];
    const slots = body.filter((c) => c !== "|").length;
    const span = el("span", "strip" + (outline ? " k" : "") + (slots > DENSE ? " dense" : ""));
    for (const c of body) {
      const block = el("i", STRIP[c]);
      if (STRIP[c].split(" ").slice(1).some((t) => t.startsWith("t"))) block.append(el("i", "f"));
      span.append(block);
    }
    box.replaceChildren(span);
    restart(span, "wipe", animate);
  }

  function bar(svg, rows, size, animate) {
    svg.replaceChildren(...rows.map(([cls, x, width]) => {
      const [y, height, rx] = RAM[size][cls];
      const rect = document.createElementNS(SVG, "rect");
      rect.setAttribute("class", cls);
      rect.setAttribute("x", x);
      rect.setAttribute("y", y);
      rect.setAttribute("width", width === null ? MARK_WIDTH[cls] : width);
      rect.setAttribute("height", height);
      if (rx) rect.setAttribute("rx", rx);
      return rect;
    }));
    restart(svg, "wipe", animate);
  }

  // the memory map: [track, bar, gpu left, gpu right, gpu short, ram left, ram right, ram short, alt]
  function map(fig, m, animate) {
    track(fig.querySelector(".htrack"), D.track[m[0]], animate);
    bar(fig.querySelector(".ram-bar"), D.bar[m[1]], "map", animate);
    rich(fig.querySelector(".gl"), m[2]);
    const gr = fig.querySelector(".gr");
    gr.textContent = text(m[3]);
    gr.classList.toggle("short", m[4] === 1);
    rich(fig.querySelector(".rl"), m[5]);
    const rr = fig.querySelector(".rr");
    rr.textContent = text(m[6]);
    rr.classList.toggle("short", m[7] === 1);
    fig.querySelector(".hdraw").setAttribute("aria-label", text(m[8]));
  }

  function overflow(tb) {
    tb.classList.toggle("ovf", tb.scrollWidth > tb.clientWidth + 1);
  }

  function terminal(box, x, over, animate) {
    const ai = state.ai;
    box.querySelector(".tt").textContent = text(x.hd);
    const body = box.querySelector(".tl");
    const before = new Set([...body.querySelectorAll("button.x")].map((b) => b.dataset.cmd));
    const lines = [].concat(D.head, x.pull, over, x.up, x.agent[ai], x.tail).map(text);
    body.replaceChildren(...lines.map((line) => {
      if (!line.startsWith("$ ")) return el("div", "c", line);
      const cmd = line.slice(2);
      const b = el("button", "x");
      b.type = "button";
      b.dataset.cmd = cmd;
      b.setAttribute("aria-label", text(D.ui.copyLabel) + cmd);
      const p = el("span", "p", "$");
      p.setAttribute("aria-hidden", "true");
      const label = el("span", "cp", text(D.ui.copy));
      label.setAttribute("aria-hidden", "true");
      b.append(p, el("span", "t", cmd), label);
      if (animate && before.size && !before.has(cmd)) b.classList.add("chg");
      return b;
    }));
    overflow(box.querySelector(".tb"));
  }

  function notes(ul, list) {
    ul.replaceChildren(...list.map((i) => {
      const li = el("li");
      rich(li, i);
      return li;
    }));
  }

  function syncControls() {
    const [gpu, ram] = D.machines[state.mi];
    const values = { gpu: String(gpu), ram: String(ram), agent: D.agents[state.ai], sort: state.sort };
    for (const input of document.querySelectorAll("[data-k]")) {
      const value = values[input.dataset.k];
      if (value === undefined) continue;
      if (input.type === "radio") input.checked = input.value === value;
      else input.value = value;
    }
  }

  // --- the home (spec 9.3) -------------------------------------------------------------------------------------

  const ranking = model ? [] : [...document.querySelectorAll(".shelf")].map((list) => [list, [...list.children]]);

  function renderHome(animate) {
    const mi = state.mi;
    const ai = state.ai;
    const laptop = !state.described;
    const touched = state.described || ai !== defaultAgent;
    const H = laptop ? D.heroLap[ai] : D.hero[mi][ai];
    const x = D.setups[H[1]];
    const line = document.getElementById("fh-state");
    line.querySelector(".st").textContent = text(state.bad === "machine" ? D.ui.badMachine
      : state.bad === "agent" ? D.ui.badAgent : touched ? -1 : D.ui.untouched);
    document.getElementById("reset").hidden = !touched;
    const answer = document.getElementById("answer");
    say(answer.querySelector(".fh-verdict"), H[0], animate);
    answer.querySelector(".pick-h").textContent = text(x.tt);
    answer.querySelector(".reason").textContent = text(H[2]);
    answer.querySelector(".f-sp").textContent = text(x.sp);
    answer.querySelector(".spu").hidden = x.spu < 0;
    answer.querySelector(".f-cx").textContent = text(x.cx);
    answer.querySelector(".f-dl").textContent = text(x.dl);
    map(answer.querySelector(".hmap"), [H[11], H[12], H[3], H[4], H[5], H[6], H[7], H[8], H[9]], animate);
    const moe = answer.querySelector(".hmoe");
    moe.hidden = x.mq < 0;
    moe.querySelector("summary").textContent = text(x.mq);
    rich(moe.querySelector("p"), x.ml);
    answer.querySelector(".hagents").textContent = text(H[10]);

    for (const row of document.querySelectorAll(".row")) {
      const k = D.ids.indexOf(row.dataset.model);
      const [si, vi] = D.rows[k][mi];
      const r = D.setups[si];
      row.querySelector(".eng").textContent = text(r.on);
      strip(row.querySelector(".stripbox"), D.strip[r.strip[mi]], animate);
      row.querySelector(".galt").textContent = text(r.galt);
      row.querySelector(".sp b").textContent = text(r.sp);
      row.querySelector(".sp .spu").hidden = r.spu < 0;
      row.querySelector(".cx b").textContent = text(r.cx);
      row.querySelector(".dl b").textContent = text(r.dl);
      row.querySelector(".agp").textContent = text(D.heads.ag[ai]);
      row.querySelector(".ag").textContent = text(r.ag[ai]);
      row.querySelector(".vdp").textContent = text(D.heads.vd[laptop ? 0 : 1]);
      const vd = row.querySelector(".vd");
      say(vd, D.vshort[vi], animate);
      vd.classList.toggle("ok", vi === 0);
      const pick = x.m === k;
      row.classList.toggle("pick", pick);
      row.querySelector(".pick-tag").hidden = !pick;
    }
    for (const [list, rows] of ranking) {
      if (state.sort === "rank") {
        list.append(...rows);
        continue;
      }
      const own = new Map(rows.map((li) => [D.ids.indexOf(li.dataset.model), li]));
      for (const k of D.order[state.sort][mi]) if (own.has(k)) list.append(own.get(k));
    }
    for (const head of document.querySelectorAll(".strip-head")) {
      head.textContent = text(laptop ? D.heads.stripLap : D.heads.strip[mi]);
    }
    for (const head of document.querySelectorAll(".ag-head")) head.textContent = text(D.heads.ag[ai]);
    for (const head of document.querySelectorAll(".vd-head")) head.textContent = text(D.heads.vd[laptop ? 0 : 1]);

    const start = document.getElementById("get-started");
    start.querySelector(".disk").textContent = text(x.disk);
    rich(start.querySelector(".ep"), x.ep);
    terminal(start.querySelector(".term"), x, x.over[mi], animate);
    notes(start.querySelector(".tnotes"), x.notes[ai]);

    for (const cell of document.querySelectorAll("a.cell[data-s]")) {
      const c = D.setups[+cell.dataset.s];
      cell.querySelector(".ags").textContent = text(c.cag[ai]);
      strip(cell.querySelector(".stripbox"), D.strip[c.strip[mi]], animate);
    }
  }

  // --- a model page (spec 9.3) ---------------------------------------------------------------------------------

  function renderModel(animate) {
    const mi = state.mi;
    const ai = state.ai;
    const laptop = !state.described;
    const si = state.engine === null ? D.rows[mi] : state.engine;
    const x = D.setups[si];
    const f = x.fit[mi];
    const panel = document.getElementById("your-machine");
    panel.querySelector("h2").textContent = text(D.panel[laptop ? 0 : 1]);
    const on = panel.querySelector(".verdict .on");
    const label = text(x.label[laptop ? 0 : 1]);
    if (on.textContent !== label) on.textContent = label;   // a live region: an unchanged label keeps its text node
    say(panel.querySelector(".verdict .vt"), D.verdict[f[0]], animate);
    const lap = x.fitLap;
    map(panel.querySelector(".hmap"), [f[1], f[2], laptop ? lap[0] : f[3], laptop ? lap[1] : f[4], f[5], x.rl,
      laptop ? lap[2] : f[6], f[7], laptop ? lap[3] : x.alt[mi]], animate);
    const moe = panel.querySelector("p.moe");
    rich(moe, x.moe);
    moe.hidden = x.moe < 0;
    document.querySelector(".figs .fe").textContent = text(x.en);
    document.querySelector(".f-sp").textContent = text(x.sp);
    document.querySelector(".figs .spu").hidden = x.spu < 0;
    document.querySelector(".f-cx").textContent = text(x.cx);
    document.querySelector(".f-dl").textContent = text(x.dl);

    const fitNote = document.querySelector(".pre-note:not(.agent-note)");
    rich(fitNote, f[8]);
    fitNote.hidden = f[8] < 0;
    const agentNote = document.querySelector(".agent-note");
    const pick = laptop ? D.pickLap[ai] : D.pick[mi][ai];
    const note = x.anote[ai];
    agentNote.replaceChildren(...nodes(note), ...(note < 0 || pick < 0 ? [] : [" ", ...nodes(pick)]));
    agentNote.hidden = note < 0;
    terminal(document.querySelector(".term"), x, f[9], animate);
    notes(document.querySelector(".tnotes"), x.notes[ai]);

    for (const row of document.querySelectorAll(".er[data-rt]")) {
      const i = D.setups.findIndex((y) => y.rt === row.dataset.rt);
      const r = D.setups[i].fit[mi];
      row.classList.toggle("cur", i === si);
      row.querySelector("input").checked = i === si;
      const word = row.querySelector(".vw");
      say(word, D.vshort[r[0]], animate);
      word.classList.toggle("ok", r[0] === 0);
      strip(row.querySelector(".stripbox"), D.strip[r[10]], animate);
    }
    for (const block of document.querySelectorAll(".per")) block.classList.toggle("cur", block.dataset.rt === x.rt);
    for (const row of document.querySelectorAll(".ar")) row.classList.toggle("mine", row.dataset.a === D.agents[ai]);
  }

  function render(animate) {
    syncControls();
    if (model) renderModel(animate);
    else renderHome(animate);
    carry();
  }

  // --- events ----------------------------------------------------------------------------------------------------

  document.addEventListener("change", (ev) => {
    const input = ev.target.closest("[data-k]");
    if (!input) return;
    const key = input.dataset.k;
    state.bad = "";   // the bad-value line shows until the next change (spec 4.1)
    if (key === "engine") {
      state.engine = D.setups.findIndex((x) => x.rt === input.value);
      history.replaceState(history.state, "", location.pathname + query() + "#engine-" + input.value);
      render(true);
      return;
    }
    if (key === "sort") {
      state.sort = input.value;
    } else if (key === "agent") {
      state.ai = D.agents.indexOf(input.value);
    } else {
      const [gpu, ram] = D.machines[state.mi];
      state.mi = machineIndex(key === "gpu" ? +input.value : gpu, key === "ram" ? +input.value : ram);
      state.described = true;
    }
    writeURL();
    render(true);
  });

  const reset = document.getElementById("reset");
  if (reset) {
    reset.addEventListener("click", () => {
      state.mi = D.home;
      state.ai = defaultAgent;
      state.described = false;
      state.bad = "";
      writeURL();
      render(true);
      document.querySelector("[data-k=gpu]:checked").focus();   // the button hides itself; focus goes to the GPU chip
    });
  }

  const timers = new WeakMap();
  document.addEventListener("click", (ev) => {
    const b = ev.target.closest("button.x");
    if (!b) return;
    const cmd = b.dataset.cmd;
    const label = b.querySelector(".cp");
    const done = (word, said) => {
      label.textContent = text(word);
      live.textContent = said;
      b.classList.add("done");
      clearTimeout(timers.get(b));
      timers.set(b, setTimeout(() => {
        b.classList.remove("done");
        label.textContent = text(D.ui.copy);
      }, 2000));
    };
    // started inside a promise, so a missing navigator.clipboard (an insecure context) also reaches the fallback
    const copied = () => done(D.ui.copied, text(D.ui.copiedLive) + cmd);
    Promise.resolve().then(() => navigator.clipboard.writeText(cmd)).then(copied, () => {
      const range = document.createRange();
      range.selectNodeContents(b.querySelector(".t"));
      const sel = getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
      done(D.ui.selected, text(D.ui.copyFailed));
    });
  });

  // spec 4.4: a jump inside a model page scrolls and focuses its target and keeps the address, so the engine stays
  document.addEventListener("click", (ev) => {
    const a = ev.target.closest("a[data-scroll]");
    if (!a) return;
    const target = document.getElementById(a.hash.slice(1));
    if (!target) return;
    ev.preventDefault();
    target.scrollIntoView({ behavior: reduced.matches ? "auto" : "smooth", block: "start" });
    target.focus({ preventScroll: true });
  });

  // only a fragment that names an engine that runs changes the engine; any other keeps the chosen one
  function followHash() {
    const engine = engineFromHash();
    if (engine !== null) state.engine = engine;
  }

  window.addEventListener("hashchange", () => {
    if (!model) return;
    const before = state.engine;
    followHash();
    if (state.engine !== before) render(false);
  });
  window.addEventListener("popstate", () => {
    readURL();
    if (model) followHash();
    render(false);
  });
  window.addEventListener("resize", () => {
    for (const tb of document.querySelectorAll(".tb")) overflow(tb);
  });

  for (const node of document.querySelectorAll("[data-js-only]")) node.hidden = false;
  const engines = document.querySelector(".engs");
  if (engines) {
    engines.setAttribute("role", "radiogroup");
    engines.setAttribute("aria-labelledby", engines.closest("section").querySelector("h2").id);
  }
  readURL();
  if (model) followHash();
  render(false);
})();
