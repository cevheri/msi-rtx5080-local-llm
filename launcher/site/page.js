"use strict";
// The runtime matrix page script. Every rule has a tested Python twin in the build (launcher/site/data.py): this
// script compares the attributes the build wrote on each cell and row, orders rows by the places the build computed,
// chooses the pick cards, the key parts and the divider's place by the rules of data.picks and data.view, copies the
// cards, role words, rule lines and separators the build rendered, and keeps the whole view in the URL. It writes
// text only through textContent, append and replaceChildren, makes nodes only with createElement and cloneNode,
// never sets a style, and stores nothing in the browser.
(() => {
  const KEYS = ["use", "engine", "status", "pass", "ctx", "think", "tools", "ram", "q", "sort"];
  const GROUPS = ["engine", "status", "think"];
  const SELECTS = ["pass", "ctx", "tools", "ram"];
  const NOUNS = {use: "a view", engine: "an engine", status: "a status", pass: "a client", ctx: "a context",
                 think: "a thinking type", tools: "a tools value", ram: "a RAM value", sort: "a sort"};
  // The search trims ASCII whitespace only, as data.ASCII_SPACE does: String.prototype.trim and str.strip differ.
  const EDGE_SPACE = /^[ \t\n\r\f\v]+|[ \t\n\r\f\v]+$/g;
  const REFUSED = "This browser does not let the page update its address, so links to views are off.";
  // The live words follow a view change after this pause, so a walk through the View select announces where it stops.
  const QUIET = 250;

  const byId = (id) => document.getElementById(id);
  const drawer = byId("drawer");
  const drawerBody = byId("drawer-body");
  const sheet = byId("sheet");
  const matrix = byId("matrix");
  const tbody = matrix.tBodies[0];
  const search = byId("q");
  const useSelect = byId("use");
  const sortSelect = byId("sort");
  const answer = byId("answer");
  const answerPicks = byId("answer-picks");
  const live = byId("live");
  const drawerLive = byId("drawer-live");
  const sheetLive = byId("sheet-live");
  const notice = byId("notice");
  const filtersButton = byId("open-filters");
  const divider = byId("grp");
  const keyLine = byId("key");
  const keyParts = [...keyLine.children].filter((part) => part.hasAttribute("data-k"));
  const classParts = keyParts.filter((part) => part.hasAttribute("data-cls"));
  const pickList = byId("pick-list");
  const noPicks = byId("no-picks");
  const picksSection = byId("picks");
  const longContext = Number(pickList.dataset.long);
  const checkboxes = [...sheet.querySelectorAll('input[type="checkbox"]')];
  const boxes = (name) => checkboxes.filter((box) => box.name === name);
  const templates = [...byId("phrases").content.children];
  const cardTemplates = [...byId("cards").content.children];
  const rows = [...tbody.rows].filter((row) => row !== divider);
  const cellsOf = (row) => [...row.cells].filter((cell) => cell.tagName === "TD");
  const heads = [...matrix.tHead.rows[0].cells].filter((th) => th.hasAttribute("data-engine"));
  const engineIds = heads.map((th) => th.dataset.engine);
  const tiles = [...matrix.querySelectorAll("a.tile")];
  const clientIds = [...byId("pass").options].map((option) => option.value).filter((v) => v && v !== "every");
  const pairs = Number(matrix.dataset.pairs);
  const narrow = matchMedia("(max-width: 47.99rem)");

  // The script, not the markup, sets autofocus before the first showModal(), because Chrome logs a console line for
  // a static autofocus on every load with a hash; Chrome 157 then focuses button#close, the dialog's first control.
  drawer.autofocus = true;

  function phrase(kind, target) {
    const found = templates.find((span) => span.dataset.kind === kind && span.dataset.for === target);
    if (!found) {
      throw new Error("template#phrases holds no " + kind + " phrase for " + JSON.stringify(target));
    }
    return found;
  }

  function copied(node) {
    return [...node.childNodes].map((child) => child.cloneNode(true));
  }

  function counted(n, one, many) {
    return n + " " + (n === 1 ? one : many);
  }

  // data._joined: "a", "a and b", "a, b and c".
  function joined(words) {
    return words.length < 2 ? words.join("") : words.slice(0, -1).join(", ") + " and " + words[words.length - 1];
  }

  function place(el, key) {
    const value = el.getAttribute("data-o-" + key);
    return value === null ? null : Number(value);
  }

  function detailsId(cell) {
    return "d-" + cell.dataset.engine + "-" + cell.closest("tr").dataset.model;
  }

  function tileFor(id) {
    return tiles.find((tile) => tile.getAttribute("href") === "#" + id) || null;
  }

  // Notices: unknown URL keys and values, and a refused history call.
  let noticeLines = [];
  let refusedShown = false;
  // The query of the history entry the controls show; popstate re-reads the URL only when an entry holds another.
  let shownSearch = location.search;

  function showNotice() {
    byId("notice-text").textContent = noticeLines.join(" ");
    notice.hidden = noticeLines.length === 0;
  }

  function remember(method, url) {
    try {
      history[method](null, "", url);
      shownSearch = location.search;
      return true;
    } catch (error) {
      if (!refusedShown) {
        refusedShown = true;
        noticeLines.push(REFUSED);
        showNotice();
      }
      return false;
    }
  }

  // The view: the controls' state, with null for a filter at its default.
  function chosenUse() {
    return useSelect.options[useSelect.selectedIndex] || useSelect.options[0];
  }

  function group(name) {
    const all = boxes(name);
    const on = all.filter((box) => box.checked).map((box) => box.value);
    return on.length === all.length ? null : on;
  }

  function state() {
    const option = chosenUse();
    // An empty data-roles is a view without roles: the inverse of the build's " ".join.
    const roles = option.dataset.roles ? option.dataset.roles.split(" ") : [];
    const view = {use: option.value, useSort: option.dataset.sort, roles: roles,
                  search: search.value.replace(EDGE_SPACE, "").toLowerCase(), sort: sortSelect.value};
    for (const name of GROUPS) {
      view[name] = group(name);
    }
    for (const name of SELECTS) {
      view[name] = byId(name).value || null;
    }
    return view;
  }

  // data.filter_count: the search does not count.
  function filtersSet(view) {
    return GROUPS.filter((name) => view[name] !== null).length + SELECTS.filter((name) => view[name] !== null).length;
  }

  function keeps(cell, view) {
    if (view.use && !(cell.dataset.use || "").split(" ").includes(view.use)) {
      return false;
    }
    if (view.search && !cell.dataset.search.includes(view.search)) {
      return false;
    }
    if (view.engine && !view.engine.includes(cell.dataset.engine)) {
      return false;
    }
    if (view.status && !view.status.includes(cell.dataset.status)) {
      return false;
    }
    if (!view.pass && !view.ctx && !view.think && !view.tools && !view.ram) {
      return true;
    }
    if (cell.dataset.status !== "runs") {
      return false;
    }
    if (view.pass === "every") {
      if (!clientIds.every((client) => cell.getAttribute("data-c-" + client) === "passed")) {
        return false;
      }
    } else if (view.pass && cell.getAttribute("data-c-" + view.pass) !== "passed") {
      return false;
    }
    if (view.ctx && Number(cell.dataset.ctx) < Number(view.ctx)) {
      return false;
    }
    if (view.think && !view.think.includes(cell.dataset.think)) {
      return false;
    }
    if (view.tools && cell.dataset.tools !== view.tools) {
      return false;
    }
    return !view.ram || Number(cell.dataset.cap) <= Number(view.ram);
  }

  function smallest(cells, key) {
    let best = null;
    for (const cell of cells) {
      if (place(cell, key) !== null && (best === null || place(cell, key) < place(best, key))) {
        best = cell;
      }
    }
    return best;
  }

  function isRowKey(key) {
    return rows.length > 0 && rows[0].hasAttribute("data-o-" + key);
  }

  // The pick cards (data.picks): each role of the view names the in-view Runs cell with the smallest place in its
  // order among the cells it admits; a role whose cell a card already holds adds its key to that card; the next
  // cells of the current sort that no card holds fill the empty places; at most three cards.
  function ranked(cell) {
    return cell.closest("tr").hasAttribute("data-rank");
  }

  // The fill order: a cell order as it is; a row order (engine difference, model ID) one cell per model, its first
  // in the speed order, models in the row order.
  function inOrder(runs, key) {
    const out = [...runs];
    if (!isRowKey(key)) {
      return out.sort((a, b) => place(a, key) - place(b, key));
    }
    out.sort((a, b) => place(a.closest("tr"), key) - place(b.closest("tr"), key) || place(a, "speed") - place(b, "speed"));
    return out.filter((cell, n) => n === 0 || out[n - 1].closest("tr") !== cell.closest("tr"));
  }

  function picksFor(view, runs) {
    const timed = runs.filter((cell) => cell.hasAttribute("data-speed"));
    const found = [];
    for (const role of view.roles) {
      if (role === "engines") {
        const per = engineIds.map((id) => smallest(timed.filter((cell) => cell.dataset.engine === id), "speed"))
          .filter((cell) => cell !== null);
        per.sort((a, b) => place(a, "speed") - place(b, "speed"));
        for (const cell of per) {
          found.push(["engine-" + cell.dataset.engine, cell]);
        }
      } else if (role === "rank") {
        found.push([role, smallest(runs.filter(ranked), "rank")]);
      } else if (role === "fastest") {
        found.push([role, smallest(timed, "speed")]);
      } else if (role === "fastest-top") {
        found.push([role, smallest(timed.filter(ranked), "speed")]);
      } else if (role === "rank-long") {
        found.push([role, smallest(runs.filter((cell) => ranked(cell) && Number(cell.dataset.ctx) >= longContext),
                                   "rank")]);
      } else if (role === "default") {
        found.push([role, runs.find((cell) => cell.hasAttribute("data-default")) || null]);
      } else if (role === "size") {
        found.push([role, smallest(runs, "size")]);
      } else if (role === "gap") {
        const byRow = new Map();
        for (const cell of timed) {
          const row = cell.closest("tr");
          byRow.set(row, [...(byRow.get(row) || []), cell]);
        }
        const gapped = [...byRow.entries()].filter((entry) => entry[1].length >= 2);
        gapped.sort((a, b) => place(a[0], "gap") - place(b[0], "gap"));
        found.push([role, gapped.length ? smallest(gapped[0][1], "speed") : null]);
      } else {
        throw new Error("unknown card role " + role);
      }
    }
    const cards = [];
    for (const [role, cell] of found) {
      if (cell === null) {
        continue;
      }
      const hit = cards.find((card) => card.cell === cell);
      if (hit) {
        hit.roles.push(role);
      } else {
        cards.push({roles: [role], cell});
      }
    }
    for (const cell of inOrder(runs, view.sort)) {
      if (cards.length >= 3) {
        break;
      }
      if (!cards.some((card) => card.cell === cell)) {
        cards.push({roles: ["next-" + view.sort], cell});
      }
    }
    return cards.slice(0, 3);
  }

  // data.role_label: fastest-top and the engine roles drop beside fastest, rank-long beside rank; the first kept
  // role's words, then the others' data-mid, joined with ", " and " and ".
  function roleLabel(roles) {
    const kept = roles.filter((role) => !((role === "fastest-top" || role.startsWith("engine-")) &&
                                          roles.includes("fastest")) && !(role === "rank-long" && roles.includes("rank")));
    return joined(kept.map((role, n) => n === 0 ? phrase("role", role).textContent : phrase("role", role).dataset.mid));
  }

  // The cards: clones of template#cards, replaced only when the wanted pairs or labels differ, so focus inside a card
  // survives a view change that keeps it. Returns the picks words of data.picks_line, "" without cards.
  function showPicks(cards, anyRows) {
    const wanted = cards.map((card) => ({id: detailsId(card.cell), label: roleLabel(card.roles), roles: card.roles}));
    const now = [...pickList.children].map((li) => ({id: li.dataset.for, label: li.querySelector(".role").textContent}));
    if (wanted.length !== now.length || wanted.some((card, n) => card.id !== now[n].id || card.label !== now[n].label)) {
      pickList.replaceChildren(...wanted.map((card) => {
        const source = cardTemplates.find((li) => li.dataset.for === card.id);
        if (!source) {
          throw new Error("template#cards holds no card for " + card.id);
        }
        const li = source.cloneNode(true);
        li.querySelector(".role").textContent = card.label;
        return li;
      }));
    }
    const shownCards = [...pickList.children];
    shownCards.forEach((li, n) => {
      const gap = li.querySelector("p.pg");
      if (gap) {
        gap.hidden = !wanted[n].roles.includes("gap");
      }
    });
    pickList.hidden = cards.length === 0;
    noPicks.hidden = cards.length > 0 || !anyRows;
    // An empty view leaves no named region holding only its hidden heading; #answer says that nothing matches.
    picksSection.hidden = !anyRows;
    const words = wanted.map((card, n) => card.label + ": " + shownCards[n].querySelector("h3 a").textContent);
    return words.length ? " Picks: " + words.join("; ") + "." : "";
  }

  // The count line and the picks words are a live region: the view changes at once, its words after a pause. The
  // open sheet makes #answer inert, so there its words wait for the close and #sheet-live gets the match line.
  let quietTimer = null;
  let pending = null;
  let unsaid = null;

  function flushWords() {
    clearTimeout(quietTimer);
    quietTimer = null;
    if (pending) {
      sheetLive.textContent = pending.match;
      unsaid = pending;
      pending = null;
    }
    if (unsaid && !sheet.open) {
      answerPicks.textContent = unsaid.picks;
      answer.replaceChildren(...unsaid.count, answerPicks);
      unsaid = null;
    }
  }

  // The key (data.view's key): the parts of the in-view Runs cells, and the built separators shown or hidden.
  function showKey(runs) {
    const present = new Set();
    for (const cell of runs) {
      for (const id of clientIds) {
        present.add(cell.getAttribute("data-c-" + id));
      }
    }
    const marks = runs.length ? classParts.filter((part) => present.has(part.dataset.k)) : [];
    const on = {
      counts: runs.length > 0,
      letters: marks.length > 0,
      setup: runs.some((cell) => cell.hasAttribute("data-setup")),
      fastest: runs.some((cell) => cell.hasAttribute("data-fast")),
      bars: runs.length > 0,
    };
    for (const part of keyParts) {
      part.hidden = part.hasAttribute("data-cls") ? !marks.includes(part) : !on[part.dataset.k];
    }
    marks.forEach((part, n) => {
      part.querySelector(".sep").hidden = n === marks.length - 1;
    });
    const counts = keyParts.find((part) => part.dataset.k === "counts");
    if (counts) {
      counts.querySelector(".sep-more").hidden = marks.length === 0;
      counts.querySelector(".sep-end").hidden = marks.length > 0;
    }
    keyLine.hidden = keyParts.every((part) => part.hidden);
  }

  function apply(immediate) {
    const view = state();
    const engineOn = (engine) => !view.engine || view.engine.includes(engine);
    for (const th of heads) {
      th.hidden = !engineOn(th.dataset.engine);
    }
    const shown = [];
    const out = [];
    for (const row of rows) {
      const cells = cellsOf(row);
      const kept = cells.filter((cell) => keeps(cell, view));
      row.hidden = kept.length === 0;
      for (const cell of cells) {
        const on = kept.includes(cell);
        cell.hidden = !engineOn(cell.dataset.engine);
        cell.querySelector("a.tile").hidden = !on;
        cell.querySelector("span.out").hidden = on;
        const section = byId(detailsId(cell));
        if (section) {
          section.classList.toggle("out", !on);
        }
      }
      (kept.length ? shown : out).push({row, kept});
    }
    const key = view.sort;
    if (isRowKey(key)) {
      shown.sort((a, b) => place(a.row, key) - place(b.row, key));
    } else {
      const rank = (entry) => {
        const first = smallest(entry.kept, key);
        return first ? [0, place(first, key)] : [1, place(entry.row, "name")];
      };
      shown.sort((a, b) => {
        const x = rank(a);
        const y = rank(b);
        return x[0] - y[0] || x[1] - y[1];
      });
    }
    out.sort((a, b) => place(a.row, "name") - place(b.row, "name"));
    const order = [...shown, ...out].map((entry) => entry.row);
    // The divider (data.view's divider): in the rank order, before the first unranked row in view when no row in view
    // after it is ranked; otherwise hidden at the end of the body.
    let first = -1;
    if (key === "rank") {
      first = shown.findIndex((entry) => !entry.row.hasAttribute("data-rank"));
      if (first >= 0 && shown.slice(first).some((entry) => entry.row.hasAttribute("data-rank"))) {
        first = -1;
      }
    }
    divider.hidden = first < 0;
    order.splice(first < 0 ? order.length : first, 0, divider);
    divider.cells[0].colSpan = 1 + heads.filter((th) => !th.hidden).length;
    if (order.some((row, n) => tbody.rows[n] !== row)) {
      for (const row of order) {
        tbody.append(row);
      }
    }
    matrix.dataset.use = view.use;
    matrix.dataset.sort = view.sort;

    const name = chosenUse().textContent;
    const inView = shown.flatMap((entry) => entry.kept);
    const runs = inView.filter((cell) => cell.dataset.status === "runs");
    let count;
    let picked = "";
    if (!shown.length) {
      count = copied(phrase("empty", ""));
      showPicks([], false);
    } else {
      const plain = !view.use && !filtersSet(view) && !view.search;
      count = [counted(shown.length, "model", "models") + "; " +
        (plain ? runs.length + " of " + pairs + " pairs run." : counted(inView.length, "pair matches", "pairs match") +
          ".")];
      picked = showPicks(picksFor(view, runs), true);
    }
    pending = {count, picks: picked, match: counted(shown.length, "model matches.", "models match.")};
    clearTimeout(quietTimer);
    if (immediate) {
      flushWords();
    } else {
      quietTimer = setTimeout(flushWords, QUIET);
    }

    showKey(runs);
    const rule = byId("use-rule");
    rule.replaceChildren(...(view.use ? copied(phrase("rule", view.use)) : []));
    rule.hidden = !view.use;
    const sortLabel = sortSelect.options[sortSelect.selectedIndex].textContent;
    byId("caption").textContent = "Decode speed, context and client results of each model on each engine. View: " +
      name + "; sorted by: " + sortLabel + ".";
    const active = filtersSet(view);
    filtersButton.textContent = active ? "Filters (" + active + ")" : "Filters";
    filtersButton.toggleAttribute("data-on", active > 0);
    byId("show").textContent = "Show " + counted(shown.length, "model", "models");
    const printed = active + (view.search ? 1 : 0);
    byId("print-view").textContent = "View: " + name + "; sorted by: " + sortLabel +
      (printed ? "; " + counted(printed, "filter", "filters") + " active" : "") + ".";
  }

  // The URL: keys in KEYS order, values as data IDs, lists in data order, defaults left out.
  function base() {
    return location.href.split(/[?#]/)[0];
  }

  function query(view) {
    const parts = [];
    for (const key of KEYS) {
      let value = null;
      if (key === "use") {
        value = view.use || null;
      } else if (GROUPS.includes(key)) {
        value = view[key] === null ? null : view[key].map(encodeURIComponent).join(",");
      } else if (key === "q") {
        value = view.search ? encodeURIComponent(view.search) : null;
      } else if (key === "sort") {
        value = view.sort === view.useSort ? null : view.sort;
      } else {
        value = view[key];
      }
      if (value !== null) {
        parts.push(key + "=" + (GROUPS.includes(key) || key === "q" ? value : encodeURIComponent(value)));
      }
    }
    return parts.length ? "?" + parts.join("&") : "";
  }

  function viewUrl() {
    return base() + query(state()) + location.hash;
  }

  let searchTimer = null;

  function writeUrl() {
    clearTimeout(searchTimer);
    searchTimer = null;
    remember("replaceState", viewUrl());
  }

  function update() {
    apply(false);
    writeUrl();
  }

  function read() {
    const params = new URLSearchParams(location.search);
    const lines = [];
    const unknown = (key, value) => lines.push("This link asks for " + NOUNS[key] + " this page does not list: " +
                                               (value === "" ? "an empty value" : value) + ".");
    for (const key of KEYS) {
      if (!params.has(key)) {
        continue;
      }
      const value = params.get(key);
      if (params.getAll(key).length > 1) {
        lines.push("This link repeats the key " + key + "; the page reads its first value only.");
      }
      if (key === "use") {
        const option = [...useSelect.options].find((item) => item.value === value);
        if (option) {
          useSelect.value = value;
          sortSelect.value = option.dataset.sort;
        } else {
          unknown(key, value);
        }
      } else if (GROUPS.includes(key)) {
        const all = boxes(key);
        // "key=" checks none; an empty item of a longer list, such as a trailing comma, names nothing.
        const wanted = value.split(",").filter((item) => item !== "");
        const bad = wanted.filter((item) => !all.some((box) => box.value === item));
        for (const item of bad) {
          unknown(key, item);
        }
        if (wanted.length && bad.length === wanted.length) {
          continue;
        }
        for (const box of all) {
          box.checked = wanted.includes(box.value);
        }
      } else if (key === "q") {
        search.value = value;
      } else {
        const select = key === "sort" ? sortSelect : byId(key);
        if ([...select.options].some((option) => option.value === value)) {
          select.value = value;
        } else {
          unknown(key, value);
        }
      }
    }
    for (const key of new Set(params.keys())) {
      if (!KEYS.includes(key)) {
        lines.push("This link holds a key this page does not know: " + key + ".");
      }
    }
    return lines;
  }

  // Back or Forward to an entry with another query: set every control to its default, then read that entry's query.
  function reread() {
    useSelect.value = "";
    sortSelect.value = chosenUse().dataset.sort;
    search.value = "";
    for (const box of checkboxes) {
      box.checked = true;
    }
    for (const name of SELECTS) {
      byId(name).value = "";
    }
    noticeLines = read();
    if (refusedShown) {
      noticeLines.push(REFUSED);
    }
    showNotice();
    apply(true);
  }

  // Clear filters (the sheet's foot and the empty view's line): the search and every filter, keeping view and sort.
  function clearFilters() {
    search.value = "";
    for (const box of checkboxes) {
      box.checked = true;
    }
    for (const name of SELECTS) {
      byId(name).value = "";
    }
    update();
    flushWords();
    if (!sheet.open) {
      search.focus();
    }
  }

  // Sort and Copy link sit in the control row from 48rem and at the end of the Filters sheet below it, where the
  // search shows the short placeholder the build wrote, so the field never shows a cut word.
  const longPlaceholder = search.placeholder;

  function moveControls() {
    const target = narrow.matches ? byId("sheet-slot") : byId("ctl");
    const moved = [byId("sortc"), byId("copy-link")];
    const held = document.activeElement;
    const had = moved.some((node) => node.contains(held));
    target.append(...moved);
    search.placeholder = narrow.matches ? search.dataset.short : longPlaceholder;
    // A move drops focus; it returns to the moved control where that can take it (shown and not inert), else to the
    // open sheet's Close button or the Filters button.
    if (had && document.activeElement !== held) {
      held.focus();
      if (document.activeElement !== held) {
        (sheet.open ? byId("sheet-close") : filtersButton).focus();
      }
    }
  }

  // The Filters sheet: a modal dialog that writes no history entry; Close, Show, Escape and a backdrop click close
  // it, and on close the pending words are announced and focus returns to the Filters button.
  filtersButton.addEventListener("click", () => {
    if (!sheet.open) {
      sheet.showModal();
    }
  });
  byId("sheet-close").addEventListener("click", () => sheet.close());
  byId("show").addEventListener("click", () => sheet.close());
  sheet.addEventListener("close", () => {
    flushWords();
    filtersButton.focus();
  });
  sheet.addEventListener("click", (event) => {
    if (event.target !== sheet) {
      return;
    }
    const box = sheet.getBoundingClientRect();
    if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top ||
        event.clientY > box.bottom) {
      sheet.close();
    }
  });

  // Details in the dialog: a click or key on a tile or a card's link pushes the hash, and closing that dialog goes
  // back; a hash at load opens its details without a history call, and closing that dialog replaces the entry with
  // the URL without the hash; a link inside the dialog replaces the entry.
  let opener = null;
  let placeholder = null;
  let current = null;
  let pushed = false;

  function sectionOf(hash) {
    if (!hash.startsWith("#d-")) {
      return null;
    }
    let found;
    try {
      found = document.getElementById(decodeURIComponent(hash.slice(1)));
    } catch (error) {
      return null;
    }
    return found && found.tagName === "SECTION" && found.classList.contains("details") ? found : null;
  }

  function show(section) {
    if (current !== section) {
      if (current) {
        placeholder.replaceWith(current);
      }
      placeholder = document.createElement("span");
      placeholder.hidden = true;
      section.before(placeholder);
      drawerBody.append(section);
      current = section;
      drawer.setAttribute("aria-labelledby", section.querySelector("h3").id);
      drawer.scrollTop = 0;
    }
    if (!drawer.open) {
      drawer.showModal();
    }
  }

  function hide() {
    if (!current) {
      return false;
    }
    placeholder.replaceWith(current);
    current = null;
    placeholder = null;
    if (drawer.open) {
      drawer.close();
    }
    // A card the view replaced while its link opened the dialog is gone, so focus goes to the table.
    const target = opener && opener.isConnected && opener.getClientRects().length ? opener : matrix;
    target.focus();
    return true;
  }

  function closeDialog() {
    const wasPushed = pushed;
    if (!hide()) {
      return;
    }
    pushed = false;
    if (wasPushed) {
      history.back();
    } else {
      remember("replaceState", base() + query(state()));
    }
  }

  drawer.addEventListener("cancel", (event) => {
    event.preventDefault();
    closeDialog();
  });
  drawer.addEventListener("close", closeDialog);
  drawer.addEventListener("click", (event) => {
    if (event.target !== drawer) {
      return;
    }
    const box = drawer.getBoundingClientRect();
    if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top ||
        event.clientY > box.bottom) {
      closeDialog();
    }
  });
  byId("close").addEventListener("click", closeDialog);
  window.addEventListener("popstate", () => {
    if (location.search !== shownSearch) {
      shownSearch = location.search;
      reread();
    }
    const section = sectionOf(location.hash);
    if (section) {
      if (!drawer.open) {
        opener = tileFor(section.id);
      }
      show(section);
      pushed = true;
    } else {
      hide();
      pushed = false;
    }
  });

  // Copy: the command's data-copy or the URL of this view, "Copied" for 2 s, and a polite announcement.
  const labels = new WeakMap();

  // An open modal dialog makes everything outside it inert, #live included, so each dialog has its own live region.
  function announce(text) {
    (drawer.open ? drawerLive : sheet.open ? sheetLive : live).textContent = text;
  }

  async function copy(button) {
    const isLink = button.id === "copy-link";
    const text = isLink ? viewUrl() : button.getAttribute("data-copy");
    try {
      await navigator.clipboard.writeText(text);
    } catch (error) {
      if (isLink) {
        announce("Copy failed; copy the link from the address bar.");
      } else {
        const range = document.createRange();
        range.selectNodeContents(button.previousElementSibling);
        const selection = getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
        announce("Copy failed; the command is selected, press Ctrl+C to copy it.");
      }
      return;
    }
    if (!labels.has(button)) {
      labels.set(button, button.textContent);
    }
    button.textContent = "Copied";
    setTimeout(() => {
      button.textContent = labels.get(button);
    }, 2000);
    announce(isLink ? "Link copied." : "Command copied.");
  }

  document.addEventListener("click", (event) => {
    const target = event.target;
    // Clear filters is no link, so it clears on a modified click too, as the shipped #clear listener did; the
    // modifier guard below keeps the browser's own behaviour on a modified click of a link.
    if (target.closest("button.clear, #clear")) {
      clearFilters();
      return;
    }
    if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) {
      return;
    }
    const button = target.closest("button.copy, #copy-link");
    if (button) {
      copy(button);
      return;
    }
    const link = target.closest("a[href]");
    if (!link) {
      return;
    }
    const href = link.getAttribute("href");
    if (current && drawer.contains(link) && href.startsWith("#m-")) {
      event.preventDefault();
      closeDialog();
      return;
    }
    const section = sectionOf(href);
    if (!section) {
      return;
    }
    event.preventDefault();
    if (drawer.open) {
      show(section);
      remember("replaceState", href);
      byId("close").focus();
      return;
    }
    opener = link;
    if (searchTimer !== null) {
      writeUrl();
    }
    show(section);
    pushed = remember("pushState", href);
  });

  document.addEventListener("change", (event) => {
    const target = event.target;
    if (target === useSelect) {
      sortSelect.value = chosenUse().dataset.sort;
      update();
    } else if (GROUPS.includes(target.name) || SELECTS.includes(target.id) || target === sortSelect) {
      update();
    }
  });
  search.addEventListener("input", () => {
    apply(false);
    clearTimeout(searchTimer);
    searchTimer = setTimeout(writeUrl, 250);
  });
  byId("dismiss").addEventListener("click", () => {
    noticeLines = [];
    showNotice();
    search.focus();
  });
  narrow.addEventListener("change", moveControls);
  // Printing opens the disclosures of the details, so nothing printed hides behind a summary.
  window.addEventListener("beforeprint", () => {
    for (const disclosure of document.querySelectorAll("section.details details")) {
      disclosure.open = true;
    }
  });

  // Load: show the controls and the Copy buttons, place Sort and Copy link for the width, read the URL, apply the
  // view with its words at once, show the notice, then open the hash's details without a history call.
  byId("controls").hidden = false;
  for (const button of document.querySelectorAll("button.copy")) {
    button.hidden = false;
  }
  moveControls();
  noticeLines = read();
  const initial = sectionOf(location.hash);
  if (!initial && /^#d-/i.test(location.hash)) {
    let asked = location.hash.slice(1);
    try {
      asked = decodeURIComponent(asked);
    } catch (error) {
      // A malformed percent escape is named as written.
    }
    noticeLines.push("This link asks for details this page does not list: " + asked + ".");
  }
  apply(true);
  showNotice();
  if (initial) {
    opener = tileFor(initial.id);
    show(initial);
  } else if (location.search !== query(state())) {
    remember("replaceState", viewUrl());
  }
})();
