"use strict";
// The runtime matrix page script. Every rule lives in the build (launcher/site/data.py): this script compares the
// attributes the build wrote on each cell and row, orders rows by the places the build computed, copies the phrases
// the build rendered into template#phrases, and keeps the whole view in the URL. It writes text only through
// textContent and append, makes nodes only with createElement and cloneNode, and stores nothing in the browser.
(() => {
  const KEYS = ["use", "engine", "status", "pass", "ctx", "think", "tools", "ram", "q", "sort"];
  const GROUPS = ["engine", "status", "think"];
  const SELECTS = ["pass", "ctx", "tools", "ram"];
  const NOUNS = {use: "a use case", engine: "an engine", status: "a status", pass: "a client", ctx: "a context",
                 think: "a thinking type", tools: "a tools value", ram: "a RAM value", sort: "a sort"};
  const REFUSED = "This browser does not let the page update its address, so links to views are off.";

  const byId = (id) => document.getElementById(id);
  const drawer = byId("drawer");
  const drawerBody = byId("drawer-body");
  const matrix = byId("matrix");
  const tbody = matrix.querySelector("tbody");
  const search = byId("q");
  const sortSelect = byId("sort");
  const answer = byId("answer");
  const live = byId("live");
  const notice = byId("notice");
  const radios = [...document.querySelectorAll('input[name="use"]')];
  const checkboxes = [...document.querySelectorAll('input[type="checkbox"]')];
  const boxes = (name) => checkboxes.filter((box) => box.name === name);
  const templates = [...byId("phrases").content.children];
  const rows = [...tbody.children];
  const cellsOf = (row) => [...row.children].filter((child) => child.tagName === "TD");
  const heads = [...matrix.querySelectorAll("th")].filter((th) => th.hasAttribute("data-engine"));
  const tiles = [...document.querySelectorAll("a.tile")];
  const clientIds = [...byId("pass").options].map((option) => option.value).filter((v) => v && v !== "every");

  // The script, not the markup, sets autofocus before the first showModal(), because Chrome logs a console line for
  // a static autofocus on every load with a hash; Chrome 157 then focuses button#close, the dialog's first control.
  drawer.autofocus = true;

  function phrase(kind, target) {
    const found = templates.find((span) => span.dataset.kind === kind && span.dataset.for === target);
    if (!found) {
      throw new Error("template#phrases holds no " + kind + " phrase for " + JSON.stringify(target));
    }
    return found.cloneNode(true);
  }

  function counted(n, one, many) {
    return n + " " + (n === 1 ? one : many);
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
    return radios.find((radio) => radio.checked) || radios[0];
  }

  function group(name) {
    const all = boxes(name);
    const on = all.filter((box) => box.checked).map((box) => box.value);
    return on.length === all.length ? null : on;
  }

  function state() {
    const radio = chosenUse();
    const view = {use: radio.value, useSort: radio.dataset.sort, search: search.value.trim().toLowerCase(),
                  sort: sortSelect.value};
    for (const name of GROUPS) {
      view[name] = group(name);
    }
    for (const name of SELECTS) {
      view[name] = byId(name).value || null;
    }
    return view;
  }

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

  function apply() {
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
    const byName = (a, b) => place(a.row, "name") - place(b.row, "name");
    if (rows.length && rows[0].hasAttribute("data-o-" + key)) {
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
    out.sort(byName);
    const order = [...shown, ...out].map((entry) => entry.row);
    if (order.some((row, n) => tbody.children[n] !== row)) {
      for (const row of order) {
        tbody.append(row);
      }
    }

    const radio = chosenUse();
    const chip = radio.closest("label").textContent.trim();
    answer.textContent = "";
    if (!shown.length) {
      answer.append(phrase("empty", ""));
    } else {
      const runs = shown.flatMap((entry) => entry.kept.filter((cell) => cell.dataset.status === "runs"));
      const parts = [];
      let head;
      if (view.useSort === "gap") {
        head = chip + ": " + counted(shown.length, "model", "models") + ".";
        const gapped = shown.filter((entry) => templates.some((span) => span.dataset.kind === "gap" &&
                                                               span.dataset.for === entry.row.id));
        gapped.sort((a, b) => place(a.row, "gap") - place(b.row, "gap"));
        if (gapped.length) {
          parts.push(phrase("gap", gapped[0].row.id));
        }
      } else {
        head = chip + ": " + counted(shown.length, "model", "models") + ", " +
          counted(runs.length, "recipe that runs", "recipes that run") + ".";
        if (!runs.length) {
          parts.push(phrase("no-best", ""));
        } else {
          const best = smallest(runs, "rank");
          const fastest = smallest(runs, "speed");
          parts.push(best.closest("tr").hasAttribute("data-rank") ? phrase("best", detailsId(best)) :
            phrase("no-best", ""));
          if (fastest.hasAttribute("data-speed")) {
            parts.push(phrase("fastest", detailsId(fastest)));
          }
        }
      }
      answer.append(head);
      for (const part of parts) {
        answer.append(" ", part);
      }
    }

    const rule = byId("use-rule");
    rule.textContent = "";
    rule.append(phrase("rule", view.use));
    const sortLabel = sortSelect.options[sortSelect.selectedIndex].textContent;
    byId("caption").textContent = "Each model on each engine, sorted by: " + sortLabel + ".";
    const active = filtersSet(view) + (view.search ? 1 : 0);
    byId("print-view").textContent = "View: " + chip + "; sorted by: " + sortLabel +
      (active ? "; " + counted(active, "filter", "filters") + " active" : "") + ".";
    byId("filters-label").textContent = active ? "Filters (" + active + " active)" : "Filters";
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
    apply();
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
        const radio = radios.find((item) => item.value === value);
        if (radio) {
          radio.checked = true;
          sortSelect.value = radio.dataset.sort;
        } else {
          unknown(key, value);
        }
      } else if (GROUPS.includes(key)) {
        const all = boxes(key);
        // "key=" checks none (ruling 14); an empty item of a longer list, such as a trailing comma, names nothing.
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

  // Back or Forward to an entry with another query (the skip link and the gap phrase's row link each push one, and a
  // later change replaces it): set every control to its default, then read that entry's query.
  function reread() {
    radios[0].checked = true;
    sortSelect.value = radios[0].dataset.sort;
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
    apply();
  }

  function clearFilters() {
    search.value = "";
    for (const box of checkboxes) {
      box.checked = true;
    }
    for (const name of SELECTS) {
      byId(name).value = "";
    }
    update();
  }

  // Details in the dialog (Details, and the lead's ruling on the history): a click or key on a tile pushes the
  // hash, and closing that dialog goes back; a hash at load opens its details without a history call, and closing
  // that dialog replaces the entry with the URL without the hash; a link inside the dialog replaces the entry.
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
    const target = opener && opener.getClientRects().length ? opener : matrix;
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

  // Copy (Details): the command's data-copy or the URL of this view, "Copied" for 2 s, and a polite announcement.
  const labels = new WeakMap();

  function announce(text) {
    live.textContent = text;
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
    if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) {
      return;
    }
    const target = event.target;
    const button = target.closest("button.copy, #copy-link");
    if (button) {
      copy(button);
      return;
    }
    if (target.closest("button.clear")) {
      clearFilters();
      search.focus();
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
    if (target.name === "use") {
      sortSelect.value = target.dataset.sort;
      update();
    } else if (GROUPS.includes(target.name) || SELECTS.includes(target.id) || target === sortSelect) {
      update();
    }
  });
  search.addEventListener("input", () => {
    apply();
    clearTimeout(searchTimer);
    searchTimer = setTimeout(writeUrl, 250);
  });
  byId("clear").addEventListener("click", clearFilters);
  byId("dismiss").addEventListener("click", () => {
    noticeLines = [];
    showNotice();
    search.focus();
  });

  // Load (Structure, URL state): show the controls, read the URL, apply the view, then open the hash's details.
  for (const id of ["uses", "toolbar"]) {
    byId(id).hidden = false;
  }
  for (const button of document.querySelectorAll("button.copy")) {
    button.hidden = false;
  }
  if (!matchMedia("(min-width: 48rem)").matches) {
    byId("versions").open = false;
    byId("legend").open = false;
  }
  noticeLines = read();
  if (matchMedia("(min-width: 64rem)").matches || filtersSet(state()) > 0) {
    byId("filters").open = true;
  }
  apply();
  showNotice();
  const initial = sectionOf(location.hash);
  if (initial) {
    opener = tileFor(initial.id);
    show(initial);
  } else if (location.search !== query(state())) {
    remember("replaceState", viewUrl());
  }
})();
