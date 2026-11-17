"""Throwaway reference prototype for the llmctl Kimi design: find, replace and insert llmctl's two tables and the
top-level default_model in a Kimi config.toml, by line edits checked with tomllib. Not the implementation.

usage: python3 -I kimi_proto.py <repo> <exp dir> <out dir>
"""

import json
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO, EXP, OUT = (Path(a).resolve() for a in sys.argv[1:4])
sys.path.insert(0, str(REPO))
from launcher.catalog import load_catalog  # noqa: E402
from launcher.model import Target  # noqa: E402
from launcher.runtimes import ADAPTERS  # noqa: E402

PROVIDER = "llmctl"
PREFIX = PROVIDER + "/"


def toml_value(value) -> str:
    if isinstance(value, bool):
        raise ValueError("no booleans")
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    raise ValueError(type(value))


def alias(t: Target) -> str:
    return PREFIX + t.served


def provider_table(t: Target) -> dict:
    return {"type": "openai", "base_url": f"http://127.0.0.1:{t.runtime.host_port}/v1", "api_key": "no-key-required",
            "model_source": "static"}


def model_table(t: Target) -> dict:
    meta = t.recipe.clients
    caps = [] if t.recipe.chat_only else ["tool_use"]
    fields = {"provider": PROVIDER, "model": t.served, "display_name": f"{t.recipe.id} on {t.label} (llmctl)",
              "max_context_size": t.recipe.context, "max_output_size": meta.output, "capabilities": caps}
    if meta.thinking == "effort":
        caps.append("thinking")
        fields["support_efforts"] = [e for e in meta.efforts if e != "none"]
        fields["default_effort"] = "medium"
        if "none" in meta.efforts:
            fields["off_effort"] = "none"
    return fields


def table_lines(header: str, table: dict) -> list[str]:
    return [header] + [f"{k} = {toml_value(v)}" for k, v in table.items()]


def render(t: Target) -> str:
    lines = table_lines(f"[providers.{PROVIDER}]", provider_table(t)) + [""] + \
        table_lines(f"[models.{toml_value(alias(t))}]", model_table(t))
    return "\n".join(lines) + "\n"


def split(text):
    parts = text.split("\n")
    out = [p + "\n" for p in parts[:-1]]
    if parts[-1]:
        out.append(parts[-1])
    return out


def bare(line):
    line = line[:-1] if line.endswith("\n") else line
    return line[:-1] if line.endswith("\r") else line


@dataclass
class Item:
    kind: str  # trivia, header, statement
    start: int
    end: int
    path: tuple = ()
    array: bool = False
    data: dict | None = None


def header_path(body):
    data = tomllib.loads(body)
    path, node = [], data
    while isinstance(node, dict) and len(node) == 1:
        key = next(iter(node))
        path.append(key)
        node = node[key]
        if node == {}:
            return tuple(path), False
        if node == [{}]:
            return tuple(path), True
    raise ValueError(body)


def layout(text):
    lines = split(text)
    items, i = [], 0
    while i < len(lines):
        body = bare(lines[i])
        stripped = body.strip()
        if not stripped or stripped.startswith("#"):
            items.append(Item("trivia", i, i))
            i += 1
            continue
        if stripped.startswith("["):
            path, array = header_path(body)
            items.append(Item("header", i, i, path, array))
            i += 1
            continue
        for j in range(i + 1, len(lines) + 1):
            try:
                data = tomllib.loads("".join(lines[i:j]))
            except tomllib.TOMLDecodeError:
                continue
            items.append(Item("statement", i, j - 1, tuple(data), data=data))
            i = j
            break
        else:
            raise ValueError(f"line {i + 1}: no statement end")
    return lines, items


def tables(items):
    """(path, array, start, end) per header: end is the last line of its last statement."""
    found = []
    for index, item in enumerate(items):
        if item.kind != "header":
            continue
        end = item.end
        for nxt in items[index + 1:]:
            if nxt.kind == "header":
                break
            if nxt.kind == "statement":
                end = nxt.end
        found.append((item.path, item.array, item.start, end))
    return found


def owned(path):
    return (len(path) >= 2 and path[0] == "providers" and path[1] == PROVIDER) or \
        (len(path) >= 2 and path[0] == "models" and isinstance(path[1], str) and path[1].startswith(PREFIX))


def strip(data):
    rest = {k: v for k, v in data.items() if k != "default_model" or not (isinstance(v, str) and False)}
    rest = dict(data)
    rest.pop("default_model", None)
    for key in ("providers", "models"):
        if isinstance(rest.get(key), dict):
            kept = {k: v for k, v in rest[key].items() if not (key == "providers" and k == PROVIDER)
                    and not (key == "models" and k.startswith(PREFIX))}
            if kept:
                rest[key] = kept
            else:
                del rest[key]
    return rest


def plan_text(old, t: Target, *, set_default: bool):
    old = old or ""
    data = tomllib.loads(old)
    lines, items = layout(old)
    eol = "\r\n" if lines and lines[0].endswith("\r\n") else "\n"
    root = [it for it in items if it.kind == "statement" and not any(h.kind == "header" and h.start < it.start for h in items)]
    for it in root:
        if it.path[0] in ("providers", "models"):
            raise SystemExit(f"refuse: line {it.start + 1}: {it.path[0]} at root")
    tabs = tables(items)
    prov_main = [x for x in tabs if x[0] == ("providers", PROVIDER)]
    prov_sub = [x for x in tabs if len(x[0]) > 2 and x[0][:2] == ("providers", PROVIDER)]
    mod_main = [x for x in tabs if len(x[0]) == 2 and owned(x[0]) and x[0][0] == "models"]
    mod_sub = [x for x in tabs if len(x[0]) > 2 and owned(x[0]) and x[0][0] == "models"]
    if PROVIDER in (data.get("providers") or {}) and not prov_main:
        raise SystemExit("refuse: provider not by header")
    for name in (data.get("models") or {}):
        if name.startswith(PREFIX) and not any(x[0][1] == name for x in mod_main):
            raise SystemExit("refuse: alias not by header")
    want_alias = alias(t)
    current = data.get("default_model")
    new_default = want_alias if set_default or (isinstance(current, str) and current.startswith(PREFIX)) else current
    p_lines = [l + eol for l in table_lines(f"[providers.{PROVIDER}]", provider_table(t))]
    m_lines = [l + eol for l in table_lines(f"[models.{toml_value(want_alias)}]", model_table(t))]
    edits = []  # (start, end_inclusive, new_lines) replacements; inserts as (pos, pos-1, lines)
    if prov_main:
        _, _, s, e = prov_main[0]
        edits.append((s, e, p_lines))
    for _, _, s, e in prov_sub:
        edits.append((s, e, []))
    if mod_main:
        _, _, s, e = mod_main[0]
        edits.append((s, e, m_lines))
        for _, _, s2, e2 in mod_main[1:]:
            edits.append((s2, e2, []))
    for _, _, s, e in mod_sub:
        edits.append((s, e, []))
    inserts = []  # (pos, lines, order)
    if not prov_main:
        last = [x for x in tabs if x[0][0] == "providers"]
        pos = last[-1][3] + 1 if last else len(lines)
        inserts.append((pos, p_lines, 0))
    if not mod_main:
        last = [x for x in tabs if x[0][0] == "models"]
        pos = last[-1][3] + 1 if last else len(lines)
        inserts.append((pos, m_lines, 1))
    dflt = None
    if new_default != current:
        stmt = [it for it in root if it.path == ("default_model",)]
        if stmt:
            it = stmt[0]
            if it.start != it.end or not isinstance(current, str):
                raise SystemExit("refuse: default_model not editable")
            body = bare(lines[it.start])
            # value: the last string literal on the line before an optional comment: prototype uses tomllib by trimming
            key_end = body.index("=") + 1
            rest = body[key_end:]
            lead = rest[:len(rest) - len(rest.lstrip())]
            val = rest.lstrip()
            q = val[0]
            k = 1
            while val[k] != q:
                k += 2 if (q == '"' and val[k] == "\\") else 1
            suffix = val[k + 1:]
            new_line = body[:key_end] + lead + toml_value(new_default) + suffix + eol
            edits.append((it.start, it.start, [new_line]))
        else:
            pos = root[-1].end + 1 if root else 0
            dflt = (pos, [f"default_model = {toml_value(new_default)}{eol}"], -1)
    # forward pass: at each original index, inserts first (default, provider, model), then the line or its span
    reps = {s_: (e_, nl) for s_, e_, nl in edits}
    ins = {}
    for p_, nl, o in inserts + ([dflt] if dflt else []):
        ins.setdefault(p_, []).append((o, nl))
    src = list(lines)
    if src and not src[-1].endswith("\n") and len(src) in ins:
        src[-1] += eol
    out = []
    i = 0
    while i <= len(src):
        for o, nl in sorted(ins.get(i, []), key=lambda x: x[0]):
            block = list(nl)
            if o >= 0:
                prev = out[-1] if out else None
                if prev is not None and bare(prev).strip():
                    block = [eol] + block
                if i < len(src) and bare(src[i]).strip():
                    block = block + [eol]
            out += block
        if i == len(src):
            break
        if i in reps:
            e_, nl = reps[i]
            out += nl
            i = e_ + 1
            continue
        out.append(src[i])
        i += 1
    new = "".join(out)
    nd = tomllib.loads(new)
    assert strip(nd) == strip(data), "rest changed"
    assert nd["providers"][PROVIDER] == provider_table(t)
    assert nd["models"][want_alias] == model_table(t)
    assert [a for a in nd["models"] if a.startswith(PREFIX)] == [want_alias]
    assert nd.get("default_model") == new_default
    return new


def targets():
    cat = load_catalog(REPO / "recipes", None)
    out = {}
    for rt, entry in cat.runtimes.items():
        for rid, rec in entry.recipes.items():
            out[(rt, rid)] = Target(rec, entry.runtime, ADAPTERS[rt].served_name(rec, entry.runtime), ADAPTERS[rt].LABEL)
    return out


def main():
    ts = targets()
    OUT.mkdir(parents=True, exist_ok=True)
    gem, q38, gpt_v, dsk_v = ts[("llamacpp", "gemma-4-26b-a4b")], ts[("llamacpp", "qwen3.8-27b")], ts[("vllm", "gpt-oss-20b")], ts[("vllm", "deepseek-r1-0528-qwen3-8b")]
    # 1. golden equality with E1 entries
    same = 0
    for (rt, rid), t in sorted(ts.items()):
        if render(t) == (EXP / "entries" / rt / f"{rid}.toml").read_text():
            same += 1
    print("renders equal to E1 entries:", same, "of", len(ts))
    owner = "# ~/.kimi-code/config.toml\n# Environment settings for Kimi Code.\n# This file starts empty so built-in defaults can apply.\n# Login will populate managed Kimi provider and model entries.\n\ntelemetry = false\n"
    fixtures = {"owner": owner}
    for f in sorted((EXP / "writer").glob("*.toml")):
        fixtures[f.stem] = f.read_text()
    for name in ("p3-tui-effort-qwen38/c1-after-effort.toml", "p3-tui-effort-low-qwen38/c1-after-effort.toml"):
        p = OUT.parent.parent / "runs" / name
        fixtures["p3" + ("b" if "low" in name else "a")] = p.read_text()
    for name, text in fixtures.items():
        for tname, t in (("gemma", gem), ("qwen38", q38), ("gptoss-vllm", gpt_v), ("deepseek-vllm", dsk_v)):
            for sd in (False, True):
                new = plan_text(text, t, set_default=sd)
                # composition: plan(plan(s, gemma), t) == plan(s, t)
                comp = plan_text(plan_text(text, gem, set_default=False), t, set_default=sd)
                ok = comp == new
                idem = plan_text(new, t, set_default=sd) == new
                if not ok or not idem:
                    print("MISMATCH", name, tname, sd, ok, idem)
                (OUT / f"{name}--{tname}--{'default' if sd else 'nodefault'}.toml").write_text(new)
    print("fixtures:", len(fixtures), "all planned, composition and idempotence checked")


if __name__ == "__main__":
    main()
