#!/usr/bin/env python3
"""Prototype of the edit llmctl would make to a Hermes config.yaml (experiment helper, not llmctl code).

block <config> <block-file>   replace the text between the llmctl markers, or append it at the end;
                              everything outside the markers must stay byte-identical
model <config> <provider> <default>   rewrite model.provider and model.default in place and blank
                              model.base_url (one line each, inside the top-level model: section)
check <config>                parse with PyYAML, report duplicate top-level keys and the llmctl values
"""
import re
import sys
from pathlib import Path

import yaml

BEGIN = "# >>> llmctl managed block: rewritten by llmctl on every start; edits here are lost >>>"
END = "# <<< llmctl managed block <<<"


def top_level_keys(text):
    return [m.group(1) for m in re.finditer(r"^([A-Za-z_][A-Za-z0-9_]*):", text, re.M)]


def check(path):
    text = Path(path).read_text()
    keys = top_level_keys(text)
    dups = sorted({k for k in keys if keys.count(k) > 1})
    data = yaml.safe_load(text)
    m = data.get("model", {})
    print(f"parse=ok duplicate_top_level_keys={dups} begin_markers={text.count(BEGIN)} end_markers={text.count(END)}")
    print(f"model.provider={m.get('provider')!r} model.default={m.get('default')!r} model.base_url={m.get('base_url')!r}")
    print(f"providers={list((data.get('providers') or {}).keys())} model_overrides={data.get('model_overrides')}")
    llm = (data.get("providers") or {}).get("llmctl") or {}
    print(f"providers.llmctl.api={llm.get('api')!r} default_model={llm.get('default_model')!r} extra_body={llm.get('extra_body')}")
    return dups


def block(path, block_file):
    p = Path(path)
    text = p.read_text()
    body = Path(block_file).read_text().rstrip("\n")
    new_block = f"{BEGIN}\n{body}\n{END}\n"
    if text.count(BEGIN) == 1 and text.count(END) == 1:
        start = text.index(BEGIN)
        stop = text.index(END) + len(END)
        if stop < len(text) and text[stop] == "\n":
            stop += 1
        before, after = text[:start], text[stop:]
        mode = "replaced"
    elif BEGIN not in text and END not in text:
        before, after = (text if text.endswith("\n") else text + "\n"), ""
        mode = "appended"
    else:
        sys.exit("refusing: markers are unbalanced")
    new_text = before + new_block + after
    clash = set(top_level_keys(body)) & set(top_level_keys(before + after))
    if clash:
        sys.exit(f"refusing: block keys already exist outside the block: {sorted(clash)}")
    yaml.safe_load(new_text)  # must still parse
    assert new_text.startswith(before) and new_text.endswith(after)
    p.write_text(new_text)
    print(f"{mode} block ({len(body.splitlines())} lines); outside text unchanged: True")


def model(path, provider, default):
    p = Path(path)
    text = p.read_text()
    sec = re.search(r"^model:\n(.*?)(?=^\S)", text, re.M | re.S)
    if not sec:
        sys.exit("no top-level model: section")
    s = sec.group(1)
    s2, n1 = re.subn(r'^(  provider:\s*)("?)[^"\n#]*\2', lambda m: f'{m.group(1)}"{provider}"', s, count=1, flags=re.M)
    s2, n2 = re.subn(r'^(  default:\s*)("?)[^"\n#]*\2', lambda m: f'{m.group(1)}"{default}"', s2, count=1, flags=re.M)
    s2, n3 = re.subn(r'^(  base_url:\s*)("?)[^"\n#]*\2', lambda m: f'{m.group(1)}""', s2, count=1, flags=re.M)
    if (n1, n2) != (1, 1):
        sys.exit(f"could not find model.provider/default lines: {n1},{n2}")
    new_text = text[:sec.start(1)] + s2 + text[sec.end(1):]
    data_old, data_new = yaml.safe_load(text), yaml.safe_load(new_text)
    data_old["model"].update(provider=provider, default=default, **({"base_url": ""} if n3 else {}))
    assert data_old == data_new, "unexpected change"
    p.write_text(new_text)
    print(f"model.provider/default rewritten, base_url blanked={bool(n3)}; parsed result equals the intended change")


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"check": lambda: check(sys.argv[2]), "block": lambda: block(sys.argv[2], sys.argv[3]),
     "model": lambda: model(sys.argv[2], sys.argv[3], sys.argv[4])}[cmd]()
