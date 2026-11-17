"""Check the inputs of measurement E's start before it runs: (1) the docker run file's launch arguments, context and
memory cap against the draft recipe's [server] args, context and [container] memory_gib; (2) the model overlay of
measurement C: every link points at a file of the pinned snapshot under the container's Hugging Face cache path, the
host file behind it exists, and generation_config.json equals the overlay's make record ("after").
Run with python3 -I: check_inputs_e.py DRAFT.toml CMD.sh OVERLAY_DIR MAKE.json HF_HOME_HUB"""
import json
import os
import shlex
import sys
import tomllib

draft_path, cmd_path, overlay, make_path, hub = sys.argv[1:6]
draft = tomllib.load(open(draft_path, "rb"))
words = shlex.split(open(cmd_path).read().replace("\\\n", " "), comments=True)
launch = words[words.index("sglang.launch_server") + 1:]
owned = {"--model-path", "--served-model-name", "--context-length", "--host", "--port"}
pairs, rest, i = {}, [], 0
while i < len(launch):
    w = launch[i]
    if i + 1 < len(launch) and not launch[i + 1].startswith("--"):
        pairs[w] = launch[i + 1]
        i += 2
    else:
        pairs[w] = None
        i += 1
cmd_args = {k: v for k, v in pairs.items() if k not in owned}
d_args, j, dargs = {}, 0, draft["server"]["args"]
while j < len(dargs):
    w = dargs[j]
    if j + 1 < len(dargs) and not dargs[j + 1].startswith("--"):
        d_args[w] = dargs[j + 1]
        j += 2
    else:
        d_args[w] = None
        j += 1
ok = True
print(f"draft args ({len(d_args)} flags) equal to the docker run file's launch args other than llmctl's owned flags: {cmd_args == d_args}")
ok &= cmd_args == d_args
for k in sorted(set(cmd_args) | set(d_args)):
    if cmd_args.get(k, "MISSING") != d_args.get(k, "MISSING"):
        print(f"  differs: {k}: file {cmd_args.get(k, 'MISSING')!r}, draft {d_args.get(k, 'MISSING')!r}")
ctx = int(pairs.get("--context-length"))
print(f"context: file {ctx}, draft {draft['server']['context']}: {ctx == draft['server']['context']}")
ok &= ctx == draft["server"]["context"]
mem = words[words.index("--memory") + 1]
print(f"memory cap: file {mem} (--memory-swap {words[words.index('--memory-swap') + 1]}), draft memory_gib {draft['container']['memory_gib']}: "
      f"{mem == str(draft['container']['memory_gib']) + 'g' and words[words.index('--memory-swap') + 1] == mem}")
print(f"served name: file {pairs.get('--served-model-name')}, draft {draft['model']['served_name']}")
env = [words[k + 1] for k, w in enumerate(words) if w == "-e"]
print(f"environment ({len(env)} variables): " + ", ".join(env))
links, bad = 0, []
prefix = "/root/.cache/huggingface/hub/"
snap = f"models--{draft['model']['repo'].replace('/', '--')}/snapshots/{draft['model']['revision']}/"
for name in sorted(os.listdir(overlay)):
    p = os.path.join(overlay, name)
    if os.path.islink(p):
        links += 1
        target = os.readlink(p)
        host = os.path.join(hub, target[len(prefix):]) if target.startswith(prefix) else None
        if not target.startswith(prefix + snap) or host is None or not os.path.exists(host):
            bad.append((name, target))
make = json.load(open(make_path))
gen = json.load(open(os.path.join(overlay, "generation_config.json")))
print(f"overlay: {links} links (make record: {make['links']}), all to the pinned snapshot {draft['model']['revision']} and present on the host: {not bad}")
for b in bad:
    print(f"  bad link: {b}")
sampling = {k: gen.get(k) for k in ("temperature", "top_p", "top_k", "min_p")}
print(f"generation_config.json equals the make record's 'after': {gen == make['after']}; its sampling {sampling}; "
      f"the draft's [server] sampling {draft['server'].get('sampling')}; equal: {sampling == draft['server'].get('sampling')}")
ok &= (not bad) and gen == make["after"] and links == make["links"]
print("inputs check:", "pass" if ok else "FAIL")
sys.exit(0 if ok else 1)
