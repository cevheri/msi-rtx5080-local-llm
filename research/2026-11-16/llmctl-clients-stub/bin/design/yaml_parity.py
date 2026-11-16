"""Parse every generated Hermes config with Hermes' own YAML policy (hermes_yaml.safe_load: ruamel, YAML 1.1,
duplicate keys rejected) and compare each block key with the JSON value llmctl intends. Also checks that a
duplicate owned key outside the block is rejected. usage: <hermes python> -I -B yaml_parity.py <checkout> <gen dir>"""
import json, re, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1]); sys.path.insert(1, sys.argv[3])
import hermes_yaml
gen = Path(sys.argv[2]); ok = 0; bad = []
for cfg in sorted(gen.glob("*/*/hermes-config.yaml")):
    text = cfg.read_text()
    parsed = hermes_yaml.safe_load(text)
    intended = {m.group(1): json.loads(m.group(2)) for m in re.finditer(r"(?m)^([a-z_]+): (\{.*\})$", text)}
    intended["_config_version"] = 53
    if parsed == intended: ok += 1
    else: bad.append(str(cfg))
dup = (gen / "llamacpp/qwen3.6-35b-a3b/hermes-config.yaml").read_text() + 'model: {"provider": "x"}\n'
try:
    hermes_yaml.safe_load(dup); dup_result = "accepted (unexpected)"
except Exception as exc:
    dup_result = f"rejected: {type(exc).__name__}: {str(exc).splitlines()[0]}"
print(f"{ok} of {ok + len(bad)} generated configs parse to exactly the intended values; mismatches: {bad}")
print(f"duplicate top-level key: {dup_result}")
