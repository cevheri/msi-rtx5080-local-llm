"""Parse each writer fixture with tomllib and compare llmctl's tables with the generator's values.

usage: python3 -I check_fixtures.py <writer dir> <entries/index.json>
"""
import json, re, sys, tomllib
from pathlib import Path

W = Path(sys.argv[1])
INDEX = json.loads(Path(sys.argv[2]).read_text())
GEMMA = next(r for r in INDEX if r["runtime"] == "llamacpp" and r["id"] == "gemma-4-26b-a4b")
out = {}
for path in sorted(W.glob("*.toml")):
    text = path.read_text()
    data = tomllib.loads(text)
    lines = text.splitlines()
    headers = [(i + 1, l) for i, l in enumerate(lines) if l.startswith("[")]
    prov = data.get("providers", {}).get("llmctl")
    model = data.get("models", {}).get("llmctl/local")
    out[path.name] = {
        "bytes": len(text.encode()),
        "comment_lines": sum(1 for l in lines if l.lstrip().startswith("#")),
        "blank_lines": sum(1 for l in lines if not l.strip()),
        "top_level_keys": sorted(k for k, v in data.items() if not isinstance(v, dict)),
        "tables": sorted(k for k, v in data.items() if isinstance(v, dict)),
        "default_model": data.get("default_model"),
        "thinking": data.get("thinking"),
        "providers": sorted(data.get("providers", {})),
        "models": sorted(data.get("models", {})),
        "llmctl_provider_equal": prov == GEMMA["provider"],
        "llmctl_model_equal": model == GEMMA["model"],
        "headers": headers,
        "array_style": sorted(set(re.findall(r"= (\[ ?\"tool_use\" ?\])", text))),
    }
print(json.dumps(out, indent=1))
