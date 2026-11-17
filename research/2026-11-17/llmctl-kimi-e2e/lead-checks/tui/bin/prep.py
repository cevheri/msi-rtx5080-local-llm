"""Copy the real ~/.kimi-code/config.toml into the private Kimi home by a rename, as llmctl writes it,
with llmctl's default edit (kimi_config.config_text(..., set_default=True)) when its default_model does not
name the running recipe's alias, as launcher/e2e/clients.py _Run.kimi_env does.
usage: python3 -I prep.py <repo> <recipe id> <runtime> <private kimi home>"""
import os, sys
from pathlib import Path
repo, recipe_id, runtime, home = sys.argv[1:5]
sys.path.insert(0, repo)
from launcher import kimi_config
from launcher.catalog import load_catalog, override_dir
from launcher.e2e.clients import _target
target = _target(load_catalog(Path(repo) / "recipes", override_dir(os.environ)), recipe_id, runtime)
real = kimi_config.config_path(os.environ)
with open(real, encoding="utf-8", newline="") as handle:
    text = handle.read()
copy = Path(home) / "config.toml"
edited = kimi_config.parse_state(copy, text).default != kimi_config.alias(target)
if edited:
    text = kimi_config.config_text(copy, text, target, set_default=True)
new = Path(home) / "config.toml.new"
fd = os.open(new, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
    stream.write(text)
os.replace(new, copy)
print(f"copied for {kimi_config.alias(target)}; default edit: {'yes' if edited else 'no'}")
