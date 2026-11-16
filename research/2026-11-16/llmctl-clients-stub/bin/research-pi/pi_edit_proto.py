#!/usr/bin/env python3
"""Prototype of a safe Pi config editor for llmctl, exercised on copies in the scratch folder only.

models.json: replace only providers["llmctl"]; refuse files Python's json cannot read (Pi accepts // comments
and trailing commas, Python does not); check that everything but providers.llmctl is unchanged after the
round trip; validate the candidate with Pi's own loader (`pi --offline --list-models llmctl` in a temp agent
dir) before replacing the target, because one schema error makes Pi drop every provider in the file.
settings.json: set defaultProvider/defaultModel and modelThinkingLevels["llmctl/<id>"] only; refuse comments.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

R = "$SCRATCH/clients/research-pi"
PI = "$HOME/.pi/agent/bin/pi"
SCHEMA = "https://pi.dev/schemas/models.schema.json"


class Refusal(Exception):
    pass


def plan_models(path, entry):
    if os.path.exists(path):
        text = open(path).read()
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise Refusal(f"{path} is not plain JSON ({exc}); it may have // comments, which Pi accepts but llmctl will not rewrite")
        if not isinstance(data, dict) or not isinstance(data.get("providers", {}), dict):
            raise Refusal(f"{path} has no providers object")
    else:
        data = {"$schema": SCHEMA, "providers": {}}
    new = json.loads(json.dumps(data))
    new.setdefault("providers", {})["llmctl"] = entry
    text_out = json.dumps(new, indent=2, ensure_ascii=False) + "\n"
    back = json.loads(text_out)
    rest_old = {k: v for k, v in data.items() if k != "providers"}
    rest_new = {k: v for k, v in back.items() if k != "providers"}
    others_old = {k: v for k, v in data.get("providers", {}).items() if k != "llmctl"}
    others_new = {k: v for k, v in back["providers"].items() if k != "llmctl"}
    if rest_old != rest_new or others_old != others_new or back["providers"]["llmctl"] != entry:
        raise Refusal("round trip changed content outside providers.llmctl")
    return text_out


def validate_with_pi(models_text, model_id):
    tmp = tempfile.mkdtemp(prefix="pi-validate-", dir=f"{R}/tmp-validate")
    try:
        os.makedirs(f"{tmp}/agent")
        open(f"{tmp}/agent/models.json", "w").write(models_text)
        env = {"HOME": tmp, "PI_CODING_AGENT_DIR": f"{tmp}/agent", "TMPDIR": tmp,
               "PATH": "$HOME/.nvm/versions/node/v24.22.1/bin:/usr/bin:/bin",
               "PI_OFFLINE": "1", "PI_SKIP_VERSION_CHECK": "1", "PI_TELEMETRY": "0"}
        res = subprocess.run([PI, "--offline", "--list-models", "llmctl"], env=env, cwd=tmp,
                             capture_output=True, text=True, timeout=30)
        rows = [line.split() for line in res.stdout.splitlines()[1:] if line.strip()]
        ok = (res.returncode == 0 and "errors loading models.json" not in res.stderr
              and any(r[:2] == ["llmctl", model_id] for r in rows))
        return ok, res.returncode, res.stdout.strip(), res.stderr.strip()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def plan_settings(path, provider, model_id, thinking_level, set_default):
    if os.path.exists(path):
        text = open(path).read()
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise Refusal(f"{path} is not plain JSON ({exc}); Pi itself ignores such a file")
    else:
        data = {}
    new = json.loads(json.dumps(data))
    if set_default:
        new["defaultProvider"], new["defaultModel"] = provider, model_id
    if thinking_level:
        levels = new.get("modelThinkingLevels") or {}
        levels[f"{provider}/{model_id}"] = thinking_level
        new["modelThinkingLevels"] = levels
    return json.dumps(new, indent=2, ensure_ascii=False) + "\n", data, new


def apply(path, text):
    mode = os.stat(path).st_mode & 0o777 if os.path.exists(path) else 0o644
    tmp = path + ".llmctl-tmp"
    open(tmp, "w").write(text)
    os.chmod(tmp, mode)
    os.replace(tmp, path)


if __name__ == "__main__":
    case_dir = sys.argv[1]
    entry = json.load(open(sys.argv[2]))
    model_id = entry["models"][0]["id"]
    level = sys.argv[3] if len(sys.argv) > 3 else None
    os.makedirs(f"{R}/tmp-validate", exist_ok=True)
    models_path, settings_path = f"{case_dir}/models.json", f"{case_dir}/settings.json"
    try:
        new_models = plan_models(models_path, entry)
        ok, code, out, err = validate_with_pi(new_models, model_id)
        print(f"validate: ok={ok} exit={code} stdout={out!r} stderr={err[:200]!r}")
        if not ok:
            raise Refusal("Pi rejected the candidate models.json; the target was not touched")
        new_settings, old, new = plan_settings(settings_path, "llmctl", model_id, level, True)
        unchanged = os.path.exists(models_path) and open(models_path).read() == new_models
        if not unchanged:
            apply(models_path, new_models)
        print("models.json:", "already up to date" if unchanged else "written")
        unchanged = os.path.exists(settings_path) and open(settings_path).read() == new_settings
        if not unchanged:
            apply(settings_path, new_settings)
        print("settings.json:", "already up to date" if unchanged else "written",
              "| keys changed:", sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k)))
    except Refusal as exc:
        print("REFUSED:", exc)
        sys.exit(3)
