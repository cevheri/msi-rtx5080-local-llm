"""The clients llmctl configures: OpenCode, Hermes Agent and Pi.

A client is installed when its binary is on the host's PATH and its config root exists; LLMCTL_CLIENTS picks which
installed clients llmctl touches (docs/superpowers/specs/2026-11-16-llmctl-clients-design.md, section D9).
"""

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from launcher import hermes_config, opencode_config, pi_config
from launcher.errors import ClientError, ClientWriteError, LlmctlError, UsageError
from launcher.files import FileChange, backup_once, backup_path, load_json_object, make_dirs, write_atomic
from launcher.host import Host
from launcher.model import Target

__all__ = ["NAMES", "LABELS", "ClientState", "ClientPlan", "ApplyResult", "FileChange", "Target", "client_states",
           "clients_line", "require_listed", "llmctl_providers", "plan", "dry_run_lines", "apply",
           "failure_message", "thinking_texts", "up_lines", "report", "down_lines"]

NAMES = ("opencode", "hermes", "pi")
LABELS = {"opencode": "OpenCode", "hermes": "Hermes", "pi": "Pi"}
ENV_VAR = "LLMCTL_CLIENTS"
LEFT_OUT = f"left out by {ENV_VAR}"


def _root(name: str, env: Mapping[str, str]) -> Path:
    roots: dict[str, Callable[[Mapping[str, str]], Path]] = {
        "opencode": opencode_config.config_root,
        "hermes": hermes_config.hermes_root,
        "pi": pi_config.agent_dir,
    }
    return roots[name](env)


@dataclass(frozen=True)
class ClientState:
    name: str
    label: str
    enabled: bool
    installed: bool
    reason: str | None  # None when active
    root: Path | None

    @property
    def active(self) -> bool:
        return self.enabled and self.installed


def _listed(env: Mapping[str, str]) -> list[str] | None:
    """The names LLMCTL_CLIENTS lists, or None when it is unset."""
    if ENV_VAR not in env:
        return None
    names = [item.strip() for item in env[ENV_VAR].split(",")]
    names = [item for item in names if item]
    for name in names:
        if name not in NAMES:
            raise UsageError(f"{ENV_VAR}: unknown client '{name}'; use opencode, hermes or pi, comma-separated, "
                             "or an empty value for none")
    return names


def client_states(host: Host) -> list[ClientState]:
    listed = _listed(host.env)
    states = []
    for name in NAMES:
        label = LABELS[name]
        if listed is not None and name not in listed:
            states.append(ClientState(name, label, False, False, LEFT_OUT, None))
            continue
        if host.which(name) is None:
            states.append(ClientState(name, label, True, False, f"no {name} on PATH", None))
            continue
        root = _root(name, host.env)
        if not root.is_dir():
            states.append(ClientState(name, label, True, False, f"no {root}", root))
            continue
        states.append(ClientState(name, label, True, True, None, root))
    return states


def clients_line(host: Host, states: list[ClientState]) -> str:
    if ENV_VAR in host.env and not _listed(host.env):
        return f"clients: none ({ENV_VAR} is empty)"
    active = [state.name for state in states if state.active]
    line = "clients: " + (", ".join(active) if active else "none")
    for state in states:
        if state.active:
            continue
        if not state.enabled:
            line += f" ({state.name}: {LEFT_OUT})"
        else:
            line += f" ({state.name}: not installed, {state.reason})"
    return line


def require_listed(host: Host, states: list[ClientState]) -> None:
    """Refuse when LLMCTL_CLIENTS names a client that is not installed here."""
    if ENV_VAR not in host.env:
        return None
    for state in states:
        if state.enabled and not state.installed:
            raise ClientError(f"{ENV_VAR} names {state.name}, but {state.label} is not installed here: {state.reason}")
    return None


def llmctl_providers(catalog) -> frozenset[str]:
    """The OpenCode provider of every runtime whose runtime file loaded."""
    return frozenset(entry.runtime.opencode_provider for entry in catalog.runtimes.values()
                     if entry.runtime is not None)


@dataclass(frozen=True)
class ClientPlan:
    """What up would change in the clients, planned before the server starts (spec D8)."""

    states: tuple[ClientState, ...]
    changes: tuple[FileChange, ...]
    hermes_use: bool
    pi_active: bool


@dataclass(frozen=True)
class ApplyResult:
    lines: tuple[str, ...]
    failures: tuple[str, ...]
    changed: frozenset[str]
    failed: frozenset[str]
    default_failed: frozenset[str] = frozenset()  # written, but making it the default failed (hermes profile use)


def _pi_candidate(host: Host, change: FileChange | None) -> str:
    """The models.json text Pi checks: the planned text, or the file as it is when the entry is unchanged."""
    if change is not None:
        return change.new_text
    path = pi_config.models_path(host.env)
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ClientError(f"cannot read {path}: {exc}") from exc


def _validate_pi(host: Host, target: Target, change: FileChange | None) -> str | None:
    """Let Pi check the candidate models.json."""
    return pi_config.validate_models(host, _pi_candidate(host, change), target, pi_config.models_path(host.env))


def plan(host: Host, catalog, target: Target, *, set_default: bool, check: bool) -> ClientPlan:
    """Plan every client change and raise any refusal; nothing is written, and only `check` runs pi."""
    states = client_states(host)
    require_listed(host, states)
    providers = llmctl_providers(catalog)
    active = {state.name for state in states if state.active}
    changes: list[FileChange | None] = []
    hermes_use = False
    if "opencode" in active:
        changes.append(opencode_config.plan_model_line(host.env, target, providers, set_default=set_default))
    if "hermes" in active:
        changes.append(hermes_config.plan_profile(host.env, target))
        hermes_use = set_default and hermes_config.sticky_profile(host.env) != hermes_config.PROFILE
    if "pi" in active:
        models = pi_config.plan_models(host.env, target)
        if check:
            _validate_pi(host, target, models)
        changes.append(models)
        changes.append(pi_config.plan_settings(host.env, target, set_default=set_default))
    return ClientPlan(tuple(states), tuple(change for change in changes if change is not None), hermes_use,
                      "pi" in active)


def dry_run_lines(planned: ClientPlan) -> list[str]:
    lines = []
    for change in planned.changes:
        backup = backup_path(change.path)
        if change.backup and change.old_text is not None and not os.path.lexists(backup):
            lines.append(f"would back up {change.path} to {backup}")
        lines.append(change.preview)
    if planned.hermes_use:
        lines.append("would run hermes profile use llmctl")
    if planned.pi_active:
        lines.append("dry run: nothing was written, and Pi has not checked the entry")
    else:
        lines.append("dry run: nothing was written")
    return lines


RELOAD_NOTE = "note: run `opencode reload`, or restart OpenCode, so an OpenCode that is already open sees it"


class _Applier:
    def __init__(self) -> None:
        self.backups: list[str] = []
        self.writes: list[str] = []
        self.after: list[str] = []
        self.failures: list[str] = []
        self.changed: set[str] = set()
        self.failed: set[str] = set()
        self.default_failed: set[str] = set()

    def write(self, change: FileChange) -> None:
        try:
            make_dirs(change.dirs)
            if change.backup and change.old_text is not None:
                backup = backup_once(change.path, change.old_text)
                if backup is not None:
                    self.backups.append(f"backed up {change.path} to {backup} before the first change")
            write_atomic(change.path, change.new_text, new_mode=change.mode)
        except OSError as exc:
            raise ClientWriteError(f"{change.path}: {exc.strerror or exc}") from exc
        except UnicodeError as exc:
            raise ClientWriteError(f"{change.path}: {exc}") from exc

    def opencode(self, host: Host, target: Target, providers: frozenset[str], set_default: bool) -> None:
        change = opencode_config.plan_model_line(host.env, target, providers, set_default=set_default)
        if change is None:
            return
        self.write(change)
        line = f"wrote {change.path}: {change.summary}"
        if host.run(["opencode", "reload"]).returncode == 0:
            self.writes.append(line + "; OpenCode reloaded its configuration")
        else:
            self.writes.append(line)
            self.after.append(RELOAD_NOTE)

    def hermes(self, host: Host, target: Target, set_default: bool) -> None:
        change = hermes_config.plan_profile(host.env, target)
        if change is not None:
            self.write(change)
            self.writes.append(f"wrote {change.path}: {change.summary}")
            self.changed.add("hermes")
        if set_default and hermes_config.sticky_profile(host.env) != hermes_config.PROFILE:
            # The profile file is written, so Hermes keeps its block; only the sticky default failed.
            try:
                self.after.extend(hermes_config.use_profile(host))
            except ClientWriteError as exc:
                self.failures.append(str(exc))
                self.default_failed.add("hermes")

    def pi(self, host: Host, target: Target, set_default: bool) -> None:
        # Pi checks the planned models.json before llmctl takes Pi's settings lock, so the lock is not held across
        # a pi run; inside the lock both files are planned again, and a models.json text that differs from the
        # checked one is checked again, before either file is written. A held lock or a refusal leaves both as
        # they were.
        models_path = pi_config.models_path(host.env)
        checked = _pi_candidate(host, pi_config.plan_models(host.env, target))
        warning = pi_config.validate_models(host, checked, target, models_path)
        with pi_config.settings_lock(host, pi_config.settings_path(host.env)):
            models = pi_config.plan_models(host.env, target)
            candidate = _pi_candidate(host, models)
            if candidate != checked:
                warning = pi_config.validate_models(host, candidate, target, models_path)
            if warning is not None:
                self.after.append(warning)
            settings = pi_config.plan_settings(host.env, target, set_default=set_default)
            if models is not None:
                self.write(models)
            if settings is not None:
                try:
                    self.write(settings)
                except BaseException as exc:
                    # Any exception, an interrupt included, may come after models.json was replaced.
                    if models is None or _holds(settings):
                        raise
                    problem = _put_back(models)
                    if isinstance(exc, ClientWriteError):
                        raise ClientWriteError(_put_back_message(models, exc, problem)) from exc
                    if problem is not None:
                        host.err.write(f"error: {_put_back_message(models, None, problem)}\n")
                    raise
            if models is not None:
                self.writes.append(f"wrote {models.path}: {models.summary}")
                self.changed.add("pi")
            if settings is not None:
                self.writes.append(f"wrote {settings.path}: {settings.summary}")


def _holds(change: FileChange) -> bool:
    """True when the file already holds the change's new text, so its write landed before the failure."""
    try:
        with open(change.path, encoding="utf-8", newline="") as handle:
            return handle.read() == change.new_text
    except (OSError, UnicodeError):
        return False


def _put_back(models: FileChange) -> str | None:
    """Undo the models.json write after settings.json failed, so Pi's two files keep naming the same model.

    Returns None when models.json is as it was, else why putting it back failed."""
    try:
        if models.old_text is None:
            os.remove(os.path.realpath(models.path))  # the file write_atomic created, not a symlink to it
        else:
            write_atomic(models.path, models.old_text, new_mode=models.mode)
    except (OSError, UnicodeError) as exc:
        reason = exc.strerror if isinstance(exc, OSError) and exc.strerror else exc
        return f"{models.path}: {reason}"
    return None


def _put_back_message(models: FileChange, failure: ClientWriteError | None, problem: str | None) -> str:
    if problem is None:
        return f"{failure}; {models.path} was put back as it was"
    head = f"{failure}; putting" if failure is not None else "putting"
    return (f"{head} {models.path} back failed too ({problem}), so models.json and "
            "settings.json now disagree about the llmctl model; fix them by hand or run up again")


def apply(host: Host, catalog, target: Target, *, set_default: bool) -> ApplyResult:
    """Re-read and write each active client on its own after readiness, collecting failures (spec D8)."""
    states = client_states(host)
    providers = llmctl_providers(catalog)
    applier = _Applier()
    steps: dict[str, Callable[[], None]] = {
        "opencode": lambda: applier.opencode(host, target, providers, set_default),
        "hermes": lambda: applier.hermes(host, target, set_default),
        "pi": lambda: applier.pi(host, target, set_default),
    }
    for state in states:
        if not state.active:
            continue
        try:
            steps[state.name]()
        except (LlmctlError, OSError) as exc:
            applier.failures.append(str(exc))
            applier.failed.add(state.name)
    return ApplyResult(tuple(applier.backups + applier.writes + applier.after), tuple(applier.failures),
                       frozenset(applier.changed), frozenset(applier.failed), frozenset(applier.default_failed))


def failure_message(failures: Sequence[str]) -> str:
    return f"the model is running, but {len(failures)} client file(s) were not updated: {'; '.join(failures)}"


def thinking_texts(target: Target) -> tuple[str, str, str]:
    """How OpenCode, Hermes and Pi choose this recipe's thinking (spec D8 line templates)."""
    meta = target.recipe.clients
    if meta.thinking == "toggle" and meta.default == "on":
        return "on; add #none to turn it off", "on, as the server sets it", "on; --thinking off turns it off"
    if meta.thinking == "toggle":
        if meta.medium_kwargs:
            return ("off; add #thinking to the model to turn it on, #medium for medium effort",
                    "off, as the server sets it",
                    "off; --thinking medium for medium effort, --thinking high for full thinking")
        return ("off; add #thinking to the model to turn it on", "off, as the server sets it",
                "off; --thinking medium turns it on")
    if meta.thinking == "effort":
        efforts = ", ".join(meta.efforts)
        level_map = pi_config.provider_entry(target)["models"][0]["thinkingLevelMap"]
        levels = ", ".join(level for level in pi_config.PI_LEVELS if level_map.get(level) is not None)
        return (f"medium effort; add #<level> for another ({efforts})",
                f"medium effort; hermes --reasoning <level> for another ({efforts})",
                f"medium effort; --thinking <level> for another ({levels})")
    if meta.thinking == "always":
        return ("always on, at the effort the server sets",) * 3
    if meta.thinking == "none":
        return ("none",) * 3
    raise ValueError(f"unknown thinking type {meta.thinking!r}")


def _default_text(is_default: Callable[[], bool], up_command: str) -> str:
    """`the default`, the hint that makes it one, or why llmctl cannot tell after the server started."""
    try:
        if is_default():
            return "the default"
    except ClientError as exc:
        return f"default unknown ({exc})"
    return f"not the default ({up_command} --set-default makes it one)"


def _notes(*probes: Callable[[], str | None]) -> list[str]:
    lines = []
    for probe in probes:
        try:
            line = probe()
        except ClientError as exc:
            line = f"note: {exc}"
        if line is not None:
            lines.append(line)
    return lines


def _opencode_lines(host: Host, target: Target, up_command: str, thinking: str) -> list[str]:
    lines = []
    if target.recipe.opencode and opencode_config.has_entry(host.env, target):
        model = f"{target.runtime.opencode_provider}/{target.recipe.id}"
        state = _default_text(lambda: opencode_config.current_default(host.env)[1] == model, up_command)
        lines += [
            f"OpenCode: {model}, {state}",
            "  in the OpenCode screen: type /models (or press ctrl+x m) and pick it",
            f'  for a single prompt: opencode run --model {model} "your prompt"',
            f"  thinking: {thinking}",
        ]
    return lines + opencode_config.opencode_notes(host.env, target)


def _hermes_lines(host: Host, target: Target, up_command: str, thinking: str, *, use_failed: bool = False) -> list[str]:
    env = host.env
    if use_failed:
        state = "not the default (hermes profile use llmctl failed, see the error below)"
    else:
        state = _default_text(lambda: hermes_config.sticky_profile(env) == hermes_config.PROFILE, up_command)
    lines = [f"Hermes: profile {hermes_config.PROFILE}, {state}", f"  hermes -p {hermes_config.PROFILE}",
             f"  thinking: {thinking}"]
    if target.recipe.chat_only:
        lines.append("  chat only: this profile has no tools")
    return lines + _notes(lambda: hermes_config.gateway_warning(env),
                          lambda: hermes_config.reasoning_effort_warning(env),
                          lambda: hermes_config.hermes_home_note(env))


def _pi_lines(host: Host, target: Target, up_command: str, thinking: str) -> list[str]:
    env = host.env
    model = f"{pi_config.PROVIDER}/{target.served}"
    state = _default_text(lambda: pi_config.pi_default(env) == (pi_config.PROVIDER, target.served), up_command)
    lines = [f"Pi: {model}, {state}", f"  pi --model {model}" + (" --no-tools" if target.recipe.chat_only else ""),
             f"  thinking: {thinking}"]
    if target.recipe.chat_only:
        lines.append("  chat only: Pi has no per-model tool switch, so pass --no-tools")
    return lines + _notes(lambda: pi_config.enabled_models_note(env, target))


def up_lines(host: Host, target: Target, states: Sequence[ClientState], *, up_command: str,
             changed: frozenset[str], failed: frozenset[str] = frozenset(),
             default_failed: frozenset[str] = frozenset()) -> list[str]:
    """What up prints about each active client once the server is ready (spec D8); a failed client gets one line,
    and a client whose file was written but whose default failed keeps its block."""
    texts = dict(zip(NAMES, thinking_texts(target)))
    sections: dict[str, Callable[[Host, Target, str, str], list[str]]] = {
        "opencode": _opencode_lines, "hermes": _hermes_lines, "pi": _pi_lines}
    lines = []
    for state in states:
        if not state.active:
            continue
        if state.name in failed:
            lines.append(f"{state.label}: not updated, see the error below")
        elif state.name == "hermes" and "hermes" in default_failed:
            lines += _hermes_lines(host, target, up_command, texts["hermes"], use_failed=True)
        else:
            lines += sections[state.name](host, target, up_command, texts[state.name])
    if (changed - failed) & {"hermes", "pi"}:
        named = " and ".join(LABELS[name] for name in ("hermes", "pi") if name not in failed)
        lines.append(f"note: {named} sessions that were open before keep the model they started with; "
                     "start a new session")
    return lines


NOT_RUNNING = "points at llmctl, but no model is running"


def _problem(path: Path, exc: ClientError) -> str:
    """`cannot read <path>: <reason>`, without repeating a path the message starts with."""
    message = str(exc)
    if message.startswith("cannot read "):
        return message
    for prefix in (f"{path}: ", f"{path} "):
        if message.startswith(prefix):
            message = message[len(prefix):]
            break
    return f"cannot read {path}: {message}"


def _show_model(model: object) -> str:
    return model if isinstance(model, str) else json.dumps(model)


def _opencode_report(host: Host, catalog, target: Target | None, running_known: bool):
    """(text, path, warnings) of OpenCode's status line (spec D8)."""
    path = opencode_config.config_path(host.env)
    try:
        text, data = load_json_object(path, "OpenCode")
    except ClientError as exc:
        return _problem(path, exc), None, []
    if text is None:
        return f"no config at {path}", None, []
    model = data.get("model")
    if model is None:
        return "no default model", path, []
    shown = _show_model(model)
    llmctl = opencode_config.points_at_llmctl(model, llmctl_providers(catalog))
    if target is None:
        warnings = [f"warning: OpenCode's default {shown} {NOT_RUNNING}"] if llmctl and running_known else []
        return f"default {shown}", path, warnings
    if model == f"{target.runtime.opencode_provider}/{target.recipe.id}":
        return f"default {shown}, the running model", path, []
    warnings = [f"warning: OpenCode's default {shown} is not the running model {target.recipe.id}"] if llmctl else []
    return f"default {shown}", path, warnings


def _hermes_report(host: Host, target: Target | None, running_known: bool):
    env = host.env
    path = hermes_config.profile_config(env)
    try:
        block = hermes_config.profile_block(env)
        sticky = hermes_config.sticky_profile(env)
    except ClientError as exc:
        return _problem(path, exc), None, []
    warnings = []
    if sticky == hermes_config.PROFILE and target is None and running_known:
        warnings.append(f"warning: Hermes' default profile {hermes_config.PROFILE} {NOT_RUNNING}")
    probes = [lambda: hermes_config.gateway_warning(env)]
    if block is not None:
        probes.append(lambda: hermes_config.reasoning_effort_warning(env))
    probes.append(lambda: hermes_config.hermes_home_note(env))
    warnings += _notes(*probes)
    if block is None:
        return f"profile {hermes_config.PROFILE} not created yet (the next ./llmctl up creates it)", None, warnings
    default = f"default profile: {sticky}"
    if target is not None and block == hermes_config.render_block(target):
        return (f"profile {hermes_config.PROFILE} for {target.recipe.id} on {target.label}, the running model; "
                f"{default}", path, warnings)
    described = hermes_config.block_description(block) or "an unknown model"
    middle = ", not the running model (run ./llmctl up again)" if target is not None else ""
    return f"profile {hermes_config.PROFILE} for {described}{middle}; {default}", path, warnings


def _pi_report(host: Host, target: Target | None, running_known: bool):
    env = host.env
    directory = pi_config.agent_dir(env)
    try:
        entry = pi_config.installed_entry(env)
        provider, model = pi_config.pi_default(env)
    except ClientError as exc:
        return _problem(directory, exc), None, []
    warnings = []
    if provider == pi_config.PROVIDER and target is None and running_known:
        warnings.append(f"warning: Pi's default {pi_config.PROVIDER}/{model or '?'} {NOT_RUNNING}")
    if target is not None:
        warnings += _notes(lambda: pi_config.enabled_models_note(env, target))
    if entry is None:
        return f"no {pi_config.PROVIDER} provider yet (the next ./llmctl up writes it)", directory, warnings
    models = entry.get("models")
    first = models[0] if isinstance(models, list) and models and isinstance(models[0], dict) else {}
    served = first.get("id") if isinstance(first.get("id"), str) else None
    default = "not set" if provider is None and model is None else f"{provider or '?'}/{model or '?'}"
    try:
        level = pi_config.stored_level(env, served) if served is not None else None
    except ClientError as exc:
        return _problem(directory, exc), None, warnings
    tail = f"default {default}, thinking {level or 'not set'}"
    if target is not None and entry == pi_config.provider_entry(target):
        return (f"{pi_config.PROVIDER}/{target.served} for {target.recipe.id} on {target.label}, the running model; "
                f"{tail}", directory, warnings)
    described = pi_config.entry_description(entry) or "an unknown model"
    middle = ", not the running model (run ./llmctl up again)" if target is not None else ""
    return f"{pi_config.PROVIDER}/{served or '?'} for {described}{middle}; {tail}", directory, warnings


def report(host: Host, catalog, target: Target | None, *, with_paths: bool, running_known: bool = True) -> list[str]:
    """The clients: section of status and list (spec D8, D2, L5); None as target means no model is running."""
    sections: dict[str, Callable[[], tuple[str, Path | None, list[str]]]] = {
        "opencode": lambda: _opencode_report(host, catalog, target, running_known),
        "hermes": lambda: _hermes_report(host, target, running_known),
        "pi": lambda: _pi_report(host, target, running_known),
    }
    lines = ["clients:"]
    warnings: list[str] = []
    for state in client_states(host):
        if not state.enabled:
            text, path = f"disabled by {ENV_VAR}", None
        elif not state.installed:
            text, path = f"not installed ({state.reason})", None
        else:
            text, path, found = sections[state.name]()
            warnings += found
        if with_paths and path is not None:
            text += f" ({path})"
        lines.append(f"  {state.label}: {text}")
    return lines + warnings


SWITCH_BACK = {"opencode": "OpenCode /models", "hermes": "hermes profile use default", "pi": "Pi /model"}


def _points_at_llmctl(name: str, host: Host, providers: frozenset[str]) -> bool:
    """True when the client's default points at llmctl; a file llmctl cannot read counts as not pointing."""
    env = host.env
    try:
        if name == "opencode":
            _, data = load_json_object(opencode_config.config_path(env), "OpenCode")
            return opencode_config.points_at_llmctl(data.get("model"), providers)
        if name == "hermes":
            return hermes_config.sticky_profile(env) == hermes_config.PROFILE
        return pi_config.pi_default(env)[0] == pi_config.PROVIDER
    except ClientError:
        return False


def _join(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def down_lines(host: Host, catalog, recipe_id: str) -> list[str]:
    """What down prints about the clients whose default still points at llmctl (spec D8, D6 rule 4)."""
    providers = llmctl_providers(catalog)
    pointing = [state for state in client_states(host)
                if state.active and _points_at_llmctl(state.name, host, providers)]
    if not pointing:
        return []
    names = _join([state.label for state in pointing])
    verb = "points" if len(pointing) == 1 else "point"
    gets = "gets" if len(pointing) == 1 else "get"
    return [f"{names} still {verb} at {recipe_id} and {gets} no answer until the next ./llmctl up",
            "  switch back: " + "; ".join(SWITCH_BACK[state.name] for state in pointing)]
