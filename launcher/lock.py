"""One llmctl command at a time may start, stop or download models."""

import fcntl
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from launcher.errors import PreflightError


def lock_path(env) -> Path:
    if env.get("XDG_RUNTIME_DIR"):
        return Path(env["XDG_RUNTIME_DIR"]) / "llmctl.lock"
    return Path(tempfile.gettempdir()) / f"llmctl-{os.getuid()}.lock"


@contextmanager
def hold_lock(env):
    path = lock_path(env)
    handle = open(path, "a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise PreflightError(f"another llmctl command is running (lock {path}); wait for it to finish") from None
    try:
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()
