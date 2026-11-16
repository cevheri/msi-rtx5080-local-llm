"""llmctl: start a measured local LLM recipe by name.

This module must stay importable on old Python versions, because the entry
script imports it to say which version is needed before anything else runs.
"""

MIN_PYTHON = (3, 11)


def python_problem(version_info):
    """Return a message when this Python is too old for llmctl, else None."""
    if tuple(version_info[:2]) < MIN_PYTHON:
        found = ".".join(str(part) for part in version_info[:3])
        return "llmctl needs Python 3.11 or newer; found " + found
    return None
