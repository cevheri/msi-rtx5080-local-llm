"""Errors that end an llmctl command, each with the exit code it returns."""


class LlmctlError(Exception):
    exit_code = 1


class UsageError(LlmctlError):
    exit_code = 1


class RecipeError(LlmctlError):
    exit_code = 2


class PreflightError(LlmctlError):
    exit_code = 3


class StartError(LlmctlError):
    exit_code = 4


class DownloadError(LlmctlError):
    exit_code = 5
