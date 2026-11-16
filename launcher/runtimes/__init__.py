"""The runtime adapters, in the order llmctl prefers them when a name exists in more than one."""

from launcher.runtimes import llamacpp, vllm

ADAPTERS = {"llamacpp": llamacpp, "vllm": vllm}
