# llmctl: start a measured model by name

Last verified: 2026-11-16.

`./llmctl` starts a model from this repository's recipes with one command, applies the rules from [README.md](../README.md#rules-for-every-setup-here) for you, and tells you how to reach the model from OpenCode.
It runs llama.cpp and vLLM in Docker, one model server at a time, on an NVIDIA GPU.
The recipes were measured on an MSI Raider 18 Max HX A2WI with an RTX 5080 Laptop GPU (16 GB) and 61 GiB of RAM; on another machine, read `./llmctl list` and write an override wherever yours differs.

## Quick start

```bash
./llmctl list                 # every recipe, its evidence, and what is downloaded
./llmctl pull qwen3.6         # asks, then downloads the model files, and the image if it is missing
./llmctl up qwen3.6           # checks the machine, starts the server, waits until it answers
./llmctl test                 # a reply, a tool call and a decode speed
./llmctl down                 # stops the server
```

`up` prints the OpenCode model to use, for example `llamacpp-docker/qwen3.6-35b-a3b`, which works with the provider config in [opencode.md](opencode.md#the-validated-config).
In the OpenCode screen, pick it with `/models` (or `ctrl+x m`); for a single prompt, pass it as `opencode run --model llamacpp-docker/qwen3.6-35b-a3b "..."`, because the interactive `opencode` command has no `-m` flag ([OpenCode models docs](https://opencode.ai/docs/models/)).
It needs Linux, Docker Engine, the NVIDIA Container Toolkit with a CDI spec ([host-setup.md](host-setup.md)), and Python 3.11 or newer; it uses only Python's standard library, so there is nothing to install.
Checked on the reference laptop on 2026-11-16: `up qwen3.6` was ready in 9 s and passed `test` and an OpenCode session, and `up gpt-oss --runtime vllm` was ready in 29 s with warm compile caches and passed `test` ([records](../research/2026-11-16/llmctl-e2e/)).

## Commands

| Command | What it does |
|---|---|
| `./llmctl list [--runtime R]` | Every recipe: ID, runtime, aliases, evidence, context, GPU need, RAM need, download size, whether the files and the image are present, and whether an override replaces it |
| `./llmctl show <model> [--runtime R]` | The effective recipe, its file and SHA-256, and the exact `docker run` command `up` would run |
| `./llmctl show <model> --as-override` | A complete copy of the recipe, marked as an estimate, to save in the override directory and edit |
| `./llmctl pull <model> [--runtime R] [--yes] [--hf-env-file F]` | Says how much it will download, asks, then pulls the image if it is missing and downloads the model files; without a terminal it needs `--yes` |
| `./llmctl up <model> [--runtime R] [--replace] [--dry-run]` | Runs the checks below, starts the container, waits until the server answers, and prints the endpoint, the served name, the context, the OpenCode model and the evidence |
| `./llmctl status` | The container llmctl started: its recipe, source file and SHA-256, state, port, health, memory use and the GPU memory in use |
| `./llmctl logs [-f] [--tail N]` | The server's log, the last 100 lines by default |
| `./llmctl test` | Three requests against the running server: a reply, one tool call, and a decode of up to 512 tokens |
| `./llmctl down [--remove]` | Stops the server with a 60-second grace period; the stopped container stays for its logs unless `--remove` |
| `./llmctl check` | Validates every shipped recipe and every override and lists each problem with its file |

A model is named by its recipe ID or an alias, such as `qwen3.6` for `qwen3.6-35b-a3b`.
When a name has a recipe for both runtimes, llama.cpp is used, because it is the main runtime here; `--runtime vllm` picks vLLM.
Run `./llmctl <command> --help` for every option.

## What llmctl enforces

These rules live in the code, and no recipe, runtime file or override can change them:

- Ports are published on 127.0.0.1 only, because Docker-published ports bypass ufw ([host-setup.md](host-setup.md)).
- `--memory` and `--memory-swap` are equal, from the recipe's `memory_gib`, so a container cannot swap.
- No restart policy, `--rm`, privileged mode or host network; the GPU is requested only as `--device nvidia.com/gpu=all`.
- One model server runs at a time; `up` refuses while another runs, and `up --replace` stops it first.
- llmctl never stops or removes a container it did not start, and never downloads during `up`.
- Every llama.cpp container gets `GGML_CUDA_DISABLE_GRAPHS=1`, the workaround for the CUDA-graph hang on RTX 50 laptop GPUs ([llama.cpp #27330](https://github.com/ggml-org/llama.cpp/issues/27330)); a runtime file that tries to set it is refused.
- A Hugging Face token comes from `HF_TOKEN` or from `--hf-env-file` (a file with an `HF_TOKEN=` line), and never appears in a command line, an output line, a label or `--dry-run`.
  It is sent only to huggingface.co: when a download redirects to another host, such as the storage CDN, the token is dropped.
- `pull` checks every llama.cpp file against the SHA-256 that Hugging Face publishes for it, and keeps a partial file when the network fails, so running `pull` again resumes it.

The containers carry labels that say what started them: `io.llmctl.managed`, `runtime`, `recipe`, `source`, `sha256` (of the recipe file as read), `evidence`, `runtime-source` and `runtime-sha256`.
`status` reads them back, so a running server always says which recipe file, and which version of it, started it.

## Recipes

Recipes live in [recipes/llamacpp/](../recipes/llamacpp/) and [recipes/vllm/](../recipes/vllm/), one TOML file per model, plus a `_runtime.toml` per runtime with what every model of that runtime shares.
The two runtimes never read each other's files, so a change to a llama.cpp file cannot change a vLLM command, and the reverse; the tests check this.

A llama.cpp recipe has these keys:

| Key | Meaning and bounds |
|---|---|
| `schema` | Always 1 |
| `id` | The file name without `.toml`, and the OpenCode model key |
| `name`, `aliases`, `doc` | A readable name; short names (lower-case letters, digits, `.`, `_`, `-`), unique within the runtime; the document section that shows the launch |
| `opencode`, `chat_only` | Whether [opencode.md](opencode.md) has an entry for it (false by default); whether the model has no tool calling (false by default) |
| `[model] repo`, `revision`, `files`, `size_bytes` | The Hugging Face repository, the 40-character commit it was measured with, the GGUF files (a split GGUF lists every part, and the first goes to `-m`), and their total size |
| `[container] memory_gib` | The container's memory cap, 4 to 56 GiB |
| `[server] context`, `args` | The context length (`-c`), 1024 to 1048576, and the other `llama-server` flags in order |
| `[needs] gpu_mib`, `ram_gib` | The GPU memory the server used at its measured peak, and the container's measured RAM peak (optional) |
| `[evidence] kind`, `summary` | `measured` or `estimate`, and one sentence on what was measured |
| `[evidence] date`, `hardware`, `record` | For a measured recipe: when, on what, and the record folder under `research/` |
| `[evidence] doc` | For an estimate: the document it comes from |

A vLLM recipe differs in three places: `[model]` has `include` and `exclude` patterns and a `served_name` instead of `files`; `args` are the flags after the model; and `[needs]` has only `ram_gib`, because the GPU need is the `--gpu-memory-utilization` share of the memory CUDA can use, which is vLLM's own start check.
A vLLM recipe without that flag uses `default_gpu_memory_utilization` from `recipes/vllm/_runtime.toml`, 0.92 for vLLM v0.33.0.

llmctl sets some flags itself, and a recipe that contains them is refused:

| Runtime | Flags llmctl owns | Flags refused because they would put a secret on the command line |
|---|---|---|
| llama.cpp | `-m`, `--model`, `-hf` and every `--hf-*` form, `-c`, `--ctx-size`, `-a`, `--alias`, `--host`, `--port` | `--api-key`, `--hf-token`, `-hft` |
| vLLM | the model argument, `--model`, `--revision`, `--served-model-name`, `--max-model-len`, `--host`, `--port` | `--api-key`, `--hf-token` |

A runtime file has `image`, `container_name`, `host_port`, `ready_path`, `start_timeout_s`, `opencode_provider`, and an `[env]` table; the llama.cpp file adds `served_name` (`local`), and the vLLM file adds named `volumes` and `default_gpu_memory_utilization`.
The llama.cpp runtime file may not set `GGML_CUDA_DISABLE_GRAPHS`, and the vLLM file sets `HF_MODULES_CACHE` to a writable path inside the container, because the model cache is mounted read-only there and a `--trust-remote-code` model copies its code into that cache.
Unknown keys are errors with a "did you mean" hint, because a misspelled key that was ignored would run a configuration you did not write.

## Overrides for another machine

Put your own files in `~/.config/llmctl/recipes/llamacpp/` and `~/.config/llmctl/recipes/vllm/` (`$XDG_CONFIG_HOME/llmctl/recipes` when that is set, or the directory in `LLMCTL_RECIPES_DIR`).

- A file with a shipped recipe's name replaces that recipe whole; nothing is merged across the two files, because half of one measurement beside half of another is a configuration nobody has run.
- A file with a new name adds a model.
- A `_runtime.toml` replaces that runtime's settings whole, for another image tag, port or cache volume; the rules above still hold, and when you change the port, change the OpenCode provider's `baseURL` to match.
- `./llmctl show qwen3.6 --as-override > ~/.config/llmctl/recipes/llamacpp/qwen3.6-35b-a3b.toml` writes a complete copy to start from, marked `kind = "estimate"` until you measure it; [recipes/examples/](../recipes/examples/llamacpp/qwen3.6-35b-a3b.toml) has one.
- An override that does not validate is listed by `check` and `list`, and `up` refuses that model until it is fixed or removed, instead of quietly running the shipped recipe.

On a GPU with less memory, lower `--moe-cache-mib` first and then raise `--n-cpu-moe`, as [llamacpp/README.md](../llamacpp/README.md#moe-expert-offload) explains; on a machine with less RAM, lower `--cache-ram` or `memory_gib`.

## When up refuses

`up` checks, in order: Docker, the CDI spec, the image, the model files, other containers and the port, free GPU memory, and available RAM.
Each refusal names what it found and exits with code 3.

| Message starts with | What to do |
|---|---|
| `Docker is not answering` | Start Docker: `sudo systemctl start docker` |
| `Docker refused the connection (permission denied` | Add yourself to the `docker` group and log in again |
| `no CDI spec for nvidia.com/gpu` | Install the NVIDIA Container Toolkit and generate the spec ([host-setup.md](host-setup.md)) |
| `the image ... is not on this machine` | `./llmctl pull <model>` |
| `the model files are not ready` | `./llmctl pull <model>`; for vLLM this also finishes a download that stopped partway |
| `another model server is running` | `./llmctl down`, or `./llmctl up <model> --replace` |
| `a container named ... was not started by llmctl` | Check that you do not need it, then `docker rm -f <name>` |
| `port 127.0.0.1:... is already in use` | Stop the program that holds the port |
| `... needs N MiB of free GPU memory` | Close the programs it lists, or write the override the message suggests: a smaller `--moe-cache-mib` for an offloaded MoE model, a smaller context for other llama.cpp models, a smaller `--gpu-memory-utilization` or context for vLLM |
| `... may use up to N GiB of RAM` | Close programs, or write an override with a smaller `memory_gib` or `--cache-ram` |
| `nvidia-smi was not found` or `nvidia-smi failed` | Install or load the NVIDIA driver |
| `another llmctl command is running` | Wait for it to finish |

Exit codes: 0 success; 1 a usage error, or a download you declined; 2 an invalid recipe or override; 3 a refused check; 4 a server that failed to start or become ready, or a failed `test`; 5 a failed download or checksum; 130 interrupted.

## Adding or changing a recipe

1. Measure the launch with the method in [llamacpp/benchmarking.md](../llamacpp/benchmarking.md) or [vllm/benchmarking.md](../vllm/benchmarking.md), and keep the records under `research/`.
2. Write the launch into its section of [llamacpp/models.md](../llamacpp/models.md) or [vllm/models.md](../vllm/models.md), and the recipe file into `recipes/`; set `opencode = true` only together with an entry in [opencode.md](opencode.md#the-validated-config).
3. Run `python3 -m launcher.tests.regen_golden`, read the diff of `launcher/tests/golden/`, and run `python3 -m unittest discover -s launcher/tests -t .`.

The tests check that each recipe matches its document's launch and its OpenCode entry, so a recipe, its document and the OpenCode config change together or the tests fail.
They run without Docker, a GPU or the network.
