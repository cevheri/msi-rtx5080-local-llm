# Kimi Code CLI 2.5.3 against the local llama.cpp server: hands-on check

Date: 2026-11-17.
Binary: `$HOME/.kimi-code/bin/kimi`, `kimi version 2.5.3`.
Server: llama.cpp `server-cuda-b12407`, recipe gemma-4-26b-a4b, served name `local`, context 131072, `--reasoning off` (from `./llmctl status` and `./llmctl show gemma-4-26b-a4b`); client output cap 16384 from `recipes/llamacpp/gemma-4-26b-a4b.toml` `[clients] output = 16384` (`llmctl show` does not print it).

## Result

Kimi Code CLI works with the local model in print mode: plain reply, Write plus Bash tool calls, and a shell command all succeeded on the first attempt (exit 0).
The thinking switch changes the request (`reasoning_effort`), but this server produced no reasoning in any run, so thinking cannot be turned on from Kimi for this recipe.
The only network connects were to the recording proxy on 127.0.0.1:47303; nothing under the owner's `~/.kimi-code` changed.

## Setup

- Private home: `$SCRATCH/kimi/handson/home` (HOME), `KIMI_CODE_HOME=$H/home/.kimi-code`, `KIMI_CODE_CACHE_DIR=$H/home/.cache/kimi-code`, all XDG dirs under it.
- Off switches: `telemetry = false` (config.toml), `KIMI_DISABLE_TELEMETRY=1`, `KIMI_CODE_NO_AUTO_UPDATE=1`, `KIMI_CLI_NO_AUTO_UPDATE=1`, `KIMI_CODE_REGION_MARKER=off`, `[upgrade] auto_install = false` (tui.toml).
- Isolation: `heavy bwrap` with `--unshare-net` (no route off the box), `--tmpfs $HOME` (owner's home invisible), the kimi binary bound read-only, `--clearenv`, and only the private home, the run's work folder, the run's output folder and the socket folder writable.
- Path to the model: kimi -> 127.0.0.1:47303 inside the sandbox (socat) -> unix socket `$H/net/llm.sock` -> socat -> recording proxy `tools/proxy.py` on host 127.0.0.1:47303 -> llama.cpp 127.0.0.1:8080.
- Scripts: `tools/proxy.py` (python3 -I, stdlib), `tools/run.sh` (sandbox and one attempt), `tools/inner.sh` (in-sandbox bridge, `strace -f -s 256 -e trace=connect,sendto`, `timeout 600`).
- `kimi doctor config` on the config: "OK config.toml ...
  All checked config files are valid." (`runs/doctor.txt`).

Config used (`config.used.toml`, no secrets; the api_key is a placeholder):

```toml
default_model = "llmctl/local"
telemetry = false

[thinking]
enabled = false

[providers.llmctl]
type = "openai"
base_url = "http://127.0.0.1:47303/v1"
api_key = "llmctl-local"
model_source = "static"

[models."llmctl/local"]
provider = "llmctl"
model = "local"
display_name = "gemma-4-26b-a4b (llmctl)"
max_context_size = 131072
max_output_size = 16384
capabilities = ["tool_use", "thinking"]
```

The thinking variants are in `configs/`: `d-on-plain` (enabled = true), `d-on-effort` (enabled = true, `support_efforts = ["low","medium","high"]`, `default_effort = "medium"`), `d-off-nooff` (enabled = false, same efforts, no `off_effort`), `d-off-effort` (as before plus `off_effort = "none"`).
Kimi did not rewrite config.toml in any run (`cmp` with the copy after run a-1: same; size stayed 477 bytes).

## Commands

```
tools/run.sh a-1 -p "Reply with exactly OK." --output-format stream-json
tools/run.sh b-1 -p "Create hello.py that prints hello, run it with python3, and tell me its output." --output-format stream-json
tools/run.sh c-1 -p "Use your shell tool to run exactly: cat /proc/self/cgroup. Then reply with its output." --output-format stream-json
# for each variant v: cp configs/$v.toml home/.kimi-code/config.toml
tools/run.sh $v-1 -p "What is 17 times 23? Reply with only the number." --output-format text
```

`-p` runs in auto (Never Ask) mode and approves every tool; `--auto`/`--yolo` cannot be combined with `-p`, so "only inside the folder" is enforced by the sandbox mounts, not by Kimi.
Each run's argv, stdout, stderr, exit and strace are in `runs/<label>/`; the work folders are `work/<label>/`.

## Results per run (one attempt each)

| Run | Exit | Outcome |
| --- | --- | --- |
| a-1 | 0 | `{"role":"assistant","content":"OK"}` |
| b-1 | 0 | Write `hello.py` (15 bytes, `print("hello")`), Bash `python3 hello.py` -> `hello\n`, final answer quotes `hello` |
| c-1 | 0 | Bash `cat /proc/self/cgroup` -> `0::/user.slice/user-1000.slice/user@1000.service/heavy.slice/run-p869176-i5068800.scope`: Kimi's shell commands stay in heavy.slice |
| d-on-plain-1 | 0 | `• 391`, no `reasoning_effort` sent, no reasoning returned |
| d-on-effort-1 | 0 | `• 391`, `reasoning_effort: "medium"` sent, no reasoning returned |
| d-off-nooff-1 | 1 | `error: failed to run prompt: provider.api_error: Model 'local' reasons by default but declares no off effort, so thinking cannot be turned off. Declare offEffort (for example 'none') for this model in the model catalog configuration.` No request sent. Expected config error, fixed by the next variant |
| d-off-effort-1 | 0 | `• 391`, `reasoning_effort: "none"` sent, no reasoning returned |

stderr in text mode: `kimi version 2.5.3` and `To resume this session: kimi -r <id>`; in stream-json the same facts arrive as `meta` lines on stdout.

## Request shapes (`shapes/*.json`)

All 9 requests: `POST /v1/chat/completions`, status 200.
- Body keys: `model` (`local`), `max_tokens` (16384), `stream` (true), `stream_options` (`{"include_usage": true}`), `prompt_cache_key` (the session id), `tools`, `messages`, plus `reasoning_effort` only in d-on-effort (`medium`) and d-off-effort (`none`).
- No `temperature`, `top_p`, `tool_choice`, `parallel_tool_calls`, `chat_template_kwargs` or `thinking`.
- 24 tools: Agent, AgentSwarm, AskUserQuestion, Bash, CreateGoal, CronCreate, CronDelete, CronList, Edit, ExitPlanMode, FetchURL, GetGoal, Glob, Grep, Read, SetGoalBudget, Skill, TaskList, TaskOutput, TaskStop, TodoList, UpdateGoal, WaitFor, Write.
- Messages: system (10,170 chars) then three user messages (the prompt, then 285 and 452 chars of injected context); tool turns add assistant/tool pairs.
- First request prompt_tokens 12,202 (a-1), about 4 s; follow-up requests in a session 0.4-0.6 s (prefix cache).
- Headers: `User-Agent: kimi-code-cli/2.5.3`, OpenAI SDK `X-Stainless-*` headers (Linux, x64, node v24.15.0, package 6.34.0), `Authorization` (masked in records).
  No `X-Msh-*` header.
- llama.cpp accepted `prompt_cache_key`, `stream_options` and `reasoning_effort` without an error (all 200).

## Connect targets (strace, all runs)

- AF_INET: only `127.0.0.1:47303` (1, 3, 2, 1, 1, 1 per successful run; 0 in d-off-nooff).
- AF_UNIX: `/var/run/nscd/socket` 4 times per run (libc user lookups, ENOENT).
- No DNS lookup, no other address, no `sendto` call.
- The private home has no `telemetry/`, `updates/` or `cache/client-configs/` folder, and no `bin/rg` download, consistent with no outbound attempt.
  Caveat: these runs did not use Grep or Glob and were not interactive, so the rg/fd download and the `client_configs` call found in the code were not exercised (inferred from absence).

## Side effects

- Owner's `~/.kimi-code`: no change (`diff before.txt after.txt` shows no line under it).
- `$HOME` top level outside `$SCRATCH`: two mtime changes, `~/.claude.json` (08:51:12 -> 08:54:38) and `~/.hermes` (08:51:01 -> 08:54:01).
  Neither comes from Kimi: the sandbox mounted a tmpfs over `$HOME`, so the runs could not write there.
  `.claude.json` belongs to Claude Code; `~/.hermes` was changed by another process (not inspected, per the rules).
- The private home gained `device_id`, `projects.json`, `session_index.jsonl`, `sessions/` (one per run), `file-history/`, `locks/`, `logs/`, `cache/query-store/`, and the native cache under `home/.cache/kimi-code/native/2.5.3/...`.

## Points for the llmctl design

- The entry above works as is; `api_key` placeholder and `model_source = "static"` are enough, and Kimi did not rewrite the file.
- `max_output_size` is sent as `max_tokens`.
- For a toggle recipe whose server runs `--reasoning off`, `reasoning_effort` has no visible effect; declaring `thinking` without `off_effort` but with `support_efforts` breaks `[thinking] enabled = false` with exit 1, so either omit `support_efforts` or set `off_effort`.
- An e2e can use exactly this sandbox: `-p ... --output-format stream-json`, a throwaway work folder, `timeout`, and the env switches above.
