# Kimi TUI checklist (LR4)

Last verified: 2026-11-17.

This is Lead step 7 of the llmctl Kimi plan: Kimi Code CLI 2.5.3's TUI, driven through a pseudo-terminal against the real model servers, to check what the owner sees.
Every item passed; nothing here is a defect of llmctl.

## How it ran

Every TUI session ran as one `heavy bash bin/case.sh <run> <W> <steps.json>`, one at a time.
`bin/case.sh` is the archive's `exp/bin/case.sh` adapted: `bwrap --unshare-net --unshare-ipc --unshare-uts --new-session --die-with-parent`, a tmpfs over `/tmp` and over `$HOME` with only `$HOME/.kimi-code/bin/kimi` bound back read-only, `--clearenv`, a private `HOME`, `KIMI_CODE_HOME`, `KIMI_CODE_CACHE_DIR` and `TMPDIR` in the private folder `W` (`mktemp -d -p $SCRATCH kimi-tui-XXXXXX`, every subfolder mode 0700), and `KIMI_DISABLE_TELEMETRY=1`, `KIMI_CODE_NO_AUTO_UPDATE=1`, `KIMI_CODE_REGION_MARKER=off`, `KIMI_CODE_TRUST_WORKSPACE=1`, `TERM=xterm-256color`.
Inside the sandbox, `bin/inner.sh` runs socats on 127.0.0.1:8080 and 127.0.0.1:8000 that forward to unix sockets; outside, socats forward those sockets to two copies of the archive's recording proxy `bin/proxy.py`, which forward to the real servers on 127.0.0.1:8080 (llama.cpp) and 127.0.0.1:8000 (vLLM) and save each request's shape.
So every entry's `base_url` stayed exactly as llmctl wrote it, and the TUI had no other network.
`bin/tui.py` is the archive's pty driver with two more step kinds, `touch` and `waitfile`, so that an open session can wait while `bin/item4.sh` runs `./llmctl up ... --replace` outside the sandbox.
The private `config.toml` (mode 0600) was always a copy of the real `$HOME/.kimi-code/config.toml` with llmctl's default edit, `kimi_config.config_text(..., set_default=True)`, as `_Run.kimi_env` of `launcher/e2e/clients.py` builds it, because the real file has no `default_model`; `bin/prep.py` writes it as `config.toml.new` and renames it over `config.toml`, as llmctl writes.
Every `./llmctl up` ran with `$HOME/.kimi-code/bin` on `PATH`, so it also wrote the real Kimi file.
The records keep, per item, the step file, the screen text, the request shapes (keys and the values Kimi sets; no message text, system prompt or session id), the private `config.toml` at each step, the request lines of the server log and the `llmctl up` output.
No session file, log or cache of `W` is kept, and `W` was removed.

## Items

1. `/model` lists `gemma-4-26b-a4b on llama.cpp (llmctl)  llmctl ← current` with `Thinking  On (Unsupported)  [ Off ]`, and Enter picks it: `Saved gemma-4-26b-a4b on llama.cpp (llmctl) with thinking off as default.`
   The pick appended `[thinking]` with `enabled = false` to the private file, as E4a found.
   Passed: `item1-2-model-and-answer/tui-transcript.txt`, `c0-start.toml`, `c1-after-pick.toml`.
2. The prompt got the answer `BANANA` from the real server in about 4 s.
   The proxy recorded one `POST /v1/chat/completions` with `model: "local"`, `max_tokens: 16384`, `stream: true`, `stream_options.include_usage`, 24 tools and no `reasoning_effort`, and the llama.cpp log shows one task with a 12,109-token prompt, the proxy's `prompt_tokens`.
   Passed: `item1-2-model-and-answer/request-shapes.json`, `server-log-llamacpp.txt`.
3. After `./llmctl up qwen3.8-27b --replace` and a new copy, `/effort` offered `Off  Low  [ Medium ]` with `Alt+S session-only` in its hint line.
   Left and Alt+S printed `Thinking set to low for this session only.`, the footer showed `thinking: low`, and the request carried `reasoning_effort: "low"`; the answer was `BANANA`.
   With Alt+S the selector saved nothing: the private `config.toml` stayed byte for byte as copied through the prompt and the exit (P3's Enter saved a `[thinking]` table).
   Passed: `item3-effort/`.
4. An open session after an `up`, as E8 found:
   - Alias changed: on `llmctl/local` (qwen3.8-27b), one prompt answered with `reasoning_effort: "medium"`; then `./llmctl up gpt-oss-20b --runtime vllm --replace` and a copy by rename (alias `llmctl/gpt-oss-20b`, `base_url` port 8000, bridged too).
     The next turn failed with `Error: [config.invalid] Model "llmctl/local" is not configured in config.toml.`, no request reached either proxy, the vLLM log has no `POST /v1/chat/completions`, the footer still named qwen3.8-27b, and the TUI did not write the file.
     Passed: `item4a-alias-changed/`.
   - Alias kept: a new session on `llmctl/local` (qwen3.8-27b again) answered with `reasoning_effort: "medium"`; then `./llmctl up gemma-4-26b-a4b --replace` and a copy by rename.
     The next turn went to gemma: the proxy recorded the second request on port 8080 with roles `system, user, user, assistant, user` and still `reasoning_effort: "medium"`, the old thinking level, and the new gemma container's log shows one task with a 12,133-token prompt, the proxy's `prompt_tokens`; the answer was `CHERRY`.
     The footer still showed `qwen3.8-27b on llama.cpp (llmctl) thinking: medium`, but its context total changed from 64k to 128k; E8 saw the footer keep the old values.
     The TUI did not write the file.
     Passed: `item4b-alias-kept/`.
5. End state: the `./llmctl up gemma-4-26b-a4b --replace` of item 4 left gemma-4-26b-a4b running on llama.cpp, and `./llmctl status` shows Kimi `llmctl/local for gemma-4-26b-a4b on llama.cpp, the running model; default not set`.

The notes `llmctl up` printed matched what the sessions did: `an open Kimi session on llmctl/local fails its next turn until you pick llmctl/gpt-oss-20b with /model or start a new session` before item 4's first switch, and `an open Kimi session on llmctl/local sends its next turn to gemma-4-26b-a4b on llama.cpp but keeps its old thinking level; start a new session` before its second.

Labels: items 1, 2, 3 and 4 (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, llama.cpp b12407); item 4's switch to `vllm/vllm-openai:v0.33.0` (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, vLLM v0.33.0), which got no request.
