# dsh Web UI checklist with llmctl's rows, 2026-11-17

Last verified: 2026-11-17.

This is the browser checklist of design D9 for llmctl's dsh support, Lead step 8 of docs/superpowers/plans/2026-11-17-llmctl-dsh.md.
Results are labelled (measured on the MSI Raider 18 Max HX, dsh 0.2.2-rc.3, 2026-11-17).

## How it ran

dsh 0.2.2-rc.3 ran `dsh web --port 3182 --no-open` under `heavy` and `setsid`, in a private home `$SCRATCH/lead8-<random>/`.
The private home held `home`, `agents`, `tmp` and `work`, with `DSH_HOME`, `DSH_AGENTS_HOME`, `TMPDIR` and `PWD` pointing at them and `DSH_TELEMETRY_DISABLED=1`.
`home/cordis.patch.yml` was a byte copy of the owner's `$HOME/.dsh/cordis.patch.yml` (`install -m 0600`, then `cmp` with no difference), holding llmctl's `llm-pi-ai`, `agent-default-model` and `acp` rows.
The first model was gemma-4-26b-a4b on llama.cpp `ghcr.io/ggml-org/llama.cpp:server-cuda-b12407` at 127.0.0.1:8080, served as `local`.
The second was gpt-oss-20b on vLLM `vllm/vllm-openai:v0.33.0` at 127.0.0.1:8000.
The browser was the Playwright MCP browser, driven through accessibility snapshots only, with no screenshots.
The launch address carries a token and is not recorded here.

## Results

| Item | Check | Pass | Text seen |
| --- | --- | --- | --- |
| a | The picker lists llmctl's gemma route, with Effort Off and Medium and Off preselected | Yes | Picker button "Select model, current gemma-4-26b-a4b on llama.cpp (llmctl), reasoning effort Off"; the Model submenu holds one group "llmctl" with the checked item "gemma-4-26b-a4b on llama.cpp (llmctl)"; the Effort submenu holds "Off" (checked) and "Medium" and nothing else |
| b | A new session answers "Reply with exactly the word pong." | Yes | Status "Completed", "Completed in 3s", reply paragraph "pong" |
| c | The Models page refuses a new OpenAI-compatible provider | Yes | Settings, Models, "Add model provider", tab "Custom model API", Provider ID `probe-lead8`, Base URL `http://127.0.0.1:9/v1`, protocol "OpenAI Chat Completions", no API key, model ID `probe-model`, "Create provider": the form shows "Configuration for "llm-pi-ai" is overridden by a home patch or command-line overlay"; Cancel; the private home patch still matched the owner's byte for byte afterwards |
| d | An Effort change lives only in the open session | Yes | After choosing Medium the open session's button read "reasoning effort Medium"; a new session read "reasoning effort Off"; the first session still read "reasoning effort Medium" when reopened |
| e | After the switch to gpt-oss-20b on vLLM the picker shows the new route within 20 s | Yes | A new session's picker read "Select model, current gpt-oss-20b on vLLM (llmctl), reasoning effort Medium" |

The provider form in item c needs at least one model before "Create provider" is enabled ("A custom provider needs at least one model."), so the check filled in a dummy model ID.

## Model switch

`heavy ./llmctl up gpt-oss-20b --runtime vllm --replace` exited 0 in 27 s and printed `ready: http://127.0.0.1:8000/v1, served as 'gpt-oss-20b', context 65536, output 16384`.
It stopped gemma-4-26b-a4b, then wrote the OpenCode, Hermes, Pi and dsh files, the dsh line reading `llmctl rows for gpt-oss-20b on vLLM (was gemma-4-26b-a4b on llama.cpp); dsh's default llmctl/gpt-oss-20b in every profile (was llmctl/local)`.
The new home patch was copied into the private home with `install -m 0600` to a temporary name and `mv`, so dsh saw one rename.
The picker of the next new session showed "gpt-oss-20b on vLLM (llmctl)" 2.4 s after the rename, well inside the 20 s limit.
The route's Effort came up as Medium, which is gpt-oss-20b's level in llmctl's rows.

## Cleanup

The browser was closed, and `kill -TERM` to `dsh web`'s process group stopped it: afterwards `pgrep -g <pgid>` found nothing and nothing listened on port 3182.
Before the run neither `$HOME/Documents/deepseek-harness` nor `$HOME/Documents/deepseek-harness/default-workspace` existed (`test -e` returned 1 for both).
After the run both existed, empty, both created at the moment the browser first opened the page, so `dsh web` creates its default workspace under `$HOME/Documents` even when `DSH_HOME`, `DSH_AGENTS_HOME` and `PWD` point into a private home.
Both empty folders were removed with `rmdir`, and `test -e` returned 1 for both again.
The Playwright MCP output folder was moved out of the repository into `$SCRATCH/playwright-lead8/`.
gpt-oss-20b was left running on vLLM for the next step of the plan.
