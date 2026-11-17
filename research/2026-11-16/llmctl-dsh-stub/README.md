# llmctl dsh client: stub evidence for DeepSeek Harness

Last verified: 2026-11-17.

This folder keeps the evidence behind the DeepSeek Harness (dsh) part of docs/clients.md and behind the design work for it, whose documents are kept outside the published repository.
DeepSeek Harness 0.2.2-rc.3 (npm `@deepseek-ai/dsh@0.2.2-rc.3`) ran in private homes; lane A asked who owns which patch file and what the Web UI does, lane B recorded the requests dsh sends per thinking type, and S1 to S11 checked the design's own rows and its `--dump-config` check.
The table under "Claims and files" maps every claim id of the design to the files here that back it, and "Research claims" lists the research verdicts that docs/clients.md and docs/sources.md cite by id.

## How the runs worked

Every experiment got a fresh `DSH_HOME` made with `mktemp -d` under the scratch folder, mostly with a fresh `DSH_AGENTS_HOME` and work folder too, and ran with `DSH_TELEMETRY_DISABLED=1` through `heavy`.
The owner's `~/.dsh` was never used, read or written.
Lane B and S5 ran `dsh --profile headless --patch <file> --json <task>` or `dsh --profile headless --json <task>` against a recording stub on 127.0.0.1 (`bin/laneB/stub.py`, the same file as `bin/spec/stub.py`), which answers `POST /v1/chat/completions` and `GET /v1/models` and appends one JSON line per request to a record file.
Lane B's `real-*` cases went instead through the recording proxy `bin/laneB/proxy.py` to the llama.cpp server running on 127.0.0.1.
S10 and S11 ran the final gemma-4-26b-a4b home patch against the real gemma-4-26b-a4b recipe on `ghcr.io/ggml-org/llama.cpp:server-cuda-b12407`.
Lane A drove the Web UI with a browser on private ports (3181 and 3182), never on dsh's default port 3080.
S7 and S8 booted `dsh web --port 3182 --no-open` and stopped it with SIGTERM.

## Layout

| Folder | Content |
|---|---|
| `laneA/` | Lane A's home and profile patch snapshots, `--dump-config` outputs, stderr files and reduced events |
| `laneB/runs/<case>/` | Per lane B case: `meta.txt` (exit code and seconds), `patch.yml` (the `--patch` overlay), `stderr.txt` and `events.jsonl` |
| `laneB/runs/b1-resume*/` | B1-9's resume sequences, one file set per step: `p<n>.yml` (the overlay of step n), `r<n>.err` (its stderr) and `r<n>.events.jsonl`; `b1-resume3` also has `r-p<n>.*` for resumes run with overlay n |
| `laneB/patches/` | The overlays lane B generated, one per case, B4-2's `b4-notools.yml` among them |
| `spec/` | S1 to S4d and S6 to S11 outputs and candidate home patches; S1 and S7 straces reduced as below |
| `spec/gen/` | The rows `bin/spec/gen_dsh.py` generated for all 19 recipes: `<runtime>/<id>.dsh.yml` (with the default rows) and `<id>.dsh-nodefault.yml` |
| `spec/runs/<case>/` | Per S5 case: `home.yml`, `result.json`, `stderr.txt` and `events.jsonl` |
| `shapes/laneB/`, `shapes/spec/` | Request shapes of lane B's and S5's stub records, one file per record |
| `bin/dsh_shapes.py` | The reducer that wrote every shape and event file here |
| `research-claims.json` | The research verdicts the documents cite by id ([Research claims](#research-claims)) |
| `bin/spec/`, `bin/laneA/`, `bin/laneB/` | The reviewed generator, run and stub scripts of each source |

A shape line holds `run` (the record file's stem), `n`, `kind` (`main`, `title`, `compaction` or `other`, as the stub classified it), `kind_n`, `method`, `path`, `model`, `keys` (the sorted top-level body keys), `max_tokens`, `max_completion_tokens`, `reasoning_effort`, `reasoning`, `chat_template_kwargs`, `stream`, `stream_options`, `store`, `first_role` (the role of the first message), `tools` (the count), `authorization` (the scheme word of the header), `reply` (the stub's action type) and `t` (`arrive`, `start` and `end` in seconds after the file's first request, rounded to 0.01).
A key is left out when the request lacks it; proxy records have no `n`, `method`, `headers` or action, so their shapes have none of those.
An event line keeps `type`, `phase`, `turn`, `step`, `callId`, `tool`, `status`, `input_keys` (the sorted keys of the tool input), `error` (`name` and `code`) and `reason` (`kind` and `error.code`) of one `dsh --json` event.
A file named `<name>.events.jsonl` replaces the NDJSON `<name>.out` or `<name>.jsonl` of the run.

## Claims and files

| Claim | What it shows | Files |
|---|---|---|
| A1 | A second pi-ai instance (an inserted row `llmctl`) cannot serve llmctl: `LlmError: configurable provider "amazon-bedrock" is already declared`, then `NO_ADAPTER`, exit 1 | `laneA/a1-headless-run.events.jsonl`, `laneA/a1-headless-run.err`, `laneA/a1-headless-dump.txt`, `laneA/a1-headless-dump.err` |
| A2-json | dsh reads JSON flow rows between comment markers in the profile and the home patch; the dumps exit 0, and web, headless and acp answered | `laneA/a2p-dump0.txt`, `laneA/a2p-dump0.err`, `laneA/final-gemma.yml`, `laneA/final-gptoss.yml`, `laneA/final-gemma-headless.*`, `laneA/final-gptoss-headless.*` |
| A2-profile | The Web UI rewrites a profile patch that holds an llmctl block on the first page load, on every picker change and on every Models page save; JSON lines become YAML and its rows land inside the block | `laneA/a2p-0-initial.yml`, `laneA/a2p-1-after-load.yml`, `laneA/a2p-2-after-effort.yml`, `laneA/a2p-3-after-add-provider.yml` |
| A2-home | dsh never writes the home patch (sha256 unchanged); the Models page refuses pi-ai edits while a home patch overrides `llm-pi-ai`, and a profile `llm-pi-ai` row is hidden | `laneA/a2h-home-0.yml`, `laneA/a2h-home-final.yml`, `laneA/a2h-prof-0.yml`, `laneA/a2h-prof-before-userrow.yml`, `laneA/a2h-prof-final.yml`, `laneA/a2h-web.err` |
| A2-hmr | A running web process picks up a patch rewrite (16 s for the picker, 8 s for an open session's next turn); the profile file after the reload | `laneA/a2p-4-after-hmr.yml` |
| A3-stale | A model id changed from `local` to `gpt-oss-20b` while web runs leaves blank sessions on `Select model`, and sends fail `UNKNOWN_MODEL` | `laneA/a3-home-1-model-rewritten.yml` |
| A3-follow | A default row in the block follows the running model for new and blank sessions | `laneA/a3-home-2-default-followed.yml`, `laneA/a3-expected-home.yml` |
| A4 | web, headless, acp and sdk read the home patch; acp uses the `acp` row; `sdk-minimal` has no `llm-pi-ai` row; the `acp` row makes the other profiles' dumps warn `patch: entry "acp" not found` | `laneA/a4-*`, `laneA/e0-web-dump.txt`, `laneA/e0-web-dump.err` |
| ALT | A Node helper merges `providers.llmctl` into a Web-UI-written profile row with dsh's own yaml library, byte-exact outside llmctl's lines | `laneA/nodemerge-before.yml`, `laneA/alt-profile-*.yml`, `laneA/alt-profile-*.err`, `bin/laneA/dsh_profile_merge.mjs` |
| SIDE | The first Web UI page load creates `~/Documents/deepseek-harness/default-workspace` outside `DSH_HOME` | No file: lane A observed it in its report |
| B1-0 | Without `compat`, pi-ai sends `store`, `max_completion_tokens` and the `developer` role to 127.0.0.1 | `shapes/laneB/b1-nocompat-gptoss.jsonl`, `laneB/runs/b1-nocompat-gptoss/` |
| B1-1 to B1-7 | The request per thinking type and level: toggle through `chat_template_kwargs.enable_thinking`, effort through `reasoning_effort`, always and none send nothing, undeclared levels fail `UNSUPPORTED_REASONING_EFFORT` before any request; the design summarizes these seven ids together, so the files are every `b1-*` case except `b1-nocompat-*`, `b1-prec-*` and `b1-resume*`, named by thinking type and level | `shapes/laneB/b1-*.jsonl`, `laneB/runs/b1-*/` |
| B1-2 | Bolt's medium level: `{"enable_thinking": true, "medium_effort": 1}` through `thinkingBudgets`, off and high as for toggle | `shapes/laneB/b1-bolt-*.jsonl`, `shapes/laneB/b1-bolt2-*.jsonl`, `laneB/runs/b1-bolt-*/`, `laneB/runs/b1-bolt2-*/` |
| B1-5 | An effort route without its own `reasoning` level makes pi-ai send the `off` value, which for gpt-oss is no `reasoning_effort` on the main request | `shapes/laneB/b1-gptoss-omit.jsonl`, `laneB/runs/b1-gptoss-omit/` |
| B1-8 | `agent-default-model.reasoningEffort` beats the route's `reasoning`, which beats nothing | `shapes/laneB/b1-prec-*.jsonl`, `laneB/runs/b1-prec-*/` |
| B1-9 | A level from the route survives entry changes on resume; an explicit `off` fails on a none entry; a resume onto a 64k model id fails `MODEL_SWITCH_COMPACTION_FAILED` | `shapes/laneB/b1-resume*.jsonl`, `laneB/runs/b1-resume*/` (the `off` failure in `b1-resume/r2.err`, the compaction failure in `b1-resume/r3.err` and `r3.events.jsonl`) |
| B2-1 to B2-6 | Title, subagent and compaction requests: the title uses the same route with `max_tokens` 4096 and the least supported level, runs concurrently with the main request and was served first on a single slot in 8 of 8 real runs; compaction uses only the route's `reasoning` | `shapes/laneB/b2-*.jsonl`, `laneB/runs/b2-*/`, `shapes/laneB/real-*.jsonl`, `laneB/runs/real-*/`, `shapes/laneB/t1-toggle-off.jsonl`, `laneB/runs/t1-toggle-off/` |
| B3-1 | Without `maxTokens`, pi-ai picks `max_tokens` itself: 128,000 with no sizes at all (`b3-cap-nosizes`), and a value from the context window and the prompt when only `contextWindow` is set (`b3-cap-nomax`, `b3-cap-prompt-*`) | `shapes/laneB/b3-cap-*.jsonl`, `laneB/runs/b3-cap-*/` |
| B3-2 | Proactive compaction starts at `min(0.8 x window, window - maxTokens - 65,536)` and not at all when that is not positive, which dsh logs only once per model through its own logger and headless does not print | `shapes/laneB/b3-comp-*.jsonl`, `laneB/runs/b3-comp-*/`; source `dsh-compaction-basic/src/config.ts:75` |
| B3-4 | Queue time counts as idle time for `streamIdleTimeoutMs` | `shapes/laneB/b3-idle-timeout*.jsonl`, `laneB/runs/b3-idle-timeout*/` |
| B3-5 | `TIMEOUT` is retried 5 times behind the abandoned attempt on one slot; a `retryPolicy` without `TIMEOUT` fails once | `shapes/laneB/b3-retry-policy.jsonl`, `laneB/runs/b3-retry-policy/`, `laneB/patches/b3-retry-policy.yml` |
| B3-6 | A `max_tokens` above vLLM's `max_model_len` is refused with HTTP 400 (the stub answered with vLLM's message) | `shapes/laneB/b3-400-maxtokens.jsonl`, `laneB/runs/b3-400-maxtokens/` |
| B3-7 | A stopped server fails each turn with `TRANSPORT` `ECONNREFUSED` after the retries, exit 1 after 15.8 s; no request reached a server, so there is no shape file | `laneB/runs/b3-refused/` |
| B3-8 | 16,384 output tokens at Qwen3.8-27B's measured 36 tok/s take about 455 s, which sets the `streamIdleTimeoutMs` estimate | No file: arithmetic on measured numbers (llamacpp/models.md) |
| B4-1 | A model has no tool switch: a `tools: false` model key was dropped silently and 23 tools were still sent | `shapes/laneB/b4-model-toolskey.jsonl`, `laneB/runs/b4-model-toolskey/` |
| B4-2 | A `--patch` that disables 14 tool rows sends no `tools` | `shapes/laneB/b4-notools.jsonl`, `laneB/runs/b4-notools/`, `laneB/patches/b4-notools.yml` |
| B4-3 | On vLLM without a tool parser, a request with tools fails with HTTP 400 `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set`; the stub replayed vLLM's message, and the source is [vllm/renderers/online_renderer.py lines 269-280 at v0.33.0](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/renderers/online_renderer.py#L269-L280) | `shapes/laneB/b4-vllm-tools-400.jsonl`, `laneB/runs/b4-vllm-tools-400/` |
| B-T | Bolt's chat template tests `medium_effort` only for truth (read in the template, not rendered) | No file: the template of [nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF) as `bin/laneB/gguf_template.py` prints it, lines 60 and 212 |
| S1 | `dsh --profile acp --dump-config` in a private `DSH_HOME` holding the candidate home patch: exit 0, empty stderr, 0.07 to 0.08 s in four timed runs; the three owned rows appear under `patched by <home patch>`; writes only inside the private home | `spec/s1.stdout`, `spec/s1.stderr`, `spec/s1-candidate.yml`, `spec/s1t-*.err`, `spec/s1.strace` |
| S2 | Owned row ids that the installed dsh lacks: exit 0 with `patch: entry "llm-pi-ai-x" not found` per id on stderr | `spec/s2.*`, `spec/s2-candidate.yml` |
| S3 | `[]` before the block: exit 1, `YAMLException: end of the stream or a document separator is expected` | `spec/s3.*`, `spec/s3-candidate.yml` |
| S4 | A broken user row: exit 1, `missed comma between flow collection entries` | `spec/s4.*`, `spec/s4-candidate.yml` |
| S4b | Appending to a file whose last line has no newline turns the user's `disabled: true` into the string `true# dsh rows for the model llmctl runs (docs/llmctl.md#clients).`, which the dump prints unquoted as `disabled: true# dsh rows ...` (`spec/s4b.stdout` line 176) | `spec/s4b.*`, `spec/s4b-candidate.yml` |
| S4c | A home patch of comments only: exit 1, `must be a top-level YAML array of loader patch entries` | `spec/s4c.*`, `spec/s4c-candidate.yml` |
| S4d | A user `llm-pi-ai` row before the block: exit 0, and llmctl's row wins without a word | `spec/s4d.*`, `spec/s4d-candidate.yml` |
| S5 | All 19 generated home patches through `dsh --profile headless --json` against the stub on port 18090: 19 of 19 exit 0 with `pong` in about 1.0 s, with the request shapes per thinking type; with owner-style profile rows under llmctl's held default (`migration-profile-rows`) the home rows win | `spec/s5-summary.jsonl`, `spec/runs/*/`, `spec/gen/`, `shapes/spec/*.jsonl` |
| S6 | `dsh --version` prints `0.2.2-rc.3`, exit 0, in 0.07 s and writes nothing | `spec/s6-*` |
| S7 | `dsh web --port 3182 --no-open` in a private home is ready in 1.5 s; outside the home it writes only under `/tmp`, and one empty `dsh-spill-*` folder stays after SIGTERM | `spec/s7.stdout`, `spec/s7.stderr`, `spec/s7.marker`, `spec/s7.strace`, `bin/spec/s8_web_boot.py` |
| S8 | Valid rows boot with an empty stderr; `contextWindow: "big"` boots with `dsh: warning: 1 entry did not activate` | `spec/s8-valid.yml`, `spec/s8-invalid.yml`, `bin/spec/s8_web_boot.py` |
| S9 | `dsh --profile sdk < /dev/null` exits 0 in 1.1 s; an invalid entry still exits 0 and prints the `ValidationError` path on stderr | `spec/s9-*` |
| S10 | The final gemma home patch against the running llama.cpp server: `pong`, exit 0, 1.6 s, 5,247 cached prompt tokens | `spec/s10-home.yml`, `spec/s10.err`, `spec/s10.events.jsonl` |
| S11 | The e2e file task through the final gemma rows on the real server: exit 0 in 5.1 s with `todo_write`, `write` (the first call failed `SANDBOX_ESCALATION_INVALID_ARGUMENT`, the second completed) and `read` | `spec/s11.err`, `spec/s11.events.jsonl` |

## Research claims

On 2026-11-16 three research lanes (distribution, configuration and runtime) read the dsh sources and the npm packages, and a verify lane re-read their 117 claims and added findings of its own; their working file stays outside the repository.
[research-claims.json](research-claims.json) keeps exactly the entries that docs/clients.md and docs/sources.md cite, each reduced to its `id`, `claim`, `verdict` (`confirmed`, `partly`, or a finding of the verify lane), the verify lane's `correction` where it has one, and `sources`, the URLs and npm package paths it rests on; no entry held anything private, so no field was left out.
An `<lane>-<n>` id is claim n of that lane, and `extra-<n>` is the verify lane's finding n.

| Id | Verdict | Claim in short | Cited in |
|---|---|---|---|
| `configuration-0` | confirmed | npm `@deepseek-ai/dsh`: `latest` 0.2.2-rc.1, `next` 0.2.2-rc.3, Node 24.9 or newer, no `npm-shrinkwrap.json` | docs/sources.md |
| `configuration-4` | confirmed | A custom model API has exactly four protocols, `openai-completions` among them | docs/clients.md, Rows per thinking type |
| `configuration-7` | confirmed | A keyless local server needs a placeholder key or an `Authorization` header | docs/clients.md, Rows per thinking type |
| `configuration-23` | partly | Proactive compaction needs a positive window minus output minus 65,536 headroom; the correction says overflow recovery still compacts | docs/clients.md, Context, compaction and timeouts |
| `configuration-25` | confirmed | No per-model or per-route switch turns tool calling off | docs/clients.md, Chat-only recipes |
| `runtime-1` | confirmed | Every release is a prerelease, and `latest` is 0.2.2-rc.1 | docs/clients.md, Versions; docs/sources.md |
| `runtime-13` | confirmed | Headless exit codes: 1 for a failed or aborted turn, an empty line and 1 without an assistant message | docs/clients.md, Silent fallbacks and exit codes |
| `runtime-14` | confirmed | The `--json` event types, and an `error` event without `final` for a failure outside a turn | docs/clients.md, Silent fallbacks and exit codes |
| `runtime-17` | confirmed | Feedback-triggered session upload is on by default in the web, headless, sdk and acp profiles | docs/clients.md, Privacy and side effects |
| `runtime-18` | confirmed | A rating, note, withdrawal or `/feedback` uploads the whole session prefix for every provider | docs/clients.md, Privacy and side effects |
| `runtime-19` | confirmed | `DSH_TELEMETRY_DISABLED` with any non-empty value turns the upload off; `DSH_*` cannot come from `.env` | docs/clients.md, Privacy and side effects |
| `runtime-26` | partly | `web_search` uses the DeepSeek Search API; the correction says `web_fetch` is not limited to public addresses | docs/clients.md, Privacy and side effects |
| `runtime-29` | partly | With telemetry off, no DeepSeek key and only a local route, no built-in path sends prompts off the machine; the correction adds `web_fetch` to any URL; a synthesis of the other claims, so its `sources` is empty | docs/clients.md, Privacy and side effects |
| `extra-0` | verify finding | `web_fetch` reaches 127.0.0.1 and private networks without approval, and reads are unconfined | docs/clients.md, Privacy and side effects |
| `extra-1` | verify finding | The privacy policy's custom-model sentence against the shipped feedback upload | docs/sources.md |
| `extra-5` | verify finding | Context-overflow recovery still compacts on the 64k recipes | docs/clients.md, Context, compaction and timeouts |
| `extra-8` | verify finding | Commands dsh runs are placed by `systemd-run --user --scope` outside `heavy.slice` | docs/clients.md, Privacy and side effects |

## Rerun

Every rerun needs DeepSeek Harness 0.2.2-rc.3 (`npm install -g @deepseek-ai/dsh@0.2.2-rc.3`) on Node 24.9 or newer, the `heavy` wrapper, and `SCRATCH` set to a work folder of your own.
Never point `DSH_HOME` at a real home: each command below makes a private one.

- Generated rows: `python3 -I bin/spec/gen_dsh.py $SCRATCH/llmctl-dsh-stub-rerun/gen` writes the rows of every recipe from the repository's recipes, `python3 -I bin/spec/gen_dsh.py --home llamacpp gemma-4-26b-a4b` prints one complete home patch, and `--no-tools` prints the chat-only overlay.
- S5: `mkdir -p $SCRATCH/llmctl-dsh-stub-rerun/out && python3 -I bin/spec/stub.py 18090 $SCRATCH/llmctl-dsh-stub-rerun/out/stub.json &`, then `python3 -I bin/spec/s5_run.py [case ...]`; it writes `out/rec/`, `out/runs/`, `homes/` and `work/` under `$SCRATCH/llmctl-dsh-stub-rerun/` and refuses to start when `SCRATCH` is unset.
- S7 and S8: `python3 -I bin/spec/s8_web_boot.py <case> <home patch file>` boots `dsh web` on port 3182 in a private home under `$SCRATCH/llmctl-dsh-stub-rerun/`, stops it, and prints the launch line with the token redacted.
- S1 to S4d and S6: `DSH_HOME=$(mktemp -d -p $SCRATCH) DSH_TELEMETRY_DISABLED=1 heavy dsh --profile acp --dump-config` with the candidate copied to `$DSH_HOME/cordis.patch.yml` first, and `heavy dsh --version`.
- Lane B: the scripts expect `$SCRATCH/laneB.GwYhns` with `bin/`, `cfg/`, `rec/`, `runs/`, `patches/`, `homes/` and `work/`; start `bin/laneB/stub.py 18080 $SCRATCH/laneB.GwYhns/cfg/stub.json`, point it at a case with `bin/laneB/setcfg.py <case>`, write the overlay with `bin/laneB/gen_patch.py <out.yml> <type> [key=value ...]` and run `bin/laneB/run_case.sh <case> <patch.yml> <task>`.
  The Python scripts name the lane B folder as the literal `$SCRATCH/laneB.GwYhns`, which the redaction wrote in place of the local path; set it to your own folder before a run.
- Lane A: `bin/laneA/web-start.sh <DSH_HOME> <tag> [port]` and `bin/laneA/web-stop.sh` start and stop `dsh web` on ports 3181 and 3182 and never print the token; `bin/laneA/entries.py`, `final_text.py` and `rewrite_block.py` write and rewrite the rows, and `bin/laneA/acp_probe.py` drives the acp profile over stdio.
- Shapes and events: `python3 -I bin/dsh_shapes.py shapes <record.jsonl> <out.jsonl>` and `python3 -I bin/dsh_shapes.py events <stdout.jsonl> <out.jsonl>`.

## Privacy

Raw stub records hold dsh's system prompts, which name the private scratch paths, so only request shapes were kept, and every `dsh --json` stream was reduced to the event fields above.
Left out entirely: every private dsh home (`laneA/home-*`, `specdsh/home-*`, `specdsh/homes`, `laneB/homes`), which hold `.credentials.yaml`, sessions and storages; the agents homes and work folders; lane A's browser artifacts; every `*.pid`; the `dsh web` stdout files `a2h-web.out` and `a2p-web.out`, which hold the launch token; lane B's `cfg/` (generated filler prompts and stub configs); lane B's `templates/` (chat templates copied out of GGUF files; B-T names the repository instead); lane B's `src-vllm/` (copies of vLLM source; B4-3 links the file at the tag instead); lane B's `runs/web-dump.*` (a full web dump of the bundled configuration, which A4's dumps already cover); S5's last `stub.json`, which `s5_run.py` rewrites per case; the stub and proxy logs; the `home.txt` and `work.txt` path files of lane B's runs; and every `__pycache__/`.
From `s1.strace` and `s7.strace` only the `connect`, `rename`, `renameat2`, `unlink`, `unlinkat`, `mkdir` and `mkdirat` lines and the write-mode `openat` lines were kept.
`spec/s7.stdout` is kept because it holds only `token=<redacted>`.
The home folder became `$HOME` and the scratch folder `$SCRATCH`; `bin/spec/gen_dsh.py` finds the repository from its own path, and `s5_run.py` and `s8_web_boot.py` write under `$SCRATCH/llmctl-dsh-stub-rerun/` instead of beside themselves.
The only key-like string left is the placeholder `Bearer no-key-required` of the generated rows.
