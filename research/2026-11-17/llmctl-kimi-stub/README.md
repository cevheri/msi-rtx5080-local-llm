# llmctl Kimi client: stub evidence for Kimi Code CLI

Last verified: 2026-11-17.

This folder keeps the evidence behind the Kimi Code CLI part of docs/clients.md and behind the design work for it, whose documents are kept outside the published repository.
Kimi Code CLI 2.5.3, the native executable `$HOME/.kimi-code/bin/kimi`, ran in private homes inside an offline sandbox: E1 to E8 settled the open risks of the research, P1 to P3 checked the design's own questions, the prototype tried the design's file edits on files Kimi wrote, and the hands-on runs used one hand-written entry against the running llama.cpp server.
The table under "Claims and files" maps every experiment id to the files here that back it, and "Research claims" describes the research verdicts that the documents cite by id.

## How the runs worked

Every Kimi command of E1 to E8 ran as one `heavy` invocation of `exp/bin/case.sh`, one case at a time.
The sandbox was `bwrap --unshare-net --unshare-ipc --unshare-uts --new-session --die-with-parent` with a tmpfs over `/tmp` and over `$HOME`, only `$HOME/.kimi-code/bin/kimi` bound back read-only, and `--clearenv`.
Each case had its own private `HOME`, `KIMI_CODE_HOME` and `KIMI_CODE_CACHE_DIR`, and the three switches `KIMI_DISABLE_TELEMETRY=1`, `KIMI_CODE_NO_AUTO_UPDATE=1` and `KIMI_CODE_REGION_MARKER=off`.
Inside the sandbox a socat listened on 127.0.0.1 at the entry's own port (8080 for llama.cpp, 8000 for vLLM) and forwarded to a unix socket; outside, a second socat forwarded that socket to the recording stub `exp/bin/stub.py` on host 127.0.0.1:18491, so every entry's `base_url` stayed exactly as generated.
The stub records one reduced shape per request in `runs/<case>/records.jsonl`: body keys, `model`, `max_tokens`, `reasoning_effort`, `stream_options`, whether `prompt_cache_key` is present, the tool count and names, the message roles, and per message only key names and sizes; no message text is kept.
`exp/bin/inner.sh` times the command inside the sandbox (`runs/<case>/time`) and, unless a case says otherwise, records `connect`, `sendto` and `sendmsg` with strace (`runs/<case>/strace.txt`).
P1 to P3 ran the same way with the copies in `spec/bin/`.
The hands-on runs used the same kind of sandbox, but went through the recording proxy `handson/tools/proxy.py` to the gemma-4-26b-a4b recipe on the llama.cpp image `server-cuda-b12407`, which was already running; its README describes the path in full.
The prototype `spec/proto/kimi_proto.py` ran with Python's `tomllib` and no server, and ran no Kimi command.
No model server was started or stopped for any of this, and the owner's `~/.kimi-code` was read only for its `config.toml`.
The private homes, work folders and socket folders were deleted after the runs, because Kimi's session files there hold whole system prompts, and every kept record had the scratch root, the home folder, the user name and the host name replaced with `$SCRATCH`, `$HOME`, `<user>` and `<host>` (`exp/bin/sanitize.py`).

## Layout

| Path | Content |
|---|---|
| `exp/findings.md` | The E1 to E8 report: what ran, the results, the evidence and the labels of each experiment |
| `exp/bin/` | The E1 to E8 scripts: `case.sh` and `inner.sh` (the sandbox), `stub.py` (the recording stub), `gen_kimi.py` (E1), `e2.py`, `e2c.sh`, `e3.py`, `e4.sh`, `e4_web.sh`, `e4_registry.sh`, `e5e7.py`, `e6.sh`, `tui.py` with its `tui-*.json` step files, `check_fixtures.py`, `tree_bytes.py` and `sanitize.py` |
| `exp/entries/` | E1's generated tables, `<runtime>/<id>.toml` for each of the 19 recipes, and `index.json` |
| `exp/writer/` | E4's eight files, the hand-made inputs `f0` and `f2` and the six files Kimi wrote from them, with `.stat` files, `index.json` (how each was made) and `check.json` (the `tomllib` check) |
| `exp/records/` | Exact `kimi provider list --json` outputs, among them the one for a file that is not valid TOML |
| `exp/neg/` | The negative candidates of E6 |
| `exp/agents/` | E3's agent files |
| `exp/registry/` | The two custom registry documents E4b and E4d served inside the sandbox |
| `exp/runs/<case>/` | Per case: `argv`, `exit`, `stdout`, `stderr`, `time`, and where they apply `records.jsonl`, `stub.log`, `strace.txt`, TUI steps and transcripts and `config.toml` snapshots; `<case>.stub.json` is the stub's script for that case, and `*-summary*.jsonl` sums up a series |
| `spec/bin/` | The P1 to P3 scripts: `p1.py`, `p2.py`, the sandbox and stub copies, and the TUI step files |
| `spec/agents/llmctl-chat-only.md` | The agent file P1 ran in print mode |
| `spec/runs/` | P1 to P3's cases as in `exp/runs/`, with `p1-summary.jsonl` and `p2-summary.jsonl` |
| `spec/proto/` | The prototype `kimi_proto.py` and three of its 88 outputs, the ones docs/clients.md cites |
| `handson/` | The hands-on README, `config.used.toml`, the thinking variants in `configs/`, `runs/<label>/`, the request shapes in `shapes/`, and the scripts `tools/run.sh`, `tools/inner.sh` and `tools/proxy.py` |
| `bin/sea_extract.py` | The research lane's extractor of the bundle from the Kimi binary |
| `research-claims.json` | The research claims the documents cite by id ([Research claims](#research-claims)) |

## The bundle

Many claims cite lines of `main.cjs`.
main.cjs is the bundle that bin/sea_extract.py extracts from the Kimi Code 2.5.3 binary: 23,705,892 bytes, sha256 0fc8a727c50dda755dcc19105ccaae0d29c0626acf85f42ccb3d3700c8fffc57 (local-K02); it is not archived, because it is Moonshot's code.
`python3 -I bin/sea_extract.py $HOME/.kimi-code/bin/kimi <out dir>` writes it again from an installed 2.5.3 binary as `<out dir>/main.cjs`, together with the binary's assets, and prints its size and sha256 for comparison.

## Claims and files

| Claim | What it shows | Files |
|---|---|---|
| E1 | The generated tables of every recipe: 19 entries, 10 toggle, 3 effort, 5 always and 1 none, the two DeepSeek-R1 recipes with `capabilities = []` | `exp/entries/`, `exp/bin/gen_kimi.py` |
| E2 | Kimi accepts all 19 entries in `provider list --json` and answers `OK` through each in 0.57 to 0.63 s; the request keys per thinking type, reasoning replayed under the server's own field, and what the global `[thinking]` switches send | `exp/runs/e2-summary-all.jsonl`, `exp/runs/e2-*/`, `exp/runs/e2r-*/`, `exp/runs/e2t-*/`, `exp/runs/e2c-*/`, `exp/records/provider-list-*.json`, `exp/bin/e2.py`, `exp/bin/e2c.sh` |
| E3 | A chat-only recipe without an agent file sends 24 tools and exits 1 on a vLLM-like HTTP 400; with `tools: []` no `tools` key is sent and the run exits 0 | `exp/runs/e3-summary.jsonl`, `exp/runs/e3-*/`, `exp/agents/`, `exp/bin/e3.py` |
| E4a | Kimi's first `default_model` rewrites the whole file without its comments (web route and TUI route); an existing `default_model` is edited in place | `exp/writer/a-web-set-default.toml`, `exp/writer/a2-tui-model-pick.toml`, `exp/writer/a3-web-switch-default.toml`, `exp/runs/e4a-web-set-default/`, `exp/runs/e4a3-web-switch-default/`, `exp/runs/e8-tui/c1-after-pick.toml`, `exp/bin/e4.sh`, `exp/bin/e4_web.sh` |
| E4b | `kimi provider add` rewrites the file and puts the new provider between llmctl's two tables | `exp/writer/b-provider-add.toml`, `exp/runs/e4b-provider-add/`, `exp/registry/v1/api.json`, `exp/bin/e4_registry.sh` |
| E4c | `kimi provider remove` removes by line edits, keeps llmctl's tables byte for byte and leaves blank lines | `exp/writer/c-provider-remove.toml`, `exp/runs/e4c-provider-remove/` |
| E4d | A `kimi web` start refreshes a registry provider and leaves llmctl's `model_source = "static"` tables byte for byte | `exp/writer/d-web-start-refresh.toml`, `exp/runs/e4d-web-start-refresh/`, `exp/registry/v2/api.json` |
| E5 | A `default_model` that names a missing alias makes print mode exit 1 in about 0.5 s with no request, even with a valid `-m`; no default plus `-m` fails too | `exp/runs/e5e7-summary-e5.jsonl`, `exp/runs/e5-*/`, `exp/bin/e5e7.py` |
| E6 | `kimi --version` and `kimi provider list --json` take 0.27 to 0.32 s each and write 1.93 MB into a fresh cache; `provider list` exits 0 with empty maps on a file that is not valid TOML | `exp/runs/e6-summary.jsonl`, `exp/runs/e6-*/`, `exp/records/`, `exp/neg/`, `exp/runs/neg-*/`, `exp/runs/doctor-*/`, `exp/bin/e6.sh` |
| E7 | A stopped server makes print mode exit 1 after 151.2 s of 10 attempts | `exp/runs/e5e7-summary-e7.jsonl`, `exp/runs/e7-refused-json/`, `exp/runs/e7-reset-by-bridge/` |
| E8 | An open TUI: a changed alias fails the next turn, a kept alias uses the new entry with the old thinking level, no write-back; an idle TUI looked up `api.kimi.com` and `code.kimi.com` with all three switches set | `exp/runs/e8-tui/`, `exp/runs/e8-tui-same/`, `exp/runs/e8-tui-m/`, `exp/runs/e8-tui-idle-strace/`, `exp/runs/e8-probe/`, `exp/bin/tui.py`, `exp/bin/tui-*.json` |
| P1 | The TUI ignores `--agent-file` with `tools: []` and still sends 24 tools and Kimi's default system prompt; print mode with the shipped file sends no `tools`, an 87-character system prompt, and answers `PONG-ONE` | `spec/runs/p1-summary.jsonl`, `spec/runs/p1-*/`, `spec/agents/llmctl-chat-only.md`, `spec/bin/p1.py`, `spec/bin/tui-p1.json` |
| P2 | Print mode sends the global `[thinking] effort` verbatim as `reasoning_effort`, also outside an entry's efforts and for an entry without the `thinking` capability | `spec/runs/p2-summary.jsonl`, `spec/runs/p2-*/`, `spec/bin/p2.py` |
| P3 | The TUI's `/effort` for an effort entry offers `Off  Low  [ Medium ]` and saves `[thinking]` directly after llmctl's last line | `spec/runs/p3-tui-effort-qwen38/`, `spec/runs/p3-tui-effort-low-qwen38/`, `spec/bin/tui-p3.json`, `spec/bin/tui-p3b.json` |
| Prototype | Finding, replacing and inserting llmctl's tables on 11 inputs times 4 targets times with and without the default: all 88 results parse back as intended, a second plan changes nothing, and the 19 rendered tables equal E1's files byte for byte | `spec/proto/kimi_proto.py`, `spec/proto/out/owner--gemma--nodefault.toml`, `spec/proto/out/owner--gemma--default.toml`, `spec/proto/out/b-provider-add--gptoss-vllm--nodefault.toml` |
| Hands-on | A hand-written entry against the running gemma-4-26b-a4b server: a plain reply, a `Write` plus `Bash` run and a shell command passed on the first attempt, 9 requests all HTTP 200, only 127.0.0.1 contacted | `handson/README.md`, `handson/config.used.toml`, `handson/configs/`, `handson/runs/`, `handson/shapes/`, `handson/tools/` |

The labels are "(measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, recording stub)" for runs answered by the stub, "(measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, no server)" for runs that needed none, and "(measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, llama.cpp b12407)" for the hands-on runs.
The hands-on README names `work/<label>/` and a `before.txt` and `after.txt` listing; neither is here, because the work folders were throwaway and the listings named files in the owner's home.

## Research claims

[research-claims.json](research-claims.json) keeps exactly the research claims that the documents cite: 37 claims of the docs lane and 34 of the binary lane, each reduced to `id`, `claim`, `verdict` and `sources`.
An id `docs-K036` is claim K036 of the docs lane, which read the kimi.com, GitHub and npm pages, and an id `local-K25` is claim K25 of the binary lane, which read the 2.5.3 binary and its bundle and ran it offline; the prefixes keep the two id spaces apart.
A docs claim's `sources` is the page it quotes.
A local claim's `sources` is the command output or bundle lines it rests on, with `main.cjs:<lines>` meaning lines of the bundle above and, where it cites them, the note on that bundle first.
A path that starts with `$SCRATCH/` in `sources` names a working file of the research that was not archived (Kimi's help and version output, module and URL lists from the bundle, the sandbox homes).
Every `verdict` is the result of the independent source check of 2026-11-17, described in the Kimi Code CLI section of [docs/sources.md](../../../docs/sources.md#kimi-code-cli): it starts `Holds`, `Holds with corrections: <correction>` or `Does not hold: <why>`, followed by `; checked` and how.
Of the 71 claims, 67 hold, 4 hold with corrections (docs-K152, docs-K154, local-K07 and local-K09) and none does not hold.
A docs claim was checked against the kimi.com page and its GitHub Pages copy fetched that day, or for docs-K007 and docs-K008 against the npm registry record; a local claim was checked by reading its cited lines of the bundle above, whose sha256 matched, or the command record it cites.

## Rerun

Every rerun needs Kimi Code CLI 2.5.3 at `$HOME/.kimi-code/bin/kimi`, `bwrap`, `socat`, `strace`, the `heavy` wrapper, and `SCRATCH` set to a work folder of your own.
The scripts write `runs/`, `homes/`, `work/` and `net/` next to their own `bin/`, so copy the folder first and run the copy: `cp -a research/2026-11-17/llmctl-kimi-stub $SCRATCH/kimi-rerun`.
Each case makes its own private Kimi home; never point `HOMEDIR` or `KIMI_CODE_HOME` at a real home.

- E1: `python3 -I -B $SCRATCH/kimi-rerun/exp/bin/gen_kimi.py <repository> $SCRATCH/kimi-rerun/exp/entries` writes the tables of every recipe from the repository's recipes.
- E2: `python3 -I $SCRATCH/kimi-rerun/exp/bin/e2.py [plain|reasoning|thinking|all] [case filter]`; E3: `python3 -I $SCRATCH/kimi-rerun/exp/bin/e3.py`; E5 and E7: `python3 -I $SCRATCH/kimi-rerun/exp/bin/e5e7.py [e5|e7|all]`; E6: `bash $SCRATCH/kimi-rerun/exp/bin/e6.sh`.
- One case by hand: `STUBCFG=<stub json> BPORT=8080 heavy bash $SCRATCH/kimi-rerun/exp/bin/case.sh <case> kimi -p "Reply with exactly OK." --output-format stream-json`.
- E4's fixtures check: `python3 -I $SCRATCH/kimi-rerun/exp/bin/check_fixtures.py $SCRATCH/kimi-rerun/exp/writer $SCRATCH/kimi-rerun/exp/entries/index.json`.
- P1 and P2: `python3 -I $SCRATCH/kimi-rerun/spec/bin/p1.py` and `python3 -I $SCRATCH/kimi-rerun/spec/bin/p2.py`.
- Prototype: `python3 -I $SCRATCH/kimi-rerun/spec/proto/kimi_proto.py <repository> $SCRATCH/kimi-rerun/exp $SCRATCH/kimi-rerun/spec/proto/out`.
- Records for publication: `python3 -I $SCRATCH/kimi-rerun/exp/bin/sanitize.py "$SCRATCH" "$HOME" "$(hostname)" "$(id -un)" <folder>` before anything is copied back.

## Privacy

Left out, because they are Moonshot's: the extracted bundle and its assets, Kimi's built-in skill texts and the raw page downloads of the docs lane.
Left out, because they are private or hold whole prompts: every private Kimi home with its `device_id`, sessions, logs and caches, every work, net and socket folder, the hands-on `diff.txt`, which lists the owner's home, and the research lane's own notes and full claim files, which describe the owner's own Kimi home in places.
The records were searched for Kimi's default system prompt, secrets, e-mail addresses and IP addresses other than 127.0.0.1.
What remains is placeholder keys (`no-key-required`, `llmctl-local`, `placeholder`), `127.0.0.53` (the local DNS stub in E8's idle TUI strace), `user@1000.service` (a systemd unit in the hands-on c-1 cgroup line) and paths such as `/home/runner` from the binary's build path.
