# Lead checks L2 and L5, 2026-11-17

The lead's records of the L2 migration and the L5 checks of the llmctl dsh plan, copied here after the e2e run and reduced so that they hold no prompt text, reply text, session id or working folder.
Measured on the MSI Raider 18 Max HX, llama.cpp b12407, dsh 0.2.2-rc.3, 2026-11-17.

- [dry-run.txt](dry-run.txt): `LLMCTL_CLIENTS=dsh ./llmctl up gemma-4-26b-a4b --set-default --dry-run`, with the `would back up` and `would write` lines and the dry-run closing line.
- [up.txt](up.txt): the same command without `--dry-run`, which found gemma running, started nothing, backed up the home patch and added llmctl's rows after the owner's rows.
- [web.err](web.err) and [headless.err](headless.err): the stderr of `dsh --profile web --dump-config` and `dsh --profile headless --dump-config`, each the single acp warning.
- [dump-patched-by.txt](dump-patched-by.txt): every `patched by` header of the two dumps with the row below it, which shows `agent-default-model`, `llm-pi-ai` and `session-telemetry-otel` patched by the home patch.
- [pong.events.jsonl](pong.events.jsonl): the private-home headless session asked for `pong`, reduced to events; it ends with a `final` event of kind `completed`.
- [cgroup.events.jsonl](cgroup.events.jsonl): the L5 session that ran `cat /proc/self/cgroup` through dsh's bash tool, reduced to events.
- [cgroup-tool-result.jsonl](cgroup-tool-result.jsonl): that bash call's tool result, kept readable: the command ran in `app.slice/dsh-subprocess-<pid>-<id>.scope` under the user manager, outside `heavy.slice`.
- [connect-session.events.jsonl](connect-session.events.jsonl): the L5 session run under strace without `DSH_TELEMETRY_DISABLED`, reduced to events.
- [connect.strace.txt](connect.strace.txt): the connect lines of that strace, two calls, both to 127.0.0.1 port 8080.
