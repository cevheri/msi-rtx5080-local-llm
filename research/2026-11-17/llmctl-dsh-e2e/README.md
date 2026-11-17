# llmctl clients e2e with DeepSeek Harness, 2026-11-17

Last verified: 2026-11-17.

This folder holds the second full run of the llmctl clients e2e on 2026-11-17: every shipped recipe, 11 on llama.cpp b12407 and 8 on vLLM v0.33.0, with OpenCode v2.3.0, Hermes Agent v0.23.1, Pi 1.5.2 and DeepSeek Harness (dsh) 0.2.2-rc.3, then the `--set-default` round trip.
The curated results are in [docs/clients.md](../../../docs/clients.md#verified-on-this-laptop).
The driver wrote `summary.md` and `records/`; do not edit them by hand.

The command was:

```bash
heavy python3 -m launcher.e2e.clients --out research/2026-11-17/llmctl-dsh-e2e --matrix all --set-default-round-trip
```

## Files

- [summary.md](summary.md): the driver's table, one row per step with its pass mark, reasons and seconds; 284 of 294 rows read Yes, and the driver exited 1.
- [records/](records/): one file per command or probe, numbered in run order; records 001 to 005 hold the version lines of Hermes, Pi, OpenCode, dsh and `nvidia-smi`.
  Each record shows the private work folder as `$WORK` and the home folder as `$HOME`.
- [lead-checks/](lead-checks/README.md): the lead's records of the L2 migration and the L5 checks, described below.
- The first full run of the day, whose kept records are in [../llmctl-dsh-e2e-run1/](../llmctl-dsh-e2e-run1/README.md), is described below.

## Lead checks (L2 and L5)

(measured on the MSI Raider 18 Max HX, llama.cpp b12407, vLLM v0.33.0, dsh 0.2.2-rc.3, 2026-11-17)

The L2 migration on 2026-11-17: the lead's hand-made rows were removed from the web and headless profile patches, `LLMCTL_CLIENTS=dsh ./llmctl up gemma-4-26b-a4b --set-default` added llmctl's block to the home patch with nothing started, the telemetry row kept its bytes, `dsh --dump-config` of web and headless exited 0 with only the acp warning, and a private-home headless session answered pong.
The L5 cgroup check: a command run by dsh's bash tool inside a heavy-run session landed in `app.slice/dsh-subprocess-<pid>-<id>.scope` under the user manager, outside `heavy.slice`, so heavy's memory cap does not cover dsh's tool commands.
The L5 connect check: a headless session under `strace -f -e trace=connect` without `DSH_TELEMETRY_DISABLED` made two connect calls, both to 127.0.0.1, no DNS lookup and no other address (the owner's home patch disables `session-telemetry-otel`).
A search of the records for `TIMEOUT` and for queued steps found none.
The records of these checks are in [lead-checks/](lead-checks/README.md).

## First run

The first full run on 2026-11-17 exited 1 with 280 Yes and 14 No rows.
Twelve of its No rows were client sessions, ten failing through model behaviour and two through the known K2 Horizon 7B rejection on vLLM ("Assistant message is missing a thinking field"), and two were probe rows that held three failed probes.
K2 Horizon 7B's vLLM probe A got HTTP 400 "Unsupported reasoning_effort 'none'", so its vLLM recipe gained the `--reasoning-effort-map` of the gpt-oss-20b recipe.
vLLM deepseek-r1-0528-qwen3-8b's probe A got a garbled title from the known tokenizer failure, and its probe B got HTTP 400 because the driver sent tools to a chat-only recipe, so the driver no longer sends probe B to a chat-only recipe.
Every up, status, config, dsh-config, dsh-web, gateway and restore row of that run read Yes.
Its summary and the records the documents cite are in [../llmctl-dsh-e2e-run1/](../llmctl-dsh-e2e-run1/README.md); the records in this folder are those of the second run.
