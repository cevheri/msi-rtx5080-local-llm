# Lead checks of the Kimi e2e

Last verified: 2026-11-17.

The TUI checklist is in [tui/README.md](tui/README.md).

## L-c: one Kimi print-mode session under strace

The lead ran it after the e2e, with gemma-4-26b-a4b started by `./llmctl up gemma-4-26b-a4b` on llama.cpp `server-cuda-b12407`.
The session ran as the owner runs Kimi: `KIMI_CODE_NO_AUTO_UPDATE=1` from the environment, `KIMI_DISABLE_TELEMETRY` and `KIMI_CODE_REGION_MARKER` unset, and `telemetry = false` from a byte copy of the owner's `config.toml` in a private Kimi home.
The copy got llmctl's default edit, `default_model = "llmctl/local"`, because print mode needs a default even with `-m` (Kimi's E5 in `../../llmctl-kimi-stub/README.md`).
The command was `strace -f -e trace=connect,sendto` around `kimi -p "Reply with exactly OK." --output-format stream-json`, under `heavy`, with a private `HOME`, `TMPDIR` and `KIMI_CODE_CACHE_DIR`.
Result: exit 0 with the answer `OK`, one AF_INET connect to 127.0.0.1 port 8080, two AF_UNIX connects to the local nscd socket, which does not exist (ENOENT), no `sendto` call and no DNS lookup (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, llama.cpp b12407).
The server log shows the request as one task of 12,205 tokens in 4,677 ms.
[connect.txt](connect.txt) holds the connect lines with their counts; the private home, its session files and the full strace were removed after the check.
