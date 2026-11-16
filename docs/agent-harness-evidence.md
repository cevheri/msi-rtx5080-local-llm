# Agent-harness evidence from LibreDB Studio

Last verified: 2026-11-16.

The user's LibreDB Studio agent harness passed 39 local Ollama models and one hosted Gemini model 30/30 in a real tool-calling loop, which is the first first-hand tool-reliability evidence this knowledge base has.
On this laptop, six of them are worth measuring first: Nemotron 3.5 Lightning, Qwen3.5-9B, Gemma 4 12B, Ornith-1.0-9B, Ministral 3 8B and Qwen2.5-7B.
All six ran here on 2026-11-16: Lightning on llama.cpp, and the five Ollama tags on Ollama 0.40.16, where each fit fully on the GPU, decoded at 99-119 tok/s and passed the LibreDB `tool_calls` probe, a tool-call test and a short OpenCode session ([Try here first](#try-here-first)).
The evidence confirms the KB's thinking-off rule and its ranks 2, 5 and 9, but it does not test coding quality and it does not by itself change the top 10.
The timings come from an M5 Max with 614 GB/s unified memory: models that fit in 16 GB of VRAM should run somewhat faster here, and models that spill to DDR5 will run slower.

## What the harness measures

LibreDB Studio's Agent mode drives a model through six surfaces (Investigate, Optimize, Assess, Operate, Analyze, Plan) against an embedded SQLite sample ([docs/llms/methodology.md](https://github.com/libredb/libredb-studio/blob/c419b8bf1b156cff7a05bacf5cce5b3c0eb4b666/docs/llms/methodology.md)).
A model is listed as supported only when each surface reached the verdict `answered` five consecutive times, so a listed model has 30 passes out of 30 runs ([docs/llms/README.md](https://github.com/libredb/libredb-studio/blob/c419b8bf1b156cff7a05bacf5cce5b3c0eb4b666/docs/llms/README.md)).
A pass means the model called tools, read what came back, kept going, and finished by filing claims that cite what it read.
That is the failure mode OpenCode cares about most: missing, malformed or abandoned tool calls in a multi-turn loop.
Five consecutive passes per surface is a reliability test, not a best-of-N score.

Its limits matter as much as its result.
The verifier checks that a run read, presented and cited data, and SQL correctness is "largely unchecked" ([docs/llms/methodology.md](https://github.com/libredb/libredb-studio/blob/c419b8bf1b156cff7a05bacf5cce5b3c0eb4b666/docs/llms/methodology.md)).
The prompts are small, so no run exercised 64k-scale contexts like OpenCode's (its first request was 3,648 tokens in a restricted test here, and it needs at least 64k).
The loop is guarded: Studio adds report reminders, an unreported-call ceiling, optional retries of empty turns and plan statements, and per-model reasoning suppression ([docs/llms/model-tuning.md](https://github.com/libredb/libredb-studio/blob/c419b8bf1b156cff7a05bacf5cce5b3c0eb4b666/docs/llms/model-tuning.md)).
OpenCode has none of these guards, so a model that needed a guard to pass can still fail in OpenCode.
Studio also falls back to a prose tool protocol when an endpoint lacks native tool calls, decided per endpoint, so a pass does not by itself prove native `tool_calls` emission ([docs/llms/setup.md](https://github.com/libredb/libredb-studio/blob/c419b8bf1b156cff7a05bacf5cce5b3c0eb4b666/docs/llms/setup.md), around line 95).
The default request is temperature 0 and top_p 1, overriding the Ollama tag defaults, while the KB recipes use model-card sampling ([docs/llms/model-tuning.md](https://github.com/libredb/libredb-studio/blob/c419b8bf1b156cff7a05bacf5cce5b3c0eb4b666/docs/llms/model-tuning.md)).
A few profiles override it: bonsai-2:2b and :4b at 0.3, gpt-oss:20b and qwen3:8b at 0.8 / top_p 0.9 on optimize only, and a separate qwen3.8:27b profile at 0.3 ([src/lib/agent/model-tuning/measured-profiles.json](https://github.com/libredb/libredb-studio/blob/c419b8bf1b156cff7a05bacf5cce5b3c0eb4b666/src/lib/agent/model-tuning/measured-profiles.json)).

## The Mac setup and why its timings do not transfer

All figures below are (measured on the M5 Max, libredb-studio) unless labelled otherwise.

| Item | Value | Source |
|---|---|---|
| Machine | Apple M5 Max, 64 GB unified memory, macOS high-power mode on AC | [docs/llms/methodology.md](https://github.com/libredb/libredb-studio/blob/c419b8bf1b156cff7a05bacf5cce5b3c0eb4b666/docs/llms/methodology.md), lines 45-50 |
| Memory bandwidth | 614 GB/s: 64 GB is sold only with the 40-core-GPU M5 Max; the 32-core part is 460 GB/s and 36 GB only | [apple.com MacBook Pro specs](https://www.apple.com/macbook-pro/specs/) |
| Engine | Ollama 0.32.13 and 0.33.0 through the OpenAI-compatible `/v1` endpoint; bonsai-2 on 0.40.6 (17-22 Oct 2026) and lfm3 on 0.40.13 | [docs/llms/methodology.md](https://github.com/libredb/libredb-studio/blob/c419b8bf1b156cff7a05bacf5cce5b3c0eb4b666/docs/llms/methodology.md), [docs/llms/bonsai/bonsai-2.md](https://github.com/libredb/libredb-studio/blob/c419b8bf1b156cff7a05bacf5cce5b3c0eb4b666/docs/llms/bonsai/bonsai-2.md), [docs/llms/liquid/lfm3.md](https://github.com/libredb/libredb-studio/blob/c419b8bf1b156cff7a05bacf5cce5b3c0eb4b666/docs/llms/liquid/lfm3.md) |
| Dates | 22-23 August 2026 per the methodology, but some models were chosen or released later (qwen3.8:27b on 2026-10-09, Granite 4.2 on 2026-08-25) | [docs/llms/methodology.md](https://github.com/libredb/libredb-studio/blob/c419b8bf1b156cff7a05bacf5cce5b3c0eb4b666/docs/llms/methodology.md), line 10; [src/lib/agent/model-tuning/measured-profiles.json](https://github.com/libredb/libredb-studio/blob/c419b8bf1b156cff7a05bacf5cce5b3c0eb4b666/src/lib/agent/model-tuning/measured-profiles.json); [Granite 4.2 verification record](../research/2026-11-15/verification/verify_IBM_Granite_4.2_8B.json) |
| Per-turn limit | 90 s shipped; nine models carry 150 s | [docs/llms/README.md](https://github.com/libredb/libredb-studio/blob/c419b8bf1b156cff7a05bacf5cce5b3c0eb4b666/docs/llms/README.md) |
| Context | Not recorded, except OLLAMA_CONTEXT_LENGTH=32768 for six models including qwen3.6:35b | [docs/llms/qwen/qwen3.6.md](https://github.com/libredb/libredb-studio/blob/c419b8bf1b156cff7a05bacf5cce5b3c0eb4b666/docs/llms/qwen/qwen3.6.md), lines 68-69 |
| Thermal | A long sweep lost 18 of 270 runs to heat; every lost cell was re-measured on a rested machine | [docs/llms/methodology.md](https://github.com/libredb/libredb-studio/blob/c419b8bf1b156cff7a05bacf5cce5b3c0eb4b666/docs/llms/methodology.md), lines 80-90 |

Four facts stop the seconds from transferring.

1. The durations are whole agent runs (several turns, prefill, decode and tool calls), reported as the median and slowest passing run, not tok/s.
   The methodology itself calls them a best case and says the pass/fail column is the transferable part.
2. Bandwidth differs.
   For a model that fits fully in 15.47 GiB of CUDA memory, decode is bandwidth-bound on both machines and this GPU has 896/614 = 1.46x the Mac's bandwidth (estimate).
   Runs that are dominated by thinking length or tool latency shrink by less than that.
3. Fit differs.
   The Mac holds up to about 48 GiB of model in fast memory (estimate).
   Here, anything over about 15 GiB with its KV cache spills to DDR5-6400 (102.4 GB/s peak, configured speed unverified), and the spilled part runs several times slower than on the Mac.
   For a spilling model, expect slower runs and a higher risk of hitting the per-turn limit (estimate).
4. Engine and build differ.
   gemma4:12b went from 5/5 to 1/5 on one cell between Ollama 0.32.13 and 0.33.0 with the same digest ([docs/llms/methodology.md](https://github.com/libredb/libredb-studio/blob/c419b8bf1b156cff7a05bacf5cce5b3c0eb4b666/docs/llms/methodology.md), lines 52-67).
   Ollama is at 0.40.16 now, and the KB's recipes use llama.cpp, often at a different quant and with a different tool parser.

Context also differs.
Ollama's default context is 4k on GPUs under 24 GiB, 32k at 24-48 GiB and 256k at 48 GiB or more ([Ollama docs](https://docs.ollama.com/context-length)).
The Mac probably ran at 32k or 256k (unknown, since Metal's working set on a 64 GB Mac is close to the 48 GiB threshold), while this laptop gets 4k unless OLLAMA_CONTEXT_LENGTH is set.
Any Ollama run here needs OLLAMA_CONTEXT_LENGTH set explicitly: at least 65536 for OpenCode, and at least 32768 to approach the Mac's likely harness condition; the KV cache for that context has to fit too (see [opencode.md](opencode.md)).

## All harness models on this laptop

The libredb README table has 40 rows: 39 local Ollama tags and one hosted Gemini model.
The README says 51 models are supported, so 11 supported models have no published timings yet and are not covered here.
"Mac median / slowest" is (measured on the M5 Max, libredb-studio), in seconds, from the README table.
"Ollama size" is the size on the [ollama.com library](https://ollama.com/library) tags page as checked on 2026-11-16, with the registry manifest at `https://registry.ollama.ai/v2/library/<name>/manifests/<tag>`.
"Fits here", "Best runtime here" and "Speed here vs Mac" are (estimate) except in the cells marked (measured on this laptop), which are llama.cpp runs of the KB recipes for gpt-oss:20b, qwen3.8:latest, qwen3.6:35b and nemotron-3.5-lightning:30b, and Ollama 0.40.16 runs of qwen2.5:7b, qwen3.5:9b, gemma4:12b, ministral-3:8b and ornith:9b.
Fit is judged against 15.47 GiB of CUDA memory at 64k context unless the row says otherwise.

| Tag | Mac median / slowest | Ollama size | Fits here (estimate) | Best runtime here | Speed here vs Mac (estimate) | KB status | Recommendation |
|---|---|---|---|---|---|---|---|
| qwen2.5:7b | 6 / 21 | 4.7 GB | Yes, about 9 GiB at 64k f16 KV; native 32k, 64k needs YaRN; 6.5 GiB peak at 32k (measured on this laptop) | Ollama for the harness; llama.cpp with YaRN for 64k | 119.4 tok/s on Ollama 0.40.16 at 32k (measured on this laptop), against an estimated ceiling of about 190 | Not evaluated | Try here first; measured |
| qwen2.5:14b | 18 / 65 | 9.0 GB | 32k with q8_0 KV; 64k only with q4_0 KV plus YaRN | Ollama at 32k with q8_0 KV; llama.cpp for 64k | Faster while resident, ceiling about 100 vs 68 tok/s | Not evaluated | Worth trying |
| qwen2.5:32b | 20 / 36 | 20 GB | No, about half the layers in RAM | None recommended | Much slower, about 3-4x | Not evaluated | Skip here |
| qwen2.5-coder:14b | 20 / 37 | 9.0 GB | As qwen2.5:14b | Ollama, after a native tool_calls probe | Faster while resident | Not evaluated; third-party Tools task failed | Worth trying |
| qwq:32b | 76 / 360 | 20 GB | No, about half the layers in RAM | None recommended | Much slower; 90 s turns likely break | Not evaluated | Skip here |
| qwen3:4b | 72 / 160 | 2.5 GB | Yes, about 12.3 GiB at 64k f16 KV | Ollama | Faster | Not evaluated | Worth trying |
| qwen3:8b | 36 / 108 | 5.2 GB | Yes at its native 40,960; 64k needs YaRN | Ollama at 40k | Faster | Not evaluated | Skip here |
| qwen3:14b | 41 / 303 | 9.3 GB | 40k with q8_0 KV; 64k too tight | Ollama at 40k with q8_0 KV | Faster while resident | Not evaluated | Skip here |
| qwen3-coder:30b | 11 / 21 | 19 GB | No, experts partly in RAM | llama.cpp, --n-cpu-moe about 20, q8_0 KV | Slower or similar | Excluded (coding scores) | Worth trying |
| qwen3.5:4b | 11 / 167 | 3.3-4.0 GB | Yes, about 6 GiB at 64k | Ollama | Faster | Not evaluated | Skip here |
| qwen3.5:9b | 33 / 98 | 6.6-7.6 GB | Yes, 10.4 GiB peak at 128k with f16 KV (measured on this laptop) | Ollama at 128k | 98.7 tok/s on Ollama 0.40.16 (measured on this laptop) | Honorable mention (vLLM NVFP4) | Try here first; measured |
| qwen3.5:27b | 78 / 239 | 17-20 GB | No at Q4_K_M; a 3-bit GGUF would fit | llama.cpp 3-bit, if at all | Slower on Ollama | Excluded (superseded by Qwen3.8-27B) | Skip here |
| qwen3.6:27b | 82 / 252 | 18-19 GB | No at Q4_K_M | llama.cpp 3-bit, if at all | Slower on Ollama | Excluded (superseded by Qwen3.8-27B) | Skip here |
| qwen3.6:35b | 13 / 172 | 23 GB | Not as one GPU load; the KB recipe keeps 10.5 GiB of experts in RAM and peaks at 15,190 MiB at 128k (measured on this laptop) | llama.cpp KB recipe, `--n-cpu-moe 25` plus a 3,200 MiB expert cache, 122.5 tok/s (measured on this laptop) | Unknown: the harness recorded no Mac tok/s | Rank 2 | Already in KB plan |
| qwen3.8:latest | 72 / 195 | 18 GB | No at Q4_K_M; UD-Q3_K_XL fits, 14,790 MiB peak at 64k (measured on this laptop) | llama.cpp KB recipe (UD-Q3_K_XL), 36.4 tok/s (measured on this laptop) | Slower on Ollama; faster at 3-bit in VRAM | Rank 1 | Already in KB plan |
| gemma4:e4b | 34 / 152 | 6.6-9.5 GB | Yes, about 8 GiB at 64k | Ollama | Faster or similar | Excluded (agentic scores) | Worth trying |
| gemma4:12b | 11 / 63 | 7.7-8.0 GB | Yes, 10.2 GiB peak at 64k with f16 KV (measured on this laptop) | Ollama, thinking off | 117.7 tok/s on Ollama 0.40.16, which drafts with the model's MTP head (measured on this laptop) | Honorable mention | Try here first; measured |
| gemma4:26b | 36 / 92 | 16-19 GB | Only as QAT Q4_0 on llama.cpp; the Ollama default spills | llama.cpp KB recipe | Faster on llama.cpp in VRAM; slower on the Ollama default | Rank 5 | Already in KB plan |
| gemma4:31b | 107 / 398 | 19-20 GB | No, 35-45% of a dense model in RAM | None recommended | Much slower, about 3x | Excluded | Skip here |
| gemini-3.5-flash-lite | 10 / 21 | Hosted | Hosted only | Gemini API | Not applicable | Out of scope (local KB) | Hosted only |
| ministral-3:8b | 13 / 26 | 6.0 GB | Yes, 10.6 GiB peak at 64k with q8_0 KV (measured on this laptop) | Ollama with q8_0 KV | 99.9 tok/s on Ollama 0.40.16, but 35 after a 58k prompt (measured on this laptop) | Excluded (agentic scores) | Try here first; measured |
| ministral-3:14b | 16 / 56 | 9.1 GB | Tight, about 14.5 GiB at 64k q8_0 KV | Ollama with q8_0 KV, or llama.cpp with --no-mmproj | Faster if nothing spills | Excluded (agentic scores) | Worth trying |
| magistral:24b | 22 / 47 | 14 GB | No; context capped at 40,960 | None for OpenCode | Slower | Not evaluated | Skip here |
| granite4.1:8b | 11 / 22 | 5.3 GB | Yes, about 10.8 GiB at 64k q8_0 KV | Ollama with q8_0 KV | Faster | Not evaluated | Worth trying |
| granite4.1:30b | 26 / 52 | 17 GB | No, dense, about 40% in RAM | None recommended | Much slower, about 3x | Not evaluated | Skip here |
| granite4.2:8b | 46 / 116 | 5.3 GB | Yes | llama.cpp or vLLM (the Ollama tag has no template or tools badge) | Faster per token | Excluded (agentic scores) | Skip here |
| cogito:14b | 11 / 20 | 9.0 GB | 32k with q8_0 KV; 64k needs q4_0 KV | Ollama at 32k | Faster while resident | Not evaluated | Worth trying |
| cogito:32b | 23 / 39 | 20 GB | No, dense, about 45% in RAM | None recommended | Much slower, 3-4x | Not evaluated | Skip here |
| gpt-oss:20b | 37 / 169 | 14 GB | Yes: 13,360 MiB at 128k on llama.cpp (measured on this laptop) | llama.cpp KB recipe, 166 tok/s (measured on this laptop) | Faster | Rank 9 | Already in KB plan |
| nemotron-3.5-lightning:30b | 29 / 145 | 19-25 GB | Not as one GPU load; the KB recipe keeps 7.4 GiB of experts in RAM and peaks at 15,106 MiB at 128k (measured on this laptop) | llama.cpp KB recipe, Q4_0, `--n-cpu-moe 25` plus a 3,200 MiB expert cache, 143.6 tok/s (measured on this laptop) | Unknown: the harness recorded no Mac tok/s | Honorable mention | Measured here; the harness's tool loop is not repeated |
| glm-4.7-flash:latest | 34 / 85 | 19 GB | No, experts partly in RAM; large KV | llama.cpp KB recipe, --n-cpu-moe 20, thinking off | Slower | Honorable mention | Worth trying |
| nemotron3:33b | 21 / 119 | 21-28 GB | No; about 3 GiB of it is vision and audio encoders | llama.cpp with the text-only GGUF | Slower | Not evaluated | Skip here |
| nemotron-3-nano:30b | 46 / 190 | 18-24 GB | No, experts partly in RAM | llama.cpp Q4_0 with --n-cpu-moe | Slower | Not ranked; superseded in discovery | Skip here |
| bonsai-2:2b | 52 / 324 | 911 MB | Yes, about 2-3 GiB | Ollama 0.40.6 or later | Similar or faster | Not evaluated | Skip here |
| bonsai-2:4b | 46 / 198 | 1.6 GB | Yes, about 4 GiB | Ollama 0.40.6 or later | Similar or faster | Not evaluated | Skip here |
| bonsai-2:9b | 55 / 113 | 3.2 GB | Yes, about 6 GiB | Ollama 0.40.6 or later, thinking off | Faster or similar | Not evaluated (27B sibling excluded) | Worth trying |
| lfm3:1.7b | 84 / 333 | 1.1-1.7 GB | Yes, about 2-4.5 GiB | Ollama 0.40.13 or later | Decode faster; the Mac likely ran a different artifact | Not evaluated | Skip here |
| ornith:9b | 31 / 118 | 5.6 GB | Yes, 9.3 GiB peak at 128k with f16 KV (measured on this laptop) | Ollama | 101.2 tok/s on Ollama 0.40.16 (measured on this laptop), against an estimated ceiling of about 159 | Not evaluated (Ornith-1.5-9B excluded) | Try here first; measured |
| ornith:35b | 36 / 150 | 21 GB | No, experts partly in RAM | llama.cpp, --n-cpu-moe about 22 | Slower | Not evaluated | Skip here |
| muse-glimmer:latest | 115 / 285 | 18 GB | No at Q4_K_M; UD-Q3_K_XL would fit | llama.cpp UD-Q3_K_XL, if at all | Slower at the tested quant | Excluded | Skip here |

The per-model pages behind each row are under [docs/llms/<vendor>/<family>.md](https://github.com/libredb/libredb-studio/blob/c419b8bf1b156cff7a05bacf5cce5b3c0eb4b666/docs/llms/<vendor>/<family>.md): `qwen/qwen2.5.md`, `qwen/qwen2.5-coder.md`, `qwen/qwq.md`, `qwen/qwen3.md`, `qwen/qwen3-coder.md`, `qwen/qwen3.5.md`, `qwen/qwen3.6.md`, `qwen/qwen3.8.md`, `gemma/gemma4.md`, `gemini/gemini-3.5.md`, `mistral/ministral-3.md`, `mistral/magistral.md`, `granite/granite4.1.md`, `granite/granite4.2.md`, `cogito/cogito.md`, `openai/gpt-oss.md`, `nvidia/nemotron-3.5-lightning.md`, `nvidia/nemotron3.md`, `nvidia/nemotron-3-nano.md`, `zhipu/glm-4.7-flash.md`, `bonsai/bonsai-2.md`, `liquid/lfm3.md`, `ornith/ornith.md` and `muse/muse-glimmer.md`.
The per-model settings are in [src/lib/agent/model-tuning/measured-profiles.json](https://github.com/libredb/libredb-studio/blob/c419b8bf1b156cff7a05bacf5cce5b3c0eb4b666/src/lib/agent/model-tuning/measured-profiles.json).

### Notes on individual rows

These notes carry the caveats that a table cell cannot.

- qwen2.5 and qwen2.5-coder: the GGUFs carry a 32,768-token context and no YaRN keys, so OLLAMA_CONTEXT_LENGTH=65536 alone does not enable YaRN; 64k needs llama.cpp `--rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 32768` ([Qwen2.5-7B-Instruct card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)).
  The 14b ceilings are about 100 vs 68 tok/s computed in bytes (estimate).
  Measured on 2026-11-16 on qwen2.5:7b with Ollama 0.40.16 at 32k: the LibreDB probe and a tool test passed, and the OpenCode session passed in 6 s after one recovery, because the model sent `write` and `read` as parallel calls, the read ran before the file existed, and it then read the file again ([record](../research/2026-11-16/ollama-qwen2.5-7b/)).
  Decode was 119.4 tok/s at a short context and 96 after a 24k prompt, with prefill at 5,044 tok/s for 8k tokens.
- qwen2.5-coder:14b: TechFuelHQ's Ollama run on a desktop RTX 5080 failed its two-turn Tools task ([community-benchmarks.md](community-benchmarks.md)), while the harness has 30/30 on first attempts.
  The two can both be true if Studio used its prose fallback, so run a native `tool_calls` probe before relying on it.
- qwen2.5:32b passes only with one plan-statement ask; without it the plan cell is 4/5 ([docs/llms/qwen/qwen2.5.md](https://github.com/libredb/libredb-studio/blob/c419b8bf1b156cff7a05bacf5cce5b3c0eb4b666/docs/llms/qwen/qwen2.5.md)).
- qwen3:4b is today the Qwen3-4B-Thinking-2507 build with a 256K context ([registry manifest](https://registry.ollama.ai/v2/library/qwen3/manifests/4b)), always thinking, which explains its slow runs for its size.
- qwen3:8b: the profile records only 3/5 on optimize at temperature 0.8, and attributes the 0/5 at temperature 0 to a false "no object" reply from Studio's own tool rather than to the model.
  Its 30/30 rests on the page alone.
  The skip verdict rests on its 40,960-token native context.
- qwen3:14b: the page says plan lost four runs in five, while the profile says 4/5 with one loss; the profile wording is the more likely one.
- qwen3-coder:30b: 30/30 at the compiled defaults, every cell on its first attempt, zero argument-shape errors, and the fastest tool caller in its lane ([docs/llms/qwen/qwen3-coder.md](https://github.com/libredb/libredb-studio/blob/c419b8bf1b156cff7a05bacf5cce5b3c0eb4b666/docs/llms/qwen/qwen3-coder.md)).
  The KB excluded it on SWE-bench Verified 31.8 and TB2.1 13.5, so it is a sub-agent candidate, not a coding pick.
- qwen3.5:9b: it needed a 150 s turn limit and one empty-turn retry.
  The KB also holds contrary evidence: the SQL benchmark scored Qwen3.5-9B UD-Q6_K_XL on llama.cpp at 5/25 with "mostly malformed tool calls" ([community-benchmarks.md](community-benchmarks.md)).
  The Ollama tag uses Ollama's own `qwen3.5` parser, so test that path here first and do not assume the llama.cpp result.
  Measured on 2026-11-16 on Ollama 0.40.16 with thinking off: the LibreDB `tool_calls` probe, a `get_weather` tool test and an OpenCode write-and-read session (7 s) all passed, so Ollama's own `qwen3.5` path works here ([record](../research/2026-11-16/ollama-qwen3.5-9b/)).
  Decode was 98.7 tok/s at a short context, 93 after an 8.6k prompt, 81 after 32k and 58 after 107k, and prefill 4,452 tok/s at 8.6k.
- qwen3.6:27b reached 30/30 on Ollama 0.32.13 with a 150 s limit and plan reasoning suppressed.
  Agent-side suppression was added later because Ollama 0.33.0 dropped optimize to 1/5; with it optimize passes in 76 s.
  It still spills here and is superseded.
- qwen3.6:35b: the page gives the slowest run as 29 s and the README as 172 s, probably because the README counts runs from before the settings were applied.
  With thinking on, optimize timed out at 200-350 s; it passed only with reasoning suppressed on plan and agent turns, at OLLAMA_CONTEXT_LENGTH=32768.
- qwen3.8:latest passed at Q4_K_M with thinking at the template default.
  The KB's rank-1 recipe uses UD-Q3_K_XL, medium effort and the llama.cpp Qwen3-Coder parser, none of which the harness tested.
  The KB's fit of about 15.2 GiB was checked against 15.92 GiB; against the 15.47 GiB CUDA reports, the margin is about 0.27 GiB (estimate).
  Measured on 2026-11-16, the recipe peaked at 14,790 MiB including the desktop, about 1 GiB below the 15,841 MiB that CUDA can use ([llamacpp/models.md](../llamacpp/models.md#1-qwen38-27b)).
- gemma4:12b passed only with reasoning suppressed on every turn and a 150 s limit; at the defaults it reached 15/30.
  It is engine-sensitive: the 1/5 on Ollama 0.33.0 held in three configurations.
  Measured on 2026-11-16 on Ollama 0.40.16 with thinking off: the LibreDB probe (at the default thinking), a tool test and an OpenCode session (5 s) passed ([record](../research/2026-11-16/ollama-gemma4-12b/)).
  Decode was 117.7 tok/s with the MTP draft that Ollama turns on for this tag (`--spec-type draft-mtp`), 109 after a 9.9k prompt and 84 after 37k and after 60k; the 0.33.0 regression was not retested.
- gemma4:e4b passed at the compiled defaults with no tuning, every cell on its first attempt.
- gemma4:26b needed an unreported-call ceiling of 10 and an empty-turn retry.
  The sources disagree on the count: the page says fifteen losses, the profile says about 15 assess runs across several configurations lost up to half their runs, and the one loss whose last turn was read was an empty completion after 11 tool calls.
  An empty turn deep in a loop is the risk to test in OpenCode, which has no such retry.
- gemma4 tags: the Mac's disk sizes match several variants (MLX, older GGUF, QAT), and the tags were republished from September 2026 builds, so which bytes the Mac ran is unknown.
- gemma4:31b needed 150 s turns on the Mac.
  With decode about 3x slower here, single optimize turns would likely exceed the 179,999 ms maximum (estimate).
- ministral-3:8b and :14b both passed 30/30 at the compiled defaults, every cell on its first attempt.
  q8_0 KV is 8.5 bits per element, so the q8_0 figures in the table include that.
  Measured on 2026-11-16 on ministral-3:8b with Ollama 0.40.16 and q8_0 KV: the LibreDB probe, a tool test and an OpenCode session (6 s) passed ([record](../research/2026-11-16/ollama-ministral-3-8b/)).
  Decode was 99.9 tok/s at a short context but fell to 80 after an 8.9k prompt, 50 after 32k and 35 after 58k, because all 34 layers carry attention; prefill fell from 3,944 to 1,132 tok/s over the same range.
- magistral:24b needed a second report reminder; its profile also sets retryUnreadStop, and investigate closed on a re-roll.
- granite4.1:8b read 24/30 at the defaults (empty turns and malformed tool objects) and passed only with refusal examples and an empty-turn retry.
- granite4.2:8b: the current Ollama tag has no template layer and no tools badge ([registry manifest](https://registry.ollama.ai/v2/library/granite4.2/manifests/8b)), so whether the 30/30 used native tool calls is unknown.
  The tag is dated 27 August 2026, after the stated measurement window.
- cogito:14b had the shortest slowest run on the roster (20 s).
  At 64k with q8_0 KV it comes to about 15.4 GiB, at the limit, so use q4_0 KV or 32k.
- gpt-oss:20b passed only with reasoning suppressed on agent turns and temperature 0.8 / top_p 0.9 on optimize.
  On this laptop, 166 tok/s is about 87% of the desktop-scaled estimate (about 190) and about 69% of the KB's bandwidth ceiling (about 240) ([llamacpp/models.md](../llamacpp/models.md#9-gpt-oss-20b), [community-benchmarks.md](community-benchmarks.md)).
  The profile gives a 31 s median and plan 15 s, while the page and README give 37 s and 16 s.
- nemotron-3.5-lightning:30b passed with thinking on and only a 150 s limit, the cleanest record of the 30B Nemotrons.
  It spends the budget it gets: the plan median went from 67 s to 144 s when the limit went from 90 s to 150 s.
- glm-4.7-flash:latest needed plan reasoning suppressed (plan was 0/5 with thinking on), a second report reminder and an unread-stop retry; agent-side suppression made optimize worse (0/3).
- nemotron3:33b is Nemotron 3 Nano Omni 30B-A3B Reasoning; its own page calls optimize "locked but marginal" (later sweeps 5/5, 5/5, 4/5, 4/5, 3/5).
- nemotron tags: the GGUF manifests a Linux pull gets today are byte-identical to the older quant-specific tags (`nemotron3:33b-q4_K_M`, `nemotron-3-nano:30b-a3b-q4_K_M`, `nemotron-3.5-lightning:30b-a3b-q4_K_M`); the recent re-pushes added MLX variants.
  Pull the explicit quant tags and record the digests.
- bonsai-2:2b and :4b needed temperature 0.3 and raised hold limits, and the 2b also a server-side fix.
  The vendor's looping and malformed-tool-call note in the KB is about the 27B only.
- lfm3:1.7b: its Ollama card says it is not recommended for agentic coding ([ollama.com/library/lfm3](https://ollama.com/library/lfm3)).
  The page says it refuses Ollama below 0.40.12, while the manifest requires 0.40.13.
- ornith:9b is Ornith-1.0-9B (repo `ornith`), not the Ornith-1.5-9B (repo `ornith-1.5`) that the KB excluded.
  It passed with one extra plan turn and thinking left on.
  The Mac page lists 5.5 GB and the tag 5.6 GB, which is either rounding or a different digest.
  Measured on 2026-11-16 on Ollama 0.40.16 with thinking at its default (on): the LibreDB probe, a tool test and an OpenCode session (11 s, with no reasoning in its turns) passed ([record](../research/2026-11-16/ollama-ornith-9b/)).
  Decode was 101.2 tok/s at a short context, 98 after an 8.6k prompt, 84 after 32k and 60 after 107k; the GGUF reports the `qwen35` architecture, like qwen3.5:9b.
- ornith:35b passed only with reasoning suppressed on every turn.
  Its GGUF is 0.58 GiB larger than the KB's Qwen3.6-35B-A3B Q4_K_M, so start at --n-cpu-moe 22 for 128k (estimate).
- muse-glimmer:latest needed plan reasoning suppressed, a 150 s limit and a second report reminder; at the tested Q4_K_M it would put its runs at risk of the per-turn limit here (estimate).

## Try here first

These six give the most information per hour of GPU time, in this order.

1. nemotron-3.5-lightning:30b on llama.cpp with the KB recipe ([llamacpp/models.md](../llamacpp/models.md)).
   It is the only Nemotron with first-hand agent-loop evidence, it passed with thinking on and no behavioural guards, and the KB already has a recipe.
   Its result decides whether it should sit above Bolt (rank 7, vendor scores only).
   Measured on 2026-11-16 on llama.cpp: 143.6 tok/s at 128k, 128 tok/s after a 104k prompt, and the tool test and an OpenCode session passed ([llamacpp/models.md](../llamacpp/models.md#nemotron-35-lightning-30b-a3b-fallback-if-bolt-does-not-fit)); Bolt, measured the same day, decoded at 73.5 tok/s and also passed, so on this laptop Lightning is twice as fast and Bolt's lead rests on NVIDIA's own scores.
2. qwen3.5:9b on Ollama at 128k.
   It is a KB honorable mention, fits fully with room to spare, and gives a zero-build path that matches the evidence.
   The conflicting 5/25 malformed-tool-call result on llama.cpp makes it the best test of whether the parser path matters.
   Measured on 2026-11-16 on Ollama 0.40.16 at 128k with thinking off: 98.7 tok/s and a 10.4 GiB GPU peak, and the native probe, a tool test and an OpenCode session passed, so on Ollama's own parser path its tool calls work ([record](../research/2026-11-16/ollama-qwen3.5-9b/)).
3. gemma4:12b on Ollama with thinking off.
   It is the fastest Gemma in the harness and fits fully at 64k-128k, but its evidence holds only with reasoning suppressed and is engine-sensitive.
   Measured on 2026-11-16 at 64k with thinking off: 117.7 tok/s with the MTP draft that Ollama enables and a 10.2 GiB GPU peak, and all three checks passed ([record](../research/2026-11-16/ollama-gemma4-12b/)).
4. ornith:9b on Ollama.
   It is the cleanest pass in its lane with almost no tuning, a coding-agent fine-tune, and fits fully at 128k.
   The KB's exclusion targeted a different model (Ornith-1.5) and runtime (vLLM NVFP4); the unverified-org caveat remains.
   Measured on 2026-11-16 at 128k with thinking at its default: 101.2 tok/s and a 9.3 GiB GPU peak, and all three checks passed ([record](../research/2026-11-16/ollama-ornith-9b/)).
5. ministral-3:8b on Ollama with q8_0 KV.
   It passed every cell on its first attempt at the defaults, so it is the best candidate for a fast tool-calling sub-agent.
   Measured on 2026-11-16 at 64k: 99.9 tok/s at a short context but 50 after 32k and 35 after 58k, with a 10.6 GiB GPU peak, and all three checks passed; it suits short sub-agent tasks better than long sessions ([record](../research/2026-11-16/ollama-ministral-3-8b/)).
6. qwen2.5:7b on Ollama.
   It is the cheapest cross-check that the harness reproduces on this GPU, and its third-party desktop RTX 5080 throughput (177-180 tok/s, [community-benchmarks.md](community-benchmarks.md)) gives a known ceiling.
   Measured on 2026-11-16 at 32k: 119.4 tok/s, 67% of the 177-180 tok/s that TechFuelHQ measured on a desktop RTX 5080 with Ollama 0.32.1, and all three checks passed, the OpenCode one after a parallel-call recovery ([record](../research/2026-11-16/ollama-qwen2.5-7b/)).

The four models already in the KB plan (ranks 1, 2, 5 and 9) stay on the existing measurement list in [model-selection.md](model-selection.md#what-to-measure-first).

## What this changes or confirms in the KB

The lead will make these edits; this document does not change other files.

Confirmed:

- Rank 2, Qwen3.6-35B-A3B: tool reliability holds only with thinking off on agent turns, which supports the KB's `--reasoning off` server default.
- Rank 9, gpt-oss-20b: 30/30 in an agent loop at low or no reasoning partly offsets its weak independent agentic scores.
  On Ollama, `think=false` removes the Reasoning line from the Harmony prompt, which is not the same as llama.cpp `--reasoning-effort low`; the mapping is untested.
- Rank 5, Gemma 4 26B-A4B: it finishes agent loops, but it can return an empty completion deep in a loop.
  That belongs in the rank-5 caveats beside llama.cpp #29655, and in the OpenCode smoke tests.
- The rule to start toggle models with thinking off: gemma4:12b, ornith:35b, qwen3.6:35b and gpt-oss:20b lost cells with thinking on and passed with it suppressed on agent turns (and on plan turns for the first three), and glm-4.7-flash and muse-glimmer needed it off on the plan turn.
  It is a default, not an absolute: glm-4.7-flash got worse on optimize with agent-side thinking off (0/3), and nemotron-3.5-lightning passed with thinking on.
- The exclusions of dense 27B-33B models at Q4_K_M (Qwen3.5-27B, Qwen3.6-27B, Gemma 4 31B, Muse Glimmer 30B): they pass, but they spill here.

Changed or worth re-judging:

- Ministral 3 (8B and 14B), Gemma 4 E4B and Qwen3-Coder-30B-A3B are excluded for weak agentic or coding scores, but all passed 30/30 at the compiled defaults, every cell on its first attempt (the qwen3-coder profile does set retryUnreadStop).
  Their exclusion reason should say coding quality, not tool reliability, and they could become honorable mentions as tool-calling sub-agents.
- Nemotron 3.5 Lightning vs Bolt: Lightning has first-hand agent-loop evidence and Bolt does not.
- Ornith: [model-selection.md](model-selection.md) and [vllm/models.md](../vllm/models.md) should distinguish Ornith-1.0-9B (passed here, Ollama GGUF) from Ornith-1.5-9B (excluded).
- Qwen3.5-9B: add an Ollama or llama.cpp Q4_K_M path beside the vLLM NVFP4 recipe, and record the parser conflict.
- Done on 2026-11-16: [llamacpp/models.md](../llamacpp/models.md#1-qwen38-27b) no longer claims that qwen3.8:27b's digest differs from 27b-q4_K_M (the live manifests for `latest`, `27b` and `27b-q4_K_M` share one model blob), and the rank-1 fit is measured rather than recomputed: about 1 GiB below the 15,841 MiB that CUDA can use.
- Gemma 4 26B-A4B: the KB names `gemma4:26b-a4b-it-qat` as the Ollama tag to use without checking it with its 1.11 GiB projector; the estimate is about 16-16.7 GiB at 64k, so it probably spills a few layers.
- Add with reasons: Qwen2.5 and QwQ (32k or 40k native context, superseded), qwen3:8b and qwen3:14b (40,960 native context), Magistral Small 2506 (40,960 cap), Granite 4.1 30B and Cogito v1 32B (dense, do not fit), Nemotron 3 Nano 30B (superseded by Lightning), Nemotron 3 Nano Omni (Lightning plus encoders), LFM3-1.7B (its own card advises against agentic coding).
- [opencode.md](opencode.md) line 201 says small models fail agent loops through bad tool calls; the harness shows 1.7B-4B models (lfm3:1.7b, bonsai-2:2b, qwen3:4b, qwen3.5:4b) passing in short guarded loops, so the sentence should say it refers to OpenCode-style long-context loops.
- Add the harness to [sources.md](sources.md) and [community-benchmarks.md](community-benchmarks.md) as a first-hand evidence class, with the 614 GB/s transfer rule.

Not covered: K2 Horizon MoVA 36B A4B (rank 3), K2 Horizon 7B (rank 4), Qwen3-Coder-Next (rank 6), Nemotron 3.5 Bolt (rank 7), Groot-19B-A3B (rank 8) and DeepSeek-R1-0528-Qwen3-8B (rank 10) have no harness result, and rank 1 is covered only at a different quant and runtime.
Because the harness does not test code edits or long contexts, it is not grounds to reorder the top 10 by itself.

## Gaps and what to measure on this laptop

Gaps in the evidence:

- The libredb pages record no tag digests, no runner (GGUF or MLX) and, for most models, no context length.
- The 11 supported models without a published timing row are not covered.
- Whether a pass used native tool calls or the prose fallback is not recorded, which matters most for granite4.2:8b and qwen2.5-coder:14b.
- Source inconsistencies: methodology.md says six models carry 150 s while README.md says nine; the qwen3:14b, qwen3.6:35b, gemma4:26b and granite4.2:8b pages disagree with the profile file or the README in places noted above.
- Only the harness models measured on 2026-11-16 (the rows marked measured) have fit and speed figures from this laptop, and the configured DDR5 speed is unverified, so every other fit and speed figure is an estimate.
- The measured Ollama runs used 0.40.16, not the 0.32.13 and 0.33.0 builds of the Mac evidence, so engine regressions such as gemma4:12b's on 0.33.0 are not retested.
- Whether Ollama clamps num_ctx to a GGUF's native context (qwen2.5 at 32k, qwen3:8b and 14b at 40,960) is unverified.

Measurements to run, one at a time:

1. Pull a pinned Ollama image (`ollama/ollama:0.33.0` to match the evidence, then `0.40.16`) and record the version and model digests with every result.
   Done on 2026-11-16 with `0.40.16` only; the version, digests and server logs are in the records, and `0.33.0` was not pulled.
2. For each "Try here first" model, record GPU memory, `ollama ps` split and decode tok/s at 64k (128k where it fits).
   Done on 2026-11-16: all five Ollama tags ran 100% on the GPU (see the rows marked measured).
3. Run the native `tool_calls` probe from [docs/llms/setup.md](https://github.com/libredb/libredb-studio/blob/c419b8bf1b156cff7a05bacf5cce5b3c0eb4b666/docs/llms/setup.md) for each tag before any longer test.
   Done on 2026-11-16: all five returned a native `inspect_operations` call.
4. Repeat the six libredb surfaces against this laptop's servers, including the llama.cpp UD-Q3_K_XL Qwen3.8-27B server, to see whether pass/fail holds at the KB's quant and parser.
5. Run an OpenCode session per candidate with thinking off, watching for empty turns deep in the loop and multi-line edit arguments.
   Partly done on 2026-11-16: a short write-and-read session passed on all five; long sessions and edits are still open.
6. Read the DDR5 configured speed with `dmidecode` so the spill estimates can be recomputed.

## Sources

- [docs/llms/README.md](https://github.com/libredb/libredb-studio/blob/c419b8bf1b156cff7a05bacf5cce5b3c0eb4b666/docs/llms/README.md)
- [docs/llms/methodology.md](https://github.com/libredb/libredb-studio/blob/c419b8bf1b156cff7a05bacf5cce5b3c0eb4b666/docs/llms/methodology.md)
- [docs/llms/setup.md](https://github.com/libredb/libredb-studio/blob/c419b8bf1b156cff7a05bacf5cce5b3c0eb4b666/docs/llms/setup.md)
- [docs/llms/model-tuning.md](https://github.com/libredb/libredb-studio/blob/c419b8bf1b156cff7a05bacf5cce5b3c0eb4b666/docs/llms/model-tuning.md)
- [docs/llms/testing-your-own.md](https://github.com/libredb/libredb-studio/blob/c419b8bf1b156cff7a05bacf5cce5b3c0eb4b666/docs/llms/testing-your-own.md)
- [src/lib/agent/model-tuning/measured-profiles.json](https://github.com/libredb/libredb-studio/blob/c419b8bf1b156cff7a05bacf5cce5b3c0eb4b666/src/lib/agent/model-tuning/measured-profiles.json)
- The per-family pages listed under the table.
- [apple.com MacBook Pro specs](https://www.apple.com/macbook-pro/specs/), for the M5 Max bandwidth.
- [Ollama context length docs](https://docs.ollama.com/context-length).
- Ollama library tag pages: [qwen2.5](https://ollama.com/library/qwen2.5/tags), [qwen2.5-coder](https://ollama.com/library/qwen2.5-coder/tags), [qwq](https://ollama.com/library/qwq/tags), [qwen3](https://ollama.com/library/qwen3/tags), [qwen3-coder](https://ollama.com/library/qwen3-coder/tags), [qwen3.5](https://ollama.com/library/qwen3.5/tags), [qwen3.6](https://ollama.com/library/qwen3.6/tags), [qwen3.8](https://ollama.com/library/qwen3.8/tags), [gemma4](https://ollama.com/library/gemma4/tags), [ministral-3](https://ollama.com/library/ministral-3/tags), [magistral](https://ollama.com/library/magistral/tags), [granite4.1](https://ollama.com/library/granite4.1/tags), [granite4.2](https://ollama.com/library/granite4.2/tags), [cogito](https://ollama.com/library/cogito/tags), [gpt-oss](https://ollama.com/library/gpt-oss/tags), [nemotron-3.5-lightning](https://ollama.com/library/nemotron-3.5-lightning/tags), [glm-4.7-flash](https://ollama.com/library/glm-4.7-flash/tags), [nemotron3](https://ollama.com/library/nemotron3/tags), [nemotron-3-nano](https://ollama.com/library/nemotron-3-nano/tags), [bonsai-2](https://ollama.com/library/bonsai-2/tags), [lfm3](https://ollama.com/library/lfm3/tags), [ornith](https://ollama.com/library/ornith/tags), [muse-glimmer](https://ollama.com/library/muse-glimmer/tags).
- Ollama registry manifests at `https://registry.ollama.ai/v2/library/<name>/manifests/<tag>`, for blob sizes, quants, parsers and required versions.
- Model configs on Hugging Face for the KV arithmetic, for example [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/config.json) and [gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it/raw/main/config.json).
- [TechFuelHQ RTX 5080 LLM throughput](https://techfuelhq.com/data/rtx-5080-llm-throughput/), third-party desktop numbers.
- KB documents: [model-selection.md](model-selection.md), [community-benchmarks.md](community-benchmarks.md), [opencode.md](opencode.md), [sources.md](sources.md), [llamacpp/models.md](../llamacpp/models.md), [vllm/models.md](../vllm/models.md).
- This laptop's gpt-oss-20b measurements: [research/2026-11-16/](../research/2026-11-16/README.md).

## Maintenance

- Re-check when the libredb pages change, when a model here is measured on this laptop, or when the Ollama tags are republished.
- Replace (estimate) labels with (measured on this laptop) as results arrive, and keep the Mac numbers labelled (measured on the M5 Max, libredb-studio).
- Record the Ollama image version and model digest beside every harness result, because the engine version can flip a cell.
- Do not edit anything under `$HOME/projects/libredb`; it is another project and is read-only from here.
- Update the "Last verified" line on every change.
