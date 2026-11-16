# Research records, 2026-11-15

Last verified: 2026-11-16.

These are the raw, machine-generated records behind this knowledge base.
The curated and reviewed output is the set of documents indexed in the root [README.md](../../README.md); these files are the evidence trail behind it.
Do not edit them by hand: they record what each research agent returned on 2026-11-15.
Before the repository went public on 2026-11-16, a script replaced local paths in these records: the session's temporary folder became `$SCRATCH` (`$SESSION` for its parent) and the home folder `$HOME`, and the GPU UUID in the toolkit install log was masked; the original folders no longer exist.

## Runs

| Run | What it did | Agents | Records |
|---|---|---|---|
| Discovery | Twelve evidence lanes (platform docs, model families, registries, leaderboards) plus a completeness critic | 13 | [discovery/](discovery/) |
| Verification | 22 candidates, each with a verification record and two adversarial checks (feasibility and facts); four platform resolvers; three judges; a synthesis; an audit | 75 | [verification/](verification/) |
| Source review | The 19 user-provided sources, each reviewed and then independently checked | 38 | [source-reviews/](source-reviews/) |
| New candidates and knowledge base build | Nine new candidates verified and challenged; three judges re-ranked all candidates; a synthesis; eleven documents written and reviewed; a consistency pass | 54 | [new-candidates/](new-candidates/), [kb-build/](kb-build/) |
| No-toolkit probe | GPU access from plain Docker without the NVIDIA Container Toolkit, run by the lead | 0 | [probes/](probes/) |

## Files

- `baseline.md`: the verified baseline used to brief the later runs.
- `discovery/<lane>.json`: one file per discovery lane; `completeness-critic.json` lists the gaps and the shortlist it proposed.
- `verification/verify_<model>.json`, `feasibility_<model>.json` and `facts_<model>.json`: the per-model record and its two adversarial checks.
- `verification/platform_<topic>.json`: the platform resolvers for vLLM, llama.cpp and Ollama images, the host and OpenCode.
- `verification/judge_<lens>.json`, `synthesis.json` and `audit.json`: the first ranking and its audit; `compact-old.json` is the condensed view of the 22 records that the second ranking used.
- `source-reviews/<key>.json`: the "review" and the independent "check" for each user-provided source.
- `new-candidates/verify_<model>.json`, `feasibility_<model>.json` and `facts_<model>.json`: the nine new candidates.
- `new-candidates/agent-saved/`: copies the verifiers saved themselves; they can differ slightly from the returned records, which are authoritative.
- `kb-build/judge_<lens>.json`, `synthesis.json`, `write_<doc>.json`, `review_<doc>.json`, `consistency.json` and `result.json`: the second ranking, the writer and reviewer report for each document, the consistency pass and the workflow result.
- `probes/`: the no-toolkit GPU test log and its probe script.

## Caveats

- Nothing in these records was measured on this laptop, except the probe in `probes/` and the host observations in `baseline.md`.
- The first ranking in `verification/synthesis.json` is superseded by `kb-build/synthesis.json`, which [docs/model-selection.md](../../docs/model-selection.md) follows.
- Third-party content is summarized, not copied; follow the URLs in the records for the original sources.
