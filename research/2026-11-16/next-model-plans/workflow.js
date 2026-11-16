export const meta = {
  name: 'next-model-launch-plans',
  description: 'Size verified llama.cpp launch plans for the next eight local models from live GGUF headers, then cross-check them',
  phases: [
    { title: 'Plan', detail: 'one agent per model: GGUF header by range request, VRAM and RAM plan, flags at b12407, OpenCode entry' },
    { title: 'Critique', detail: 'one agent cross-checks all plans against the measured budget' },
  ],
}

const S = args.scratch
const KB = '$HOME/projects/llm'

const COMMON = `You are preparing an exact, runnable llama.cpp launch plan for one model on this laptop, before the lead measures it. You do not run the model.

Hard limits:
- Do not run docker, do not touch the GPU, do not start any server, and do not download model weights. The only download allowed is the GGUF header by HTTP range request (at most 64 MiB), anonymously; never use the user's HF token.
- Do not edit anything under ${KB}; it is read-only for you. Write only inside your own new directory ${S}/headers/<model key>/, and keep downloaded bytes there.
- Run Python that reads downloaded files with python3 -I and pass paths as arguments. No heavy commands are needed.
- Write in English, never use an em dash or an en dash, and return only the structured result.
- Advisory time budget: about 15 minutes.

Machine facts, measured on this laptop on 2026-11-16 (treat these as ground truth):
- GPU: RTX 5080 Laptop, sm_120, PCIe 5.0 x16. nvidia-smi shows 16,303 MiB, the driver reserves 462 MiB, so CUDA can use 15,841 MiB. The desktop holds about 112 MiB. Plan for at least 500 MiB free at the nvidia-smi peak, so the predicted nvidia-smi peak (which includes the desktop) must be at most 15,341 MiB.
- llama.cpp overhead beyond weights on the GPU plus the KV cache (CUDA context, compute buffers at the default -ub 512, recurrent state): 0.79 GiB for Qwen3.8-27B (dense, 248k vocab) and 0.72 GiB for Qwen3.6-35B-A3B (hybrid MoE, 248k vocab). The CUDA pool grew only 52 MiB more during 8k to 107k prompts. token_embd stays in host memory. Use about 0.8 GiB unless the architecture or vocabulary clearly needs more, and say why.
- --moe-cache-mib N allocates exactly N MiB of VRAM. On Qwen3.6-35B-A3B, with -lm none (pinned host experts) and -t 8: --n-cpu-moe 21 without cache decoded 68 tok/s; 21 plus 1,536 MiB cache 112 tok/s; 25 plus 3,200 MiB 120 tok/s; all 40 blocks in RAM plus 9,600 MiB 126 tok/s but prefill fell from 1,990 to 1,679 tok/s and pinned RAM rose by 6 GiB. CPU share 0.95 was worse than 0 with pinned experts. -lm none doubled prefill (934 to 2,112 tok/s at 8.6k) and puts host experts in anonymous pinned memory that counts against the container cap. -t 8 (one per P-core; auto chose 4) added 2%.
- RAM: about 40 GiB free with the desktop. Container cap = pinned host experts + host token_embd + prompt cache (--cache-ram default 8192 MiB) + about 2 GiB process. Measured Qwen3.6 container peak was 14.9 GiB with 10.5 GiB of pinned experts and a 107k prompt cached.
- Image: ghcr.io/ggml-org/llama.cpp:server-cuda-b12407, 47 commits after v1.0.0. The lead adds the docker flags (--user, the HF cache at /hf, -e GGML_CUDA_DISABLE_GRAPHS=1, 127.0.0.1:8080, --memory equal to --memory-swap) and passes the cached file with -m. You provide only the llama-server arguments, with MODEL_PATH for the -m value, plus --alias local, -np 1, --jinja and -fa on.

Method:
1. Read the model's recipe in ${KB}/llamacpp/models.md (the section named below), and the parts of ${KB}/llamacpp/README.md on MoE expert offload, KV quantization, thinking control, prompt cache, tool calling and known issues, and the OpenCode conventions and existing entries in ${KB}/docs/opencode.md.
2. Fetch the header: mkdir -p ${S}/headers/<model key>, then curl -sL --fail -r 0-33554431 -o ${S}/headers/<model key>/head.gguf "https://huggingface.co/<repo>/resolve/main/<file>". Run python3 -I ${S}/bench/gguf_plan.py ${S}/headers/<model key>/head.gguf <context> <KV bytes per element: 2 for f16, 1.0625 for q8_0>. If it reports that the header is longer than the downloaded part, refetch with -r 0-67108863. The script prints the metadata, tensor totals by type, expert bytes per block, the non-expert GPU weights and a GPU total for every --n-cpu-moe value. Its KV line handles full-attention intervals and per-layer KV-head arrays only: compute sliding-window caches (check how llama.cpp sizes them at b12407, src/llama-kv-cache-iswa.cpp and --swa-full), Mamba or SSM state, MLA or any other layout by hand from the printed metadata, and show the arithmetic. Remember --n-cpu-moe counts blocks from 0, including dense blocks.
3. Verify at llama.cpp tag b12407 on GitHub (gh api or raw.githubusercontent.com): the architecture is supported, how the chat template and the tool-call parser are selected in common/chat.cpp, which reasoning controls apply (--reasoning, --reasoning-effort, --chat-template-kwargs, template kwargs such as enable_thinking or reasoning_effort), and any model-specific flag the recipe uses. Check the live state of every llama.cpp issue or PR the recipe cites with gh api repos/ggml-org/llama.cpp/issues/<n>.
4. Recommend one launch for the measurement: the KB's target context and KV type unless the header shows they do not fit or a materially better safe value exists. For an all-on-GPU model, confirm the fit and the margin. For an offloaded MoE model, recommend -lm none, -t 8 and a split between --n-cpu-moe and --moe-cache-mib that follows the Qwen3.6 finding (give spare VRAM to the cache rather than only to whole blocks) while keeping 500 MiB free at the peak and keeping pinned RAM reasonable; describe one alternative split worth a short comparison run. Show the VRAM breakdown and the predicted nvidia-smi peak in MiB.
5. Give the RAM plan and the --memory cap, the thinking mode and how to switch it per request, the sampling defaults from the model card, the tool-calling path, an OpenCode model entry in the shape used in docs/opencode.md (modelID "local", name, capabilities, limit.context, limit.output, variants if the model toggles thinking), probe prompt sizes that fit the context (for example "8000,30000" at 64k, "8000,30000,100000" at 128k), and an expected decode range with the reasoning.
6. List every way the KB recipe is wrong or stale, with evidence.`

const PLAN_SCHEMA = {
  type: 'object',
  properties: {
    key: { type: 'string' },
    repo: { type: 'string' },
    file: { type: 'string' },
    size_bytes: { type: 'number' },
    header_ok: { type: 'boolean', description: 'true if the GGUF header was fetched and parsed' },
    header_facts: { type: 'string', description: 'arch, block count, experts and experts used, dense blocks, attention layout (full, sliding window, SSM), KV heads and head sizes, vocab size, native context, per-block expert GiB, non-expert GPU GiB, token_embd GiB' },
    kv_math: { type: 'string', description: 'KV and recurrent-state arithmetic at the recommended context and KV type' },
    recommended: {
      type: 'object',
      properties: {
        context: { type: 'number' },
        kv_type: { type: 'string' },
        n_cpu_moe: { type: ['number', 'null'] },
        moe_cache_mib: { type: ['number', 'null'] },
        load_mode: { type: 'string', description: 'mmap (default) or none' },
        threads: { type: ['number', 'null'] },
        vram_breakdown: { type: 'string' },
        predicted_peak_mib: { type: 'number', description: 'nvidia-smi peak including the desktop' },
        predicted_free_mib: { type: 'number', description: '15841 minus predicted_peak_mib' },
        memory_cap: { type: 'string', description: 'docker --memory value, for example 20g' },
        predicted_container_ram_gib: { type: 'number' },
        server_args: { type: 'array', items: { type: 'string' }, description: 'llama-server arguments after the image name, with MODEL_PATH as the -m value' },
      },
      required: ['context', 'kv_type', 'n_cpu_moe', 'moe_cache_mib', 'load_mode', 'threads', 'vram_breakdown', 'predicted_peak_mib', 'predicted_free_mib', 'memory_cap', 'predicted_container_ram_gib', 'server_args'],
    },
    alternative: { type: 'string', description: 'a second launch worth one short comparison run, with its predicted peak, or an empty string' },
    thinking: { type: 'string', description: 'toggle, always or none; the server default flag; how to switch it per request' },
    per_request_thinking_on: { type: 'string', description: 'JSON fragment to add to a chat request to turn thinking on, or an empty string' },
    tool_calling: { type: 'string' },
    sampling: { type: 'string' },
    opencode_entry: { type: 'string', description: 'JSON object for this model under providers.llamacpp-docker.models, keyed by the entry name' },
    probe_long_tokens: { type: 'string' },
    expected_speed: { type: 'string' },
    issues: { type: 'array', items: { type: 'object', properties: { id: { type: 'string' }, state: { type: 'string' }, note: { type: 'string' } }, required: ['id', 'state', 'note'] } },
    kb_discrepancies: { type: 'array', items: { type: 'string' } },
    risks: { type: 'array', items: { type: 'string' } },
    sources: { type: 'array', items: { type: 'string' } },
  },
  required: ['key', 'repo', 'file', 'size_bytes', 'header_ok', 'header_facts', 'kv_math', 'recommended', 'alternative', 'thinking', 'per_request_thinking_on', 'tool_calling', 'sampling', 'opencode_entry', 'probe_long_tokens', 'expected_speed', 'issues', 'kb_discrepancies', 'risks', 'sources'],
}

const CRITIC_SCHEMA = {
  type: 'object',
  properties: {
    per_model: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          key: { type: 'string' },
          verdict: { type: 'string', enum: ['ok', 'fix', 'unsafe'] },
          corrections: { type: 'array', items: { type: 'string' } },
          corrected_server_args: { type: 'array', items: { type: 'string' }, description: 'the full corrected argument list if anything in server_args must change, otherwise empty' },
        },
        required: ['key', 'verdict', 'corrections', 'corrected_server_args'],
      },
    },
    cross_cutting: { type: 'array', items: { type: 'string' } },
    measurement_order: { type: 'array', items: { type: 'string' } },
  },
  required: ['per_model', 'cross_cutting', 'measurement_order'],
}

phase('Plan')
const results = await parallel(args.models.map(m => () =>
  agent(`${COMMON}\n\nYour model: key "${m.key}", repo ${m.repo}, file ${m.file} (${m.size} bytes), KB recipe section "${m.section}" in ${KB}/llamacpp/models.md. ${m.note}`,
    { label: `plan:${m.key}`, phase: 'Plan', schema: PLAN_SCHEMA })))
const plans = results.filter(Boolean)
const missing = args.models.filter((m, i) => !results[i]).map(m => m.key)
log(`${plans.length}/${args.models.length} plans returned` + (missing.length ? `; missing: ${missing.join(', ')}` : ''))

phase('Critique')
const critique = await agent(`You are the critic for eight llama.cpp launch plans prepared for an RTX 5080 Laptop (CUDA can use 15,841 MiB; desktop about 112 MiB; every plan must keep at least 500 MiB free at the nvidia-smi peak, so a predicted peak above 15,341 MiB is unsafe; llama.cpp overhead beyond weights and KV was measured at 0.72-0.79 GiB on two Qwen models; about 40 GiB of RAM is free; image ghcr.io/ggml-org/llama.cpp:server-cuda-b12407; -lm none pins host experts and doubled prefill; --moe-cache-mib at CPU share 0 nearly doubled decode on Qwen3.6-35B-A3B). The plans are below as JSON.

Check each plan independently and adversarially: redo the VRAM and KV arithmetic from its header facts (sliding-window and SSM layers included), check that every llama-server flag exists at b12407 (read common/arg.cpp at that tag on GitHub if in doubt), that the reasoning and tool-calling claims match common/chat.cpp at b12407 and the model's chat template, that the container cap covers pinned experts plus the prompt cache plus the process, and that the OpenCode entry follows the conventions in $HOME/projects/llm/docs/opencode.md. Give verdict ok, fix or unsafe per model, with concrete corrections and, if any argument must change, the full corrected argument list. Then list cross-cutting problems, and give a measurement order. The files arrive in this download order: gemma-4-26b-a4b, k2-horizon-7b, nemotron-3.5-lightning, k2-horizon-mova, deepseek-r1-0528-qwen3-8b, groot-19b-a3b, nemotron-3.5-bolt, qwen3-coder-next. Do not run docker or touch the GPU, do not edit $HOME/projects/llm, write in English without em or en dashes.

Missing plans: ${missing.length ? missing.join(', ') : 'none'}.

PLANS:
${JSON.stringify(plans, null, 1)}`,
  { label: 'critic', phase: 'Critique', schema: CRITIC_SCHEMA })

return { plans, critique, missing }
