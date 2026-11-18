#!/usr/bin/env bash
# One M16 start of a draft llama.cpp recipe, through the copy of main with no client (LLMCTL_CLIENTS empty) and C's
# override folder: record the docker command (llmctl show), trace GPU and cgroup memory every second and docker stats
# every 2 s, start it with llmctl up, then run the repository probe (decode and first token at 8k, 30k and a near-full
# prompt), llmctl test, the smoke and grammar requests of seq_c.py (Hermes' title request, a forced tool request, the
# api probe: plain, streamed, a tool call and its second turn, thinking per type, json_schema), and a code-generation
# decode, then save the server log and remove the container with llmctl down --remove.
# Usage: run_llamacpp_c.sh <id> <run> <context> <thinking-type> <long-tokens>; needs SCRATCH and HOME.
set -uo pipefail
: "${SCRATCH:?}"
id=$1; run=$2; context=$3; thinking=$4; long=$5
C=$SCRATCH/sglang/measure-c; S=$C/scripts; LM=$SCRATCH/llmctl-main/llmctl
R=$C/llamacpp-$id; logs=$R/logs; shapes=$R/shapes; cmds=$R/cmds
mkdir -p "$logs" "$shapes" "$cmds"
export LLMCTL_CLIENTS= LLMCTL_RECIPES_DIR=$C/overrides
for other in llamacpp sglang vllm sglang-measure vllm-measure llamacpp-measure; do
  if [ "$(docker inspect --format '{{.State.Status}}' "$other" 2>/dev/null)" = running ]; then echo "refusing: container $other is running"; exit 9; fi
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if [ "$used" -gt 600 ]; then echo "refusing: $used MiB of GPU memory in use before the start"; exit 9; fi
cat > "$cmds/$run.sh" <<CMD
#!/usr/bin/env bash
# M16 $run: the draft \$SCRATCH/sglang/measure-c/overrides/llamacpp/$id.toml started through the copy of main, no client.
set -euo pipefail
LLMCTL_CLIENTS= LLMCTL_RECIPES_DIR="\$SCRATCH/sglang/measure-c/overrides" "\$SCRATCH/llmctl-main/llmctl" up $id --runtime llamacpp
CMD
chmod +x "$cmds/$run.sh"
cp -p "$C/overrides/llamacpp/$id.toml" "$logs/$run.recipe.toml"
{
  echo "gpu_used_before_mib=$used"
  echo "enforced_power_limit=$(nvidia-smi --query-gpu=enforced.power.limit --format=csv,noheader | tr -d ' ')"
  echo "download_running=$(docker ps --format '{{.Names}}' | grep -c '^dl-')"
} > "$logs/$run.meta"
"$LM" show "$id" --runtime llamacpp > "$logs/$run.show.txt" 2>&1
bash "$S/trace_c.sh" llamacpp "$logs/$run-trace.csv" 5400 > /dev/null 2>&1 &
tracer=$!
( while true; do echo "$(date +%s.%N | cut -c1-14) $(docker stats --no-stream --format '{{.MemUsage}} {{.CPUPerc}}' llamacpp 2>/dev/null)"; sleep 2; done ) > "$logs/$run.docker-stats.log" 2>&1 &
dstats=$!
t0=$(date +%s.%N); echo "$t0" > "$logs/$run.t0"
"$LM" up "$id" --runtime llamacpp > "$logs/$run.up.txt" 2>&1
rc=$?
t1=$(date +%s.%N)
echo "up_exit=$rc up_seconds=$(echo "$t1 - $t0" | bc)" >> "$logs/$run.meta"
cat "$logs/$run.up.txt"
if [ "$rc" = 0 ]; then
  echo "gpu_right_after_ready=$(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | tr -d ' ')" >> "$logs/$run.meta"
  bash "$S/procmem.sh" llamacpp "at ready" > "$logs/$run.procmem.txt" 2>&1
  python3 -I "$S/llm_probe.py" --base http://127.0.0.1:8080/v1 --model local --label "llamacpp-$id-$run" \
    --runs 3 --long-tokens "$long" --out "$shapes/$run-probe.json" > "$shapes/$run-probe-summary.json" 2> "$logs/$run.probe.err"
  echo "probe_exit=$?" >> "$logs/$run.meta"
  "$LM" test > "$logs/$run.test.txt" 2>&1
  echo "test_exit=$?" >> "$logs/$run.meta"
  python3 -I "$S/seq_c.py" --base http://127.0.0.1:8080 --kind llamacpp --container llamacpp --model local \
    --context "$context" --thinking "$thinking" --phases grammar,smoke --out-dir "$shapes" --run "$run" > "$logs/$run.seq.txt" 2>&1
  echo "seq_exit=$?" >> "$logs/$run.meta"
  curl -s --noproxy '*' http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"local","max_tokens":3000,"messages":[{"role":"user","content":"Write a Python module that implements an LRU cache class with get and put, type hints and docstrings, followed by five pytest tests."}]}' \
    > "$shapes/$run-code-decode.json"
  bash "$S/procmem.sh" llamacpp "at the end" >> "$logs/$run.procmem.txt" 2>&1
fi
docker inspect --format '{{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} started={{.State.StartedAt}} finished={{.State.FinishedAt}}' llamacpp > "$logs/$run.state.txt" 2>&1
docker logs llamacpp > "$logs/$run.full.log" 2>&1
"$LM" down --remove > "$logs/$run.down.txt" 2>&1
echo "down_exit=$?" >> "$logs/$run.meta"
kill "$tracer" "$dstats" 2>/dev/null; wait "$tracer" "$dstats" 2>/dev/null
tr '\r' '\n' < "$logs/$run.full.log" | grep -E 'print_timing|prompt eval time|eval time|total time|listening|model loaded|n_ctx_slot|n_ctx|n_threads|llama_kv_cache|KV self size|CUDA0 (model|KV|compute) buffer|load_tensors|W llama|E llama|error|warn' > "$logs/$run.server-timings.log"
cat "$logs/$run.state.txt" "$logs/$run.down.txt"
echo "GPU peak $(awk -F, 'NR>1 && $2!="" {print $2}' "$logs/$run-trace.csv" | sort -n | tail -1) MiB; docker stats peak $(awk '{print $2}' "$logs/$run.docker-stats.log" | grep -E 'iB$' | sort -h | tail -1)"
