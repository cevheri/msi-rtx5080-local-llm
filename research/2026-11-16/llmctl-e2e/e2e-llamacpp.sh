#!/usr/bin/env bash
# End-to-end check of llmctl with the default model: up, status, test, an OpenCode session without -m, down.
set -u
WT=$HOME/projects/llm/.worktrees/llmctl
OUT=$SCRATCH/llmctl/e2e/llamacpp-qwen3.6
mkdir -p "$OUT/proj"
cd "$WT" || exit 1

step() { echo "== $* (at $(date +%T))"; }

step "llmctl up qwen3.6"
S=$(date +%s)
./llmctl up qwen3.6 > "$OUT/up.txt" 2>&1
echo "up exit=$? after $(( $(date +%s) - S )) s"
cat "$OUT/up.txt"

step "llmctl status"
./llmctl status > "$OUT/status.txt" 2>&1
echo "status exit=$?"
cat "$OUT/status.txt"

step "llmctl test"
./llmctl test > "$OUT/test.txt" 2>&1
echo "test exit=$?"
cat "$OUT/test.txt"

step "OpenCode session without -m"
cd "$OUT/proj" || exit 1
rm -f hello.py
cat > opencode.json <<'EOF'
{
  "$schema": "https://opencode.ai/config.json",
  "permission": { "bash": "deny", "external_directory": "deny", "webfetch": "deny", "websearch": "deny", "task": "deny", "question": "deny" }
}
EOF
S=$(date +%s)
timeout 900 heavy ~/.opencode/bin/opencode run --standalone --archived --auto --format json \
  --title "llmctl e2e" \
  "Create a file named hello.py containing a function add(a, b) that returns a + b, and a main block that prints add(2, 3). Then read the file back and tell me how many lines it has. Do not run any shell commands." \
  > "$OUT/opencode-events.jsonl" 2> "$OUT/opencode.err"
echo "opencode exit=$? after $(( $(date +%s) - S )) s"
cat -n hello.py 2>&1 | head -8
python3 -I - "$OUT/opencode-events.jsonl" <<'EOF'
import json, sys
for line in open(sys.argv[1]):
    try:
        e = json.loads(line)
    except ValueError:
        continue
    p = e.get("part", {}) or {}
    if e.get("type") == "tool_use":
        print("tool_use:", p.get("tool"), p.get("state", {}).get("status"))
    elif e.get("type") == "text":
        print("answer:", p.get("text", "")[:160].replace("\n", " "))
    elif e.get("type") == "error":
        print("error:", json.dumps(e)[:300])
EOF

cd "$WT" || exit 1
step "llmctl down"
./llmctl down > "$OUT/down.txt" 2>&1
echo "down exit=$?"
cat "$OUT/down.txt"
docker ps -a --filter label=io.llmctl.managed=true --format '{{.Names}} {{.State}}'
nvidia-smi --query-gpu=memory.used --format=csv,noheader
