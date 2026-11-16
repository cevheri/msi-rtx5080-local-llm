#!/bin/bash
# Runs inside the overlay sandbox. Two stubs, then the C2 sequence on profile c2.
S=$SCRATCH/clients/research-hermes
RUN=$S/runs/cfg/c2; F=$HOME/.hermes/profiles/c2/config.yaml; B=$S/bin/llmctl_block.py
P="Create hello.py that prints hello, then read it back and show its contents"
mkdir -p $RUN/cap8080 $RUN/cap8000 $RUN/work $RUN/snap
coproc A { exec python3 -I $S/bin/stub_server.py --port 8080 --flavor llamacpp --model local --ctx 131072 --capture-dir $RUN/cap8080 2>$RUN/stub8080.log; }
read -r -t 15 l1 <&"${A[0]}"
coproc V { exec python3 -I $S/bin/stub_server.py --port 8000 --flavor vllm --model gpt-oss-20b --ctx 65536 --capture-dir $RUN/cap8000 2>$RUN/stub8000.log; }
read -r -t 15 l2 <&"${V[0]}"
echo "stubs: $l1 $l2"
cnt() { echo "captures 8080=$(ls $RUN/cap8080 | wc -l) 8000=$(ls $RUN/cap8000 | wc -l)"; }
snap() { cp $F $RUN/snap/$1.yaml; }
snap 00-copy
echo "### 1 model lines (llmctl-style regex edit)"; python3 -I $B model $F custom:llmctl local
echo "### 2 gateway.standalone via hermes config set (sandbox needs it)"; hermes -p c2 config set gateway.standalone true; echo rc=$?
echo "### 3 append marked block (llama.cpp Qwen3.6 thinking off)"; python3 -I $B block $F $S/configs/blocks/llamacpp-qwen36-off.yaml; snap 03-block-appended
echo "### 4 check"; python3 -I $B check $F; hermes -p c2 config get providers.llmctl.api; echo rc=$?
echo "### 5 run against 8080"; cnt; (cd $RUN/work && hermes -p c2 -z "$P" -t file > $RUN/run5.out 2> $RUN/run5.err); echo rc=$?; cnt; cat $RUN/work/hello.py
echo "### 6 hermes-side writes after the block exists"; hermes -p c2 config set agent.reasoning_effort ""; echo rc=$?; hermes -p c2 config set zz_new_top.key 1 --force; echo rc=$?; hermes -p c2 config set providers.llmctl.models.local.context_length 120000; echo rc=$?; snap 06-after-hermes-writes
echo "### 7 replace block (vLLM gpt-oss) + model lines"; python3 -I $B block $F $S/configs/blocks/vllm-gptoss.yaml; python3 -I $B model $F custom:llmctl gpt-oss-20b; python3 -I $B check $F; snap 07-replaced
echo "### 8 run against 8000"; rm -f $RUN/work/hello.py; cnt; (cd $RUN/work && hermes -p c2 -z "$P" -t file > $RUN/run8.out 2> $RUN/run8.err); echo rc=$?; cnt; cat $RUN/work/hello.py
echo "### 9 duplicate top-level model: key appended (what Hermes does with it)"; cp $F $RUN/snap/09-before-dup.yaml; { echo; cat $S/configs/blocks/dup-model.yaml; } >> $F; hermes -p c2 config get model.default; echo rc=$?; (cd $RUN/work && hermes -p c2 -z "say hi" -t file > $RUN/run9.out 2> $RUN/run9.err); echo rc=$?; tail -3 $RUN/run9.err; cp $RUN/snap/09-before-dup.yaml $F
kill $A_PID $V_PID 2>/dev/null
