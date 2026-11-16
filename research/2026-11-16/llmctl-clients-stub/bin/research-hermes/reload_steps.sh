#!/bin/bash
# Inside the overlay sandbox: one profile, one session resumed across config rewrites.
S=$SCRATCH/clients/research-hermes
RUN=$S/runs/reload; F=$HOME/.hermes/profiles/rl/config.yaml; C=$S/configs/reload
mkdir -p $RUN/cap8080 $RUN/cap8000 $RUN/work
coproc A { exec python3 -I $S/bin/stub_server.py --port 8080 --flavor llamacpp --model local --ctx 131072 --capture-dir $RUN/cap8080 2>$RUN/stub8080.log; }
read -r -t 15 l1 <&"${A[0]}"
coproc V { exec python3 -I $S/bin/stub_server.py --port 8000 --flavor vllm --model gpt-oss-20b --ctx 65536 --capture-dir $RUN/cap8000 2>$RUN/stub8000.log; }
read -r -t 15 l2 <&"${V[0]}"
put() { cp $C/$1.yaml $F.tmp && mv $F.tmp $F; echo "config -> $1"; }
mark() { echo "--- $1: captures 8080=$(ls $RUN/cap8080 | wc -l) 8000=$(ls $RUN/cap8000 | wc -l)"; }
cd $RUN/work
put A-ctk-on; mark before-1
hermes -p rl chat -q "Create hello.py that prints hello, then read it back and show its contents" -t file -Q --source tool > $RUN/1.out 2> $RUN/1.err; echo "turn1 rc=$?"
SID=$(grep -o 'session_id: [^ ]*' $RUN/1.err | tail -1 | cut -d' ' -f2); echo "session=$SID"; mark after-1
put B-ctk-off
hermes -p rl chat --resume "$SID" -q "say hi" -t file -Q --source tool > $RUN/2.out 2> $RUN/2.err; echo "turn2 (resume, extra_body changed) rc=$?"; mark after-2
put C-ctk-off-noreason
hermes -p rl chat --resume "$SID" -q "say hi again" -t file -Q --source tool > $RUN/3.out 2> $RUN/3.err; echo "turn3 (resume, model_overrides added) rc=$?"; mark after-3
put D-vllm
hermes -p rl chat --resume "$SID" -q "and once more" -t file -Q --source tool > $RUN/4.out 2> $RUN/4.err; echo "turn4 (resume, endpoint+model changed) rc=$?"; mark after-4
hermes -p rl -z "fresh session hi" -t file > $RUN/5.out 2> $RUN/5.err; echo "turn5 (new session) rc=$?"; mark after-5
put B-ctk-off
hermes -p rl -z "fresh session back on llama" -t file > $RUN/6.out 2> $RUN/6.err; echo "turn6 (new session, back to B) rc=$?"; mark after-6
echo "gateway starts: $(wc -l < $HOME/.hermes/profiles/rl/gateway-starts.log)"
kill $A_PID $V_PID 2>/dev/null
