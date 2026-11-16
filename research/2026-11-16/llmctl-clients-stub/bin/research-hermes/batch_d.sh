#!/bin/bash
S=$SCRATCH/clients/research-hermes
C=$S/configs/cases; P="Create hello.py that prints hello, then read it back and show its contents"
R=("$S/bin/run_case.sh" hd)
"${R[@]}" d1-chat-q-streamjson $C/b1s-baseline-standalone.yaml llamacpp local 131072 "" chat -q "$P" -t file --format stream-json --source tool --max-turns 10
"${R[@]}" d2-chat-q-text $C/b1s-baseline-standalone.yaml llamacpp local 131072 "" chat -q "$P" -t file -Q --source tool --max-turns 10
"${R[@]}" d3-oneshot-usage $C/b1s-baseline-standalone.yaml llamacpp local 131072 "" -z "$P" -t file --usage-file $S/runs/hd/d3-usage.json
