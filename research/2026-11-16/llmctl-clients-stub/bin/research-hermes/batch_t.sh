#!/bin/bash
S=$SCRATCH/clients/research-hermes
C=$S/configs/cases; P="Create hello.py that prints hello, then read it back and show its contents"
R=("$S/bin/run_case.sh" ht)
"${R[@]}" t1-title-mo-bare $C/r03-mo-bare.yaml llamacpp local 131072 "" chat -q "$P" -t file -Q --source tool
"${R[@]}" t2-title-vllm-reject-none $C/r14-vllm-gptoss.yaml vllm gpt-oss-20b 65536 "--reject-effort none" chat -q "$P" -t file -Q --source tool
