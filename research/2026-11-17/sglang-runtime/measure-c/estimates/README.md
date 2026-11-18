# Pinned-store estimates before each model's first start (measurement C)

Last verified: 2026-11-17.

The brief's rule: estimate each model's store from the expert bytes of its checkpoint index times 1.84, plus about 5 GiB, and start nothing whose estimate exceeds 34 GiB.
The expert bytes come from the safetensors shard headers that each checkpoint index names (`$SCRATCH/sglang/measure-c/scripts/expert_bytes.py`; `<id>.expert-bytes.json` here), and each estimate was made before the model's first start.
After the gpt-oss check (start pe-round1, 20:38 UTC) measured the store with `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1` at 10.37 GiB for 9.467 GiB of checkpoint expert tensors, the factor became 1.096, as the brief allows once the setting is adopted.

| Model | Expert tensors in the checkpoint | MoE layers, per layer | Estimate with 1.84 | Estimate with 1.096 (used) | Measured store (cgroup shmem) |
|---|---|---|---|---|---|
| gpt-oss-20b (B's baseline) | 9.467 GiB | 24, 403.95 MiB | 22.4 GiB (B measured 19.19 GiB of store) | 15.4 GiB | 10.37 GiB with the setting |
| qwen3.6-35b-a3b (`nvidia/Qwen3.6-35B-A3B-NVFP4`) | 16.875 GiB | 40, 432.01 MiB | 36.0 GiB, above 34 | 23.5 GiB | see the model's record |
| nemotron-3.5-lightning (`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`) | 15.389 GiB (the MTP layer's 2.379 GiB of BF16 experts excluded) | 23, 685.13 MiB | 33.3 GiB | 21.9 GiB | see the model's record |
| gemma-4-26b-a4b (`cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4`) | 11.964 GiB | 30, 408.38 MiB | 27.0 GiB | 18.1 GiB | see the model's record |
| k2-horizon-mova-36b (`cyankiwi/K2-Horizon-MoVA-36B-A4B-AWQ-INT4`) | 14.291 GiB routed (and 4.065 GiB of value experts, which are attention weights and not paged) | 45, 325.2 MiB | 31.3 GiB | 20.7 GiB | see the model's record |

All checkpoint figures are measured from the shard headers on 2026-11-17; the estimates are arithmetic (estimate).
