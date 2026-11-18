"""One-rank NCCL probe for the SGLang v0.5.23 image: how much GPU memory the first all_reduce on a one-rank
torch NCCL group takes, and which NCCL settings change it.

Run inside the image with the GPU and no network:
  python3 -I /probe/nccl_probe.py                      runs every variant in VARIANTS, one child process each
  python3 -I /probe/nccl_probe.py --only NAME [NAME]   runs the named variants
Each child mimics the sampler: a one-rank process group, a one-rank device group made with new_group, and an
int64 all_reduce(MIN) on that group, as layers/sampler.py:_sync_token_ids_across_tp does.
"""
import json
import os
import subprocess
import sys
import time

# SGLang's engine sets these for a single-node start without symmetric memory (entrypoints/engine.py:1898-1921).
SGLANG_ENV = {"NCCL_CUMEM_ENABLE": "0", "NCCL_NVLS_ENABLE": "0", "CUDA_DEVICE_MAX_CONNECTIONS": "8"}

DEBUG = {"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,ALLOC"}
# run1 showed the 512 MiB as "transport/net.cc:675 (sharedNetBuffersInit) Cuda Calloc Size 536870912" with
# "64 p2p channels" and "P2P Chunksize set to 524288": 64 x 8 steps x 512 KiB x 2 = 512 MiB, so these settings
# should remove or shrink it (NCCL parameter names as listed by strings on the image's libnccl.so.2).
VARIANTS = {
    "baseline": {},
    "debug": DEBUG,
    "net_shared_buffers_0": dict(DEBUG, NCCL_NET_SHARED_BUFFERS="0"),
    "max_nchannels_1": dict(DEBUG, NCCL_MAX_NCHANNELS="1"),
    "max_p2p_nchannels_1": dict(DEBUG, NCCL_MAX_P2P_NCHANNELS="1"),
    "p2p_net_chunksize_64k": dict(DEBUG, NCCL_P2P_NET_CHUNKSIZE="65536"),
}


def mib(nbytes):
    return round(nbytes / 2**20, 1)


def child():
    import torch
    import torch.distributed as dist

    out = {"variant": os.environ.get("PROBE_VARIANT"), "env": {k: v for k, v in os.environ.items()
                                                                if k.startswith(("NCCL_", "TORCH_NCCL", "SYNC_"))}}
    torch.cuda.set_device(0)
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    out["free_after_context_mib"] = mib(torch.cuda.mem_get_info()[0])
    init_file = f"/tmp/nccl-init-{os.getpid()}"
    dist.init_process_group(backend="nccl", init_method=f"file://{init_file}", world_size=1, rank=0)
    out["free_after_init_process_group_mib"] = mib(torch.cuda.mem_get_info()[0])
    group = dist.new_group([0], backend="nccl")
    out["free_after_new_group_mib"] = mib(torch.cuda.mem_get_info()[0])
    ids = torch.ones(4, dtype=torch.int64, device="cuda")
    t0 = time.time()
    try:
        dist.all_reduce(ids, op=dist.ReduceOp.MIN, group=group)
        torch.cuda.synchronize()
        out["first_all_reduce"] = "ok"
    except Exception as err:  # recorded, the probe is about this failure
        out["first_all_reduce"] = repr(err)[:400]
    out["first_all_reduce_s"] = round(time.time() - t0, 3)
    out["free_after_first_all_reduce_mib"] = mib(torch.cuda.mem_get_info()[0])
    dist.all_reduce(ids, op=dist.ReduceOp.MIN, group=group)
    torch.cuda.synchronize()
    out["free_after_second_all_reduce_mib"] = mib(torch.cuda.mem_get_info()[0])
    out["torch_reserved_mib"] = mib(torch.cuda.memory_reserved())
    out["nccl_version"] = ".".join(str(x) for x in torch.cuda.nccl.version())
    out["taken_by_first_all_reduce_mib"] = round(out["free_after_new_group_mib"] - out["free_after_first_all_reduce_mib"], 1)
    print("PROBE-RESULT " + json.dumps(out), flush=True)
    dist.destroy_process_group()


def main():
    names = sys.argv[sys.argv.index("--only") + 1:] if "--only" in sys.argv else list(VARIANTS)
    for name in names:
        env = dict(os.environ, **SGLANG_ENV, **VARIANTS[name], PROBE_VARIANT=name)
        proc = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--child"], env=env,
                              capture_output=True, text=True, timeout=300)
        print(f"===== variant {name} exit={proc.returncode}", flush=True)
        print(proc.stdout, flush=True)
        print(proc.stderr[-20000:], flush=True)


if __name__ == "__main__":
    if "--child" in sys.argv:
        child()
    else:
        main()
