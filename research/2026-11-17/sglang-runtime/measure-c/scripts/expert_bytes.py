"""Sum the bytes of the routed-expert tensors of a safetensors snapshot from the shard headers (the checkpoint index
names the shards; each header gives every tensor's dtype, shape and byte range), per MoE layer and in total, and the
bytes of everything else. Reads only the headers. Run with python3 -I: expert_bytes.py <snapshot-dir> <expert-regex>
where the regex has one group for the layer number, for example 'layers\\.(\\d+)\\.mlp\\.experts\\.\\d+\\.'"""
import json
import os
import re
import struct
import sys
from collections import defaultdict

snap, pattern = sys.argv[1], re.compile(sys.argv[2])
index = os.path.join(snap, "model.safetensors.index.json")
if os.path.exists(index):
    shards = sorted(set(json.load(open(index))["weight_map"].values()))
else:
    shards = sorted(f for f in os.listdir(snap) if f.endswith(".safetensors"))
per_layer, kinds = defaultdict(int), defaultdict(int)
expert_total = other_total = 0
examples = {}
for shard in shards:
    with open(os.path.join(snap, shard), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        size = end - start
        m = pattern.search(name)
        if m:
            expert_total += size
            per_layer[int(m.group(1))] += size
            suffix = re.sub(r"\d+", "N", name.split("experts", 1)[1])
            kinds[suffix] += size
            examples.setdefault(suffix, (name, meta["dtype"], meta["shape"]))
        else:
            other_total += size
G = 2**30
layers = sorted(per_layer)
print(json.dumps({
    "snapshot_shards": len(shards),
    "moe_layers": len(layers),
    "expert_bytes": expert_total, "expert_gib": round(expert_total / G, 3),
    "other_bytes": other_total, "other_gib": round(other_total / G, 3),
    "per_layer_mib": sorted({round(v / 2**20, 2) for v in per_layer.values()}),
    "by_tensor_kind_mib": {k: round(v / 2**20, 1) for k, v in sorted(kinds.items())},
    "examples": examples,
}, indent=1))
