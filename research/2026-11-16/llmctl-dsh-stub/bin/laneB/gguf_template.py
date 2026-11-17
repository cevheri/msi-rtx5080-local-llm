#!/usr/bin/env python3
"""Print tokenizer.chat_template from a GGUF header: gguf_template.py <file.gguf> [key]."""
import struct
import sys

TYPES = {0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4), 5: ("i", 4), 6: ("f", 4),
         7: ("?", 1), 10: ("Q", 8), 11: ("q", 8), 12: ("d", 8)}


def read_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n)


def read_value(f, vtype, keep):
    if vtype == 8:
        s = read_str(f)
        return s.decode("utf-8", "replace") if keep else None
    if vtype == 9:
        (atype,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        if atype == 8:
            for _ in range(count):
                read_str(f)
        elif atype in TYPES:
            f.seek(TYPES[atype][1] * count, 1)
        else:
            raise ValueError(f"nested array type {atype}")
        return f"<array {count}>"
    fmt, size = TYPES[vtype]
    (v,) = struct.unpack("<" + fmt, f.read(size))
    return v


def main():
    path = sys.argv[1]
    want = sys.argv[2] if len(sys.argv) > 2 else "tokenizer.chat_template"
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise SystemExit("not a GGUF file")
        (version,) = struct.unpack("<I", f.read(4))
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
        for _ in range(n_kv):
            key = read_str(f).decode("utf-8", "replace")
            (vtype,) = struct.unpack("<I", f.read(4))
            value = read_value(f, vtype, keep=(key == want))
            if key == want:
                print(value)
                return
    raise SystemExit(f"{want} not found")


if __name__ == "__main__":
    main()
