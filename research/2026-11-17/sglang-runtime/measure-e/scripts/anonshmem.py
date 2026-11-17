"""Peak of cgroup anon + shmem (the D5 ram_gib rule) and of anon + shmem + kernel (non-reclaimable, no swap) in a
trace_c.sh CSV, each with the moment (seconds after the first sample) and the split there; plus the anon and shmem peaks.
Run with python3 -I: anonshmem.py TRACE.csv [t0-epoch]"""
import csv
import sys

G = 2**30
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r.get("cg_anon_b")]
t0 = float(sys.argv[2]) if len(sys.argv) > 2 else float(rows[0]["epoch"])


def best(key):
    top = max(rows, key=key)
    return key(top) / G, top


a_s, r1 = best(lambda r: float(r["cg_anon_b"]) + float(r["cg_shmem_b"]))
nr, r2 = best(lambda r: float(r["cg_anon_b"]) + float(r["cg_shmem_b"]) + float(r["cg_kernel_b"]))
fmt = lambda r: (f"at {float(r['epoch']) - t0:.1f} s: anon {float(r['cg_anon_b'])/G:.2f}, shmem {float(r['cg_shmem_b'])/G:.2f}, "
                 f"kernel {float(r['cg_kernel_b'])/G:.2f}, file {float(r['cg_file_b'])/G:.2f}, current {float(r['cg_current_b'])/G:.2f} GiB")
print(f"anon+shmem peak {a_s:.2f} GiB {fmt(r1)}")
print(f"anon+shmem+kernel peak {nr:.2f} GiB {fmt(r2)}")
print(f"anon peak {max(float(r['cg_anon_b']) for r in rows)/G:.2f} GiB; shmem peak {max(float(r['cg_shmem_b']) for r in rows)/G:.2f} GiB; "
      f"memory.peak {max(float(r['cg_peak_b']) for r in rows)/G:.2f} GiB")
