"""Peak of the non-reclaimable cgroup memory in a trace_b.sh CSV: anon + shmem + kernel per sample (the container has
no swap, so page cache is the only part the kernel can reclaim). Run with python3 -I: nonreclaim.py TRACE.csv"""
import csv
import sys

G = 2**30
best, at, rows = None, None, list(csv.DictReader(open(sys.argv[1])))
for r in rows:
    try:
        v = (float(r["cg_anon_b"]) + float(r["cg_shmem_b"]) + float(r["cg_kernel_b"])) / G
    except (KeyError, ValueError):
        continue
    if best is None or v > best:
        best, at = v, r
if at:
    print(f"non-reclaimable peak {best:.2f} GiB: anon {float(at['cg_anon_b'])/G:.2f}, shmem {float(at['cg_shmem_b'])/G:.2f}, "
          f"kernel {float(at['cg_kernel_b'])/G:.2f}, file {float(at['cg_file_b'])/G:.2f}, current {float(at['cg_current_b'])/G:.2f} GiB; "
          f"peak anon {max(float(r['cg_anon_b']) for r in rows if r['cg_anon_b'])/G:.2f} GiB; memory.peak {max(float(r['cg_peak_b']) for r in rows if r['cg_peak_b'])/G:.2f} GiB")
