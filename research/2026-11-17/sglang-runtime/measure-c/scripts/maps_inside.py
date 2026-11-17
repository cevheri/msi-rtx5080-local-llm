"""Run inside a container (docker exec <name> python3 -I - < maps_inside.py): for every process whose resident set is
above 100 MiB, sum /proc/<pid>/smaps by mapping name (anonymous mappings as [anon]) and print the mappings above 64 MiB
with Rss, shared and private bytes, Locked, the flags of the largest such mapping, and the mount of a path; this shows
where a pinned host store lives (anonymous memory, a shared mapping, a /dev/shm file or a driver mapping)."""
import os
import re

MiB = 1024
for pid in sorted((p for p in os.listdir("/proc") if p.isdigit()), key=int):
    try:
        status = open(f"/proc/{pid}/status").read()
        rss = int(re.search(r"VmRSS:\s+(\d+)", status).group(1))
        name = re.search(r"Name:\s+(\S+)", status).group(1)
        smaps = open(f"/proc/{pid}/smaps").read().splitlines()
    except (OSError, AttributeError):
        continue
    if rss < 100 * MiB:
        continue
    groups, current = {}, None
    for line in smaps:
        m = re.match(r"^[0-9a-f]+-[0-9a-f]+ (\S+) \S+ \S+ \d+\s*(.*)$", line)
        if m:
            path = m.group(2).strip() or "[anon]"
            current = groups.setdefault(path, {"perms": set(), "Rss": 0, "Shared": 0, "Private": 0, "Locked": 0, "maps": 0, "flags": ""})
            current["perms"].add(m.group(1))
            current["maps"] += 1
            continue
        if current is None:
            continue
        key, _, value = line.partition(":")
        value = value.strip()
        if key == "Rss":
            current["Rss"] += int(value.split()[0])
        elif key in ("Shared_Clean", "Shared_Dirty"):
            current["Shared"] += int(value.split()[0])
        elif key in ("Private_Clean", "Private_Dirty"):
            current["Private"] += int(value.split()[0])
        elif key == "Locked":
            current["Locked"] += int(value.split()[0])
        elif key == "VmFlags" and not current["flags"]:
            current["flags"] = value
    print(f"pid={pid} name={name} VmRSS={rss // MiB} MiB")
    for path, g in sorted(groups.items(), key=lambda kv: -kv[1]["Rss"]):
        if g["Rss"] < 64 * MiB:
            continue
        print(f"  {path}: Rss {g['Rss'] // MiB} MiB, shared {g['Shared'] // MiB} MiB, private {g['Private'] // MiB} MiB, "
              f"locked {g['Locked'] // MiB} MiB, {g['maps']} mappings, perms {sorted(g['perms'])}, first VmFlags [{g['flags']}]")
print("mounts:")
for line in open("/proc/self/mounts"):
    if " /dev/shm " in line or "tmpfs /tmp" in line:
        print("  " + line.strip())
