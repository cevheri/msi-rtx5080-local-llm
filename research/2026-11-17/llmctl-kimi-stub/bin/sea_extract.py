import sys, struct, os, re, hashlib
binpath, outdir = sys.argv[1], sys.argv[2]
NOTE_OFF = 0x7540188 + 4  # desc start (after 'NODE_SEA_BLOB\0' name padding)
with open(binpath, 'rb') as f:
    data = f.read()
# locate note header to be robust
import re as _re
m = _re.search(rb'\x0e\x00\x00\x00....\x00\x00\x00\x00NODE_SEA_BLOB\x00', data, _re.S)
idx = m.start() + 12
namesz, descsz, ntype = struct.unpack_from('<III', data, idx - 12)
desc = idx + ((namesz + 3) // 4) * 4
print('note at', hex(idx - 12), 'namesz', namesz, 'descsz', descsz, 'desc_off', hex(desc))
p = desc
def u32():
    global p; v = struct.unpack_from('<I', data, p)[0]; p += 4; return v
def u8():
    global p; v = data[p]; p += 1; return v
def u64():
    global p; v = struct.unpack_from('<Q', data, p)[0]; p += 8; return v
def sv():
    global p; n = u64(); s = data[p:p+n]; p += n; return s
magic = u32(); flags = u32(); ext = u8()
print('magic', hex(magic), 'flags', hex(flags), 'exec_argv_ext', ext)
code_path = sv(); print('code_path', code_path.decode())
code = sv(); print('main code bytes', len(code), 'sha256', hashlib.sha256(code).hexdigest())
os.makedirs(outdir, exist_ok=True)
with open(os.path.join(outdir, 'main.cjs'), 'wb') as o: o.write(code)
if flags & 4:
    cc = sv(); print('code cache bytes', len(cc))
if flags & 8:
    n = u64(); print('assets', n)
    ad = os.path.join(outdir, 'assets'); os.makedirs(ad, exist_ok=True)
    for i in range(n):
        k = sv(); v = sv()
        name = re.sub(r'[^A-Za-z0-9._-]', '_', k.decode('utf-8', 'replace'))
        with open(os.path.join(ad, f'{i:03d}_{name}'), 'wb') as o: o.write(v)
        print(f'  asset {i}: {k.decode("utf-8","replace")!r} {len(v)} bytes magic={v[:8].hex()}')
if flags & 16:
    n = u64(); print('exec_argv', [sv().decode() for _ in range(n)])
print('end', hex(p), 'note end', hex(desc + descsz))
