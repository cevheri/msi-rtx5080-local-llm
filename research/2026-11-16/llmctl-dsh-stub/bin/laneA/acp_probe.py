#!/usr/bin/env python3
"""Drive `dsh --profile acp` over stdio (newline-delimited JSON-RPC) for lane A.
usage: acp_probe.py <DSH_HOME> <workdir> <rewrite-cmd-json|-> [prompt]
Prints: model option per session/new, before and after an optional home-patch rewrite, and a prompt result."""
import json, os, subprocess, sys, threading, time, queue

home, work, rewrite = sys.argv[1], sys.argv[2], sys.argv[3]
prompt = sys.argv[4] if len(sys.argv) > 4 else None
env = dict(os.environ, DSH_HOME=home, DSH_TELEMETRY_DISABLED='1')
p = subprocess.Popen(['heavy', 'dsh', '--profile', 'acp'], cwd=work, env=env,
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
q = queue.Queue()
def reader():
    for line in p.stdout:
        q.put(line)
threading.Thread(target=reader, daemon=True).start()
err_lines = []
threading.Thread(target=lambda: [err_lines.append(l) for l in p.stderr], daemon=True).start()
nid = [0]
updates = []
def call(method, params, timeout=120):
    nid[0] += 1
    mid = nid[0]
    p.stdin.write(json.dumps({'jsonrpc': '2.0', 'id': mid, 'method': method, 'params': params}) + '\n')
    p.stdin.flush()
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            line = q.get(timeout=1)
        except queue.Empty:
            continue
        msg = json.loads(line)
        if msg.get('id') == mid and ('result' in msg or 'error' in msg):
            return msg
        if 'method' in msg and 'id' in msg:   # agent -> client request (permissions etc.)
            p.stdin.write(json.dumps({'jsonrpc': '2.0', 'id': msg['id'], 'error': {'code': -32601, 'message': 'not supported by probe'}}) + '\n'); p.stdin.flush()
        elif 'method' in msg:
            updates.append(msg)
    raise SystemExit(f'timeout waiting for {method}')

def model_option(result):
    out = []
    for opt in result.get('configOptions', []):
        if opt.get('category') == 'model' or 'model' in str(opt.get('id', '')).lower():
            vals = []
            for o in opt.get('options', []):
                if 'options' in o:
                    vals += [x.get('value') for x in o['options']]
                else:
                    vals.append(o.get('value'))
            out.append({'id': opt.get('id'), 'currentValue': opt.get('currentValue'), 'values': vals})
    return out

r = call('initialize', {'protocolVersion': 1, 'clientCapabilities': {}})
print('initialize:', 'ok' if 'result' in r else r)
r1 = call('session/new', {'cwd': work, 'mcpServers': []})
print('session/new #1:', json.dumps(model_option(r1.get('result', {}))) if 'result' in r1 else r1)
if rewrite != '-':
    subprocess.run(json.loads(rewrite), check=True)
    time.sleep(5)
    r2 = call('session/new', {'cwd': work, 'mcpServers': []})
    print('session/new #2 after rewrite:', json.dumps(model_option(r2.get('result', {}))) if 'result' in r2 else r2)
if prompt:
    sid = r1['result']['sessionId']
    t0 = time.time()
    rp = call('session/prompt', {'sessionId': sid, 'prompt': [{'type': 'text', 'text': prompt}]}, timeout=300)
    texts = []
    for u in updates:
        upd = u.get('params', {}).get('update', {})
        if upd.get('sessionUpdate') == 'agent_message_chunk':
            c = upd.get('content', {})
            if c.get('type') == 'text':
                texts.append(c.get('text', ''))
    print('session/prompt:', json.dumps(rp.get('result', rp.get('error'))), 'text=', repr(''.join(texts)), f'{time.time()-t0:.1f}s')
p.stdin.close()
try:
    p.wait(timeout=30)
except subprocess.TimeoutExpired:
    p.terminate(); p.wait(timeout=10)
print('exit:', p.returncode)
print('stderr:', ''.join(err_lines)[:1500])
