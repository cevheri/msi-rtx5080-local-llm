"""Pair streamed replays with the raw output text that SGLang's request log holds for them. Run with python3 -I.

Usage: logged_outputs.py <replay folder>
Reads server.log (docker logs --timestamps, taken while /configure_logging had log_requests on), joins the pieces that
docker splits long lines into, takes each finished request's id and output text, and prints one JSON line per
stream-<i> replay: what the event stream delivered (tool call names, content, reasoning_content, finish_reason,
usage) and the raw text the model generated, special tokens included.
"""
import json
import pathlib
import re
import sys

folder = pathlib.Path(sys.argv[1])
log = (folder / "server.log").read_text(errors="replace")
log = re.sub(r"\n?\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+Z ", "", log)
outputs = {}
for piece in log.split("out={'text': '")[1:]:
    text = piece.split("', 'output_ids'", 1)[0]
    found = re.search(r"'meta_info': \{'id': '([0-9a-f]+)'", piece)
    if found:
        outputs[found.group(1)] = text
for sse in sorted(folder.glob("stream-*.sse"), key=lambda p: int(p.stem.split("-")[1])):
    rid = re.search(r'"id":"([0-9a-f]+)"', sse.read_text()).group(1)
    joined = json.loads((folder / (sse.stem + ".json")).read_text())
    print(json.dumps({"replay": sse.stem, "id": rid,
                      "tool_calls": [call["name"] for call in joined.get("tool_calls") or []],
                      "content": joined.get("content"), "reasoning_content": joined.get("reasoning_content"),
                      "finish_reason": joined.get("finish_reason"), "matched_stop": joined.get("matched_stop"),
                      "usage": joined.get("usage"), "raw_text": outputs.get(rid)}, ensure_ascii=False))
