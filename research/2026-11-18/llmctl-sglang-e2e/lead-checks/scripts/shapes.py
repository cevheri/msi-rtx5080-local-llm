"""Describe captured chat exchanges without their private parts. Run with python3 -I.

Usage: shapes.py <http folder from strace_http.py> <folder> <shown name>
Every occurrence of <folder> (the scratch folder) in a shown text becomes <shown name> before the text is cut.
One block per exchange: its time, the request's parameters other than messages and tools, the number of tools, each
message's role and length (with the first 100 characters of a user message, and none of the system prompt, which
names local paths), then the reply's status line and, for an error, its body.
"""
import json
import pathlib
import sys


def main() -> None:
    folder, work, shown = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    for request in sorted(folder.glob("*-request.json")):
        record = json.loads(request.read_text())
        body = record["body"] if isinstance(record["body"], dict) else {}
        reply = (folder / request.name.replace("-request.json", "-response.txt")).read_bytes().decode(errors="replace")
        head, _, rest = reply.partition("\r\n\r\n")
        params = {k: v for k, v in body.items() if k not in ("messages", "tools")}
        print(f"exchange {record['n']} at {record['time']}: {record['head'][0]}")
        print(f"  parameters {json.dumps(params)}, tools {len(body.get('tools') or [])}")
        for message in body.get("messages", []):
            content = message.get("content")
            text = content if isinstance(content, str) else json.dumps(content)
            extra = {k: len(json.dumps(v)) for k, v in message.items() if k not in ("role", "content")}
            line = f"  {message.get('role')}: {len(text or '')} chars"
            if extra:
                line += f", other fields (JSON chars) {extra}"
            if message.get("role") == "user":
                line += " | " + (text or "").replace(work, shown)[:100].replace("\n", " ")
            print(line)
        status = head.split("\r\n")[0]
        print(f"  reply: {status}")
        if " 200 " not in status:
            print(f"  body: {rest.strip()}")


if __name__ == "__main__":
    main()
