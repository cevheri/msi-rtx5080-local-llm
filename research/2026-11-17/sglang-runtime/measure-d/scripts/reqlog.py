"""Pull the sampling evidence out of an SGLang log written with --log-requests --log-requests-level 2:
for every "Receive OpenAI" line the request keys the client sent (and any sampling field), and for every "Receive:"
line the rid and the sampling_params dict the server built, plus the "Using default chat sampling params" line.
Model text is left out. Run with python3 -I: reqlog.py <raw.log> <out.txt>"""
import re
import sys

STAMP = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]")
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty", "frequency_penalty",
                 "max_tokens", "max_completion_tokens", "seed", "response_format", "tool_choice")


def braced(text, start):
    """The {...} block that starts at text[start], with nested braces and quoted strings respected."""
    depth, i, quote = 0, start, None
    while i < len(text):
        c = text[i]
        if quote:
            if c == "\\":
                i += 1
            elif c == quote:
                quote = None
        elif c in "'\"":
            quote = c
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
        i += 1
    return text[start:start + 300] + " [unterminated]"


def main():
    out = []
    for line in open(sys.argv[1], errors="replace"):
        stamp = (STAMP.match(line) or [None, "?"])[1]
        if "Using default chat sampling params" in line:
            out.append(line.strip()[:400])
        elif "Receive OpenAI: obj=" in line:
            body = line.split("Receive OpenAI: obj=", 1)[1]
            keys = re.findall(r"'([a-z_]+)': ", body[:4000])
            top = [k for k in dict.fromkeys(keys) if k in ("model", "messages", "max_tokens", "stream", "tools",
                                                            "tool_choice", "response_format", "reasoning_effort",
                                                            "chat_template_kwargs") or k in SAMPLING_KEYS]
            sampling = {k: re.search(rf"'{k}': ([^,}}]+)", body).group(1) for k in SAMPLING_KEYS
                        if re.search(rf"'{k}': ([^,}}]+)", body) and k not in ("max_tokens", "tool_choice", "response_format")}
            out.append(f"[{stamp}] Receive OpenAI: keys seen {top}; sampling fields sent {sampling or 'none'}")
        elif "Receive: obj=" in line:
            rid = re.search(r"rid='([^']+)'", line)
            pos = line.find("sampling_params=")
            params = braced(line, pos + len("sampling_params=")) if pos >= 0 else "no sampling_params"
            out.append(f"[{stamp}] Receive: rid={rid.group(1) if rid else '?'} sampling_params={params}")
    open(sys.argv[2], "w").write("\n".join(out) + "\n")
    print(sys.argv[2].rsplit("/", 1)[-1], len(out), "lines")


if __name__ == "__main__":
    main()
