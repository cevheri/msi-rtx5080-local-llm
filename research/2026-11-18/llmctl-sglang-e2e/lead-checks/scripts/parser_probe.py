"""Run fixed gpt-oss outputs through SGLang's own gpt-oss parsers. Run with python3 -I.

Usage: parser_probe.py <n>-request.json (for its tools)
For each text: /separate_reasoning with reasoning_parser gpt-oss, then /parse_function_call with tool_call_parser
gpt-oss on the text that the reasoning parser left, the order in which serving_chat.py runs them for a non-streamed
reply. Prints one JSON line per text.
"""
import json
import pathlib
import sys
import urllib.request

BASE = "http://127.0.0.1:30000"
CALL = '<|message|>{"path":"hello.py","offset":1,"limit":2000}<|call|>'
ANALYSIS = "<|channel|>analysis<|message|>Now read the file and count lines.<|end|><|start|>assistant"
TEXTS = {
    "A1 analysis, then to=functions.read<|constrain|>json": ANALYSIS + "<|channel|>commentary to=functions.read<|constrain|>json" + CALL,
    "A2 analysis, then to=functions.read <|constrain|>json": ANALYSIS + "<|channel|>commentary to=functions.read <|constrain|>json" + CALL,
    "A3 analysis, then to=functions.read json": ANALYSIS + "<|channel|>commentary to=functions.read json" + CALL,
    "A4 analysis, then to=functions.read<|channel|>commentary <|constrain|>json": ANALYSIS + "<|channel|>commentary to=functions.read<|channel|>commentary <|constrain|>json" + CALL,
    "A5 analysis, then to=functions.read<|channel|>commentary json": ANALYSIS + "<|channel|>commentary to=functions.read<|channel|>commentary json" + CALL,
    "A6 analysis, then analysis to=functions.read code": ANALYSIS + "<|channel|>analysis to=functions.read code" + CALL,
    "F1 first, to=functions.read<|constrain|>json": "<|channel|>commentary to=functions.read<|constrain|>json" + CALL,
    "F2 first, to=functions.read <|constrain|>json": "<|channel|>commentary to=functions.read <|constrain|>json" + CALL,
    "F3 first, to=functions.read json": "<|channel|>commentary to=functions.read json" + CALL,
}


def post(path: str, body: dict) -> dict:
    request = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def main() -> None:
    tools = json.loads(pathlib.Path(sys.argv[1]).read_text())["body"]["tools"]
    for name, text in TEXTS.items():
        separated = post("/separate_reasoning", {"text": text, "reasoning_parser": "gpt-oss"})
        parsed = post("/parse_function_call", {"text": separated["text"], "tool_call_parser": "gpt-oss",
                                               "tools": tools})
        print(json.dumps({"case": name, "text": text, "reasoning_text": separated["reasoning_text"],
                          "text_after_reasoning": separated["text"], "calls": parsed["calls"],
                          "normal_text": parsed["normal_text"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
