"""Send chat requests one after another and print one JSON line per request. Run with python3 -I.

Usage: chat_requests.py <url> <model> <count> <max_tokens> <prompt>
Each line holds the UTC start time, the seconds, the HTTP status and the reply body as the server sent it.
"""
import datetime
import json
import sys
import time
import urllib.error
import urllib.request


def main() -> None:
    url, model, count, max_tokens, prompt = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
    for number in range(1, count + 1):
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        start = time.monotonic()
        request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                status, text = response.status, response.read().decode()
        except urllib.error.HTTPError as exc:
            status, text = exc.code, exc.read().decode(errors="replace")
        seconds = round(time.monotonic() - start, 3)
        try:
            reply = json.loads(text)
        except ValueError:
            reply = text
        print(json.dumps({"request": number, "start_utc": stamp, "seconds": seconds, "status": status,
                          "body": body, "reply": reply}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
