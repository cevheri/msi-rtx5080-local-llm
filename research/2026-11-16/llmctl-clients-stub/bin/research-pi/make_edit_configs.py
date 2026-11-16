#!/usr/bin/env python3
"""models.json variants for the editing-mechanics tests (configs/D*/models.json, port placeholder __PORT__)."""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
toggle = json.load(open(f"{ROOT}/configs/toggle/models.json"))
llmctl = toggle["providers"]["llmctl"]
other = {"name": "my other server", "baseUrl": "http://127.0.0.1:__PORT__/v1", "api": "openai-completions",
         "apiKey": "other-placeholder", "models": [{"id": "other-model", "contextWindow": 32768, "maxTokens": 4096}]}


def write(name, text):
    os.makedirs(f"{ROOT}/configs/{name}", exist_ok=True)
    with open(f"{ROOT}/configs/{name}/models.json", "w") as fh:
        fh.write(text)


def dumps(providers, extra=None):
    doc = {"$schema": toggle["$schema"], **(extra or {}), "providers": providers}
    return json.dumps(doc, indent=2) + "\n"


# D1: // line comments and trailing commas (a user-edited file), plus another user provider
d1 = """{
  // user's own notes
  "$schema": "https://pi.dev/schemas/models.schema.json",
  "providers": {
    "my-other": { // a provider the user wrote by hand
      "name": "my other server",
      "baseUrl": "http://127.0.0.1:__PORT__/v1",
      "api": "openai-completions",
      "apiKey": "other-placeholder",
      "models": [{ "id": "other-model", "contextWindow": 32768, "maxTokens": 4096, }],
    },
    "llmctl": %s,
  },
}
""" % json.dumps(llmctl, indent=2).replace("\n", "\n    ")
write("D1-line-comments", d1)

# D2: a /* block */ comment
write("D2-block-comment", dumps({"my-other": other, "llmctl": llmctl}).replace('"providers"', '/* block */ "providers"', 1))

# D3: unknown keys at root, provider and model level
l3 = json.loads(json.dumps(llmctl))
l3["x-llmctl"] = {"managed": True, "recipe": "qwen3.6-35b-a3b", "runtime": "llamacpp"}
l3["models"][0]["x-llmctl-recipe"] = "qwen3.6-35b-a3b"
write("D3-unknown-keys", dumps({"my-other": other, "llmctl": l3}, extra={"x-root-note": "kept?"}))

# D4: one invalid value in llmctl (maxTokens 0), next to a valid user provider
l4 = json.loads(json.dumps(llmctl))
l4["models"][0]["maxTokens"] = 0
write("D4-invalid-value", dumps({"my-other": other, "llmctl": l4}))

# D4b: wrong type (reasoning as a string) in llmctl
l4b = json.loads(json.dumps(llmctl))
l4b["models"][0]["reasoning"] = "yes"
write("D4b-wrong-type", dumps({"my-other": other, "llmctl": l4b}))

# D4c: an unknown thinkingFormat value in llmctl compat
l4c = json.loads(json.dumps(llmctl))
l4c["models"][0]["compat"]["thinkingFormat"] = "chat_template"
write("D4c-bad-enum", dumps({"my-other": other, "llmctl": l4c}))

# D5: duplicate model id inside llmctl (second entry wins?)
l5 = json.loads(json.dumps(llmctl))
second = json.loads(json.dumps(l5["models"][0]))
second["name"] = "SECOND ENTRY WITH SAME ID"
second["contextWindow"] = 65536
l5["models"].append(second)
write("D5-duplicate-id", dumps({"llmctl": l5}))

# D6: unset $VAR apiKey
l6 = json.loads(json.dumps(llmctl))
l6["apiKey"] = "$LLMCTL_TEST_UNSET_KEY"
write("D6-unset-env-key", dumps({"llmctl": l6}))

# D7: missing baseUrl (semantic error found after the schema check)
l7 = json.loads(json.dumps(llmctl))
del l7["baseUrl"]
write("D7-missing-baseurl", dumps({"my-other": other, "llmctl": l7}))
print(sorted(d for d in os.listdir(f"{ROOT}/configs") if d.startswith("D")))
