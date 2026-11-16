"""Every shipped recipe must match the launch its document shows, and the OpenCode config in docs/opencode.md.

When one of these fails, fix whichever side disagrees with the measurement record; never loosen the test.
"""

import json
import re
import shlex
import unittest

from launcher.catalog import load_catalog
from launcher.runtimes import ADAPTERS
from launcher.tests.fakes import REPO


def slug(heading: str) -> str:
    """GitHub's anchor rule, as the knowledge base's link checker applies it."""
    text = re.sub(r"`|\*\*|\*", "", heading)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    return re.sub(r"[^\w\- ]", "", text.lower()).replace(" ", "-")


def section(doc: str) -> list[str]:
    path, anchor = doc.split("#", 1)
    lines, found, depth = [], False, 0
    for line in (REPO / path).read_text().splitlines():
        heading = re.match(r"^(#{2,6})\s+(.*?)\s*$", line)
        if heading:
            if found and len(heading.group(1)) <= depth:
                break
            if slug(heading.group(2)) == anchor:
                found, depth = True, len(heading.group(1))
                continue
        if found:
            lines.append(line)
    if not found:
        raise AssertionError(f"no section #{anchor} in {path}")
    return lines


def last_run_block(lines: list[str]) -> list[str] | None:
    blocks, current, inside = [], [], False
    for line in lines:
        if line.startswith("```"):
            if inside:
                blocks.append("\n".join(current))
                current = []
            inside = not inside
            continue
        if inside:
            current.append(line)
    blocks = [block for block in blocks if "docker run -d" in block]
    return shlex.split(blocks[-1].replace("\\\n", " ")) if blocks else None


def take(tokens: list[str], flag: str) -> str | None:
    if flag not in tokens:
        return None
    index = tokens.index(flag)
    value = tokens[index + 1]
    del tokens[index:index + 2]
    return value


def llamacpp_launch(doc: str) -> dict:
    lines = section(doc)
    tokens = last_run_block(lines)
    if tokens is None:
        model_line = next(line for line in lines if "**Model lines:**" in line)
        server = shlex.split(re.search(r"`([^`]+)`", model_line).group(1))
        memory = re.search(r"--memory (\d+)g", model_line).group(1)
    else:
        memory = tokens[tokens.index("--memory") + 1].rstrip("g")
        server = tokens[next(i for i, t in enumerate(tokens) if t.startswith("ghcr.io/ggml-org/llama.cpp:")) + 1:]
    hf = take(server, "-hf")
    launch = {"memory": int(memory), "repo": hf.split(":")[0] if hf else None, "file": take(server, "--hf-file"),
              "alias": take(server, "--alias"), "context": int(take(server, "-c"))}
    take(server, "-lv")
    launch["args"] = [token for token in server if token != "--no-mmproj"]
    return launch


def vllm_launch(doc: str) -> dict:
    tokens = last_run_block(section(doc))
    server = tokens[next(i for i, t in enumerate(tokens) if t.startswith("vllm/vllm-openai:")) + 1:]
    return {"memory": int(tokens[tokens.index("--memory") + 1].rstrip("g")), "repo": server.pop(0),
            "served": take(server, "--served-model-name"), "context": int(take(server, "--max-model-len")),
            "args": server}


def opencode_providers() -> dict:
    text = (REPO / "docs/opencode.md").read_text()
    block = re.search(r"```json\n(.*?)\n```", text[text.index("## The validated config"):], re.S).group(1)
    return json.loads(block)["providers"]


class DocsConsistencyTest(unittest.TestCase):
    catalog = load_catalog(REPO / "recipes", None)

    def test_llamacpp_docs_match(self):
        for recipe in self.catalog.runtimes["llamacpp"].recipes.values():
            with self.subTest(recipe=recipe.id):
                launch = llamacpp_launch(recipe.doc)
                self.assertEqual(launch["alias"], "local")
                self.assertEqual((launch["memory"], launch["context"]), (recipe.memory_gib, recipe.context))
                self.assertEqual(launch["args"], list(recipe.args))
                if launch["repo"]:
                    self.assertEqual(launch["repo"], recipe.model.repo)
                if launch["file"]:
                    self.assertEqual(launch["file"], recipe.model.files[0])

    def test_vllm_docs_match(self):
        for recipe in self.catalog.runtimes["vllm"].recipes.values():
            with self.subTest(recipe=recipe.id):
                launch = vllm_launch(recipe.doc)
                self.assertEqual((launch["repo"], launch["served"]), (recipe.model.repo, recipe.model.served_name))
                self.assertEqual((launch["memory"], launch["context"]), (recipe.memory_gib, recipe.context))
                self.assertEqual(launch["args"], list(recipe.args))

    def test_opencode_config_matches(self):
        providers = opencode_providers()
        for name, entry in self.catalog.runtimes.items():
            models = providers[entry.runtime.opencode_provider]["models"]
            for recipe in entry.recipes.values():
                with self.subTest(runtime=name, recipe=recipe.id):
                    model = models.get(recipe.id)
                    if not recipe.opencode:
                        self.assertIsNone(model, "the recipe says opencode = false but the config has an entry")
                        continue
                    self.assertIsNotNone(model, "the recipe says opencode = true but the config has no entry")
                    self.assertEqual(model["modelID"], ADAPTERS[name].served_name(recipe, entry.runtime))
                    self.assertEqual(model["limit"]["context"], recipe.context)
            started = {recipe.id for recipe in entry.recipes.values() if recipe.opencode}
            self.assertEqual(set(models), started, f"{entry.runtime.opencode_provider} lists a model without a recipe")


if __name__ == "__main__":
    unittest.main()
