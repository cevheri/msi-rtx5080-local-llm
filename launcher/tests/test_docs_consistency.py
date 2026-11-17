"""Every shipped recipe must match the launch its document shows, and the OpenCode config in docs/opencode.md.

When one of these fails, fix whichever side disagrees with the measurement record; never loosen the test.
"""

import json
import re
import shlex
import unittest

from launcher import dsh_config, opencode_config
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


def heading_slugs(path) -> set[str]:
    """The anchors of every Markdown heading in a file, outside fenced blocks."""
    slugs, inside = set(), False
    for line in path.read_text().splitlines():
        if line.startswith("```"):
            inside = not inside
            continue
        heading = None if inside else re.match(r"^#{1,6}\s+(.*?)\s*$", line)
        if heading:
            slugs.add(slug(heading.group(1)))
    return slugs


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


def opencode_snippet(doc: str) -> dict | None:
    """The "OpenCode entry" JSON snippet of a recipe's document section, or None when it has none."""
    lines = section(doc)
    start = next((index for index, line in enumerate(lines) if line.startswith("OpenCode entry")), None)
    if start is None:
        return None
    opening = next(index for index in range(start + 1, len(lines)) if lines[index].startswith("```"))
    if lines[opening] != "```json":
        raise AssertionError(f"{doc}: the block after the OpenCode entry line is not ```json")
    closing = next(index for index in range(opening + 1, len(lines)) if lines[index].startswith("```"))
    snippet = json.loads("{" + "\n".join(lines[opening + 1:closing]) + "}")
    if len(snippet) != 1:
        raise AssertionError(f"{doc}: the OpenCode entry snippet holds {len(snippet)} keys, not one")
    return next(iter(snippet.values()))


def fenced_blocks(lines: list[str]) -> list[str]:
    """The text of every fenced block in the lines, each ending with a newline."""
    blocks, current, inside = [], [], False
    for line in lines:
        if line.startswith("```"):
            if inside:
                blocks.append("".join(f"{text}\n" for text in current))
                current = []
            inside = not inside
            continue
        if inside:
            current.append(line)
    return blocks


# The first cells of the dsh rows in docs/llmctl.md "When up refuses" (dsh design, Failure modes).
DSH_REFUSALS = (
    "`DSH_HOME is relative`",
    "`... line N: the llmctl block holds a line llmctl did not write`",
    "`... line N: dsh reads this file as one YAML list`",
    "`... line N: the llm-pi-ai row is set outside the llmctl block`",
    "`... line N: the ... row is set outside the llmctl block while llmctl holds or sets dsh's default`",
    "`dsh's profile patches set the llm-pi-ai row`",
    "`dsh cannot send ... per level`",
    "`dsh --version failed` or `dsh --dump-config failed`",
    "`dsh rejects the home patch llmctl would write for ...`",
    "`dsh ... has no ... row, so llmctl's row would not apply`",
    "`dsh did not apply llmctl's ... row`",
    "`cannot check ... with dsh`",
    "`of the new home patch failed`",
)


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

    def test_opencode_entries_match_client_metadata(self):
        providers = opencode_providers()
        for name, entry in self.catalog.runtimes.items():
            models = providers[entry.runtime.opencode_provider]["models"]
            for recipe in entry.recipes.values():
                with self.subTest(runtime=name, recipe=recipe.id):
                    documented = []
                    if recipe.opencode:
                        self.assertIn(recipe.id, models, "the recipe says opencode = true but the config has no entry")
                        documented.append(models[recipe.id])
                    snippet = opencode_snippet(recipe.doc)
                    if snippet is not None:
                        documented.append(snippet)
                    self.assertTrue(documented, "no documented OpenCode entry")
                    expected = opencode_config.expected_entry(recipe, ADAPTERS[name].served_name(recipe, entry.runtime))
                    for model in documented:
                        self.assertEqual({key: value for key, value in model.items() if key != "name"}, expected)

    def test_every_opencode_entry_has_a_recipe(self):
        providers = opencode_providers()
        for entry in self.catalog.runtimes.values():
            models = providers[entry.runtime.opencode_provider]["models"]
            started = {recipe.id for recipe in entry.recipes.values() if recipe.opencode}
            self.assertEqual(set(models), started, f"{entry.runtime.opencode_provider} lists a model without a recipe")

    def test_measured_recipes_have_opencode_entries(self):
        # Lead review L8 item 4: every measured recipe gets its entry in the validated config.
        for name, entry in self.catalog.runtimes.items():
            for recipe in entry.recipes.values():
                if recipe.evidence.kind != "measured":
                    continue
                with self.subTest(runtime=name, recipe=recipe.id):
                    self.assertTrue(recipe.opencode, "a measured recipe needs opencode = true and its entry")

    def test_clients_doc_examples_match_golden(self):
        lines = (REPO / "docs/clients.md").read_text().splitlines()
        named = []
        for index, line in enumerate(lines):
            match = re.search(r"`launcher/tests/golden/clients/((?:llamacpp|vllm)/[^`]+)`", line)
            if match is None:
                continue
            name = match.group(1)
            named.append(name)
            with self.subTest(golden=name):
                opening = index + 1
                while opening < len(lines) and not lines[opening].strip():
                    opening += 1
                self.assertTrue(opening < len(lines) and lines[opening].startswith("```"),
                                f"docs/clients.md line {index + 1}: no fenced block before the next prose line")
                closing = next(i for i in range(opening + 1, len(lines)) if lines[i].startswith("```"))
                shown = "\n".join(lines[opening + 1:closing]) + "\n"
                golden = (REPO / "launcher/tests/golden/clients" / name).read_text()
                # Compared as text, so the document shows the layout llmctl writes, not only the same values.
                self.assertEqual(shown, golden)
        for required in ("llamacpp/qwen3.6-35b-a3b.hermes.yaml", "llamacpp/qwen3.6-35b-a3b.pi.json",
                         "llamacpp/qwen3.6-35b-a3b.dsh.yml", "vllm/gpt-oss-20b.hermes.yaml",
                         "vllm/gpt-oss-20b.pi.json", "vllm/gpt-oss-20b.dsh.yml"):
            self.assertIn(required, named)

    def test_no_tools_block_matches(self):
        blocks = fenced_blocks(section("docs/clients.md#chat-only-recipes"))
        self.assertTrue(blocks, "docs/clients.md#chat-only-recipes has no fenced block")
        # Compared as text, so a reader can copy the block into a --patch file as it is.
        self.assertEqual(blocks[0], dsh_config.no_tools_patch())

    def test_every_dsh_refusal_has_a_row(self):
        first_cells = [line.split("|")[1].strip() for line in section("docs/llmctl.md#when-up-refuses")
                       if line.startswith("|")]
        for start in DSH_REFUSALS:
            with self.subTest(start=start):
                self.assertTrue(any(start in cell for cell in first_cells),
                                f"no row of docs/llmctl.md#when-up-refuses starts with {start}")

    def test_compaction_list_matches_recipes(self):
        text = "\n".join(section("docs/clients.md#context-compaction-and-timeouts"))
        small = [recipe.id for entry in self.catalog.runtimes.values() for recipe in entry.recipes.values()
                 if recipe.context - recipe.clients.output <= dsh_config.COMPACTION_HEADROOM]
        self.assertIn(f"{len(small)} of 19 recipes", text)
        for recipe_id in small:
            with self.subTest(recipe=recipe_id):
                self.assertIn(recipe_id, text)

    def test_doc_anchors_in_messages_resolve(self):
        pattern = re.compile(r"\b((?:docs|llamacpp|vllm)/[A-Za-z0-9_.-]+\.md)#([A-Za-z0-9_-]+)")
        found = 0
        for source in sorted((REPO / "launcher").rglob("*.py")):
            if (REPO / "launcher/tests") in source.parents:
                continue
            for path, anchor in pattern.findall(source.read_text()):
                found += 1
                with self.subTest(source=str(source.relative_to(REPO)), link=f"{path}#{anchor}"):
                    self.assertTrue((REPO / path).is_file(), f"{path} does not exist")
                    self.assertIn(anchor, heading_slugs(REPO / path))
        self.assertGreater(found, 0)


if __name__ == "__main__":
    unittest.main()
