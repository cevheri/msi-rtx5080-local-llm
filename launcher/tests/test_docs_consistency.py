"""Every shipped recipe must match the launch its document shows, and the OpenCode config in docs/opencode.md.

When one of these fails, fix whichever side disagrees with the measurement record; never loosen the test.
"""

import json
import re
import shlex
import unittest

from launcher import dsh_config, kimi_config, markdown, opencode_config
from launcher.catalog import load_catalog
from launcher.hfcache import repo_dirname
from launcher.markdown import heading_slugs
from launcher.runtimes import ADAPTERS
from launcher.tests.fakes import REPO


def section(doc: str) -> list[str]:
    try:
        return markdown.section(REPO, doc)
    except LookupError as exc:
        raise AssertionError(str(exc)) from None


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
    image = next(i for i, t in enumerate(tokens) if t.startswith("vllm/vllm-openai:"))
    server = tokens[image + 1:]
    mounts = [tokens[i + 1] for i in range(image)
              if tokens[i] == "-v" and tokens[i + 1].endswith(":/llmctl/tokenizer:ro")]
    return {"memory": int(tokens[tokens.index("--memory") + 1].rstrip("g")), "repo": server.pop(0),
            "served": take(server, "--served-model-name"), "context": int(take(server, "--max-model-len")),
            "tokenizer": take(server, "--tokenizer"), "tokenizer_mount": mounts[0] if mounts else None,
            "args": server}


def sglang_launch(doc: str) -> dict:
    tokens = last_run_block(section(doc))
    server = tokens[next(i for i, t in enumerate(tokens) if t.startswith("lmsysorg/sglang:")) + 1:]
    if server[:3] != ["python3", "-m", "sglang.launch_server"]:
        raise AssertionError(f"{doc}: the image is not followed by python3 -m sglang.launch_server")
    del server[:3]
    return {"memory": int(tokens[tokens.index("--memory") + 1].rstrip("g")),
            "model_path": take(server, "--model-path"), "served": take(server, "--served-model-name"),
            "context": int(take(server, "--context-length")), "host": take(server, "--host"),
            "port": int(take(server, "--port")), "args": server}


def opencode_providers() -> dict:
    text = (REPO / "docs/opencode.md").read_text()
    block = re.search(r"```json\n(.*?)\n```", text[text.index("## The validated config"):], re.S).group(1)
    return json.loads(block)["providers"]


def opencode_cannot_talk_to(recipe, runtime) -> bool:
    """OpenCode always sends tool_choice auto, which a runtime that refuses tools without a parser answers with HTTP 400."""
    return recipe.chat_only and runtime.refuses_tools_without_parser


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

# The first cells of the Kimi rows in docs/llmctl.md "When up refuses" (Kimi design, Failure modes).
KIMI_REFUSALS = (
    "`KIMI_CODE_HOME is set but empty`",
    "`KIMI_CODE_HOME is relative`",
    "`... is not valid TOML`",
    "`...: \"providers\" is not a table` or `...: \"models\" is not a table`",
    "`... line N: providers is written as an inline table or with dotted keys`",
    "`...: llmctl's provider is written with dotted keys or as an inline table`",
    "`could not change the top-level default_model in ... safely`",
    "`kimi --version failed` or `kimi provider list --json failed`",
    "`Kimi ... does not read llmctl's entry for ...`",
    "`cannot check ... with kimi`",
    "`...: the file changed while llmctl wrote it, probably through Kimi`",
    "`of the new config.toml failed`",
)

# The first cells of the SGLang rows in docs/llmctl.md "When up refuses" (SGLang design, Failure modes; plan ruling 6).
SGLANG_ROWS = (
    "`llmctl's cache folder ... is not an absolute path`",
    "`cannot write the model overlay`",
    "`generation_config.json in snapshot ... is not a JSON object`",
    "`the KV pool holds ... tokens, fewer than the context`",
    "`only ... MiB of GPU memory is free after the start`",
    "`could not read internal_states[0].memory_usage.token_capacity`",
    "`POST http://127.0.0.1:30000/flush_cache answered`",
)

# The first cells of the vLLM tokenizer overlay rows in docs/llmctl.md "When up refuses" (plan rulings 36 and 37).
VLLM_TOKENIZER_ROWS = (
    "`cannot write the tokenizer overlay`",
    "`tokenizer_config.json in snapshot ... is missing`",
    "`tokenizer_config.json in snapshot ... is not a JSON object`",
)

# Ruling 11 of the Kimi plan: the documents cite only claims archived with a verdict that holds.
KIMI_CLAIMS = REPO / "research/2026-11-17/llmctl-kimi-stub/research-claims.json"
KIMI_CLAIM_DOCS = ("docs/clients.md", "docs/llmctl.md", "docs/sources.md", "README.md")


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
                self.assertEqual(launch["tokenizer"], "/llmctl/tokenizer" if recipe.model.tokenizer_class else None)
                self.assertEqual(launch["tokenizer_mount"] is not None, recipe.model.tokenizer_class is not None)

    def test_sglang_docs_match(self):
        for recipe in self.catalog.runtimes["sglang"].recipes.values():
            with self.subTest(recipe=recipe.id):
                launch = sglang_launch(recipe.doc)
                self.assertEqual((launch["memory"], launch["context"]), (recipe.memory_gib, recipe.context))
                self.assertEqual(launch["served"], recipe.model.served_name)
                self.assertEqual(launch["args"], list(recipe.args))
                self.assertEqual((launch["host"], launch["port"]), ("0.0.0.0", 30000))
                self.assertEqual(launch["model_path"], f"/root/.cache/huggingface/hub/{repo_dirname(recipe.model.repo)}"
                                                       f"/snapshots/{recipe.model.revision}")

    def test_sglang_sampling_lines_match(self):
        for recipe in self.catalog.runtimes["sglang"].recipes.values():
            with self.subTest(recipe=recipe.id):
                lines = [line for line in section(recipe.doc) if line.startswith("Sampling: llmctl pins")]
                self.assertEqual(len(lines), 1, "no single line starting 'Sampling: llmctl pins'")
                shown = json.loads(re.search(r"`([^`]+)`", lines[0]).group(1))
                self.assertEqual(shown, dict(recipe.model.sampling))

    def test_sglang_recipes_follow_their_twins(self):
        llamacpp_flags = {"--temp": "temperature", "--top-p": "top_p", "--top-k": "top_k", "--min-p": "min_p",
                          "--repeat-penalty": "repetition_penalty"}
        for recipe in self.catalog.runtimes["sglang"].recipes.values():
            with self.subTest(recipe=recipe.id):
                self.assertEqual(recipe.evidence.kind, "measured")
                self.assertTrue((REPO / recipe.evidence.record).is_dir(), recipe.evidence.record)
                # Plan ruling 18: the vLLM recipe of the same ID, else the llama.cpp one.
                twin = (self.catalog.runtimes["vllm"].recipes.get(recipe.id)
                        or self.catalog.runtimes["llamacpp"].recipes.get(recipe.id))
                self.assertIsNotNone(twin, "no vLLM or llama.cpp recipe of the same ID")
                if (twin.model.repo, getattr(twin.model, "revision", None)) == (recipe.model.repo,
                                                                                 recipe.model.revision):
                    for key in ("include", "exclude", "size_bytes"):
                        self.assertEqual(getattr(recipe.model, key), getattr(twin.model, key), key)
                args = list(twin.args)
                if twin.runtime == "vllm":
                    index = args.index("--override-generation-config")
                    pinned = {key: value for key, value in json.loads(args[index + 1]).items()
                              if key in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty")}
                else:
                    pinned = {name: float(args[args.index(flag) + 1]) for flag, name in llamacpp_flags.items()
                              if flag in args}
                self.assertEqual(dict(recipe.model.sampling), pinned)

    def test_opencode_providers_include_sglang(self):
        provider = opencode_providers()["sglang-docker"]
        self.assertEqual(provider["settings"]["baseURL"], "http://127.0.0.1:30000/v1")
        self.assertEqual(provider["compatibility"]["reasoningField"], "reasoning_content")

    def test_opencode_entries_match_client_metadata(self):
        providers = opencode_providers()
        for name, entry in self.catalog.runtimes.items():
            models = providers[entry.runtime.opencode_provider]["models"]
            for recipe in entry.recipes.values():
                if opencode_cannot_talk_to(recipe, entry.runtime):
                    continue  # test_chat_only_recipes_on_refusing_runtimes_have_no_opencode_entry covers it
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
                if recipe.evidence.kind != "measured" or opencode_cannot_talk_to(recipe, entry.runtime):
                    continue
                with self.subTest(runtime=name, recipe=recipe.id):
                    self.assertTrue(recipe.opencode, "a measured recipe needs opencode = true and its entry")

    def test_chat_only_recipes_on_refusing_runtimes_have_no_opencode_entry(self):
        # Lead ruling after the e2e of 2026-11-18 (record 411): OpenCode v2.3.0 sends tool_choice auto even with
        # tools false, and a runtime that refuses tools without a parser answers HTTP 400, so such a recipe has no entry.
        providers = opencode_providers()
        refused = set()
        for name, entry in self.catalog.runtimes.items():
            models = providers[entry.runtime.opencode_provider]["models"]
            for recipe in entry.recipes.values():
                if not opencode_cannot_talk_to(recipe, entry.runtime):
                    continue
                refused.add((name, recipe.id))
                with self.subTest(runtime=name, recipe=recipe.id):
                    self.assertFalse(recipe.opencode, "OpenCode cannot talk to this recipe, so opencode must be false")
                    self.assertNotIn(recipe.id, models, "docs/opencode.md must not list this recipe")
                    self.assertIsNone(opencode_snippet(recipe.doc), "the recipe's section must not show an OpenCode entry")
        self.assertEqual(refused, {("vllm", "deepseek-r1-0528-qwen3-8b")})

    def test_clients_doc_examples_match_golden(self):
        lines = (REPO / "docs/clients.md").read_text().splitlines()
        named = []
        for index, line in enumerate(lines):
            match = re.search(r"`launcher/tests/golden/clients/((?:llamacpp|vllm|sglang)/[^`]+)`", line)
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
                         "llamacpp/qwen3.6-35b-a3b.dsh.yml", "llamacpp/qwen3.6-35b-a3b.kimi.toml",
                         "vllm/gpt-oss-20b.hermes.yaml", "vllm/gpt-oss-20b.pi.json", "vllm/gpt-oss-20b.dsh.yml",
                         "vllm/gpt-oss-20b.kimi.toml"):
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

    def test_kimi_agent_file_block_matches(self):
        blocks = fenced_blocks(section("docs/clients.md#kimi-chat-only-recipes"))
        self.assertTrue(blocks, "docs/clients.md#kimi-chat-only-recipes has no fenced block")
        # Compared as text, so a reader can save the block as the agent file that llmctl ships and up prints.
        self.assertEqual(blocks[0], kimi_config.AGENT_FILE.read_text(encoding="utf-8"))

    def test_every_kimi_refusal_has_a_row(self):
        first_cells = [line.split("|")[1].strip() for line in section("docs/llmctl.md#when-up-refuses")
                       if line.startswith("|")]
        for start in KIMI_REFUSALS:
            with self.subTest(start=start):
                self.assertTrue(any(start in cell for cell in first_cells),
                                f"no row of docs/llmctl.md#when-up-refuses starts with {start}")

    def test_every_vllm_tokenizer_refusal_has_a_row(self):
        first_cells = [line.split("|")[1].strip() for line in section("docs/llmctl.md#when-up-refuses")
                       if line.startswith("|")]
        for start in VLLM_TOKENIZER_ROWS:
            with self.subTest(start=start):
                self.assertTrue(any(start in cell for cell in first_cells),
                                f"no row of docs/llmctl.md#when-up-refuses starts with {start}")

    def test_every_sglang_refusal_has_a_row(self):
        first_cells = [line.split("|")[1].strip() for line in section("docs/llmctl.md#when-up-refuses")
                       if line.startswith("|")]
        for start in SGLANG_ROWS:
            with self.subTest(start=start):
                self.assertTrue(any(start in cell for cell in first_cells),
                                f"no row of docs/llmctl.md#when-up-refuses starts with {start}")

    def test_kimi_claim_ids_resolve(self):
        verdicts = {claim["id"]: claim["verdict"] for claim in json.loads(KIMI_CLAIMS.read_text())}
        found = 0
        for doc in KIMI_CLAIM_DOCS:
            for claim_id in re.findall(r"\b(docs-K\d{3}|local-K\d{2})\b", (REPO / doc).read_text()):
                found += 1
                with self.subTest(doc=doc, claim=claim_id):
                    self.assertIn(claim_id, verdicts, "the claim is not in the archive's research-claims.json")
                    self.assertTrue(verdicts[claim_id].startswith("Holds"), verdicts[claim_id])
        self.assertGreater(found, 0)

    def test_compaction_list_matches_recipes(self):
        text = "\n".join(section("docs/clients.md#context-compaction-and-timeouts"))
        small = [recipe.id for entry in self.catalog.runtimes.values() for recipe in entry.recipes.values()
                 if recipe.context - recipe.clients.output <= dsh_config.COMPACTION_HEADROOM]
        total = sum(len(entry.recipes) for entry in self.catalog.runtimes.values())
        self.assertIn(f"{len(small)} of {total} recipes", text)
        for recipe_id in small:
            with self.subTest(recipe=recipe_id):
                self.assertIn(recipe_id, text)

    def test_readme_recipe_table_matches_the_catalog(self):
        header = "| Model ID | Aliases | " + " | ".join(f"{adapter.LABEL} | Context | Decode tok/s"
                                                     for adapter in ADAPTERS.values()) + " |"
        width = 2 + 3 * len(ADAPTERS)
        lines = section("README.md#quick-start")
        self.assertIn(header, lines, "README.md#quick-start has no recipe table")
        start = lines.index(header) + 2
        rows = {}
        for line in lines[start:]:
            if not line.startswith("|"):
                break
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            self.assertEqual(len(cells), width, line)
            recipe_id = cells[0].strip("`")
            self.assertNotIn(recipe_id, rows, f"{recipe_id} has more than one row")
            rows[recipe_id] = cells
        shipped = {recipe_id for entry in self.catalog.runtimes.values() for recipe_id in entry.recipes}
        self.assertEqual(set(rows), shipped)
        for recipe_id, cells in rows.items():
            with self.subTest(recipe=recipe_id):
                aliases = []
                for column, name in ((2 + 3 * index, name) for index, name in enumerate(ADAPTERS)):
                    recipe = self.catalog.runtimes[name].recipes.get(recipe_id)
                    if recipe is None:
                        self.assertEqual(cells[column:column + 3], ["No", "-", "-"], name)
                        continue
                    self.assertEqual(recipe.context % 1024, 0, f"{name}: context {recipe.context} is not whole k")
                    speed = f"{recipe.evidence.decode_tok_s:g}" if recipe.evidence.decode_tok_s else "-"
                    self.assertEqual(cells[column:column + 3], ["Yes", f"{recipe.context // 1024}k", speed], name)
                    aliases += [alias for alias in recipe.aliases if alias not in aliases]
                self.assertEqual(cells[1], ", ".join(f"`{alias}`" for alias in aliases) or "-")
        prose = "\n".join(line for line in lines if not line.startswith("|"))
        for name, entry in self.catalog.runtimes.items():
            for recipe_id, recipe in entry.recipes.items():
                if not recipe.evidence.decode_tok_s:
                    with self.subTest(unmeasured=f"{recipe_id} on {name}"):
                        self.assertIn(f"`{recipe_id}`", prose, "an unmeasured recipe needs a note outside the table")
        from launcher.site import render  # inside the test, so only this test errors while the module is missing
        text = (REPO / "README.md").read_text(encoding="utf-8")
        self.assertEqual(render.readme_block(text), "\n" + render.readme_table(self.catalog) + "\n",
                         "README.md's recipe table is stale: run python3 -m launcher.site readme")

    def test_runtime_coverage_matches_the_catalog(self):
        # D17 of the SGLang design: every model of the README table has one cell per engine, and a cell starts Runs:
        # exactly when a measured recipe ships (ruling 28); an estimate recipe's cell starts Failed:.
        header = "| Model ID | llama.cpp b12407 | vLLM v0.33.0 | SGLang v0.5.23 |"
        lines = section("docs/model-selection.md#runtime-coverage")
        self.assertIn(header, lines, "docs/model-selection.md#runtime-coverage has no coverage table")
        rows = []
        for line in lines[lines.index(header) + 2:]:
            if not line.startswith("|"):
                break
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            self.assertEqual(len(cells), 4, line)
            rows.append(cells)
        readme = section("README.md#quick-start")
        table = next(index for index, line in enumerate(readme) if line.startswith("| Model ID | Aliases |"))
        readme_ids = []
        for line in readme[table + 2:]:
            if not line.startswith("|"):
                break
            readme_ids.append(line.strip("|").split("|")[0].strip().strip("`"))
        self.assertEqual([cells[0].strip("`") for cells in rows], readme_ids)
        link = re.compile(r"\]\(\.\./([^)#\s]+)(?:#[^)\s]*)?\)")
        dated = re.compile(r"measured (\d{4}-\d{2}-\d{2})")
        for cells in rows:
            for column, name in ((1, "llamacpp"), (2, "vllm"), (3, "sglang")):
                cell, recipe = cells[column], self.catalog.runtimes[name].recipes.get(cells[0].strip("`"))
                with self.subTest(model=cells[0], runtime=name):
                    starts = [word for word in ("Runs:", "Failed:", "Does not fit:") if cell.startswith(word)]
                    self.assertEqual(len(starts), 1, cell)
                    for target in link.findall(cell):
                        self.assertTrue((REPO / target).exists(), f"{target} does not exist")
                    measured = recipe is not None and recipe.evidence.kind == "measured"
                    self.assertEqual(starts[0] == "Runs:", measured, cell)
                    if measured:
                        speed = f"{recipe.evidence.decode_tok_s:g} tok/s"
                        for text in (f"{recipe.context // 1024}k", speed, f"measured {recipe.evidence.date}",
                                     f"](../{recipe.doc})"):
                            self.assertIn(text, cell)
                        e2e = re.search(r"e2e (\d{4}-\d{2}-\d{2}):", cell)
                        self.assertIsNotNone(e2e, "a Runs: cell names its e2e date")
                        self.assertRegex(cell, rf"\]\(\.\./research/{e2e.group(1)}/[^)/\s]+/summary\.md\)")
                        if name == "vllm" and recipe.model.tokenizer_class:
                            self.assertGreater(cell.find("tokenizer overlay"), cell.find(speed), cell)
                    elif recipe is not None:
                        record = re.search(r"research/(\d{4}-\d{2}-\d{2})/[^/\s]+/", recipe.evidence.summary)
                        self.assertIsNotNone(record, "an estimate recipe's summary names its failure record")
                        self.assertEqual(starts[0], "Failed:", cell)
                        self.assertIn(f"measured {record.group(1)}", cell)
                        self.assertIn(f"](../{record.group(0)}", cell)
                    if starts[0] == "Failed:":
                        date = dated.search(cell)
                        self.assertIsNotNone(date, "a Failed: cell names the date of its measurement")
                        self.assertRegex(cell, rf"\]\(\.\./research/{date.group(1)}/[^)/\s]+/")
                    if starts[0] == "Does not fit:":
                        self.assertIn("(estimate)", cell)

    def test_doc_anchors_in_messages_resolve(self):
        pattern = re.compile(r"\b((?:docs|llamacpp|vllm|sglang)/[A-Za-z0-9_.-]+\.md)#([A-Za-z0-9_-]+)")
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
