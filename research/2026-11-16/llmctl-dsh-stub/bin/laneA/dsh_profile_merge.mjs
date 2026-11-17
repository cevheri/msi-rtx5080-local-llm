// Lane A feasibility probe: merge providers.llmctl (and optionally the default) into a Web-UI-written
// dsh profile patch with dsh's own `yaml` library and file lock, the way dsh-config-editor edits it.
// usage: node dsh_profile_merge.mjs <dsh package.json> <profile dir> <provider-json> [default-json]
import { createRequire } from 'node:module'
import { readFile } from 'node:fs/promises'
import { join } from 'node:path'
const [dshPkg, profileDir, providerJson, defaultJson] = process.argv.slice(2)
const req = createRequire(dshPkg)
const { parseDocument, isMap, isSeq } = req('yaml')
const { withFileLock, writeFileAtomic } = await import(req.resolve('@deepseek-ai/dsh-atomic-write'))
const path = join(profileDir, 'cordis.patch.yml')
const lastRow = (doc, id) => doc.contents.items.findLastIndex((item, i) => isMap(item) && doc.getIn([i, 'id']) === id && !item.has('insert'))
await withFileLock(join(profileDir, 'package.json'), async () => {
  const before = await readFile(path, 'utf8')
  const doc = parseDocument(before, { customTags: [{ tag: 'tag:yaml.org,2002:js', resolve: v => v }] })
  if (doc.errors[0] !== undefined) throw doc.errors[0]
  if (!isSeq(doc.contents)) throw new Error(`${path}: not a YAML sequence`)
  doc.contents.flow = false
  let i = lastRow(doc, 'llm-pi-ai')
  if (i < 0) doc.add(doc.createNode({ id: 'llm-pi-ai', config: { providers: { llmctl: JSON.parse(providerJson) } } }))
  else doc.setIn([i, 'config', 'providers', 'llmctl'], doc.createNode(JSON.parse(providerJson)))
  if (defaultJson) {
    const j = lastRow(doc, 'agent-default-model')
    if (j < 0) doc.add(doc.createNode({ id: 'agent-default-model', config: JSON.parse(defaultJson) }))
    else doc.setIn([j, 'config'], doc.createNode(JSON.parse(defaultJson)))
  }
  await writeFileAtomic(path, String(doc), { mode: 0o600 })
})
console.log('merged into', path)
