import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { test } from 'node:test'
import { restoreMaterialOriginals } from '../scripts/restore-material-originals.mjs'
import { sceneSourceCache } from '../scripts/scene-source-cache.mjs'

test('restoration deduplicates originals, identifies missing upgrades, and resumes without changing the scene', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'caustic-restore-'))
  const preview = join(dir, 'preview')
  const bytes = new Map()
  function ref(name, upgrade) {
    const data = Buffer.from('verified ' + name)
    const url = 'https://dl.polyhaven.org/model/' + name + '.png'
    bytes.set(url, data)
    return { url, bytes: data.length, md5: createHash('md5').update(data).digest('hex'), upgrade }
  }
  const orm = ref('orm'), normal = ref('normal'), color = ref('color', true)
  const refs = [orm, normal, color]
  const catalog = refs.map(source => ({ ...source, size: source.bytes }))
  const calls = []
  let online = false
  const sources = sceneSourceCache(join(dir, 'cache'), async url => {
    calls.push(url)
    return new Response(online || url === orm.url ? bytes.get(url) : 'wrong bytes')
  }, { catalog, sourcePath: '' })
  try {
    await mkdir(preview)
    await writeFile(join(preview, 'scene.obj'), 'keep geometry')
    await writeFile(join(preview, 'source-textures.json'), JSON.stringify({ version: 1, materials: {
      rock: { metallicRoughnessMap: orm, occlusionMap: { ...orm, upgrade: true }, normalMap: normal, map: color },
      pier: { normalMap: { ...normal, upgrade: true } },
    } }))
    const restore = () => restoreMaterialOriginals(preview, sources, () => {})
    assert.deepEqual(await restore(), { total: 3, installed: 1, available: 1, missing: 1, missingUpgrades: 1 })
    assert.deepEqual(await readdir(join(preview, 'originals')), [orm.md5 + '.png'])
    assert.equal(calls.length, 3)
    online = true
    await writeFile(join(dir, '.install-preview.lock'), 'test lock')
    await assert.rejects(restore(), /Another process/)
    assert.deepEqual(await readdir(join(preview, 'originals')), [orm.md5 + '.png'])
    await rm(join(dir, '.install-preview.lock'))
    const before = calls.length
    assert.deepEqual(await restore(), { total: 3, installed: 2, available: 3, missing: 0, missingUpgrades: 0 })
    assert.equal(calls.length, before)
    for (const source of refs) assert.deepEqual(await readFile(join(preview, 'originals', source.md5 + '.png')), bytes.get(source.url))
    assert.equal((await restore()).installed, 0)
    assert.equal(await readFile(join(preview, 'scene.obj'), 'utf8'), 'keep geometry')
    assert.deepEqual((await readdir(dir)).sort(), ['cache', 'preview'])
  } finally { await rm(dir, { recursive: true, force: true }) }
})
