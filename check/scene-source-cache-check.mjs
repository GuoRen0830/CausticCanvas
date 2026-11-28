import assert from 'node:assert/strict'
import { test } from 'node:test'
import { mkdir, mkdtemp, readdir, rm, writeFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { tmpdir } from 'node:os'
import { basename, dirname, join } from 'node:path'
import { sceneSourceCache } from '../scripts/scene-source-cache.mjs'

test('local original packs verify official file identities and reject same-name substitutions', async () => {
  const temp = await mkdtemp(join(tmpdir(), 'caustic-originals-'))
  const api = 'https://api.polyhaven.com/files/probe', url = 'https://dl.polyhaven.org/textures/probe.png'
  const data = Buffer.from('signed reference texture'), signature = createHash('md5').update(data).digest('hex')
  const catalog = { probe: { Rough: { '1k': { png: { url, size: data.length, md5: signature } } } } }
  // A deflated ZIP member named textures/probe.png containing the same bytes.
  const zip = 'UEsDBBQAAAAIAAAAIQDKmhC1GgAAABgAAAASAAAAdGV4dHVyZXMvcHJvYmUucG5nK85Mz0tNUShKTUstSs1LTlUoSa0oKS1KBQBQSwECFAAUAAAACAAAACEAypoQtRoAAAAYAAAAEgAAAAAAAAAAAAAAgAEAAAAAdGV4dHVyZXMvcHJvYmUucG5nUEsFBgAAAAABAAEAQAAAAEoAAAAAAA=='
  const offline = async () => { throw new Error('No network') }
  try {
    const dir = join(temp, 'media'), archive = join(temp, 'download.zip')
    await mkdir(dir)
    await writeFile(join(dir, 'probe.png'), data)
    await writeFile(archive, Buffer.from(zip, 'base64'))
    for (const [i, sourcePath] of [dir, archive].entries()) {
      const cache = sceneSourceCache(join(temp, 'cache' + i), offline, { sourcePath, catalog })
      assert.equal(await cache.hasLocal(url), true)
      assert.deepEqual(await (await cache.fetch(api)).json(), catalog.probe)
      const response = await cache.fetch(url)
      assert.equal(response.headers.get('Content-Type'), 'image/png')
      assert.deepEqual(Buffer.from(await response.arrayBuffer()), data)
      assert.equal(cache.counts.downloaded, 0)
    }
    const installed = join(temp, 'installed')
    await mkdir(installed)
    await writeFile(join(installed, signature + '.png'), data)
    const restored = sceneSourceCache(join(temp, 'persisted'), offline, { originalsDirectory: installed, sourcePath: '', catalog })
    assert.equal(await restored.hasLocal(url), true)
    assert.deepEqual(Buffer.from(await (await restored.fetch(url)).arrayBuffer()), data)
    assert.equal(restored.counts.downloaded, 0)
    await writeFile(join(dir, 'probe.png'), Buffer.alloc(data.length))
    await assert.rejects(sceneSourceCache(join(temp, 'reject'), offline, { sourcePath: dir, catalog }).fetch(url), /No network/)
    await assert.rejects(sceneSourceCache(join(temp, 'corrupt'), async () => new Response('wrong'), { catalog }).fetch(url), /checksum mismatch/)
  } finally {
    if (dirname(temp) === tmpdir() && basename(temp).startsWith('caustic-originals-')) await rm(temp, { recursive: true, force: true })
  }
})

test('an interrupted scene download can be retried and replayed offline with its MIME type and URL', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'caustic-sources-'))
  const url = 'https://api.polyhaven.com/files/treasure_chest'
  try {
    const broken = sceneSourceCache(directory, async () => new Response(new ReadableStream({
      start(controller) { controller.error(new Error('interrupted transfer')) },
    })))
    await assert.rejects(broken.fetch(url), /interrupted transfer/)
    assert.deepEqual(await readdir(directory), [])
    const live = sceneSourceCache(directory, async () => Response.json({ files: ['chest.gltf'] }))
    assert.deepEqual(await (await live.fetch(url)).json(), { files: ['chest.gltf'] })
    const offline = sceneSourceCache(directory, async () => { throw new Error('No network') })
    const cached = await offline.fetch(url)
    assert.equal(cached.url, url)
    assert.ok(cached.headers.get('Content-Type').startsWith('application/json'))
    assert.deepEqual(await cached.json(), { files: ['chest.gltf'] })
    assert.deepEqual(offline.counts, { downloaded: 0, reused: 1 })
    await assert.rejects(offline.fetch(url, { headers: { Range: 'bytes=0-3' } }), /No network/)
  } finally {
    if (dirname(directory) === tmpdir() && basename(directory).startsWith('caustic-sources-')) {
      await rm(directory, { recursive: true, force: true })
    }
  }
})
