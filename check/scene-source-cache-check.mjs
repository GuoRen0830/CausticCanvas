import assert from 'node:assert/strict'
import { test } from 'node:test'
import { mkdtemp, readdir, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { basename, dirname, join } from 'node:path'
import { sceneSourceCache } from '../scripts/scene-source-cache.mjs'

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
