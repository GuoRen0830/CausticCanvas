import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { test } from 'node:test'
import { createServer } from 'vite'

test('verified sources replace only matching baked materials and persist across offline scene loads', async () => {
  const site = await createServer({ configFile: false, server: { middlewareMode: true, hmr: false, watch: null } })
  const fetchBefore = globalThis.fetch
  const cacheBefore = Object.getOwnPropertyDescriptor(globalThis, 'caches')
  const storage = new Map()
  const cache = {
    match: async key => storage.get(key)?.clone(),
    put: async (key, response) => storage.set(key, response.clone()),
  }
  function installCache(open) {
    Object.defineProperty(globalThis, 'caches', { configurable: true, value: { open } })
  }
  try {
    installCache(async () => cache)
    const { loadCoveSourceTextures } = await site.ssrLoadModule('/src/geometry/cove-source-textures.ts')
    // These synthetic signatures test the identity verifier independently of the publisher.
    const image = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aOF0AAAAASUVORK5CYII=', 'base64')
    const sources = new Map()
    function original(name, bytes) {
      const url = 'https://dl.polyhaven.org/file/ph-assets/' + name
      sources.set(url, bytes)
      return { url, md5: createHash('md5').update(bytes).digest('hex'), bytes: bytes.length }
    }
    const arm = original('source-arm.png', image)
    const normal = original('source-normal.png', Buffer.concat([image, Buffer.alloc(131_132)]))
    const bakedAlbedo = new File(['local'], 'base.png')
    const bakedNormal = new File(['local'], 'pier-planks-normal.png')
    const bakedRoughness = new File(['local'], 'pier-planks-roughness.png')
    const fallback = new Map([
      ['pier', { color: [1, 1, 1], map: bakedAlbedo, normalMap: bakedNormal, normalYSign: 1,
        roughness: 0.78, metallic: 0, roughnessMap: bakedRoughness }],
      ['wetPier', { color: [0.9, 0.9, 0.9], roughness: 0.4 }],
      ['no-uv', { color: [0.6, 0.6, 0.6] }],
    ])
    const manifest = { version: 1, materials: {
      pier: { metallicRoughnessMap: { ...arm, roughness: 0.7, metallic: 0.8 },
        occlusionMap: { ...arm, occlusionStrength: 0.9 },
        normalMap: { ...normal, normalScale: 0.75, normalYSign: -1, clamp: true,
          uvTransform: [1, 2, 0.1, 0], sampling: { uvSet: 1 } } },
      wetPier: { metallicRoughnessMap: { ...arm, roughness: 0.33 } },
      obsolete: { normalMap: normal },
    } }
    let corrupt = true
    const fetched = []
    globalThis.fetch = async (url, options) => {
      assert.equal(options.credentials, 'omit')
      fetched.push(url)
      const data = Buffer.from(sources.get(url))
      if (corrupt && url === normal.url) data[data.length - 1] ^= 1
      return new Response(data)
    }
    const loaded = await loadCoveSourceTextures(fallback, manifest)
    assert.equal(loaded.sourceTextures.missing, 2)
    assert.equal(loaded.materials.get('pier'), fallback.get('pier'))
    assert.equal(fetched.length, 0)
    const state = loaded.sourceTextures
    const [partial, concurrent] = await Promise.all([state.restore(fallback), state.restore(fallback)])
    assert.deepEqual([...fetched].sort(), [arm.url, normal.url].sort())
    assert.equal(partial.restored, 1)
    assert.equal(concurrent.missing, 1)
    assert.match(partial.warnings.join(' '), /source-normal\.png: File checksum mismatch/)
    const pier = partial.materials.get('pier')
    assert.equal(pier.roughness, 0.7)
    assert.equal(pier.metallic, 0.8)
    assert.equal(pier.occlusionStrength, 0.9)
    assert.equal(pier.occlusionMap, pier.metallicRoughnessMap)
    assert.equal(pier.occlusionMap, partial.materials.get('wetPier').metallicRoughnessMap)
    assert.equal(partial.materials.get('wetPier').roughness, 0.33)
    assert.equal(pier.map, bakedAlbedo)
    assert.equal(pier.normalMap, bakedNormal)
    assert.equal(pier.roughnessMap, undefined, 'A verified packed map replaces the local standalone roughness')
    assert.equal(partial.materials.get('no-uv'), fallback.get('no-uv'))
    assert.equal(fallback.get('pier').roughness, 0.78)
    assert.equal(fallback.get('pier').occlusionMap, undefined)
    corrupt = false
    const complete = await state.restore(partial.materials)
    assert.equal(complete.missing, 0)
    assert.equal(complete.restored, 1)
    assert.deepEqual(complete.warnings, [])
    assert.equal(fetched.at(-1), normal.url)
    const detailed = complete.materials.get('pier')
    assert.equal(detailed.normalMap.type, 'image/png')
    assert.equal(detailed.normalScale, 0.75)
    assert.equal(detailed.normalYSign, -1)
    assert.deepEqual(detailed.normalUVTransform, [1, 2, 0.1, 0])
    assert.deepEqual(detailed.sampling.normalMap, { uvSet: 1 })
    assert.equal(detailed.normalClamp, true)
    const count = fetched.length
    // A canceled GPU upload can retry from its last published (fallback) material set.
    const republished = await state.restore(fallback)
    assert.equal(republished.restored, 0)
    assert.equal(republished.materials.get('pier').normalMap, detailed.normalMap)
    assert.equal(fetched.length, count)
    const offline = await loadCoveSourceTextures(fallback, manifest)
    assert.equal(offline.sourceTextures.missing, 0)
    assert.equal(offline.materials.get('pier').normalMap.name, 'source-normal.png')
    assert.equal(fetched.length, count)
    for (const key of storage.keys()) if (key.includes(arm.md5)) storage.set(key, new Response('bad cache'))
    const damaged = await loadCoveSourceTextures(fallback, manifest)
    assert.equal(damaged.sourceTextures.missing, 1)
    assert.equal(damaged.materials.get('pier').metallic, 0)
    assert.equal(fetched.length, count)
    await assert.rejects(damaged.sourceTextures.restore(fallback, AbortSignal.abort()), { name: 'AbortError' })
    installCache(async () => { throw Error('disabled') })
    const transient = await loadCoveSourceTextures(fallback, manifest)
    const recovered = await transient.sourceTextures.restore(fallback)
    assert.equal(recovered.missing, 0)
    assert.match(recovered.warnings.join(' '), /session/)
    const localUrl = '/assets/verified-original.png'
    globalThis.fetch = async url => {
      assert.equal(url, localUrl, 'hydration must only request the local release asset')
      return new Response(image)
    }
    const upgradeManifest = { version: 1, materials: {
      pier: { occlusionMap: arm, metallicRoughnessMap: arm, map: { ...normal, upgrade: true } },
      wetPier: { metallicRoughnessMap: { ...arm, upgrade: true } },
    } }
    const durable = await loadCoveSourceTextures(fallback, upgradeManifest, undefined, new Map([[arm.md5, localUrl]]))
    assert.equal(durable.sourceTextures.missing, 0)
    assert.equal(durable.sourceTextures.upgrades, 1)
    assert.equal(durable.materials.get('pier').map, bakedAlbedo)
    assert.equal(durable.materials.get('pier').occlusionMap, durable.materials.get('wetPier').metallicRoughnessMap)
    assert.deepEqual(durable.localWarnings, [])
    globalThis.fetch = async () => new Response('damaged copy')
    const corruptDurable = await loadCoveSourceTextures(fallback, upgradeManifest, undefined, new Map([[arm.md5, localUrl]]))
    assert.equal(corruptDurable.sourceTextures.missing, 1)
    assert.equal(corruptDurable.sourceTextures.upgrades, 1)
    assert.match(corruptDurable.localWarnings.join(' '), /local source-arm\.png.*checksum/)
    assert.equal(corruptDurable.materials.get('pier').roughness, 0.78)
  } finally {
    globalThis.fetch = fetchBefore
    if (cacheBefore) Object.defineProperty(globalThis, 'caches', cacheBefore)
    else delete globalThis.caches
    await site.close()
  }
})
