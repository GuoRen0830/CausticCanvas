import assert from 'node:assert/strict'
import { test } from 'node:test'
import { tmpdir } from 'node:os'
import { join, dirname, resolve, basename } from 'node:path'
import { mkdtemp, readFile, readdir, rm } from 'node:fs/promises'
import { createServer } from 'vite'
import { writeScenePreview } from '../scripts/prepare-preview-scenes.mjs'

test('OBJ baking preserves per-draw textured and Kd-only material bindings', async () => {
  const vite = await createServer({ server: { middlewareMode: true, hmr: false, watch: null } })
  const temp = resolve(tmpdir()), directory = await mkdtemp(join(temp, 'caustic-bake-'))
  try {
    const { readObjAsset } = await vite.ssrLoadModule('/src/geometry/obj-materials.ts')
    const map = new File([Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX9kAAAAASUVORK5CYII=', 'base64')], 'atlas.png')
    const factor = [0.5, 0.7, 0.4], transform = [2, 3, 0.125, 0.25]
    const materials = new Map([
      ['mixed', { color: factor, map, uvTransform: transform, clamp: true }],
      ['also-textured', { color: [0.8, 0.8, 0.8], map }],
      ['unused-image', { color: [0.2, 0.2, 0.2], map: new File(['unused'], 'unused.png') }],
    ])
    const draws = [['mixed', false], ['mixed', true], ['also-textured', true], ['mixed', true], ['unused-image', false]]
      .map(([material, hasUV], i) => ({ firstIndex: 3 * i, indexCount: 3, material, hasUV }))
    const mesh = {
      vertices: new Float32Array([[0, 0, 0], [0, 0, 1], [1, 0, 0]].flatMap(p => [...p, 0, 1, 0, 1, 1, 1, 0, 2])),
      uvs: new Float32Array([0, 0, 0, 1, 1, 0]), indices: new Uint32Array(draws.flatMap(() => [0, 1, 2])), draws,
    }
    await writeScenePreview({ mesh, materials }, directory, 'sample', 11)
    assert.deepEqual(await readdir(join(directory, 'textures')), ['base-color-0.png'])
    const files = await Promise.all(['sample.obj', 'sample.mtl', 'textures/base-color-0.png'].map(async path => {
      const file = new File([await readFile(join(directory, path))], basename(path))
      Object.defineProperty(file, 'webkitRelativePath', { value: path })
      return file
    }))
    const baked = await readObjAsset(files[0], files)
    assert.deepEqual([...baked.warnings], [])
    assert.equal(baked.materials.size, 4)
    const [plain, mapped, other, reused, unused] = baked.mesh.draws.map(draw => baked.materials.get(draw.material))
    assert.deepEqual(plain.color, factor)
    assert.equal(plain.map, undefined)
    assert.deepEqual(mapped.color, factor)
    assert.deepEqual(mapped.uvTransform, transform)
    assert.equal(mapped.clamp, true)
    assert.equal(mapped.map, files[2])
    assert.equal(mapped.map, other.map)
    assert.equal(reused, mapped)
    assert.equal(unused.map, undefined)
  } finally {
    await vite.close()
    if (dirname(resolve(directory)) === temp && basename(directory).startsWith('caustic-bake-')) {
      await rm(directory, { recursive: true, force: true })
    }
  }
})
