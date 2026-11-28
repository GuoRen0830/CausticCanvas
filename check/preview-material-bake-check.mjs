import assert from 'node:assert/strict'
import { test } from 'node:test'
import { tmpdir } from 'node:os'
import { join, dirname, resolve, basename } from 'node:path'
import { mkdtemp, readFile, readdir, rm } from 'node:fs/promises'
import { createServer } from 'vite'
import { writeScenePreview } from '../scripts/prepare-preview-scenes.mjs'

test('OBJ baking round-trips PBR factors, color and linear maps, including draws without UVs', async () => {
  const vite = await createServer({ server: { middlewareMode: true, hmr: false, watch: null } })
  const temp = resolve(tmpdir()), directory = await mkdtemp(join(temp, 'caustic-bake-'))
  try {
    const { readObjAsset } = await vite.ssrLoadModule('/src/geometry/obj-materials.ts')
    const map = new File([Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX9kAAAAASUVORK5CYII=', 'base64')], 'atlas.png')
    const normal = new File([map], 'normal.png'), orm = new File([map], 'orm.png')
    const factor = [0.5, 0.7, 0.4], transform = [2, 3, 0.125, 0.25]
    const sampling = { uvSet: 1, rotation: -0.35, sampler: { addressModeU: 'mirror-repeat', minFilter: 'nearest', lodMaxClamp: 0 } }
    const original = { url: 'https://example.org/wood-normal.png', bytes: 1542, md5: '41dd557b15287cf3ef38031afde8c616' }
    const materials = new Map([
      ['mixed', { color: factor, vertexColor: true, vertexWetness: { saturatedRed: 0.65, saturatedRoughness: 0.173 },
        doubleSided: true, map, uvTransform: transform, clamp: true, metallic: 0.75, roughness: 0.42,
        normalMap: normal, normalScale: 0.55, normalYSign: -1, normalClamp: true, metallicRoughnessMap: orm,
        metallicRoughnessUVTransform: [3, 1, 0, 0.5], occlusionMap: orm, occlusionStrength: 0.65,
        occlusionUVTransform: transform, occlusionClamp: true, sampling: { occlusionMap: sampling },
        emissive: [0.1, 1.2, 0.4], emissiveMap: map }],
      ['also-textured', { color: [0.8, 0.8, 0.8], map, recoverableMaps: {
        normalMap: original, map: { ...original, upgrade: true }, occlusionMap: { ...original, sampling: { uvSet: 1 } },
      } }],
      ['unused-image', { color: [0.2, 0.2, 0.2], map: new File(['unused'], 'unused.png') }],
      ['wet stone', { color: [0.25, 0.2, 0.17], roughnessMap: orm, roughnessClamp: true }],
    ])
    const draws = [['mixed', false], ['mixed', true], ['also-textured', true], ['mixed', true], ['unused-image', false], ['wet stone', true]]
      .map(([material, hasUV], i) => ({ firstIndex: 3 * i, indexCount: 3, material, hasUV }))
    draws[1].uvMask = draws[3].uvMask = 3
    const mesh = {
      vertices: new Float32Array([[0, 0, 0], [0, 0, 1], [1, 0, 0]].flatMap(p => [...p, 0, 1, 0, 1, 0.5, 0.25, 0, 2])),
      uvs1: new Float32Array([0.75, 0.25, 0, 0.5, 0.25, 1]),
      tangents: new Float32Array([[1,0,0,-1], [1,0,0,1], [1,0,0,-1]].flat()),
      uvs: new Float32Array([0, 0, 0, 1, 1, 0]), indices: new Uint32Array(draws.flatMap(() => [0, 1, 2])), draws,
    }
    await writeScenePreview({ mesh, materials }, directory, 'sample', 11)
    const textures = (await readdir(join(directory, 'textures'))).sort()
    assert.deepEqual(textures, ['base-color-0.png', 'metallic-roughness-0.png', 'normal-0.png'])
    const files = await Promise.all(['sample.obj', 'sample.mtl', ...textures.map(p => 'textures/' + p)].map(async path => {
      const file = new File([await readFile(join(directory, path))], basename(path))
      Object.defineProperty(file, 'webkitRelativePath', { value: path })
      return file
    }))
    const baked = await readObjAsset(files[0], files)
    assert.deepEqual([...baked.warnings], [])
    assert.equal(baked.materials.size, 5)
    const [plain, mapped, other, reused, unused, wet] = baked.mesh.draws.map(draw => baked.materials.get(draw.material))
    assert.deepEqual(plain.color, factor)
    assert.equal(plain.map, undefined)
    assert.equal(plain.normalMap, undefined)
    assert.equal(plain.metallicRoughnessMap, undefined)
    assert.equal(plain.metallic, 0.75)
    assert.equal(plain.roughness, 0.42)
    assert.deepEqual(mapped.color, factor)
    assert.deepEqual(mapped.uvTransform, transform)
    assert.equal(mapped.clamp, true)
    assert.equal(mapped.map, files[2])
    assert.equal(mapped.metallicRoughnessMap, files[3])
    assert.deepEqual(mapped.metallicRoughnessUVTransform, [3, 1, 0, 0.5])
    assert.equal(mapped.normalMap, files[4])
    assert.equal(mapped.normalScale, 0.55)
    assert.equal(mapped.normalClamp, true)
    assert.equal(mapped.normalYSign, -1)
    assert.equal(wet.roughnessMap, files[3])
    assert.equal(wet.roughnessClamp, true)
    assert.equal(wet.roughness, 1)
    assert.equal(mapped.map, other.map)
    assert.equal(reused, mapped)
    assert.equal(unused.map, undefined)
    assert.equal(mapped.vertexColor, true)
    assert.deepEqual(mapped.vertexWetness, materials.get('mixed').vertexWetness)
    assert.equal(other.vertexWetness, undefined)
    assert.equal(mapped.doubleSided, true)
    assert.equal(baked.mesh.draws[1].doubleSided, true)
    assert.equal(baked.mesh.draws[2].doubleSided, undefined)
    assert.equal(other.vertexColor, false)
    assert.deepEqual([...baked.mesh.vertices.slice(6, 9)], [1, 0.5, 0.25])
    assert.equal(baked.mesh.draws[1].uvMask, 3)
    const firstMapped = baked.mesh.indices[baked.mesh.draws[1].firstIndex]
    assert.deepEqual([...baked.mesh.uvs1.slice(firstMapped * 2, firstMapped * 2 + 2)], [0.75, 0.25])
    assert.deepEqual([...baked.mesh.tangents.slice(firstMapped * 4, firstMapped * 4 + 4)], [1, 0, 0, -1])
    assert.equal(mapped.occlusionMap, mapped.metallicRoughnessMap)
    assert.equal(mapped.occlusionStrength, 0.65)
    assert.deepEqual(mapped.occlusionUVTransform, transform)
    assert.equal(mapped.occlusionClamp, true)
    assert.deepEqual(mapped.sampling.occlusionMap, sampling)
    assert.deepEqual(mapped.emissive, [0.1, 1.2, 0.4])
    assert.equal(mapped.emissiveMap, mapped.map)
    assert.equal(plain.occlusionMap, undefined)
    assert.deepEqual(JSON.parse(await readFile(join(directory, 'source-textures.json'), 'utf8')), {
      version: 1, materials: { [baked.mesh.draws[2].material]: { normalMap: original, map: { ...original, upgrade: true } } },
    })
    // Many OBJ exporters write a legacy exponent along with PBR parameters.
    // An explicit Pr takes precedence however the two lines are ordered.
    const legacyText = (await files[1].text()).replace('Pr 0.42', 'Ns 30').replace('Pr 0.42', 'Pr 0.42\nNs 30')
    const legacy = new File([legacyText], 'sample.mtl')
    const imported = await readObjAsset(files[0], [files[0], legacy, ...files.slice(2)])
    assert.equal(imported.materials.get('material-0').roughness, 0.5)
    assert.equal(imported.materials.get('material-1').roughness, 0.42)
  } finally {
    await vite.close()
    if (dirname(resolve(directory)) === temp && basename(directory).startsWith('caustic-bake-')) {
      await rm(directory, { recursive: true, force: true })
    }
  }
})
