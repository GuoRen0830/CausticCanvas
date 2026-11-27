import assert from 'node:assert/strict'
import { test } from 'node:test'
import { createServer } from 'vite'

// A pier with long piles reproduces the floating-platform regression. The
// other boxes stand in for remote scans so the layout can run offline.
const models = {
  modular_wooden_pier: [[-1.1, 6.97, -1.5, 2.2, 0.15, 3],
    ...[-0.94, 0.84].flatMap(x => [-1.42, 1.30].map(z => [x, 0, z, 0.1, 7.62, 0.1]))],
  coast_land_rocks_02: [[-1.6, 0, -1.4, 3.2, 0.06, 2.8], [-1, 0.05, -0.74, 1.65, 0.7, 1.4]],
  treasure_chest: [[0, 0, 0, 1, 0.62, 0.64]],
  wooden_lantern_01: [[0, 0, 0, 0.24, 0.39, 0.24]],
  wooden_crate_01: [[0, 0, 0, 1, 0.72, 0.82]],
  wooden_barrels_01: [[0, 0, 0, 0.6, 0.95, 0.57]],
}

function gltfFixture(name, boxes) {
  const positions = [], indices = []
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity]
  for (const [x, y, z, w, h, d] of boxes) {
    const base = positions.length / 3
    for (let corner = 0; corner < 8; corner++) {
      const p = [x + (corner & 1) * w, y + ((corner >> 1) & 1) * h, z + ((corner >> 2) & 1) * d]
      positions.push(...p)
      p.forEach((value, axis) => { min[axis] = Math.min(min[axis], value); max[axis] = Math.max(max[axis], value) })
    }
    for (const i of [0, 3, 1, 0, 2, 3, 4, 7, 6, 4, 5, 7, 0, 5, 4, 0, 1, 5,
      2, 7, 3, 2, 6, 7, 0, 6, 2, 0, 4, 6, 1, 7, 5, 1, 3, 7]) indices.push(base + i)
  }
  const indexOffset = positions.length * 4, bin = new ArrayBuffer(indexOffset + indices.length * 2)
  new Float32Array(bin, 0, positions.length).set(positions)
  new Uint16Array(bin, indexOffset).set(indices)
  const doc = {
    asset: { version: '2.0' }, scenes: [{ nodes: [0] }], nodes: [{ name, mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1, material: 0 }] }],
    buffers: [{ uri: 'geometry.bin', byteLength: bin.byteLength }],
    bufferViews: [{ buffer: 0, byteLength: indexOffset }, { buffer: 0, byteOffset: indexOffset, byteLength: indices.length * 2 }],
    accessors: [{ bufferView: 0, componentType: 5126, count: positions.length / 3, type: 'VEC3', min, max },
      { bufferView: 1, componentType: 5123, count: indices.length, type: 'SCALAR' }],
    materials: [{ name, pbrMetallicRoughness: { baseColorTexture: { index: 0 } } }],
    textures: [{ source: 0 }], images: [{ name, uri: 'diffuse.png', mimeType: 'image/png' }],
  }
  return { doc, bin }
}

test('tidal landing seats the pier, reuses cargo maps and provides paintable scenery', async () => {
  const site = await createServer({ server: { middlewareMode: true, hmr: false, watch: null } })
  const fixtures = new Map(Object.entries(models).map(([name, boxes]) => [name, gltfFixture(name, boxes)]))
  const originalFetch = globalThis.fetch
  let pierAvailable = true, remoteCargo = true
  try {
    globalThis.fetch = async url => {
      const { hostname, pathname } = new URL(url)
      if (hostname === 'api.polyhaven.com') {
        const id = pathname.split('/').at(-1), path = `https://fixture.invalid/${id}/`
        if (!pierAvailable && id === 'modular_wooden_pier') return new Response('', { status: 503 })
        if (!remoteCargo && ['wooden_crate_01', 'wooden_barrels_01', 'wooden_lantern_01'].includes(id)) {
          return new Response('', { status: 503 })
        }
        return Response.json(id === 'coast_sand_01' ? { diffuse: { '1k': { jpg: { url: path + 'diffuse.png' } } } }
          : { gltf: { '1k': { gltf: { url: path + 'model.gltf' } } } })
      }
      const source = fixtures.get(pathname.split('/')[1])
      if (pathname.endsWith('/model.gltf')) return Response.json(source.doc)
      if (pathname.endsWith('/geometry.bin')) return new Response(source.bin)
      if (pathname.endsWith('/diffuse.png')) return new Response('fixture image', { headers: { 'Content-Type': 'image/png' } })
      throw new Error('Unexpected request ' + url)
    }
    const { readCovePreview } = await site.ssrLoadModule('/src/geometry/smugglers-cove-preview.ts')
    const { placeObjOnReceiver } = await site.ssrLoadModule('/src/geometry/place-obj.ts')
    const { buildSceneGeometry, sceneVertexStride } = await site.ssrLoadModule('/src/rendering/scene-mesh.ts')
    const { buildDesignView } = await site.ssrLoadModule('/src/optics/design-view.ts')
    const { config } = await site.ssrLoadModule('/src/config.ts')
    const { quayLevel, quayFront, pierApproach } = await site.ssrLoadModule('/src/geometry/cove-quayside.ts')
    const scene = await readCovePreview(), { mesh, materials } = scene
    assert.deepEqual([...scene.warnings], [])
    assert.equal([...materials.values()].filter(m => m.map?.name === 'wooden_crate_01').length, 1)
    const decks = mesh.draws.filter(d => materials.get(d.material)?.map?.name === 'modular_wooden_pier' && !d.hasUV)
    assert.equal(decks.length, 2)
    for (const deck of decks) {
      const ys = mesh.indices.slice(deck.firstIndex, deck.firstIndex + deck.indexCount).map(i => i * sceneVertexStride + 1)
      const heights = Array.from(ys, i => mesh.vertices[i])
      assert.ok(Math.min(...heights) > -0.52 && Math.max(...heights) < 1.03)
    }
    function topAt(x, z) {
      const v = mesh.vertices, indices = mesh.indices
      let top = -Infinity
      for (let i = 0; i < indices.length; i += 3) {
        const a = indices[i]*sceneVertexStride, b = indices[i+1]*sceneVertexStride, c = indices[i+2]*sceneVertexStride
        const dx = v[b]-v[a], dz = v[b+2]-v[a+2], ex = v[c]-v[a], ez = v[c+2]-v[a+2]
        const det = dx*ez-ex*dz
        if (!det) continue
        const u = ((x-v[a])*ez-(z-v[a+2])*ex)/det, w = ((z-v[a+2])*dx-(x-v[a])*dz)/det
        if (u >= 0 && w >= 0 && u+w <= 1) top = Math.max(top, v[a+1]+u*(v[b+1]-v[a+1])+w*(v[c+1]-v[a+1]))
      }
      return top
    }
    const width = (pierApproach.z-quayFront)/4
    let previous = quayLevel
    for (let i = 0; i < 4; i++) {
      const height = topAt(pierApproach.x, quayFront+(i+0.5)*width)
      assert.ok(previous-height > 0.055 && previous-height < 0.20, 'All four steps descend from quay to pier')
      previous = height
    }
    assert.ok(Math.abs(topAt(pierApproach.x, pierApproach.z+0.11)-previous) < 0.008, 'Last tread meets actual walking deck')
    assert.ok(topAt(3.71,-2.93) > quayLevel+1.45, 'Shed roof above the stone quay')
    assert.ok(topAt(3.67,-1.77) > quayLevel+0.7, 'The hoist rises above its stone footing')
    assert.ok(topAt(-2.42,3.19) > 0.17, 'The striped pot marker breaks the surface')
    assert.ok(Math.abs(topAt(-0.5,1.5)) < 0.02, 'Open shallow water remains between the pools and the skiff')
    assert.ok(topAt(-6.6,-3.2) > 1.4, 'The covered fish counter receives overhead light')
    assert.ok(topAt(-2.83,5.25) > 2, 'The billowed lug sail presents a surface to the overhead brush')
    assert.ok(topAt(3.69,6.49) > 1.85, 'The pierhead light stands at the end of the breakwater')
    assert.ok(Math.abs(topAt(1.65,5.55)) < 0.02, 'Open water separates the lugger from the pierhead')
    const { receiver } = placeObjOnReceiver(mesh, config.lens, config.receiver)
    const view = buildDesignView(buildSceneGeometry(receiver, { mesh, receiverFloor: false }).mesh, receiver)
    const hitObjects = new Set()
    for (let i = 0; i < view.samples.length; i += 2) {
      const hit = view.samples[i]
      assert.ok(hit, 'The bay and beach should form a continuous top-light receiver')
      hitObjects.add(view.bvh.triangles[(hit - 1) * 12 + 3])
    }
    assert.ok(hitObjects.size > 18)
    pierAvailable = false
    const partial = await readCovePreview()
    assert.ok(partial.mesh.indices.length && [...partial.warnings].some(w => w.includes('modular_wooden_pier')))
    remoteCargo = false
    const authored = await readCovePreview()
    assert.equal([...authored.warnings].filter(w => w.includes('project-modeled')).length, 3)
    assert.ok([...authored.materials.keys()].some(name => name.endsWith('/lamp glass')))
  } finally {
    globalThis.fetch = originalFetch
    await site.close()
  }
})

test('coastal rock atlas is only substituted for the missing source photograph', async () => {
  const site = await createServer({ configFile: false, server: { middlewareMode: true, hmr: false, watch: null } })
  try {
    const { supplyCoveRockColor } = await site.ssrLoadModule('/src/geometry/cove-rock-material.ts')
    const failure = 'Could not load coast_land_rocks_02 base color coast_land_rocks_02_diff (fetch failed). Using the material color.'
    const asset = { mesh: {}, materials: new Map([['0', { color: [1, 1, 1] }]]), warnings: new Set(['other warning']) }
    assert.equal(await supplyCoveRockColor(asset), asset)
    asset.warnings.add(failure)
    const replaced = await supplyCoveRockColor(asset)
    assert.equal(replaced.mesh, asset.mesh)
    const map = replaced.materials.get('0').map
    assert.equal(map.type, 'image/jpeg')
    assert.deepEqual([...new Uint8Array(await map.slice(0, 3).arrayBuffer())], [255, 216, 255])
    assert.ok(replaced.warnings.has('other warning') && !replaced.warnings.has(failure))
    assert.ok([...replaced.warnings].some(w => w.startsWith('Using the project-authored')))
    asset.materials.get('0').map = new File(['source'], 'coast_land_rocks_02_diff.jpg')
    assert.equal(await supplyCoveRockColor(asset), asset)
  } finally {
    await site.close()
  }
})
