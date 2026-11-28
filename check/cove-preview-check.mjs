import assert from 'node:assert/strict'
import { test } from 'node:test'
import { readFile } from 'node:fs/promises'
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

const readDetailFile = async filename =>
  new Blob([await readFile(new URL('../scene/smugglers-cove/' + filename, import.meta.url))], { type: 'image/png' })

function gltfFixture(name, boxes) {
  const positions = [], uvs = [], indices = []
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity]
  for (const [x, y, z, w, h, d] of boxes) {
    const base = positions.length / 3
    for (let corner = 0; corner < 8; corner++) {
      const p = [x + (corner & 1) * w, y + ((corner >> 1) & 1) * h, z + ((corner >> 2) & 1) * d]
      positions.push(...p)
      uvs.push(corner & 1, (corner >> 2) & 1)
      p.forEach((value, axis) => { min[axis] = Math.min(min[axis], value); max[axis] = Math.max(max[axis], value) })
    }
    for (const i of [0, 3, 1, 0, 2, 3, 4, 7, 6, 4, 5, 7, 0, 5, 4, 0, 1, 5,
      2, 7, 3, 2, 6, 7, 0, 6, 2, 0, 4, 6, 1, 7, 5, 1, 3, 7]) indices.push(base + i)
  }
  const uvOffset = positions.length * 4, indexOffset = uvOffset + uvs.length * 4
  const bin = new ArrayBuffer(indexOffset + indices.length * 2)
  new Float32Array(bin, 0, positions.length).set(positions)
  new Float32Array(bin, uvOffset, uvs.length).set(uvs)
  new Uint16Array(bin, indexOffset).set(indices)
  const doc = {
    asset: { version: '2.0' }, scenes: [{ nodes: [0] }], nodes: [{ name, mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0, TEXCOORD_0: 2 }, indices: 1, material: 0 }] }],
    buffers: [{ uri: 'geometry.bin', byteLength: bin.byteLength }],
    bufferViews: [{ buffer: 0, byteLength: uvOffset }, { buffer: 0, byteOffset: indexOffset, byteLength: indices.length * 2 },
      { buffer: 0, byteOffset: uvOffset, byteLength: uvs.length * 4 }],
    accessors: [{ bufferView: 0, componentType: 5126, count: positions.length / 3, type: 'VEC3', min, max },
      { bufferView: 1, componentType: 5123, count: indices.length, type: 'SCALAR' },
      { bufferView: 2, componentType: 5126, count: uvs.length / 2, type: 'VEC2' }],
    materials: [{ name, pbrMetallicRoughness: { baseColorTexture: { index: 0 }, metallicRoughnessTexture: { index: 1 },
      metallicFactor: 0.64, roughnessFactor: 0.43 }, normalTexture: { index: 2, scale: 0.73 } }],
    textures: [0,1,2].map(source => ({ source })), images: [{ name, uri: 'diffuse.png', mimeType: 'image/png' },
      { name: name + '-arm', uri: 'arm.png', mimeType: 'image/png' },
      { name: name + '-normal', uri: 'normal.png', mimeType: 'image/png' }],
  }
  // Both a lower sibling and an MSFT_lod child should disappear. Keeping either would duplicate the original.
  if (name === 'treasure_chest') {
    doc.scenes[0].nodes.push(1, 2)
    doc.nodes[0].extensions = { MSFT_lod: { ids: [2] } }
    doc.nodes.push({ name: name + '_LOD1', mesh: 0 }, { name: 'reduced', mesh: 0 })
    doc.extensionsRequired = ['MSFT_lod']
  }
  return { doc, bin }
}

test('tidal landing seats the pier, reuses cargo maps and provides paintable scenery', async () => {
  const site = await createServer({ server: { middlewareMode: true, hmr: false, watch: null } })
  const fixtures = new Map(Object.entries(models).map(([name, boxes]) => [name, gltfFixture(name, boxes)]))
  const originalFetch = globalThis.fetch
  let pierAvailable = true, remoteCargo = true, completeMaps = true
  const requests = []
  try {
    globalThis.fetch = async url => {
      const { hostname, pathname } = new URL(url)
      requests.push(pathname)
      if (hostname === 'api.polyhaven.com') {
        const id = pathname.split('/').at(-1), path = `https://fixture.invalid/${id}/`
        if (!pierAvailable && id === 'modular_wooden_pier') return new Response('', { status: 503 })
        if (!remoteCargo && ['wooden_crate_01', 'wooden_barrels_01', 'wooden_lantern_01'].includes(id)) {
          return new Response('', { status: 503 })
        }
        const texture = name => ({ url: path + name, size: 13, md5: '00001111222233334444555566667777' })
        if (id === 'treasure_chest') {
          const model = size => ({ gltf: { url: path + 'model.gltf',
            include: Object.fromEntries(['diffuse.png','arm.png','normal.png'].map(p =>
              [p, texture(size === '1k' ? p : size + '/' + p)])) } })
          return Response.json({ gltf: Object.fromEntries(['1k','2k','4k'].map(size => [size, model(size)])) })
        }
        return Response.json(id === 'coast_sand_01' ? Object.fromEntries(
          [['diffuse','diffuse.png'],['nor_gl','normal.png'],['rough','rough.png'],['ao','ao.png']]
            .map(([key, image]) => [key, { '1k': { jpg: texture(image) } }]))
          : { gltf: { '1k': { gltf: { url: path + 'model.gltf',
            include: Object.fromEntries(['diffuse.png','arm.png','normal.png'].map(p => [p,texture(p)])) } } } })
      }
      const source = fixtures.get(pathname.split('/')[1])
      if (pathname.endsWith('/model.gltf')) return Response.json(source.doc)
      if (pathname.endsWith('/geometry.bin')) return new Response(source.bin)
      if (!completeMaps && pathname === '/wooden_crate_01/arm.png') return new Response('', { status: 404 })
      if (pathname === '/treasure_chest/2k/diffuse.png') return new Response('', { status: 404 })
      if (pathname.endsWith('.png')) return new Response(pathname, { headers: { 'Content-Type': 'image/png' } })
      throw new Error('Unexpected request ' + url)
    }
    const { readCovePreview } = await site.ssrLoadModule('/src/geometry/smugglers-cove-preview.ts')
    const { placeObjOnReceiver } = await site.ssrLoadModule('/src/geometry/place-obj.ts')
    const { buildSceneGeometry, sceneVertexStride } = await site.ssrLoadModule('/src/rendering/scene-mesh.ts')
    const { buildDesignView } = await site.ssrLoadModule('/src/optics/design-view.ts')
    const { config } = await site.ssrLoadModule('/src/config.ts')
    const { quayLevel, quayFront, pierApproach } = await site.ssrLoadModule('/src/geometry/cove-quayside.ts')
    const scene = await readCovePreview(undefined, { readDetailFile, hasLocal: async file => file.url.endsWith('/4k/normal.png') })
    const { mesh, materials } = scene
    assert.deepEqual([...scene.warnings], [])
    // These finishes belong to working areas on opposite sides of the harbor.
    // Check the batches the renderer will actually draw, including their maps.
    for (const [name, family] of [
      ['steamed willow', 'timber'], ['tidepool mussel', 'shell'], ['olive kelp', 'grass'],
      ['pilot signal cream', 'sailcloth'], ['fish market terracotta','fired-clay'],
      ['fish market fired clay','fired-clay'], ['fish market madder','linen'],
      ['tarred seine','rope'], ['ochre sea star','sea-star'], ['acorn barnacle','shell'],
    ]) {
      const draw = mesh.draws.find(part => part.material.endsWith('/' + name))
      assert.ok(draw?.hasUV && draw.indexCount > 0, 'Missing textured mesh for ' + name)
      const material = materials.get(draw.material)
      assert.equal(material.normalMap?.name, family + '-normal.png')
      assert.equal(material.roughnessMap?.name, family + '-roughness.png')
    }
    const sand = [...materials.entries()].find(([id]) => id.endsWith('/dry'))[1]
    assert.deepEqual([sand.normalMap.name, sand.roughnessMap.name, sand.occlusionMap.name], ['normal.png','rough.png','ao.png'])
    // Compare the tint at real vertices along three cross-sections through the sand.
    // A split mesh may have two vertices at the same position, on different
    // materials. Both are kept here so a hard jump on a shared edge is detected.
    const beachColumns = new Map()
    for (const draw of mesh.draws) {
      const material = materials.get(draw.material)
      if (material.map !== sand.map) continue
      const vertices = new Set(mesh.indices.subarray(draw.firstIndex, draw.firstIndex + draw.indexCount))
      for (const id of vertices) {
        const i = id * sceneVertexStride, v = mesh.vertices
        if (v[i + 4] < 0.8) continue // Ignore the vertical and bottom faces of the diorama.
        const x = v[i]
        if (!beachColumns.has(x)) beachColumns.set(x, [])
        const tint = material.color.map((base, k) => base * (material.vertexColor ? v[i + 6 + k] : 1))
        beachColumns.get(x).push({ z: v[i + 2], tint })
      }
    }
    const columns = [...beachColumns.keys()]
    for (const position of [-3.6, 0, 3.6]) {
      const x = columns.reduce((a, b) => Math.abs(a - position) < Math.abs(b - position) ? a : b)
      const samples = beachColumns.get(x).sort((a, b) => a.z - b.z)
      const dry = samples[0], wet = samples.at(-1)
      assert.ok(dry.tint[0] - wet.tint[0] > 0.22, 'The wet sand at the waterline is darker than the dry sand')
      for (let i = 1; i < samples.length; i++) {
        const a = samples[i - 1], b = samples[i], distance = b.z - a.z
        for (let k = 0; k < 3; k++) {
          assert.ok(Math.abs(b.tint[k] - a.tint[k]) <= 1e-5 + distance,
            'A centimeter across the visible beach should not produce a color-factor jump greater than 0.01')
        }
      }
    }
    const grass = mesh.draws.find(draw => draw.material.endsWith('/grass-olive'))
    const shell = mesh.draws.find(draw => draw.material.endsWith('/limpet shell'))
    for (const [draw, stem] of [[grass, 'grass'], [shell, 'shell']]) {
      assert.equal(draw.hasUV, true)
      const finish = materials.get(draw.material)
      assert.equal(finish.normalMap.name, stem + '-normal.png')
      assert.equal(finish.roughnessMap.name, stem + '-roughness.png')
      for (let i = draw.firstIndex; i < draw.firstIndex + draw.indexCount; i += 3) {
        const [a,b,c] = mesh.indices.subarray(i, i+3).map(index => 2*index)
        const uv = mesh.uvs
        const determinant = (uv[b]-uv[a])*(uv[c+1]-uv[a+1]) - (uv[c]-uv[a])*(uv[b+1]-uv[a+1])
        assert.ok(Math.abs(determinant) > 1e-8, 'Expected a usable tangent frame on ' + stem)
      }
    }
    const chestDraws = mesh.draws.filter(d => materials.get(d.material)?.normalMap?.name === 'treasure_chest-normal')
    assert.equal(chestDraws.reduce((n, draw) => n + draw.indexCount / 3, 0), 12, 'Use exactly one original mesh')
    const chest = materials.get(chestDraws[0].material)
    assert.deepEqual([chest.roughness,chest.metallic,chest.normalScale,chest.normalYSign], [0.43,0.64,0.73,-1])
    assert.equal(await chest.normalMap.text(), '/treasure_chest/4k/normal.png')
    assert.equal(await chest.metallicRoughnessMap.text(), '/treasure_chest/2k/arm.png')
    assert.equal(await chest.map.text(), '/treasure_chest/diffuse.png')
    assert.equal(chest.recoverableMaps.map.upgrade, true)
    assert.equal(chest.recoverableMaps.map.url, 'https://fixture.invalid/treasure_chest/2k/diffuse.png')
    assert.deepEqual(requests.filter(path => path.includes('/4k/')), ['/treasure_chest/4k/normal.png'])
    assert.equal(chest.metallicRoughnessMap.name, 'treasure_chest-arm')
    assert.equal([...materials.values()].filter(m => m.map?.name === 'wooden_crate_01').length, 1)
    const decks = mesh.draws.filter(d => /\/\d+$/.test(d.material) && materials.get(d.material)?.normalMap?.name === 'modular_wooden_pier-normal' && d.hasUV)
    assert.equal(decks.length, 2)
    for (const deck of decks) {
      const ys = mesh.indices.slice(deck.firstIndex, deck.firstIndex + deck.indexCount).map(i => i * sceneVertexStride + 1)
      const heights = Array.from(ys, i => mesh.vertices[i])
      assert.ok(Math.min(...heights) > -0.52 && Math.max(...heights) < 1.03)
    }
    function topAt(x, z, draw) {
      const v = mesh.vertices, indices = mesh.indices
      let top = -Infinity
      const start = draw?.firstIndex ?? 0, end = start + (draw?.indexCount ?? indices.length)
      for (let i = start; i < end; i += 3) {
        const a = indices[i]*sceneVertexStride, b = indices[i+1]*sceneVertexStride, c = indices[i+2]*sceneVertexStride
        const dx = v[b]-v[a], dz = v[b+2]-v[a+2], ex = v[c]-v[a], ez = v[c+2]-v[a+2]
        const det = dx*ez-ex*dz
        if (!det) continue
        const u = ((x-v[a])*ez-(z-v[a+2])*ex)/det, w = ((z-v[a+2])*dx-(x-v[a])*dz)/det
        if (u >= 0 && w >= 0 && u+w <= 1) top = Math.max(top, v[a+1]+u*(v[b+1]-v[a+1])+w*(v[c+1]-v[a+1]))
      }
      return top
    }
    const { shoreHeight } = await site.ssrLoadModule('/src/geometry/cove-shore.ts')
    const skiff = mesh.draws.find(d => d.material.endsWith('/deep teal paint')).material.split('/')[0] + '/'
    function positions(draw) {
      return Array.from(new Set(mesh.indices.subarray(draw.firstIndex,draw.firstIndex+draw.indexCount)), index => {
        const i = index*sceneVertexStride
        return mesh.vertices.subarray(i,i+3)
      })
    }
    const hull = mesh.draws.filter(d => d.material.startsWith(skiff)).flatMap(positions)
    for (const [x,y,z] of hull) assert.ok(y-shoreHeight(x,z) > 0.08, 'The whole skiff clears the sloping seabed')
    assert.ok(hull.some(p => p[1] < -0.20), 'The keel remains submerged')
    // Look down just inside the transom and stem, beyond the thwarts and oars.
    for (const [x,z] of [[1.005,1.45],[0.796,4.95]]) {
      assert.ok(topAt(x,z) > 0.025, 'A dry sole covers the water in the open ends of the boat')
    }
    const hardware = positions(mesh.draws.find(d => d.material === skiff+'verdigris'))
    const ropes = mesh.draws.filter(d => d.material.endsWith('/hemp rope')).flatMap(positions)
    for (const z of [Math.min(...hardware.map(p => p[2])),Math.max(...hardware.map(p => p[2]))]) {
      const fitting = hardware.filter(p => Math.abs(p[2]-z) < 0.05)
      assert.ok(fitting.some(p => ropes.some(q => Math.hypot(p[0]-q[0],p[1]-q[1],p[2]-q[2]) < 0.018)),
        'A physical mooring line reaches each gunwale fitting')
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
    const net = mesh.draws.find(draw => draw.material.endsWith('/tarred seine'))
    let cordHits = 0
    for (let ix = 0; ix < 21; ix++) for (let iz = 0; iz < 21; iz++) {
      const height = topAt(-7.02+0.054*ix, 2.14+0.054*iz, net)
      if (Number.isFinite(height)) {
        assert.ok(height > 0.03, 'The hung net clears the surface')
        cordHits++
      }
    }
    assert.ok(cordHits > 12 && cordHits < 0.60*21*21, 'Light can hit the physical cords and pass through the mesh openings')
    assert.ok(topAt(-2.68,-3.91) > 1, 'A draining pot is exposed to top light behind the drying nets')
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
    completeMaps = false
    const partial = await readCovePreview(undefined, { readDetailFile })
    assert.ok(partial.mesh.indices.length && [...partial.warnings].some(w => w.includes('modular_wooden_pier')))
    const scannedCrate = [...partial.materials.values()].find(m => m.map?.name === 'wooden_crate_01')
    assert.equal(scannedCrate.normalMap.name, 'wooden_crate_01-normal')
    assert.equal(scannedCrate.metallicRoughnessMap, undefined)
    assert.deepEqual(scannedCrate.recoverableMaps.metallicRoughnessMap, {
      roughness: 0.43, metallic: 0.64, uvTransform: undefined, clamp: false, sampling: undefined,
      url: 'https://fixture.invalid/wooden_crate_01/arm.png', bytes: 13, md5: '00001111222233334444555566667777',
    })
    assert.ok([...partial.warnings].some(w => w.startsWith('Source material wooden_crate_01:') && w.includes('metallic-roughness')))
    remoteCargo = false
    const authored = await readCovePreview(undefined, { readDetailFile })
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
