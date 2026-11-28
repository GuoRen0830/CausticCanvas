import assert from 'node:assert/strict'
import { test } from 'node:test'
import { createServer } from 'vite'

const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX9kAAAAASUVORK5CYII=', 'base64')
const points = new Float32Array([0,0,0, 1,0,0, 0,0,1]), texcoords = new Float32Array([0,0, 1,0, 0,1])
const colors = Buffer.from([64,255,0,255, 0,64,255,255, 255,0,64,255])
const uv1 = new Float32Array([0.25,0.5, 0.75,0.5, 0.25,0.875])
const normals = new Float32Array([0,1,0, 0,1,0, 0,1,0])
const tangents = new Float32Array([1,0,0,1, 1,0,0,1, 1,0,0,1])
const parts = [points,texcoords,colors,uv1,normals,tangents]
const geometry = Buffer.concat(parts.map(p => Buffer.from(p.buffer,p.byteOffset,p.byteLength)))

function document() {
  return {
    asset: { version: '2.0' }, scene: 0, scenes: [{ nodes: [0] }], nodes: [{ mesh: 0, translation: [2,0,0] }],
    buffers: [{ byteLength: geometry.length }],
    bufferViews: parts.map((p, i) => ({ buffer: 0, byteOffset: parts.slice(0,i).reduce((sum, a) => sum + a.byteLength,0),
      byteLength: p.byteLength })),
    accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: 1, componentType: 5126, count: 3, type: 'VEC2' },
      { bufferView: 2, componentType: 5121, normalized: true, count: 3, type: 'VEC4' },
      { bufferView: 3, componentType: 5126, count: 3, type: 'VEC2' },
      { bufferView: 4, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: 5, componentType: 5126, count: 3, type: 'VEC4' }],
    meshes: [{ primitives: [{ material: 0, attributes: { POSITION: 0, TEXCOORD_0: 1, COLOR_0: 2 } }] }],
    images: [{ mimeType: 'image/png' }], textures: [{ source: 0 }],
    materials: [{ pbrMetallicRoughness: {
      metallicFactor: 0.75, roughnessFactor: 0.26, baseColorFactor: [0.8,0.7,0.6,1],
      baseColorTexture: { index: 0 }, metallicRoughnessTexture: { index: 0, extensions: {
        KHR_texture_transform: { scale: [2,3], offset: [0.1,0.25] },
      } },
    }, normalTexture: { index: 0, scale: 0.4 } }],
  }
}

function glb(doc, binary) {
  const json = Buffer.from(JSON.stringify(doc))
  function chunk(data, type, padding) {
    const padded = (data.length + 3) & ~3
    const bytes = Buffer.alloc(8 + padded, padding)
    bytes.writeUInt32LE(padded)
    bytes.writeUInt32LE(type, 4)
    data.copy(bytes, 8)
    return bytes
  }
  const chunks = [chunk(json, 0x4e4f534a, 32), chunk(binary, 0x004e4942, 0)]
  const header = Buffer.alloc(12)
  header.writeUInt32LE(0x46546c67)
  header.writeUInt32LE(2, 4)
  header.writeUInt32LE(12 + chunks[0].length + chunks[1].length, 8)
  return new File([header, ...chunks], 'preview.glb')
}

function local(path, bytes) {
  const file = new File([bytes], path.split('/').at(-1))
  Object.defineProperty(file, 'webkitRelativePath', { value: path })
  return file
}

test('local GLB and nested glTF retain PBR factors, per-slot UVs and embedded/external maps', async () => {
  const site = await createServer({ configFile: false, server: { middlewareMode: true, hmr: false, watch: null } })
  try {
    const { readGltfAsset } = await site.ssrLoadModule('/src/geometry/gltf-asset.ts')
    const embedded = document(), binary = Buffer.concat([geometry, png])
    embedded.buffers[0].byteLength = binary.length
    embedded.images[0].bufferView = embedded.bufferViews.push({ buffer: 0, byteOffset: geometry.length, byteLength: png.length }) - 1
    const model = glb(embedded, binary)
    const external = document()
    external.buffers[0].uri = '../mesh.bin'
    external.images[0].uri = 'textures/brass%20finish.png'
    const json = local('art/scenes/sample.gltf', JSON.stringify(external))
    const geo = local('art/mesh.bin', geometry), map = local('art/scenes/textures/brass finish.png', png)
    const duplicateName = local('art/other/brass finish.png', 'wrong file')
    for (const [entry, files] of [[model,[model]], [json,[json,geo,map,duplicateName]]]) {
      const { mesh, materials, warnings } = await readGltfAsset(entry, files)
      assert.deepEqual([...warnings], [])
      assert.equal(mesh.indices.length, 3)
      assert.equal(mesh.vertices[0], 2)
      assert.deepEqual([...mesh.uvs], [0,1, 1,1, 0,0])
      assert.deepEqual([...mesh.vertices.slice(6,9)], [Math.fround(64/255),1,0])
      const mat = materials.get('0')
      assert.equal(mat.vertexColor, true)
      assert.equal(mat.metallic, 0.75)
      assert.equal(mat.roughness, 0.26)
      assert.equal(mat.normalScale, 0.4)
      assert.equal(mat.normalYSign, -1)
      assert.equal(mat.normalMap, mat.map)
      assert.equal(mat.metallicRoughnessMap, mat.map)
      assert.deepEqual(mat.metallicRoughnessUVTransform, [2,3,0.1,-2.25])
      assert.deepEqual(Buffer.from(await mat.map.arrayBuffer()), png)
    }
    await assert.rejects(readGltfAsset(json, [json,map]), /Missing companion file/)
    const inline = document()
    inline.buffers[0].uri = 'data:application/octet-stream;base64,' + geometry.toString('base64')
    inline.images[0].uri = 'data:image/png;base64,' + png.toString('base64')
    const standalone = await readGltfAsset(new File([JSON.stringify(inline)], 'inline.gltf'), [])
    assert.equal(standalone.mesh.indices.length, 3)
    assert.deepEqual(Buffer.from(await standalone.materials.get('0').map.arrayBuffer()), png)
    external.materials[0].normalTexture.texCoord = 1
    const other = local('art/scenes/other.gltf', JSON.stringify(external))
    const fallback = await readGltfAsset(other, [other,geo,map])
    assert.equal(fallback.materials.get('0').normalMap, undefined)
    assert.ok([...fallback.warnings].some(w => w.includes('without TEXCOORD_1')))

    external.extensionsRequired = ['KHR_materials_emissive_strength','KHR_texture_transform']
    const surface = external.materials[0], primitive = external.meshes[0].primitives[0]
    surface.occlusionTexture = { index: 0, texCoord: 1, strength: 0.55 }
    surface.emissiveFactor = [1,0.25,0.125]
    surface.doubleSided = true
    surface.emissiveTexture = { index: 0 }
    surface.extensions = { KHR_materials_emissive_strength: { emissiveStrength: 2 } }
    surface.normalTexture.extensions = { KHR_texture_transform: { rotation: 0.35, offset: [0.1,0.2] } }
    external.samplers = [{ wrapS: 33648, wrapT: 33071, magFilter: 9728, minFilter: 9729 }]
    external.textures[0].sampler = 0
    // One material, two primitives: only the second primitive has UV1 and
    // authored tangents. Detecting one missing stream must not disable the other.
    external.meshes[0].primitives.push({ ...primitive,
      attributes: { ...primitive.attributes, NORMAL: 4, TANGENT: 5, TEXCOORD_1: 3 } })
    external.nodes[0].scale = [-2,1,1]
    const detailed = await readGltfAsset(local('art/scenes/detailed.gltf', JSON.stringify(external)), [geo,map])
    const detail = detailed.materials.get('0')
    assert.equal(detail.doubleSided, true)
    assert.deepEqual(detailed.mesh.draws.map(d => d.doubleSided), [true,true])
    assert.deepEqual(detailed.mesh.draws.map(d => d.uvMask), [1,3])
    assert.deepEqual([...detailed.mesh.uvs1], [0,0,0,0,0,0, 0.25,0.5,0.25,0.125,0.75,0.5])
    assert.deepEqual([...detailed.mesh.tangents], [0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,1,-1,0,0,1,-1,0,0,1])
    assert.equal(detail.normalMap,detail.map)
    assert.equal(detail.occlusionMap,detail.map)
    assert.equal(detail.emissiveMap,detail.map)
    assert.deepEqual(detail.emissive,[2,0.5,0.25])
    assert.equal(detail.occlusionStrength,0.55)
    assert.equal(detail.sampling.normalMap.uvSet,1)
    assert.equal(detail.sampling.normalMap.rotation,0.35)
    assert.deepEqual(detail.normalUVTransform,[1,1,0.1,-0.2])
    assert.equal(detail.sampling.occlusionMap.uvSet,1)
    assert.deepEqual(detail.sampling.map.sampler, {addressModeU:'mirror-repeat',addressModeV:'clamp-to-edge',
      magFilter:'nearest',minFilter:'linear',mipmapFilter:'linear',lodMaxClamp:0})
    assert.ok([...detailed.warnings].every(w => w.includes('without TEXCOORD_1')))
    // Each primitive may use higher-numbered source UVs; only the number of
    // distinct streams actually referenced by this material consumes GPU slots.
    const attributes = external.meshes[0].primitives[1].attributes
    attributes.TEXCOORD_3 = attributes.TEXCOORD_1
    delete attributes.TEXCOORD_1
    surface.normalTexture.extensions.KHR_texture_transform.texCoord = 3
    surface.occlusionTexture.texCoord = 3
    const remapped = await readGltfAsset(local('art/scenes/remapped.gltf', JSON.stringify(external)), [geo,map])
    assert.deepEqual(remapped.mesh.uvs1, detailed.mesh.uvs1)
    assert.equal(remapped.materials.get('0').sampling.normalMap.uvSet, 1)
    assert.ok([...remapped.warnings].some(w => w.includes('without TEXCOORD_3')))

    // Sparse attributes can appear on static geometry, with an implicit
    // zero base (UVs) or as overrides of an authored base stream (colors).
    const sparse = document(), sparseBinary = Buffer.concat([geometry, Buffer.from([2,0,0,0]),
      Buffer.from(new Float32Array([0.5,0.75]).buffer), Buffer.from([255,128,0,255]), png])
    sparse.buffers[0].byteLength = sparseBinary.length
    const view = (byteOffset, byteLength) => sparse.bufferViews.push({ buffer: 0, byteOffset, byteLength }) - 1
    const indices = { bufferView: view(geometry.length, 1), componentType: 5121 }
    const sparseUV = sparse.accessors.push({ count: 3, type: 'VEC2', componentType: 5126,
      sparse: { count: 1, indices, values: { bufferView: view(geometry.length + 4, 8) } } }) - 1
    sparse.accessors[2].sparse = { count: 1, indices, values: { bufferView: view(geometry.length + 12, 4) } }
    sparse.images[0].bufferView = view(geometry.length + 16, png.length)
    sparse.meshes[0].primitives[0].attributes.TEXCOORD_1 = sparseUV
    sparse.materials[0].normalTexture.texCoord = 1
    const sparseAsset = await readGltfAsset(glb(sparse, sparseBinary), [])
    assert.deepEqual([...sparseAsset.mesh.uvs1], [0,1,0,1,0.5,0.25])
    assert.deepEqual([...sparseAsset.mesh.vertices.slice(28,31)], [1,Math.fround(128/255),0])
    sparseBinary[geometry.length] = 3
    await assert.rejects(readGltfAsset(glb(sparse, sparseBinary), []), /Invalid sparse glTF index/)
    external.extensionsRequired = ['KHR_draco_mesh_compression']
    await assert.rejects(readGltfAsset(glb(external, binary), []), /KHR_draco_mesh_compression/)
  } finally {
    await site.close()
  }
})
