import assert from 'node:assert/strict'
import { test } from 'node:test'
import { createServer } from 'vite'
import { mat4 } from 'wgpu-matrix'

// Exercise the real renderer's upload and publication order with a delayed image decoder.
test('replacement keeps the last complete material set on cancellation, decode failure and out-of-order completion', async () => {
  const site = await createServer({ configFile: false, server: { middlewareMode: true, hmr: false, watch: null } })
  const original = new Map(['GPUBufferUsage','GPUTextureUsage','createImageBitmap'].map(k => [k,globalThis[k]]))
  const resource = props => Object.assign({ destroy() { this.destroyed = true } },props)
  const scopes = []
  let allocationFailure
  const device = {
    limits: { maxTextureDimension2D: 2048 },
    queue: {
      writeBuffer(buffer,_,data) { buffer.data = data.slice() }, writeTexture() {}, copyExternalImageToTexture() {},
    },
    pushErrorScope(filter) { scopes.push({ filter, error: null }) },
    async popErrorScope() { return scopes.pop().error },
    createBuffer: resource, createTexture(desc) {
      if (desc.label === 'too-large.png') {
        scopes.findLast(s => s.filter === 'out-of-memory').error = { message: 'Out of GPU memory' }
        allocationFailure = resource(desc)
        allocationFailure.createView = () => allocationFailure
        return allocationFailure
      }
      return resource({ ...desc, createView() { return this } })
    },
    createBindGroup: x => x, createSampler: x => x, createShaderModule: x => x,
    createRenderPipeline: () => ({ getBindGroupLayout: i => i }),
  }
  let releaseSlow, renderer
  try {
    globalThis.GPUBufferUsage = { VERTEX: 1, INDEX: 2, UNIFORM: 4, COPY_DST: 8 }
    globalThis.GPUTextureUsage = { RENDER_ATTACHMENT: 1, TEXTURE_BINDING: 2, COPY_DST: 4 }
    globalThis.createImageBitmap = async file => {
      if (file.name === 'broken.png') throw Error('invalid bytes')
      if (file.name === 'slow.png') await new Promise(r => { releaseSlow = r })
      return { width: 1, height: 1, close() {} }
    }
    const { createSceneRenderer } = await site.ssrLoadModule('/src/rendering/scene-renderer.ts')
    const tri = { vertices: new Float32Array(33), indices: Uint32Array.of(0,1,2), uvs: new Float32Array(6),
      uvs1: new Float32Array(6), draws: [3,1].map(uvMask => ({ firstIndex: 0, indexCount: 3, material: 'paint', uvMask })) }
    const chart = mat => new Map([['paint',{ color: [1,1,1], ...mat }]])
    const next = (mat, signal = new AbortController().signal) => renderer.replaceMaterials(chart(mat), signal)
    const group = () => {
      const groups = []
      renderer.draw({ setPipeline() {}, setVertexBuffer() {}, setIndexBuffer() {}, drawIndexed() {},
        setBindGroup(i,g) { if (i === 1) groups.push(g) } })
      return groups
    }
    renderer = await createSceneRenderer(device, tri, mat4.identity(), chart({roughness:0.9}))
    const originalGroup = group()[0], slow = new File(['data'],'slow.png')
    const pending = next({ map:slow })
    assert.equal(group()[0], originalGroup)
    assert.equal(scopes.length,0, 'GPU scopes cannot span the asynchronous image decode')
    const shared = new File(['data'],'orm-emission.png')
    const sampler = { addressModeU:'mirror-repeat', addressModeV:'clamp-to-edge', minFilter:'nearest' }
    await next({ roughness:0.25, occlusionMap:shared, emissiveMap:shared, emissive:[2,1,0], occlusionStrength:0.6,
      sampling:{ occlusionMap:{ uvSet:1, sampler } } })
    const active = group(), byBinding = (g,i) => g.entries.find(e => e.binding === i).resource
    const data = byBinding(active[0],0).buffer.data
    assert.equal(originalGroup.entries[0].resource.buffer.destroyed,true)
    assert.equal(data[4],0.25)
    assert.deepEqual([...data.slice(12,16)],[2,1,0,Math.fround(0.6)])
    assert.equal(data[8],32|64)
    assert.equal(byBinding(active[1],0).buffer.data[8],64)
    assert.equal(byBinding(active[0],11).format,'rgba8unorm')
    assert.equal(byBinding(active[0],13).format,'rgba8unorm-srgb')
    assert.equal(byBinding(active[0],12).addressModeU,'mirror-repeat')
    assert.equal(byBinding(active[0],12).minFilter,'nearest')
    assert.deepEqual(renderer.mapCoverage(['occlusionMap']), { materials:1, mapped:1, triangles:2, mappedTriangles:1 })
    assert.deepEqual(renderer.mapCoverage(['roughnessMap','metallicRoughnessMap']), { materials:1, mapped:0, triangles:2, mappedTriangles:0 })
    releaseSlow()
    await assert.rejects(pending, { name:'AbortError' })
    const cancel = new AbortController()
    const aborted = next({ map:slow },cancel.signal)
    cancel.abort()
    releaseSlow()
    await assert.rejects(aborted, { name:'AbortError' })
    await assert.rejects(next({ map:new File(['!'],'broken.png') }), /Cannot decode/)
    await assert.rejects(next({ map:new File(['data'],'too-large.png') }), /Out of GPU memory/)
    assert.equal(allocationFailure.destroyed,true)
    assert.equal(scopes.length,0)
    assert.equal(group()[0], active[0])
    assert.equal(renderer.mapCoverage(['emissiveMap']).mappedTriangles, 2)
    await next({ emissive: [2,0,0] })
    assert.equal(byBinding(group()[0],0).buffer.data[9],1, 'an emissive factor enables shading without a texture')
  } finally {
    renderer?.destroy()
    for (const [k,v] of original) if (v === undefined) delete globalThis[k]; else globalThis[k] = v
    await site.close()
  }
})

test('the top-light design view uses back faces only on double-sided draws', async () => {
  const site = await createServer({ configFile: false, server: { middlewareMode: true, hmr: false, watch: null } })
  try {
    const { buildDesignView } = await site.ssrLoadModule('/src/optics/design-view.ts')
    const vertices = new Float32Array([[-1,-1], [1,-1], [0,1]].flatMap(([x,y]) => [x,y,1, 0,0,1, 1,1,1, 0,2]))
    const mesh = { vertices, indices: Uint32Array.of(0,1,2), draws: [{ firstIndex: 0, indexCount: 3 }] }
    const receiver = { width: 1, height: 1, sizeX: 1, sizeY: 1, z: 2 }
    assert.equal(buildDesignView(mesh, receiver).samples[0],0)
    mesh.draws[0].doubleSided = true
    const view = buildDesignView(mesh, receiver)
    assert.equal(view.bvh.triangles[11],1)
    assert.equal(view.samples[0],1)
    assert.equal(view.surfaceAreas[0],1)
  } finally { await site.close() }
})
