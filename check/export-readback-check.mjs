import assert from 'node:assert/strict'
import { test } from 'node:test'
import { mat4 } from 'wgpu-matrix'
import { createServer } from 'vite'

// The transport and renderer execute their normal host code. Supply known
// shader output at the texture readback boundary so coordinates and padding
// can be checked byte-for-byte on machines without a WebGPU adapter.
function readbackDevice(onMap, hardwareBlend) {
  const live = new Set(), scopes = [], tiles = []
  const attempts = []
  const noop = () => {}
  const descriptor = value => value
  const pass = { setPipeline: noop, setBindGroup: noop, draw: noop, dispatchWorkgroups: noop, end: noop }
  let origin = [0, 0]
  function track(resource) {
    live.add(resource)
    resource.destroy = () => {
      assert.ok(!resource.mapped, 'readback must be unmapped before disposal')
      assert.ok(live.delete(resource), 'destroyed a resource twice')
    }
    return resource
  }
  const device = {
    features: new Set(hardwareBlend ? ['float32-blendable'] : []),
    limits: { maxTextureDimension2D: 4096, maxStorageBufferBindingSize: 2 ** 27 },
    queue: { submit: noop, onSubmittedWorkDone: async () => {}, writeBuffer(buffer, _offset, data) {
      assert.ok(live.has(buffer), 'modified a borrowed input')
      if (buffer.label === 'Export image coordinates') origin = Array.from(data)
    } },
    pushErrorScope(filter) { scopes.push(filter) },
    async popErrorScope() { assert.ok(scopes.pop()); return null },
    createBuffer(desc) {
      return track({ ...desc, bytes: new Uint8Array(desc.size), mapped: false,
        async mapAsync() { assert.ok(!this.mapped); this.mapped = true; if (this.containsPixels) onMap(tiles.length) },
        getMappedRange() { assert.ok(this.mapped); return this.bytes.buffer },
        unmap() { assert.ok(this.mapped); this.mapped = false },
      })
    },
    createTexture: desc => track({ ...desc, createView() { return this } }),
    createShaderModule: descriptor, createRenderPipelineAsync: descriptor, createComputePipelineAsync: descriptor,
    createRenderPipeline: desc => ({ ...desc, getBindGroupLayout: i => desc.layout.bindGroupLayouts[i] }),
    createBindGroupLayout: descriptor, createPipelineLayout: descriptor, createBindGroup: descriptor,
    createCommandEncoder: () => ({
      beginRenderPass: () => pass, beginComputePass: () => pass, finish: noop,
      clearBuffer(buffer) { buffer.bytes.fill(0) },
      copyBufferToBuffer(a, i, b, j, count) {
        b.bytes.set(a.bytes.subarray(i, i + count), j)
        if (b.label === 'Export contention readback') {
          // One contested tile must be retried before its pixels are published.
          if (!attempts.length) b.bytes[0] = 1
          attempts.push(origin.slice(0, 2))
        }
      },
      copyTextureToBuffer(_texture, { buffer, bytesPerRow }, [w, h]) {
        const [x, y] = origin
        assert.equal(bytesPerRow % 256, 0)
        assert.ok(bytesPerRow >= 4*w && bytesPerRow * h <= buffer.bytes.length)
        tiles.push([x, y, w, h])
        buffer.containsPixels = true
        buffer.bytes.fill(253) // Guard bytes between rows and beyond the edge tile.
        for (let j = 0; j < h; j++) {
          for (let i = 0; i < w; i++) buffer.bytes.set([x+i, y+j, (x+i) ^ (y+j), 255], j*bytesPerRow + i*4)
        }
      },
    }),
  }
  return { device, live, scopes, tiles, attempts }
}

async function setupRoom(t) {
  const globals = {
    ImageData: class {
      constructor(width, height) { Object.assign(this, { width, height, data: new Uint8ClampedArray(4*width*height) }) }
    },
    GPUMapMode: { READ: 1 }, GPUShaderStage: { COMPUTE: 4, VERTEX: 1, FRAGMENT: 2 },
    GPUBufferUsage: { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128 },
    GPUTextureUsage: { COPY_SRC: 1, TEXTURE_BINDING: 4, RENDER_ATTACHMENT: 16 },
  }
  for (const [key, value] of Object.entries(globals)) {
    const previous = Object.getOwnPropertyDescriptor(globalThis, key)
    Object.defineProperty(globalThis, key, { configurable: true, value })
    t.after(() => { if (previous) Object.defineProperty(globalThis, key, previous); else delete globalThis[key] })
  }
  const site = await createServer({ configFile: false, server: { middlewareMode: true, hmr: false, watch: null } })
  t.after(() => site.close())
  const borrowed = { destroy() { assert.fail('destroyed borrowed scene resources') } }
  const mesh = {}
  const lens = { rows: 2, columns: 2, sizeX: 1, sizeY: 1, heightBuffer: borrowed, offsetBuffer: borrowed }
  const room = { geometry: { mesh, materials: new Map([['metal', { color: [0.6, 0.6, 0.6], metallic: 1 }]]) },
    opticalToWorld: mat4.identity(), scene: { draw() {} },
    result: { beams: { lens, buffer: borrowed }, firstHits: { mesh, buffer: borrowed, nodes: borrowed, triangles: borrowed } },
    lightPathsVisible: false, exposure: 1, environmentLight: 0, incidentIrradiance: 1, relativeIOR: 1.5 }
  return { site, room }
}

test('Room renderer assembles padded RGBA tiles, reports progress and frees an interrupted mapped readback', async t => {
  const { site, room } = await setupRoom(t)
  const { renderRoomExport } = await site.ssrLoadModule('/src/rendering/room-export.ts')
  for (const mode of ['hardware', 'software', 'cancel']) {
    const cancel = mode === 'cancel'
    const stop = new AbortController()
    const { device, scopes, live, tiles, attempts } = readbackDevice(tile => {
      if (cancel && tile === 2) stop.abort()
    }, mode !== 'software')
    const progress = []
    const width = 137, height = 133
    const output = renderRoomExport(device, room, { width, height, tileSize: 70, viewProjection: mat4.identity(),
      signal: stop.signal, onProgress: f => progress.push(f) })
    if (cancel) {
      await assert.rejects(output, { name: 'AbortError' })
      assert.equal(tiles.length, 2)
      assert.ok(progress.at(-1) < 1)
    } else {
      const image = await output
      assert.deepEqual(tiles, [[0,0,69,67], [69,0,68,67], [0,67,69,66], [69,67,68,66]])
      const expected = new Uint8ClampedArray(4*width*height)
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) expected.set([x, y, x ^ y, 255], 4*(y*width + x))
      }
      assert.deepEqual(image.data, expected)
      assert.equal(progress.at(-1), 1)
      if (mode === 'software') assert.deepEqual(attempts, [[0,0], [0,0], [69,0], [0,67], [69,67]])
    }
    assert.equal(progress[0], 0)
    assert.ok(progress.every((f, i) => f >= 0 && f <= 1 && (i === 0 || f >= progress[i-1])))
    assert.equal(live.size, 0)
    assert.deepEqual(scopes, [])
  }
})

test('Room and export use the selected material pass while Debug continues to inspect all maps', async t => {
  const { site, room } = await setupRoom(t)
  const { createBeamRenderer } = await site.ssrLoadModule('/src/rendering/beam-renderer.ts')
  const { renderRoomExport } = await site.ssrLoadModule('/src/rendering/room-export.ts')
  const { device, live } = readbackDevice(() => {}, true)
  const renderer = createBeamRenderer(device, 'rgba8unorm', room.opticalToWorld, room.exposure, room.geometry)
  const output = { width: 9, height: 7, createView() { return this } }
  const viewProjection = mat4.identity()
  const draws = []
  room.scene.draw = (_pass, mode) => draws.push(mode)
  try {
    renderer.setLightPathsVisible(false)
    function preview(view, enabled) {
      renderer.render(device.createCommandEncoder(), output, viewProjection, view, room.result, room.scene.draw, false, enabled)
    }
    preview('lit')
    preview('lit', false)
    preview('normal-map', false)
    preview('roughness', false)
    preview('lit', false)
    preview('lit', true)
    assert.deepEqual(draws, ['material', 'base-color', 'map-inspection', 'material', 'base-color', 'material'])

    const request = { width: 9, height: 7, tileSize: 4, viewProjection, signal: new AbortController().signal }
    for (const enabled of [undefined, false, true]) {
      draws.length = 0
      await renderRoomExport(device, { ...room, materialsEnabled: enabled }, request)
      // 3 columns and 2 rows, all with the same mode even at the cropped edges.
      assert.deepEqual(draws, Array(6).fill(enabled === false ? 'base-color' : 'material'))
    }
  } finally { renderer.destroy() }
  assert.equal(live.size, 0)
})
