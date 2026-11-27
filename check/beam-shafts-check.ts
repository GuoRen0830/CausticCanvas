import { mat4 } from 'wgpu-matrix'
import type { ForwardResult } from '../src/contracts'
import { createFlatLens } from '../src/geometry/lens-mesh'
import { createBeamShafts } from '../src/rendering/beam-shafts'

// Side view of real beam records. Every lit ray travels rightward from z=0.15
// to the first-hit plane at z=0.55. A foreground band can interrupt it midway.
export async function checkBeamShafts(device: GPUDevice, report: (line: string) => void) {
  const size = 128
  const spec = { columns: 17, rows: 17, sizeX: 0.06, sizeY: 0.04, thickness: 0.15 }
  const lens = createFlatLens(device, spec)
  const count = 2 * (spec.columns - 1) * (spec.rows - 1)
  const rays = new Float32Array(4 * count)
  const hits = new Uint32Array(2 * count)
  const distances = new Float32Array(hits.buffer)
  const irradiance = 3
  const dz = 1 / Math.hypot(1, 0.25)
  for (let i = 0; i < count; i++) {
    rays.set([0.25 * dz, 0, dz, irradiance * lens.sizeX * lens.sizeY / count], 4 * i)
    hits[2 * i] = 1
    distances[2 * i + 1] = (0.55 - spec.thickness) / dz
  }
  const storage = (bytes: number) => device.createBuffer({ size: bytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
  const rayBuffer = storage(rays.byteLength), hitBuffer = storage(hits.byteLength)
  const offsets = storage(8 * lens.columns * lens.rows)
  const triangles = storage(48), image = storage(4)
  const receiver = { width: 1, height: 1, sizeX: 0.3, sizeY: 0.2, z: 0.75 }
  const result: ForwardResult = { receiver, beams: { lens, buffer: rayBuffer },
    caustic: { width: 1, height: 1, kind: 'buffer', buffer: image },
    firstHits: { mesh: {}, buffer: hitBuffer, triangles } }
  device.queue.writeBuffer(triangles, 0, new Float32Array([-1,-1,0.55,0, -1,2,0.55,0, 2,-1,0.55,0]))
  device.queue.writeBuffer(image, 0, new Float32Array([2 * irradiance * lens.sizeX * lens.sizeY / (receiver.sizeX * receiver.sizeY)]))
  device.queue.writeBuffer(rayBuffer, 0, rays)
  device.queue.writeBuffer(hitBuffer, 0, hits)
  const overlay = device.createTexture({ size: [size, size], format: 'rgba16float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC })
  const depth = device.createTexture({ size: [size, size], format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING })
  const overlayView = overlay.createView(), depthView = depth.createView()
  const readback = device.createBuffer({ size: size * size * 8,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
  const shafts = createBeamShafts(device)
  const vp = mat4.create(
    6, 0, 0, 0,   0, 0, 1, 0,   0, -3, 0, 0,   0, 1.2, 0.5, 1,
  )
  const opaque = device.createRenderPipeline({
    layout: 'auto', vertex: { module: device.createShaderModule({ code: `
      @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        let p = array(vec2f(-1,-1), vec2f(3,-1), vec2f(-1,3));
        return vec4f(p[i], 0.1, 1);
      }` }), entryPoint: 'vs' },
    depthStencil: { format: 'depth24plus', depthCompare: 'always', depthWriteEnabled: true },
  })

  async function render(occluder: boolean, hasDrawing: boolean, power = irradiance) {
    const encoder = device.createCommandEncoder()
    const pass = encoder.beginRenderPass({ colorAttachments: [], depthStencilAttachment: { view: depthView,
      depthLoadOp: 'clear', depthClearValue: 1, depthStoreOp: 'store' } })
    if (occluder) {
      pass.setPipeline(opaque)
      pass.setScissorRect(0, 48, size, 16)
      pass.draw(3)
    }
    pass.end()
    const accents = encoder.beginRenderPass({ colorAttachments: [
      { view: overlayView, loadOp: 'clear', clearValue: [0, 0, 0, 0], storeOp: 'store' },
    ], depthStencilAttachment: { view: depthView, depthReadOnly: true } })
    shafts.draw(accents, vp, size, size, result, offsets, power, hasDrawing)
    accents.end()
    encoder.copyTextureToBuffer({ texture: overlay }, { buffer: readback, bytesPerRow: size * 8 }, [size, size])
    device.queue.submit([encoder.finish()])
    await readback.mapAsync(GPUMapMode.READ)
    const pixels = Float32Array.from(new Uint16Array(readback.getMappedRange()), bits => {
      const exponent = bits >> 10
      if (exponent >= 31) return NaN
      return ((bits & 1023) + (exponent ? 1024 : 0)) * 2 ** (Math.max(1, exponent) - 25)
    })
    readback.unmap()
    return pixels
  }
  // Sum radiance and locate the column carrying it. Alpha belongs to glass,
  // so the additive light path overlay must leave it zero everywhere.
  function band(pixels: Float32Array, y0: number, y1: number) {
    let light = 0, moment = 0
    for (let y = y0; y < y1; y++) for (let x = 0; x < size; x++) {
      const p = 4 * (y * size + x), value = pixels[p] + pixels[p + 1] + pixels[p + 2]
      if (!Number.isFinite(value) || value < 0 || pixels[p + 3] !== 0) throw new Error('Invalid RGB or alpha in the light ribbons.')
      light += value
      moment += value * x
    }
    return { light, x: moment / light }
  }
  try {
    const lit = await render(false, true)
    const start = band(lit, 24, 31), end = band(lit, 77, 84)
    if (!(start.light > 0 && end.light > 0 && end.x - start.x > 19) ||
        band(lit, 0, 10).light || band(lit, 102, size).light) {
      throw new Error('Light ribbons do not follow the refracted path from the exit to the first hit.')
    }
    const obscured = await render(true, true)
    if (!(band(lit, 50, 62).light > 0) || band(obscured, 50, 62).light ||
        band(obscured, 24, 31).light !== start.light || band(obscured, 77, 84).light !== end.light) {
      throw new Error('The foreground band did not hide only the occluded section of the light ribbons.')
    }
    const flood = band(await render(false, false), 0, size).light, drawn = band(lit, 0, size).light
    if (!(flood > 0 && flood < drawn)) throw new Error('Expected subtler path accents for a blank drawing.')
    device.queue.writeBuffer(image, 0, new Float32Array([0]))
    if (!(band(await render(false, true), 0, size).light < drawn * 0.5)) throw new Error('Paths into dark spill should be dimmer.')
    if (band(await render(false, true, 0), 0, size).light) throw new Error('Light paths remained with the lamp off.')
    device.queue.writeBuffer(rayBuffer, 0, new Float32Array(rays.length))
    if (band(await render(false, true), 0, size).light) throw new Error('A zero-flux beam produced a light ribbon.')
    device.queue.writeBuffer(rayBuffer, 0, rays)
    device.queue.writeBuffer(hitBuffer, 0, hits.fill(0))
    if (band(await render(false, true), 0, size).light) throw new Error('A ray with no surface hit produced a light ribbon.')
    report('PASS light ribbons: refraction, first hits, per-pixel occlusion, endpoint irradiance, blank and zero-flux states')
  } finally {
    shafts.destroy()
    for (const resource of [overlay, depth, readback, rayBuffer, hitBuffer, triangles, image, offsets, lens.heightBuffer]) resource.destroy()
  }
}
