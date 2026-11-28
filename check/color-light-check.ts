import { mat4 } from 'wgpu-matrix'
import { createLens } from '../src/geometry/lens-mesh'
import { parseObjMesh } from '../src/geometry/obj-mesh'
import { createApertureColor } from '../src/optics/aperture-color'
import { buildDesignView } from '../src/optics/design-view'
import { createForwardModel } from '../src/optics/forward-model'
import { createBeamRenderer } from '../src/rendering/beam-renderer'
import { createCausticDisplay } from '../src/rendering/caustic-display'
import { buildSceneGeometry } from '../src/rendering/scene-mesh'
import { createSceneRenderer } from '../src/rendering/scene-renderer'

// A tilted exit face shifts a collimated bundle about 4.8 cm to the right.
// Its flat receiving sheet is larger than the lit footprint, but fully painted.
export async function checkColorLight(device: GPUDevice, report: (line: string) => void) {
  const width = 64, height = 32
  const receiver = { width, height, sizeX: 0.24, sizeY: 0.12, z: 0.4 }
  const plane = parseObjMesh('v -0.12 0.06 0.3\nv 0.12 0.06 0.3\nv 0.12 -0.06 0.3\nv -0.12 -0.06 0.3\nf 1 2 3\nf 1 3 4')
  const geometry = buildSceneGeometry(receiver, { mesh: plane })
  const view = buildDesignView(geometry.mesh, receiver)
  const spec = { columns: 33, rows: 17, sizeX: 0.1, sizeY: 0.06, thickness: 0.03 }
  const heights = Float32Array.from({ length: spec.columns * spec.rows },
    (_, i) => spec.thickness + 0.35 * spec.sizeX * ((i % spec.columns) / (spec.columns - 1) - 0.5))
  const lens = createLens(device, spec, heights)
  const input = { lens, receiver, optics: { lensIOR: 1.5, outsideIOR: 1, incidentIrradiance: 3 } }
  const forward = await createForwardModel(device, input, { designView: view })
  const aperture = createApertureColor(device)
  const room = createBeamRenderer(device, 'rgba8unorm', mat4.identity(), 1, geometry)
  room.setLightPathsVisible(false)
  const scene = await createSceneRenderer(device, geometry.mesh, mat4.identity())
  const vp = mat4.scaling([2 / receiver.sizeX, -2 / receiver.sizeY, 1])
  scene.update(vp)
  const display = createCausticDisplay(device, 'rgba8unorm')
  const output = () => device.createTexture({ size: [width, height], format: 'rgba8unorm',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC })
  const surface = output(), preview = output()
  const bytes = 256 * height
  const staging = device.createBuffer({ size: 2 * bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
  const copy = (e: GPUCommandEncoder, texture: GPUTexture, offset: number) =>
    e.copyTextureToBuffer({ texture }, { buffer: staging, bytesPerRow: 256, offset }, [width, height])
  async function read(encoder: GPUCommandEncoder) {
    device.queue.submit([encoder.finish()])
    await staging.mapAsync(GPUMapMode.READ)
    const pixels = new Uint8Array(staging.getMappedRange().slice(0))
    staging.unmap()
    return pixels
  }
  function expectColor(pixels: Uint8Array, x: number, channel: number, label: string, offset = 0) {
    const i = offset + (width * (height / 2) + x) * 4
    const rgb = pixels.subarray(i, i + 3)
    if (rgb[channel] < 40 || rgb.some((v, c) => c !== channel && v > 0.25 * rgb[channel])) {
      throw new Error(`${label} at x=${x} expected channel ${channel}, received ${rgb}.`)
    }
  }
  device.pushErrorScope('validation')
  try {
    await display.prepare('rgb')
    const initial = device.createCommandEncoder()
    const result = forward.encode(initial, input)
    device.queue.submit([initial.finish()])
    const values = new Float32Array(width * height).fill(1)
    for (const revision of [0, 1]) {
      const colors = new Float32Array(3 * values.length)
      const left = revision === 0 ? 0 : 2, right = 2 - left
      for (let i = 0; i < values.length; i++) colors[3 * i + (i % width < 44 ? left : right)] = 1
      const target = { width, height, version: 1, colorVersion: revision + 1, values, colors }
      const encoder = device.createCommandEncoder()
      const { filter, image } = aperture.update(encoder, result, target, view)
      room.setColorFilter(filter)
      room.render(encoder, surface, vp, 'lit', result, scene.draw)
      display.setImage({ kind: 'rgb', view: image, width, height })
      const pass = encoder.beginRenderPass({ colorAttachments: [
        { view: preview.createView(), loadOp: 'clear', storeOp: 'store' },
      ] })
      display.draw(pass)
      pass.end()
      copy(encoder, surface, 0)
      copy(encoder, preview, bytes)
      const pixels = await read(encoder)
      for (const [label, offset] of [['Room', 0], ['Result', bytes]] as const) {
        expectColor(pixels, 36, left, label, offset)
        expectColor(pixels, 52, right, label, offset)
        const dark = offset + 4 * (width * height / 2 + 18)
        if (pixels.subarray(dark, dark + 3).some(v => v > 8)) throw new Error(label + ' lit an unreachable painted pixel.')
      }
    }
    // Keep the filter fixed and remove the refractive bend. Both patches must
    // follow their source rays left, even though the target has not moved.
    const encoder = device.createCommandEncoder()
    const straight = forward.encode(encoder, { ...input, optics: { ...input.optics, lensIOR: 1 } })
    room.render(encoder, surface, vp, 'lit', straight, scene.draw)
    copy(encoder, surface, 0)
    const moved = await read(encoder)
    expectColor(moved, 24, 2, 'Moved blue beam')
    expectColor(moved, 37, 0, 'Moved red beam')
    report('PASS chromatic caustics: first-hit filter, color-only recolor, RGB Result and Room, and fixed-filter beam motion')
  } finally {
    scene.destroy()
    room.destroy()
    display.destroy()
    forward.destroy()
    aperture.destroy()
    lens.heightBuffer.destroy()
    staging.destroy()
    surface.destroy()
    preview.destroy()
    const error = await device.popErrorScope()
    if (error) throw new Error(error.message)
  }
}
