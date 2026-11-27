import { mat4 } from 'wgpu-matrix'
import { createFlatLens } from '../src/geometry/lens-mesh'
import { parseObjMesh } from '../src/geometry/obj-mesh'
import { placeObjOnReceiver } from '../src/geometry/place-obj'
import { buildDesignView } from '../src/optics/design-view'
import { createForwardModel } from '../src/optics/forward-model'
import { createBeamRenderer, type BeamView } from '../src/rendering/beam-renderer'
import { buildSceneGeometry } from '../src/rendering/scene-mesh'
import { createSceneRenderer } from '../src/rendering/scene-renderer'
import { checkBeamVolume } from './beam-volume-check'
import { checkBeamShafts } from './beam-shafts-check'
import plate from './fixtures/base-color/plate.obj?raw'

// Pixel readback verifies the presentation controls affect the intended views
// while the shared lens, beams, first hits and solver image remain byte-identical.
export async function checkPresentation(device: GPUDevice, report: (line: string) => void) {
  const limits = { width: 64, height: 64, sizeX: 0.16, sizeY: 0.16, z: 0.5 }
  const spec = { columns: 17, rows: 17, sizeX: 0.16, sizeY: 0.16, thickness: 0.01 }
  const lens = createFlatLens(device, spec)
  const heights = Float32Array.from({ length: 17 * 17 }, (_, i) =>
    spec.thickness + 0.001 * Math.sin((i % 17) * 0.4) * Math.cos(Math.floor(i / 17) * 0.3))
  device.queue.writeBuffer(lens.heightBuffer, 0, heights)
  const mesh = parseObjMesh(plate)
  const { receiver } = placeObjOnReceiver(mesh, spec, limits)
  const input = { lens, receiver, optics: { lensIOR: 1.5, outsideIOR: 1, incidentIrradiance: 3 } }
  const geometry = buildSceneGeometry(receiver, { mesh })
  const forward = await createForwardModel(device, input, { designView: buildDesignView(geometry.mesh, receiver) })
  const scene = await createSceneRenderer(device, geometry.mesh, mat4.identity())
  const beams = createBeamRenderer(device, 'rgba8unorm', mat4.identity(), 1, geometry)
  const output = device.createTexture({ size: [64, 64], format: 'rgba8unorm',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC })
  const staging = device.createBuffer({ size: 64 * 256, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
  const vp = mat4.scaling([2 / receiver.sizeX, -2 / receiver.sizeY, 1])
  beams.setReferenceIrradiance(input.optics.incidentIrradiance)
  const encoder = device.createCommandEncoder()
  const result = forward.encode(encoder, input)
  device.queue.submit([encoder.finish()])

  async function readBuffer(buffer: GPUBuffer) {
    const copy = device.createCommandEncoder()
    copy.copyBufferToBuffer(buffer, 0, staging, 0, buffer.size)
    device.queue.submit([copy.finish()])
    await staging.mapAsync(GPUMapMode.READ)
    const bytes = new Uint8Array(staging.getMappedRange(0, buffer.size).slice(0))
    staging.unmap()
    return bytes
  }
  async function render(mode: BeamView, environment = 0, camera = vp) {
    beams.setEnvironmentLight(environment)
    scene.update(camera)
    const draw = device.createCommandEncoder()
    beams.render(draw, output, camera, mode, result, scene.draw)
    draw.copyTextureToBuffer({ texture: output }, { buffer: staging, bytesPerRow: 256 }, [64, 64])
    device.queue.submit([draw.finish()])
    await staging.mapAsync(GPUMapMode.READ)
    const bytes = new Uint8Array(staging.getMappedRange().slice(0))
    staging.unmap()
    return bytes
  }
  const equal = (a: Uint8Array, b: Uint8Array) => a.length === b.length && a.every((v, i) => v === b[i])
  const brightness = (image: Uint8Array) => image.reduce((sum, v, i) => sum + (i % 4 < 3 ? v : 0), 0)
  device.pushErrorScope('validation')
  try {
    if (result.caustic.kind !== 'buffer' || !result.firstHits) throw new Error('Expected scene optical buffers.')
    const buffers = [lens.heightBuffer, result.beams.buffer, result.firstHits.buffer, result.caustic.buffer]
    const before: Uint8Array[] = []
    for (const buffer of buffers) before.push(await readBuffer(buffer))
    beams.setLightPathsVisible(false)
    const dark = await render('lit'), filled = await render('lit', 0.05)
    if (brightness(filled) <= brightness(dark)) throw new Error('Environment light did not brighten the room.')
    // Row 25 cuts through the glass shell. Row 48 samples air below the exit.
    const side = mat4.create(10, 0, 0, 0, 0, 0, 3, 0, 0, -40, 0, 0, 0, 0.4, 0.5, 1)
    const hidden = await render('lit', 0.05, side)
    beams.setLightPathsVisible(true)
    const visible = await render('lit', 0.05, side)
    for (const [row, label] of [[25, 'prism'], [48, 'refracted light']] as const) {
      const start = row * 256, end = start + 256
      if (equal(hidden.subarray(start, end), visible.subarray(start, end))) {
        throw new Error(`The Room toggle did not show the ${label}.`)
      }
    }
    beams.setLightPathsVisible(false)
    if (!equal(hidden, await render('lit', 0.05, side))) throw new Error('Hiding the light setup did not restore the Room image.')
    for (const mode of ['irradiance', 'candidates', 'normals', 'albedo'] as const) {
      beams.setLightPathsVisible(false)
      const raw = await render(mode)
      beams.setLightPathsVisible(true)
      if (!equal(raw, await render(mode))) throw new Error(`Light paths leaked into ${mode}.`)
      const withFill = await render(mode, 1)
      if (!equal(raw, withFill)) throw new Error(`Environment light leaked into ${mode}.`)
      if (brightness(raw) === 0) throw new Error(`Empty ${mode} preview.`)
    }
    for (let i = 0; i < buffers.length; i++) {
      if (!equal(before[i], await readBuffer(buffers[i]))) throw new Error(`Presentation modified optical buffer ${i}.`)
    }
    report('PASS presentation: shared prism and light paths toggle, debug isolation, environment fill and unchanged optical buffers')
    await checkBeamVolume(device, result, report)
    await checkBeamShafts(device, report)
  } finally {
    beams.destroy(); scene.destroy(); forward.destroy()
    output.destroy(); staging.destroy(); lens.heightBuffer.destroy()
    const error = await device.popErrorScope()
    if (error) throw new Error(error.message)
  }
}
