import { mat4 } from 'wgpu-matrix'
import { sceneLamp } from '../src/config'
import { parseObjMesh } from '../src/geometry/obj-mesh'
import { buildDesignView } from '../src/optics/design-view'
import { uniformIlluminationHeights } from '../src/optimization/uniform-illumination'
import { buildSceneGeometry } from '../src/rendering/scene-mesh'
import { createSceneSession } from '../src/scene-session'
import { reference } from './optics-reference'

// A raised, full-width paintable sheet. A lens aimed at the floor instead
// would leave a dark border on this sheet. A flat lens lights only its center.
function raisedSheet(width: number, height: number) {
  const receiver = { width, height, sizeX: 0.2, sizeY: 0.12, z: 0.22 }
  const sheetZ = 0.145
  const model = parseObjMesh([
    `v -0.1 0.06 ${sheetZ}`, `v 0.1 0.06 ${sheetZ}`,
    `v 0.1 -0.06 ${sheetZ}`, `v -0.1 -0.06 ${sheetZ}`,
    'f 1 2 3', 'f 1 3 4',
  ].join('\n'))
  const geometry = buildSceneGeometry(receiver, { mesh: model })
  return { receiver, geometry, view: buildDesignView(geometry.mesh, receiver), sheetZ }
}

export function checkUniformIllumination(report: (line: string) => void) {
  const { receiver, view, sheetZ } = raisedSheet(16, 12)
  const { lens: spec, optics } = sceneLamp(receiver)
  const lens = { ...spec, columns: 60, rows: 47 }
  const heights = uniformIlluminationHeights(lens, optics, receiver, view)
  if (heights.length !== lens.rows * lens.columns ||
      !heights.every(h => Number.isFinite(h) && h > 0 && h < sheetZ)) {
    throw new Error('Initial exit surface is invalid.')
  }
  const { image } = reference({ lens, optics, receiver: { ...receiver, z: sheetZ } }, heights)
  const mean = optics.incidentIrradiance * lens.sizeX * lens.sizeY / (receiver.sizeX * receiver.sizeY)
  const received = image.reduce((sum, e) => sum + e, 0) / (mean * image.length)
  const relativeRms = Math.sqrt(image.reduce((sum, e) => sum + (e / mean - 1) ** 2, 0) / image.length)
  const dimmest = Math.min(...image) / mean
  if (received < 0.94 || received > 1.001 || relativeRms > 0.22 || dimmest < 0.45) {
    throw new Error(`Raised sheet flood: received ${received}, relative RMS ${relativeRms}, dimmest ${dimmest}`)
  }
  // Background pixels have no receiving area. A stale or absent depth at
  // those locations must not redirect the light on the visible geometry.
  const masked = { samples: view.samples.slice(), surfaceAreas: view.surfaceAreas.slice() }
  const depths = new Float32Array(masked.samples.buffer)
  for (let i = 0; i < masked.surfaceAreas.length; i += 3) {
    masked.surfaceAreas[i] = 0
    masked.samples[2 * i] = 0
    depths[2 * i + 1] = 0
  }
  const withoutDepth = uniformIlluminationHeights(lens, optics, receiver, masked)
  for (let i = 0; i < masked.surfaceAreas.length; i += 3) depths[2 * i + 1] = receiver.z * 2
  const withDepth = uniformIlluminationHeights(lens, optics, receiver, masked)
  if (!withoutDepth.every((h, i) => h === withDepth[i])) throw new Error('Background depth changed the flood lens.')
  report(`PASS initial flood on a raised rectangular sheet: ${(100 * received).toFixed(1)}% received, ${(100 * relativeRms).toFixed(1)}% relative RMS`)
}

// Exercises the artist's normal entry points at the production lens resolution.
// A clear or undo-to-empty reaches render with zero scheduled optimizer steps.
export async function checkUniformReset(device: GPUDevice, report: (line: string) => void) {
  const { receiver, geometry, view } = raisedSheet(64, 48)
  const format = 'rgba8unorm'
  const session = await createSceneSession(device, format, mat4.identity(), geometry, receiver, view)
  const texture: GPUTextureDescriptor = { format, size: [receiver.width, receiver.height],
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC }
  const output = device.createTexture(texture), preview = device.createTexture(texture)
  const staging = device.createBuffer({ size: 256 * receiver.height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
  try {
    session.setLightPathsVisible(false)
    const vp = mat4.scaling([2 / receiver.sizeX, -2 / receiver.sizeY, 1])
    const render = async (steps: number) => {
      const encoder = device.createCommandEncoder()
      session.render(encoder, output, preview, vp, 'lit', steps)
      encoder.copyTextureToBuffer({ texture: preview }, { buffer: staging, bytesPerRow: 256 },
        [receiver.width, receiver.height])
      device.queue.submit([encoder.finish()])
      await staging.mapAsync(GPUMapMode.READ)
      const pixels = new Uint8Array(staging.getMappedRange().slice(0))
      staging.unmap()
      return pixels
    }
    const original = session.state.lens.heightBuffer
    const initial = await render(0)
    const values = Float32Array.from({ length: receiver.width * receiver.height },
      (_, i) => Math.abs(i % receiver.width - receiver.width / 2) < 5 ? 1 : 0)
    const target = { width: receiver.width, height: receiver.height, version: 1, values }
    session.setTarget(target)
    const painted = await render(1)
    if (session.state.lens.heightBuffer === original ||
        !painted.some((v, i) => Math.abs(v - initial[i]) > 12)) throw new Error('The painted preview did not change.')
    session.setTarget({ ...target, version: 2, values: new Float32Array(values.length) })
    const cleared = await render(0)
    if (session.state.lens.heightBuffer !== original || session.state.lens.offsetBuffer ||
        cleared.some((v, i) => Math.abs(v - initial[i]) > 2)) {
      throw new Error('Clearing the last stroke must restore the initial lens and flood image on the next draw.')
    }
    report('PASS scene startup and painting followed by zero-step undo/clear to uniform illumination')
  } finally {
    staging.destroy()
    output.destroy()
    preview.destroy()
    session.destroy()
  }
}
