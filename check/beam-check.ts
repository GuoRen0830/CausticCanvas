import { mat4 } from 'wgpu-matrix'
import { createFlatLens } from '../src/geometry/lens-mesh'
import { createForwardModel } from '../src/optics/forward-model'
import { createBeamRenderer } from '../src/rendering/beam-renderer'
import { createSceneRenderer } from '../src/rendering/scene-renderer'
import type { BeamSource, BeamView } from '../src/rendering/beam-renderer'
import type { ForwardResult } from '../src/contracts'
import { checkSceneOcclusion } from './scene-occlusion'
import { checkSceneTraining } from './scene-training'
import { checkFirstHits } from './first-hit-check'

// Use the production geometry, optical solver, accumulation and comparison passes.
// Float output at low exposure retains enough precision to invert the display map.
export async function checkBeams(device: GPUDevice, report: (line: string) => void) {
  const lensSpec = { columns: 33, rows: 33, sizeX: 0.2, sizeY: 0.2, thickness: 0.01 }
  const receiver = { sizeX: 0.4, sizeY: 0.32, z: 0.5, width: 128, height: 80 }
  const model = mat4.multiply(mat4.translation([0, receiver.z, 0]), mat4.rotationX(Math.PI / 2))
  const lens = createFlatLens(device, lensSpec)
  const input = { lens, receiver, optics: { lensIOR: 1.5, outsideIOR: 1, incidentIrradiance: 1 } }
  const forward = (await createForwardModel(device, input))!
  const exposure = 0.0001
  const box = { center: [0.05, 0.03, 0.3], size: [0.08, 0.08, 0.1] } as const
  const beam = createBeamRenderer(device, 'rgba32float', lensSpec, receiver, model, exposure, box)
  const scene = createSceneRenderer(device, receiver, box, model)
  // View along optical +Z with screen-right +X and screen-down +Y. The
  // receiver then faces the camera and a screen row equals an optical row.
  const clip = mat4.scaling([2 / receiver.sizeX, -2 / receiver.sizeY, 1])
  const vp = mat4.multiply(clip, mat4.inverse(model))
  const { width, height } = receiver
  const output = device.createTexture({ format: 'rgba32float', size: [width, height],
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC })
  const staging = device.createBuffer({ size: width * height * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
  let result: ForwardResult | null = null
  const blank = () => {}
  function evaluate() {
    const encoder = device.createCommandEncoder()
    result = forward.encode(encoder, input)
    device.queue.submit([encoder.finish()])
    return result
  }
  async function readView(source: BeamSource, mode: BeamView, camera = vp) {
    const encoder = device.createCommandEncoder()
    scene.update(camera)
    beam.render(encoder, output, camera, source, mode, result, scene.draw, blank, 'analytic')
    encoder.copyTextureToBuffer({ texture: output }, { buffer: staging, bytesPerRow: width * 16 }, [width, height])
    device.queue.submit([encoder.finish()])
    await staging.mapAsync(GPUMapMode.READ)
    const rgba = new Float32Array(staging.getMappedRange())
    const values = Float32Array.from({ length: width * height }, (_, i) => {
      const encoded = rgba[4 * i]
      if (mode === 'lit') return encoded // Linear composite red, no display map.
      const mapped = encoded <= 0.04045 ? encoded / 12.92 : ((encoded + 0.055) / 1.055) ** 2.4
      return mapped / ((1 - mapped) * exposure)
    })
    staging.unmap()
    return values
  }
  function require(ok: boolean, message: string) { if (!ok) throw new Error(message) }
  try {
    const single = await readView('single', 'footprint')
    const pair = await readView('pair', 'footprint')
    const singles = single.filter(v => Math.abs(v - 2.4) < 0.006).length
    const doubles = pair.filter(v => Math.abs(v - 4.8) < 0.012).length
    require(singles > 50 && doubles > 10, 'Missing analytic single or doubled overlap')
    require(pair.every((v, i) => v >= single[i] - 0.006 &&
      [0, 2.4, 4.8].some(expected => Math.abs(v - expected) < 0.012)), 'Unexpected additive irradiance')
    report('PASS test beams: ' + singles + ' single-beam pixels at 2.4 W/m², ' + doubles + ' overlap pixels at 4.8 W/m²')

    async function compare(name: string, current: ForwardResult, tolerance: number) {
      if (current.caustic.kind !== 'buffer') throw new Error('Expected f32 reference buffer')
      const encoder = device.createCommandEncoder()
      encoder.copyBufferToBuffer(current.caustic.buffer, 0, staging, 0, width * height * 4)
      device.queue.submit([encoder.finish()])
      await staging.mapAsync(GPUMapMode.READ)
      const ref = new Float32Array(staging.getMappedRange().slice(0, width * height * 4))
      staging.unmap()
      const plane = await readView('solver', 'plane')
      const difference = await readView('solver', 'difference')
      let sum = 0, refSum = 0, absSum = 0, max = 0
      for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
        const i = y * width + x
        const expected = ref[i]
        const error = Math.abs(plane[i] - expected)
        require(Math.abs(difference[i] - error) < 1e-4, 'Error view disagrees at ' + i)
        sum += plane[i]; refSum += expected; absSum += error; max = Math.max(max, error)
      }
      const dA = receiver.sizeX * receiver.sizeY / (width * height)
      const relativeL1 = absSum / Math.max(refSum, 1e-20)
      require(relativeL1 < tolerance && max < 0.04, name + ' mismatch: ' + relativeL1 + ' / ' + max)
      report('PASS ' + name + ': beam ' + (sum * dA).toFixed(7) + ' W, solver ' + (refSum * dA).toFixed(7)
        + ' W, relative L1 ' + relativeL1.toExponential(3) + ', max ΔE ' + max.toExponential(3) + ' W/m²')
      return { sum: sum * dA, plane }
    }

    const flat = await compare('flat lens', evaluate(), 0.0015)
    require(Math.abs(flat.sum - 0.04) < 0.00008, 'Flat lens power')
    const screen = await readView('solver', 'footprint')
    // Every visible pixel center in this sub-rectangle lies within the aperture,
    // including shared diagonals; sampling it avoids the box.
    for (let y = 26; y < 44; y++) for (let x = 35; x < 58; x++) {
      require(Math.abs(screen[y * width + x] - 1) < 0.002, 'Flat screen hole or double count')
    }
    report('PASS visible flat floor irradiance including shared-edge ownership')

    const slopes = Float32Array.from({ length: lens.columns * lens.rows }, (_, i) => {
      const x = ((i % lens.columns) / (lens.columns - 1) - 0.5) * lens.sizeX
      const y = (Math.floor(i / lens.columns) / (lens.rows - 1) - 0.5) * lens.sizeY
      return 0.025 + 0.06 * x - 0.09 * y - 0.8 * x * x - 1.1 * y * y + 0.3 * x * y
    })
    device.queue.writeBuffer(lens.heightBuffer, 0, slopes)
    const curved = await compare('asymmetric curved lens', evaluate(), 0.003)
    require(curved.plane.some((v, i) => Math.abs(v - flat.plane[i]) > 1), 'Stale beam or height field')
    // A tilted plane translates all footprints together. Its interior has known
    // uniform irradiance, so a center sample and a texel integral must agree.
    device.queue.writeBuffer(lens.heightBuffer, 0, Float32Array.from(slopes, (_, i) =>
      0.025 + 0.06 * (i % lens.columns) * lens.sizeX / (lens.columns - 1)
            - 0.09 * Math.floor(i / lens.columns) * lens.sizeY / (lens.rows - 1)))
    const tilted = await compare('tilted lens', evaluate(), 0.002)
    const visible = await readView('solver', 'footprint')
    const blocked = await readView('solver', 'blocked')
    let matched = 0
    for (let y = 1; y < height - 1; y++) for (let x = 1; x < width - 1; x++) {
      const i = y * width + x
      if (x >= 66 && x <= 94 && y >= 36 && y <= 61) continue // Box silhouette.
      const value = tilted.plane[i]
      if (value < 0.5 || [-width, -1, 1, width].some(d => Math.abs(tilted.plane[i + d] - value) > 0.01)) continue
      require(Math.abs(visible[i] + blocked[i] - value) < 0.01, 'Tilted visible + blocked floor/plane mismatch')
      matched++
    }
    require(matched > 500, 'Not enough visible tilted-floor samples')
    report('PASS tilted visible + blocked irradiance matches the plane at ' + matched + ' samples')
    const reference = result!
    await checkSceneOcclusion(device, receiver, model, async (field, mode, camera) => {
      result = { ...reference, beams: field }
      return readView('solver', mode, camera)
    }, report)
    input.optics.incidentIrradiance = 0
    await compare('zero flux', evaluate(), 0.001)
    require((await readView('solver', 'footprint')).every(v => v === 0), 'Stale screen illumination')
    // One aperture-sized receiver texel receives 8192 equal contributions.
    // f16 blending stalls at 0.25 even though the correct sum is 1 W/m2.
    const denseSpec = { ...lensSpec, columns: 65, rows: 65 }
    const smallReceiver = { ...receiver, sizeX: lensSpec.sizeX, sizeY: lensSpec.sizeY, width: 1, height: 1 }
    const denseLens = createFlatLens(device, denseSpec)
    const denseInput = { lens: denseLens, receiver: smallReceiver, optics: { ...input.optics, incidentIrradiance: 1 } }
    const referenceModel = (await createForwardModel(device, denseInput))!
    try {
      const encoder = device.createCommandEncoder()
      result = referenceModel.encode(encoder, denseInput)
      device.queue.submit([encoder.finish()])
      const center = Math.floor(height/2)*width + Math.floor(width/2)
      const reconstructed = (await readView('solver', 'plane'))[center]
      const residual = (await readView('solver', 'difference'))[center]
      require(Math.abs(reconstructed - 1) < 0.0001 && residual < 0.0001, 'Dense receiver f32 accumulation')
      report('PASS 8192 subpixel beams in one receiver texel, E = ' + reconstructed.toFixed(6) + ' W/m2')
    } finally { referenceModel.destroy(); denseLens.heightBuffer.destroy() }
    await checkFirstHits(device, report)
    await checkSceneTraining(device, report)
  } finally {
    staging.destroy(); output.destroy(); beam.destroy(); scene.destroy(); forward.destroy(); lens.heightBuffer.destroy()
  }
}
