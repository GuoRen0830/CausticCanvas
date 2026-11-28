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
  async function render(mode: BeamView, environment = 0, camera = vp, surface = scene, lighting = beams, materialsEnabled = true) {
    lighting.setEnvironmentLight(environment)
    surface.update(camera)
    const draw = device.createCommandEncoder()
    lighting.render(draw, output, camera, mode, result, surface.draw, false, materialsEnabled)
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
    for (const mode of ['irradiance', 'candidates', 'normals', 'geometric-normals', 'albedo', 'normal-map',
      'roughness', 'metallic', 'occlusion', 'emissive', 'map-coverage'] as const) {
      beams.setLightPathsVisible(false)
      const raw = await render(mode)
      beams.setLightPathsVisible(true)
      if (!equal(raw, await render(mode))) throw new Error(`Light paths leaked into ${mode}.`)
      const withFill = await render(mode, 1)
      if (!equal(raw, withFill)) throw new Error(`Environment light leaked into ${mode}.`)
      // This untextured fixture legitimately shows black for missing maps, metal and emission.
      if (!['normal-map','map-coverage','metallic','emissive'].includes(mode) && brightness(raw) === 0) {
        throw new Error(`Empty ${mode} preview.`)
      }
    }

    // Draw the same optical plate at two roughnesses and two camera angles.
    // The shear pivots around the plate's depth, keeping its projected pixels fixed.
    const oblique = mat4.copy(vp)
    oblique[8] = 1.25 * vp[0]
    oblique[12] = -oblique[8] * mesh.vertices[2]
    const metal = (roughness: number) => new Map([['Swatches', {
      color: [0.36, 0.36, 0.36] as [number, number, number], metallic: 1, roughness,
    }]])
    const polished = metal(0.16), rough = metal(0.9)
    const plated = await createSceneRenderer(device, geometry.mesh, mat4.identity(), polished)
    let matte: typeof plated | undefined, pbr: typeof beams | undefined
    try {
      matte = await createSceneRenderer(device, geometry.mesh, mat4.identity(), rough)
      pbr = createBeamRenderer(device, 'rgba8unorm', mat4.identity(), 0.30, { ...geometry, materials: polished })
      pbr.setReferenceIrradiance(input.optics.incidentIrradiance)
      pbr.setLightPathsVisible(false)
      const center = (image: Uint8Array) => {
        let red = 0
        for (let y = 28; y < 36; y++) for (let x = 28; x < 36; x++) red += image[(y * 64 + x) * 4]
        return red / 64
      }
      const onAxis = center(await render('lit', 0, vp, plated, pbr))
      const offAxis = center(await render('lit', 0, oblique, plated, pbr))
      const roughCenter = center(await render('lit', 0, vp, matte, pbr))
      const plain = await render('lit', 0, vp, plated, pbr, false)
      if (!equal(plain, await render('lit', 0, vp, matte, pbr, false)) || center(plain) + 4 >= onAxis) {
        throw new Error('The base-color preview changed with roughness or retained the polished highlight.')
      }
      if (onAxis < offAxis + 4 || onAxis < roughCenter + 4) {
        throw new Error(`Expected a view-dependent polished highlight, got on-axis ${onAxis}, oblique ${offAxis}, rough ${roughCenter}.`)
      }
      const incident = await render('irradiance', 0, vp, plated, pbr)
      const roughIncident = await render('irradiance', 0, vp, matte, pbr)
      if (!equal(incident, roughIncident)) throw new Error('Roughness changed incident irradiance.')
      report('PASS PBR presentation: polished highlight tracks the eye, roughness broadens it, and irradiance is unchanged')

      async function swatch(rgb: string) {
        const tile = new OffscreenCanvas(1, 1)
        const paint = tile.getContext('2d')!
        paint.fillStyle = rgb
        paint.fillRect(0, 0, 1, 1)
        return new File([await tile.convertToBlob()], 'swatch.png', { type: 'image/png' })
      }
      const occlusionMap = await swatch('#000')
      const signal = new AbortController().signal
      const source = rough.get('Swatches')!
      await matte.replaceMaterials(new Map([['Swatches', { ...source, occlusionMap }]]), signal)
      if (Math.abs(center(await render('lit', 0, vp, matte, pbr)) - roughCenter) > 1) {
        throw new Error('Ambient occlusion attenuated painted light.')
      }
      pbr.setExposure(0)
      const occluded = center(await render('lit', 0.5, vp, matte, pbr))
      await matte.replaceMaterials(rough, signal)
      if (center(await render('lit', 0.5, vp, matte, pbr)) <= occluded + 4) {
        throw new Error('Ambient occlusion did not attenuate the hemisphere fill.')
      }
      const unlit = center(await render('lit', 0, vp, matte, pbr))
      await matte.replaceMaterials(new Map([['Swatches', { ...source, emissive: [2, 0.5, 0.125] }]]), signal)
      if (center(await render('lit', 0, vp, matte, pbr)) < unlit + 64) {
        throw new Error('Source emission disappeared when the scene lighting was off.')
      }
      report('PASS material relighting: AO affects ambient fill only and emission survives with both lights off')

      const checkCenter = async (mode: BeamView, expected: number[]) => {
        const bytes = await render(mode, 1, vp, matte, pbr)
        const actual = bytes.slice(4*(32*64+32), 4*(32*64+32)+3)
        if (expected.some((v,c) => Math.abs(actual[c]-v) > 2)) {
          throw new Error(`Expected ${mode} RGB ${expected}, got ${actual}.`)
        }
      }
      await checkCenter('normal-map', [0,0,0])
      await checkCenter('map-coverage', [0,0,0])
      const normalMap = await swatch('#cc80e6'), orm = await swatch('#4080c0')
      await matte.replaceMaterials(new Map([['Swatches', { ...source, normalMap, roughness: 0.5, metallic: 0.8,
        metallicRoughnessMap: orm, occlusionMap: orm, occlusionStrength: 0.5 }]]), signal)
      await checkCenter('normal-map', [204,128,230])
      await checkCenter('roughness', [64,64,64])
      await checkCenter('metallic', [154,154,154])
      await checkCenter('occlusion', [160,160,160])
      await checkCenter('emissive', [0,0,0])
      await checkCenter('map-coverage', [255,255,255])
      await checkCenter('geometric-normals', [128,255,128])
      const shading = await render('normals', 1, vp, matte, pbr)
      if (!equal(shading, await render('normals', 1, vp, matte, pbr, false))) {
        throw new Error('Switching Room materials hid the normal map from Debug.')
      }
      if (equal(shading, await render('geometric-normals', 1, vp, matte, pbr))) {
        throw new Error('The mapped shading normal was lost in the normal views.')
      }
      await matte.replaceMaterials(new Map([['Swatches', { ...source, roughnessMap: orm }]]), signal)
      await checkCenter('map-coverage', [0,255,0])
      report('PASS material views: raw normal, separate face and shading normals, effective PBR parameters and texture coverage')
    } finally {
      plated.destroy(); matte?.destroy(); pbr?.destroy()
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
