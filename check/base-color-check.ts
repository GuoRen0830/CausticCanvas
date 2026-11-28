import { mat4 } from 'wgpu-matrix'
import { readObjAsset } from '../src/geometry/obj-materials'
import { placeObjOnReceiver } from '../src/geometry/place-obj'
import { buildSceneGeometry, sceneVertexStride } from '../src/rendering/scene-mesh'
import { createSceneRenderer, type SceneSurfaceMode } from '../src/rendering/scene-renderer'
import { buildDesignView } from '../src/optics/design-view'
import plate from './fixtures/base-color/plate.obj?raw'
import mtl from './fixtures/base-color/materials/plate.mtl?raw'
import chartUrl from './fixtures/base-color/textures/orientation.png?url'

export async function checkBaseColor(device: GPUDevice, report: (line: string) => void) {
  function file(path: string, content: BlobPart) {
    const f = new File([content], path.split('/').at(-1)!)
    Object.defineProperty(f, 'webkitRelativePath', { value: 'sample/' + path })
    return f
  }
  const obj = file('plate.obj', plate), lib = file('materials/plate.mtl', mtl)
  const png = file('textures/orientation.png', await (await fetch(chartUrl)).blob())
  const decoy = file('other/orientation.png', 'wrong folder')
  const absent = await readObjAsset(obj, [obj, lib, decoy])
  if (absent.materials.get('Swatches')?.map || ![...absent.warnings].some(w => w.includes('orientation.png'))) {
    throw new Error('Missing map must use Kd and name the missing file.')
  }
  const { mesh, materials, warnings } = await readObjAsset(obj, [obj, lib, decoy, png])
  const swatches = materials.get('Swatches')!
  if (warnings.size || swatches.map !== png || !swatches.clamp) throw new Error('Incorrect relative map_Kd resolution.')
  swatches.color[0] = 0.5 // Test Kd multiplication between texture decode and output encoding.
  // The same image intentionally serves as both sRGB color and linear data.
  swatches.metallicRoughnessMap = swatches.normalMap = png
  swatches.occlusionMap = png
  swatches.occlusionStrength = 0.5
  swatches.emissive = [2, 0.5, 0.125]
  swatches.roughness = 0.75
  swatches.metallic = 0.6
  const limits = { width: 64, height: 64, sizeX: 0.16, sizeY: 0.16, z: 0.5 }
  const fitted = placeObjOnReceiver(mesh, { sizeX: 0.16, sizeY: 0.16, thickness: 0.01 }, limits)
  // The swatch pixel coordinates are calibrated on a square raster.
  const receiver = { ...limits, z: fitted.receiver.z }
  const geometry = buildSceneGeometry(receiver, { mesh })
  // Ordinary OBJ materials ignore this vertex tint. The wet-sand variant
  // below opts in and interprets red = 0.65 as a saturated surface.
  const swatchDraw = geometry.mesh.draws!.find(draw => draw.material === 'Swatches')!
  for (const i of geometry.mesh.indices.subarray(swatchDraw.firstIndex, swatchDraw.firstIndex + swatchDraw.indexCount)) {
    geometry.mesh.vertices.set([0.65, 1, 1], i * sceneVertexStride + 6)
  }
  const design = buildDesignView(geometry.mesh, receiver)
  const renderer = await createSceneRenderer(device, geometry.mesh, mat4.identity(), materials, w => { throw new Error(w) })
  const extent = [64, 64]
  const colorTarget = (format: GPUTextureFormat) => device.createTexture({ size: extent, format,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC })
  const colors = [colorTarget('rgba8unorm-srgb'), colorTarget('rgba8unorm'), colorTarget('rg32uint'), colorTarget('rg32uint')]
  const depth = colorTarget('depth24plus')
  const layerSize = 64 * 256
  const read = device.createBuffer({ size: 4 * layerSize + 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
  async function capture(mode: SceneSurfaceMode) {
    const encoder = device.createCommandEncoder()
    const pass = encoder.beginRenderPass({
      colorAttachments: colors.map(t => ({ view: t.createView(), loadOp: 'clear', storeOp: 'store' })),
      depthStencilAttachment: { view: depth.createView(), depthLoadOp: 'clear', depthStoreOp: 'discard', depthClearValue: 1 },
    })
    renderer.draw(pass, mode)
    pass.end()
    for (const [layer, attachment] of [0, 1, 3].entries()) {
      encoder.copyTextureToBuffer({ texture: colors[attachment] },
        { buffer: read, offset: layer * layerSize, bytesPerRow: attachment === 3 ? 512 : 256 }, extent)
    }
    encoder.copyTextureToBuffer({ texture: colors[2], origin: [19,51] },
      { buffer: read, offset: 4 * layerSize }, [1,1])
    device.queue.submit([encoder.finish()])
    await read.mapAsync(GPUMapMode.READ)
    const result = new Uint8Array(read.getMappedRange().slice(0))
    read.unmap()
    return result
  }
  try {
    renderer.update(mat4.scaling([2 / receiver.sizeX, -2 / receiver.sizeY, 1]))
    const pixels = await capture('material')
    const encode = (x: number) => 255 * (x < 0.0031308 ? 12.92 * x : 1.055 * x ** (1/2.4) - 0.055)
    function pixel(x: number, y: number, expected: number[], layer = 0) {
      const i = y * 64 + x
      const face = design.samples[2 * i] - 1
      if (design.bvh.triangles[12 * face + 3] !== 2) throw new Error('Material boundary split the optical object.')
      const start = layer * layerSize + (layer === 2 ? 8 : 4) * i
      const actual = pixels.slice(start, start + expected.length)
      if (expected.some((v, c) => Math.abs(v - actual[c]) > 2)) throw new Error('G-buffer layer ' + layer + ' at ' + [x,y] +
        ': expected ' + expected.map(Math.round) + ', got ' + actual)
    }
    for (const [x, y, r, g, b] of [
      [19, 27, 228, 101, 77], [43, 27, 52, 166, 136], [19, 51, 61, 104, 191], [43, 51, 232, 181, 64],
    ]) {
      pixel(x, y, [encode(0.5 * ((r/255 + 0.055) / 1.055) ** 2.4), g, b])
      pixel(x, y, [0.75 * g, 0.6 * b, 255, (255 + r) / 2], 2)
      const offset = 2 * layerSize + 8 * (y * 64 + x) + 4
      const scale = 2 ** (pixels[offset + 3] - 128) / 255
      if (swatches.emissive.some((channel, i) => Math.abs(channel - pixels[offset + i] * scale) > 0.01)) {
        throw new Error('HDR emission changed in the G-buffer.')
      }
    }
    pixel(53, 32, [0.12, 0.15, 0.22].map(encode))
    const normalX = pixels[layerSize + 4 * (51 * 64 + 19)]
    const packedGeometric = new DataView(pixels.buffer).getUint32(4 * layerSize + 4, true)
    if (normalX >= 75 || packedGeometric !== (129 << 16)) {
      throw new Error('Normal mapping must tilt the shading normal and preserve the geometric plane.')
    }
    if (pixels[2 * layerSize + 8 * (32 * 64 + 53) + 2] !== 0) throw new Error('Plain Kd entered the PBR path.')
    const base = await capture('base-color')
    if (base.subarray(0, layerSize).some((v,i) => v !== pixels[i])) throw Error('Base-color mode changed the authored albedo.')
    const sample = 51 * 64 + 19
    const n = base.subarray(layerSize + sample * 4, layerSize + sample * 4 + 3)
    const packed = base.subarray(2 * layerSize + sample * 8, 2 * layerSize + sample * 8 + 8)
    if (n[0] !== 128 || n[1] !== 128 || n[2] !== 1 || [255,0,0,255,0,0,0,0].some((v,i) => v !== packed[i])) {
      throw Error('Base-color mode retained a PBR channel, normal map or emissive term.')
    }
    const restored = await capture('material')
    if (restored.some((v,i) => v !== pixels[i])) throw Error('Re-enabling materials changed the original G-buffer.')
    const wet = new Map(materials)
    wet.set('Swatches', { ...swatches, vertexColor: true, vertexWetness: { saturatedRed: 0.65, saturatedRoughness: 0.173 } })
    await renderer.replaceMaterials(wet, new AbortController().signal)
    const coated = await capture('material'), colorOnly = await capture('base-color')
    const surfaceOffset = 2 * layerSize + 8 * sample
    if (Math.abs(coated[surfaceOffset] - 255*0.173) > 1 || coated[surfaceOffset + 3] !== 255 ||
        [sample, 27*64+19].some(p => [128,128,1].some((v,k) => Math.abs(v - coated[layerSize + 4*p + k]) > 1))) {
      throw Error('Saturated sand should have the water-edge roughness, a smooth surface normal and unoccluded ambient light.')
    }
    if (coated[4*sample] >= pixels[4*sample] ||
        coated.subarray(0, layerSize).some((value,i) => value !== colorOnly[i]) || colorOnly[surfaceOffset] !== 255) {
      throw Error('Base-color comparison should keep the wet tint and disable its gloss.')
    }
    // Red = 0.65 is halfway from the dry endpoint (1) to this wet endpoint (0.30).
    wet.set('Swatches', { ...swatches, vertexColor: true, vertexWetness: { saturatedRed: 0.30, saturatedRoughness: 0.173 } })
    await renderer.replaceMaterials(wet, new AbortController().signal)
    const halfway = await capture('material')
    if (Math.abs(halfway[surfaceOffset] - (pixels[surfaceOffset] + coated[surfaceOffset])/2) > 1) {
      throw Error('A partially wet surface should interpolate its textured roughness.')
    }
    report('PASS materials: relative paths, sRGB albedo, linear roughness/metallic/AO, HDR emission, tangent normal and geometric plane')
  } finally {
    renderer.destroy()
    read.destroy()
    depth.destroy()
    colors.forEach(t => t.destroy())
  }
}
