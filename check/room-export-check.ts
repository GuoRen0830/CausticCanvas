import { mat4 } from 'wgpu-matrix'
import { createFlatLens } from '../src/geometry/lens-mesh'
import { buildDesignView } from '../src/optics/design-view'
import { createForwardModel } from '../src/optics/forward-model'
import { renderRoomExport } from '../src/rendering/room-export'
import { createSceneRenderer } from '../src/rendering/scene-renderer'
import type { SceneGeometry } from '../src/rendering/scene-mesh'
import { encodeExportPng } from '../src/ui/image-export'

// A flat lens supplies parallel rays to a polished planar plate. Seen through
// an orthographic camera, its direct specular response must be uniform across
// the plate, including far from the center of the lens.
export async function checkRoomExport(device: GPUDevice, report: (line: string) => void) {
  const receiver = { width: 24, height: 16, sizeX: 0.16, sizeY: 0.12, z: 0.07 }
  const lens = createFlatLens(device, { ...receiver, columns: 9, rows: 9, thickness: 0.004 })
  const mesh = {
    vertices: Float32Array.from([[-0.08,0.06],[0.08,0.06],[0.08,-0.06],[-0.08,-0.06]].flatMap(
      ([x,y]) => [x, y, receiver.z, 0, 0, -1, 1, 1, 1, 0, 2])),
    indices: Uint32Array.of(0,1,2, 0,2,3),
    draws: [{ firstIndex: 0, indexCount: 6, material: 'metal' }],
  }
  const geometry: SceneGeometry = { mesh, materials: new Map([['metal', {
    color: [0.6,0.6,0.6], roughness: 0.35, metallic: 1,
  }]]) }
  const design = buildDesignView(mesh, receiver)
  const optics = { lensIOR: 1.5, outsideIOR: 1, incidentIrradiance: 1 }
  const input = { lens, receiver, optics }
  const forward = await createForwardModel(device, input, { designView: design })
  const scene = await createSceneRenderer(device, mesh, mat4.identity(), geometry.materials)
  const filter = device.createTexture({ size: [2,2], format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST })
  // Top row: red, green. Bottom row: blue, white. Row zero of the artist's
  // filter is at optical -Y, also the top edge of this particular projection.
  device.queue.writeTexture({ texture: filter }, Uint8Array.of(
    255,0,0,255, 0,255,0,255, 0,0,255,255, 255,255,255,255), { bytesPerRow: 8 }, [2,2])

  const capture = async (buffer: GPUBuffer) => {
    const read = device.createBuffer({ size: buffer.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    try {
      const e = device.createCommandEncoder()
      e.copyBufferToBuffer(buffer, 0, read, 0, buffer.size)
      device.queue.submit([e.finish()])
      await read.mapAsync(GPUMapMode.READ)
      return new Uint8Array(read.getMappedRange().slice(0))
    } finally { read.destroy() }
  }
  device.pushErrorScope('validation')
  try {
    const e = device.createCommandEncoder()
    const result = forward.encode(e, input)
    device.queue.submit([e.finish()])
    if (!result.firstHits || result.caustic.kind !== 'buffer') throw Error('Export check requires scene optical output.')
    const borrowed = [lens.heightBuffer, result.beams.buffer, result.firstHits.buffer, result.caustic.buffer]
    const before = await Promise.all(borrowed.map(capture))
    const source = { geometry, opticalToWorld: mat4.identity(), result, scene,
      exposure: 0.15, environmentLight: 0, incidentIrradiance: 1, relativeIOR: 1.5, lightPathsVisible: false }
    const vp = mat4.scaling([2/receiver.sizeX, -2/receiver.sizeY, 1])
    const signal = new AbortController().signal
    const request = { width: 73, height: 49, viewProjection: vp, signal }
    const white = await renderRoomExport(device, source, request)
    const pixel = (image: ImageData, x: number, y: number, channel: number) =>
      image.data[(y * image.width + x) * 4 + channel]
    const middle = pixel(white, 37, 23, 0)
    if (middle < 32 || Math.abs(middle - pixel(white, 13, 23, 0)) > 3 ||
        Math.abs(middle - pixel(white, 58, 23, 0)) > 3) {
      throw Error('Collimated light on the polished plate acquired a lens-center highlight or disappeared.')
    }

    const colored = { ...source, colorFilter: filter.createView() }
    const tiled = await renderRoomExport(device, colored, { ...request, tileSize: 24 })
    const single = await renderRoomExport(device, colored, request)
    const base = await renderRoomExport(device, { ...colored, materialsEnabled: false }, request)
    const bareRed = pixel(base, 18, 12, 0)
    if (bareRed < 5 || bareRed + 8 >= pixel(single, 18, 12, 0) ||
        pixel(base, 18, 12, 1) > 3 || pixel(base, 18, 12, 2) > 3) {
      throw Error('The base-color export should keep red painted illumination and remove the polished highlight.')
    }
    if (tiled.width !== 73 || tiled.height !== 49 || tiled.data.length !== 73*49*4 ||
        tiled.data.some((v,i) => i % 4 === 3 ? v !== 255 : Math.abs(v-single.data[i]) > 2)) {
      throw Error('Tiled export changed the image, inserted row padding or lost edge pixels.')
    }
    for (const [x,y,dominant] of [[18,12,0],[55,12,1],[18,37,2]]) {
      for (const c of [0,1,2]) {
        const value = pixel(tiled, x, y, c)
        if (c === dominant ? value < 32 : value > 3) throw Error('Aperture colors or exported row orientation changed.')
      }
    }
    const png = await encodeExportPng(tiled, signal)
    const bitmap = await createImageBitmap(png)
    try {
      if (png.type !== 'image/png' || bitmap.width !== tiled.width || bitmap.height !== tiled.height) {
        throw Error('Invalid PNG format or size.')
      }
      const ctx = new OffscreenCanvas(bitmap.width, bitmap.height).getContext('2d')!
      ctx.drawImage(bitmap, 0, 0)
      const rgba = ctx.getImageData(0, 0, bitmap.width, bitmap.height).data
      if (rgba.some((v,i) => Math.abs(v-tiled.data[i]) > 1)) throw Error('PNG encoding changed the Room colors.')
    } finally { bitmap.close() }

    // From this angle the plate is edge-on. Row 9 crosses the prism and
    // row 24 crosses only the colored air between the prism and the plate.
    const side = mat4.create(10,0,0,0, 0,0,3,0, 0,-18,0,0, 0,0.65,0.5,1)
    const sideRequest = { ...request, viewProjection: side }
    const dark = await renderRoomExport(device, colored, sideRequest)
    const paths = { ...colored, lightPathsVisible: true }
    const visible = await renderRoomExport(device, paths, sideRequest)
    const visibleTiled = await renderRoomExport(device, paths, { ...sideRequest, tileSize: 24 })
    for (const row of [9, 24]) {
      const start = row * request.width * 4
      if (!visible.data.subarray(start, start + request.width * 4)
          .some((v,i) => Math.abs(v - dark.data[start+i]) > 1)) {
        throw Error('The enabled prism or refracted light was missing from the export.')
      }
    }
    if (visible.data.some((v,i) => Math.abs(v - visibleTiled.data[i]) > 2)) {
      throw Error('Splitting the volume integration into tiles introduced a seam.')
    }

    const stop = new AbortController()
    try {
      await renderRoomExport(device, colored, { ...request, tileSize: 24, signal: stop.signal,
        onProgress: (fraction, stage) => {
          if (fraction > 0.2 && fraction < 1 && stage?.startsWith('Rendering tile')) stop.abort()
        } })
      throw Error('Export published an image after cancellation.')
    } catch (error) {
      if (!(error instanceof DOMException) || error.name !== 'AbortError') throw error
    }
    const after = await Promise.all(borrowed.map(capture))
    if (before.some((bytes,b) => bytes.some((v,i) => v !== after[b][i]))) {
      throw Error('Export or cancellation altered a borrowed optical buffer.')
    }
    report('PASS Room export: parallel-ray highlight, painted RGB, prism and volume, odd-sized tile seams, PNG round trip, cancellation and optical state')
  } finally {
    filter.destroy()
    scene.destroy()
    forward.destroy()
    lens.heightBuffer.destroy()
    const error = await device.popErrorScope()
    if (error) throw Error(error.message)
  }
}
