import { mat4 } from 'wgpu-matrix'
import { readObjAsset } from '../src/geometry/obj-materials'
import { placeObjOnReceiver } from '../src/geometry/place-obj'
import { buildSceneGeometry } from '../src/rendering/scene-mesh'
import { createSceneRenderer } from '../src/rendering/scene-renderer'
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
  const receiver = { width: 64, height: 64, sizeX: 0.16, sizeY: 0.16, z: 0.5 }
  placeObjOnReceiver(mesh, { sizeX: 0.16, sizeY: 0.16, thickness: 0.01 }, receiver)
  const geometry = buildSceneGeometry(receiver, { mesh })
  const design = buildDesignView(geometry.mesh, receiver)
  const renderer = await createSceneRenderer(device, geometry.mesh, mat4.identity(), materials, w => { throw new Error(w) })
  const extent = [64, 64]
  const colorTarget = (format: GPUTextureFormat) => device.createTexture({ size: extent, format,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC })
  const colors = [colorTarget('rgba8unorm-srgb'), colorTarget('rgba8unorm'), colorTarget('rg32uint')]
  const depth = colorTarget('depth24plus')
  const read = device.createBuffer({ size: 64 * 256, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
  try {
    renderer.update(mat4.scaling([2 / receiver.sizeX, -2 / receiver.sizeY, 1]))
    const encoder = device.createCommandEncoder()
    const pass = encoder.beginRenderPass({
      colorAttachments: colors.map(t => ({ view: t.createView(), loadOp: 'clear', storeOp: 'store' })),
      depthStencilAttachment: { view: depth.createView(), depthLoadOp: 'clear', depthStoreOp: 'discard', depthClearValue: 1 },
    })
    renderer.draw(pass)
    pass.end()
    encoder.copyTextureToBuffer({ texture: colors[0] }, { buffer: read, bytesPerRow: 256 }, extent)
    device.queue.submit([encoder.finish()])
    await read.mapAsync(GPUMapMode.READ)
    const pixels = new Uint8Array(read.getMappedRange())
    const encode = (x: number) => 255 * (x < 0.0031308 ? 12.92 * x : 1.055 * x ** (1/2.4) - 0.055)
    function pixel(x: number, y: number, expected: number[]) {
      const i = y * 64 + x
      const face = design.samples[2 * i] - 1
      if (design.bvh.triangles[12 * face + 3] !== 2) throw new Error('Material boundary split the optical object.')
      const actual = pixels.slice(4 * i, 4 * i + 3)
      if (expected.some((v, c) => Math.abs(v - actual[c]) > 2)) throw new Error('G-buffer RGB at ' + [x,y] +
        ': expected ' + expected.map(Math.round) + ', got ' + actual)
    }
    for (const [x, y, r, g, b] of [
      [19, 27, 228, 101, 77], [43, 27, 52, 166, 136], [19, 51, 61, 104, 191], [43, 51, 232, 181, 64],
    ]) {
      pixel(x, y, [encode(0.5 * ((r/255 + 0.055) / 1.055) ** 2.4), g, b])
    }
    pixel(53, 32, [0.12, 0.15, 0.22].map(encode))
    read.unmap()
    report('PASS Base Color: MTL-relative paths, missing image fallback, four UV corners, linear Kd and a shared optical object')
  } finally {
    renderer.destroy()
    read.destroy()
    depth.destroy()
    colors.forEach(t => t.destroy())
  }
}
