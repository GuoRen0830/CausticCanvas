import { mat4 } from 'wgpu-matrix'
import { createFlatLens } from '../src/geometry/lens-mesh'
import { createExportTransport } from '../src/rendering/export-transport'
import { buildSceneBvh } from '../src/rendering/scene-bvh'

// One active right triangular beam covers half of a one-square-metre aperture.
// Its 0.5 W flux gives exactly 1 W/m² on a front-facing diffuse receiver.
// The front plate intercepts its lower corner at x + y = -0.5, i.e. exactly
// one quarter of its area. Observe both depth layers independently so that a
// missed blocker or duplicated child footprint cannot hide in the total.
export async function checkExportLightTransport(device: GPUDevice, report: (line: string) => void) {
  const size = 64
  const signal = new AbortController().signal
  const objects: { destroy(): void }[] = []
  function keep<T extends { destroy(): void }>(value: T): T { objects.push(value); return value }
  function upload(data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>, usage: GPUBufferUsageFlags) {
    const buffer = keep(device.createBuffer({ size: data.byteLength, usage: usage | GPUBufferUsage.COPY_DST }))
    device.queue.writeBuffer(buffer, 0, data)
    return buffer
  }
  const target = (format: GPUTextureFormat, usage = 0) => keep(device.createTexture({
    size: [size, size], format, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | usage,
  }))
  device.pushErrorScope('validation')
  try {
    const lens = createFlatLens(device, { columns: 2, rows: 2, sizeX: 1, sizeY: 1, thickness: 0.02 })
    keep(lens.heightBuffer)
    const vp = mat4.scaling([2, -2, 1])
    const data = new Float32Array(44)
    const words = new Uint32Array(data.buffer)
    data.set(vp); data.set(mat4.inverse(vp), 16)
    words.set([2, 2], 32)
    data.set([1, 1, 0, 1, size, size], 34)
    data[41] = 1; data[43] = 0.02
    const frame = upload(data, GPUBufferUsage.UNIFORM)
    const albedo = target('rgba8unorm').createView()
    const normal = target('rgba8unorm').createView()
    const object = target('rg32uint').createView()
    const material = target('rg32uint').createView()
    const depth = target('depth24plus').createView()
    const receiverZ = 0.6, blockerZ = 0.3

    async function create(scene: 'split' | 'sliver', direction: number[]) {
      const vertices: number[] = [], indices: number[] = []
      function polygon(points: number[][], z: number, id: number) {
        const first = vertices.length / 11
        for (const [x, y] of points) vertices.push(x, y, z, 0, 0, -1, 1, 1, 1, 0, id)
        for (let i = 1; i + 1 < points.length; i++) indices.push(first, first + i, first + i + 1)
      }
      polygon([[-3, -2], [-3, 5], [5, -2]], receiverZ, 1)
      if (scene === 'split') {
        polygon([[-1, -1], [-1, 0.5], [0.5, -1], [0.5, -1.25]], blockerZ, 2)
      } else {
        // None of the four tilted sub-beam centers touch this small plate.
        // Only a receiver-to-aperture visibility segment can see its shadow.
        polygon([[-0.29, -0.42], [-0.29, -0.32], [-0.24, -0.32], [-0.24, -0.42]], blockerZ, 2)
        // A backward visibility ray must stop at the lens before this large plate.
        polygon([[-3, -2], [-3, 5], [5, -2]], -0.15, 3)
      }
      const bvh = buildSceneBvh({ vertices: Float32Array.from(vertices), indices: Uint32Array.from(indices) })
      const buffer = upload(Float32Array.of(...direction, 0.5, 0, 0, 1, 0), GPUBufferUsage.STORAGE)
      return keep(await createExportTransport(device, { buffer, lens }, upload(bvh.triangles, GPUBufferUsage.STORAGE),
        upload(bvh.nodes, GPUBufferUsage.STORAGE), undefined, signal))
    }
    type Transport = Awaited<ReturnType<typeof create>>
    async function capture(transport: Transport, z: number, id: number, metal = false) {
      const encoder = device.createCommandEncoder()
      const attachment = (view: GPUTextureView, clearValue: GPUColor): GPURenderPassColorAttachment =>
        ({ view, clearValue, loadOp: 'clear', storeOp: 'store' })
      // G-buffer layout from scene.wgsl. Red is unity albedo (and lamp); the
      // lit shader's red channel is linear direct radiance with no tint factor.
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          attachment(albedo, [1, 1, 1, 1]), attachment(normal, [128/255, 128/255, 1/255, 1]),
          attachment(object, [id, 0x810000, 0, 0]),
          attachment(material, [metal ? 0xffffff59 : 0xff0000ff, 0, 0, 0]),
        ],
        depthStencilAttachment: { view: depth, depthClearValue: z, depthLoadOp: 'clear', depthStoreOp: 'store' },
      })
      pass.end()
      device.queue.submit([encoder.finish()])
      const output = target(transport.format, GPUTextureUsage.COPY_SRC)
      await transport.render(frame, { albedo, normal, object, material, depth, direct: output.createView(),
        width: size, height: size },
        vp, true, signal)

      const bytesPerRow = size * 16
      const read = keep(device.createBuffer({ size: bytesPerRow * size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }))
      const copy = device.createCommandEncoder()
      copy.copyTextureToBuffer({ texture: output }, { buffer: read, bytesPerRow }, [size, size])
      device.queue.submit([copy.finish()])
      await read.mapAsync(GPUMapMode.READ)
      const data = new Float32Array(read.getMappedRange())
      const red = Float32Array.from({ length: size * size }, (_, i) => data[i * 4])
      read.unmap()
      if (red.some(v => !(v >= 0 && Number.isFinite(v)))) throw Error('Nonfinite or negative exported radiance')
      return red
    }

    const divided = await create('split', [0, 0, 1])
    const front = await capture(divided, blockerZ, 2)
    const back = await capture(divided, receiverZ, 1)
    const flux = (pixels: Float32Array) => Math.PI * pixels.reduce((sum, v) => sum + v, 0) / (size * size)
    // Finite receiver edges cut through pixel centers. Allow one pixel of
    // edge integration error, independently on the foreground and background.
    if (Math.abs(flux(front) - 0.125) > 0.006 || Math.abs(flux(back) - 0.375) > 0.006 ||
        Math.abs(flux(front) + flux(back) - 0.5) > 0.006) {
      throw Error(`Split transport has incorrect flux: front=${flux(front)}, back=${flux(back)} W`)
    }
    const corner = 8 + size * 8, middle = 24 + size * 24
    if (front[corner] < 0.3 || back[corner] > 1e-5 || back[middle] < 0.3 || front[middle] > 1e-5) {
      throw Error('A partially occluded export beam did not resolve both depth layers')
    }

    const faceOn = await capture(divided, receiverZ, 1, true)
    const tiltedTransport = await create('sliver', [0.6, 0, 0.8])
    const tilted = await capture(tiltedTransport, receiverZ, 1, true)
    // This pixel is inside both projected footprints, outside the blocker.
    const i = 35 + size * 20
    const roughness = 89 / 255
    const peak = 1 / (4 * Math.PI * roughness ** 4)
    // V is normal to the plate and L is tilted with NoL = 0.8. Its
    // perpendicular flux density is 1/NoL, which cancels the incidence.
    const a2 = roughness ** 4, nl = 0.8, nh2 = (1 + nl) / 2
    const distribution = a2 / (Math.PI * (1 - nh2 + nh2 * a2) ** 2)
    const oblique = distribution * 0.5 / (nl + Math.sqrt(nl * nl * (1 - a2) + a2))
    if (Math.abs(faceOn[i] - peak) > 0.03 * peak || Math.abs(tilted[i] - oblique) > 0.03 * oblique) {
      throw Error(`Export GGX disagrees with collimated ray direction: normal=${faceOn[i]}, tilted=${tilted[i]}`)
    }
    // The blocker casts an oblique shadow across x in [-.065,-.015].
    if (tilted[29 + size * 8] > 1e-5 || Math.abs(tilted[33 + size * 8] - oblique) > 0.03 * oblique) {
      throw Error('The export shadow segment missed a fine blocker or continued beyond the lens')
    }
  } finally {
    for (const resource of objects.reverse()) resource.destroy()
    const validation = await device.popErrorScope()
    if (validation) throw Error(validation.message)
  }
  report('PASS direct transport: foreground 1/8 W, background 3/8 W, fine occluder, finite shadows and directional GGX')
}
