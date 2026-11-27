import type { ForwardResult } from '../contracts'
import apertureUV from './shaders/aperture-uv.wgsl?raw'
import volumeShader from './shaders/beam-volume-build.wgsl?raw'

const resolution = [144, 144, 96] as const
// Stable lens-space strata. Leave headroom for the optimizer during training,
// then oversample the last evaluation to resolve the narrowest folds.
const samplesPerSide = 192
const trainingSamplesPerSide = 160
type Point = [number, number, number]

// The short axis retains a little space around narrow receivers for stray rays.
// The incident column, above optical z = 0, is rendered separately.
export function beamVolumeBounds(result: ForwardResult, receiverZ: number): { min: Point; max: Point } {
  const { lens } = result.beams
  const width = Math.max(lens.sizeX, result.receiver.sizeX)
  const height = Math.max(lens.sizeY, result.receiver.sizeY)
  const margin = 0.10 * Math.max(width, height)
  const x = 0.5 * width + margin, y = 0.5 * height + margin
  return { min: [-x, -y, 0], max: [x, y, receiverZ] }
}

export function createBeamVolume(device: GPUDevice) {
  const [width, height, depth] = resolution
  const texture = device.createTexture({
    label: 'Refracted light in the medium', dimension: '3d', size: [width, height, depth],
    format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  })
  const textureView = texture.createView()
  const sampler = device.createSampler({ minFilter: 'linear', magFilter: 'linear' })
  const whiteFilter = device.createTexture({
    label: 'White beam filter', size: [1, 1], format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  })
  device.queue.writeTexture({ texture: whiteFilter }, new Uint8Array([255, 255, 255, 255]), {}, [1, 1])
  const whiteView = whiteFilter.createView()
  const planeBytes = 4 * width * height * depth
  const accumulator = device.createBuffer({
    label: 'Volume RGB fluence-rate accumulator', size: 3 * planeBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })
  // Two half-float pairs store RGB and a broader neighbourhood reference.
  // White light only needs the first plane in each scratch buffer.
  const filteredRows = device.createBuffer({
    label: 'Transversely filtered beam rows', size: 2 * planeBytes, usage: GPUBufferUsage.STORAGE,
  })
  const params = device.createBuffer({
    label: 'Beam volume parameters', size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  // Plane-only forward models omit BVH records. Their rays reach the volume boundary.
  const emptyHits = device.createBuffer({ label: 'Unused volume hit data', size: 48, usage: GPUBufferUsage.STORAGE })
  const module = device.createShaderModule({ label: 'Refracted beam volume', code: apertureUV + '\n' + volumeShader })
  const inject = device.createComputePipeline({
    label: 'Deposit refracted beams', layout: 'auto', compute: { module, entryPoint: 'inject' },
  })
  const filterRows = device.createComputePipeline({
    label: 'Filter refracted beam rows', layout: 'auto', compute: { module, entryPoint: 'filterRows' },
  })
  const resolve = device.createComputePipeline({
    label: 'Resolve refracted beam volume', layout: 'auto', compute: { module, entryPoint: 'resolve' },
  })
  const rowGroup = device.createBindGroup({
    layout: filterRows.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: params } }, { binding: 6, resource: { buffer: accumulator } },
      { binding: 8, resource: { buffer: filteredRows } },
    ],
  })
  const resolveGroup = device.createBindGroup({
    layout: resolve.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: params } },
      { binding: 7, resource: textureView }, { binding: 8, resource: { buffer: filteredRows } },
    ],
  })
  const data = new ArrayBuffer(80)
  const floats = new Float32Array(data), words = new Uint32Array(data)
  words.set(resolution, 16)
  let boundBuffers: GPUBuffer[] = []
  let boundFilter: GPUTextureView | undefined
  let injectGroup: GPUBindGroup

  return {
    textureView, sampler,
    update(encoder: GPUCommandEncoder, result: ForwardResult, offsets: GPUBuffer, receiverZ: number,
      training = false, colorFilter?: GPUTextureView) {
      const { lens, buffer: rays } = result.beams
      const hitBuffer = result.firstHits?.buffer ?? emptyHits
      const triangles = result.firstHits?.triangles ?? emptyHits
      const buffers = [params, lens.heightBuffer, offsets, rays, hitBuffer, triangles, accumulator]
      const filter = colorFilter ?? whiteView
      if (boundFilter !== filter || buffers.some((buffer, index) => buffer !== boundBuffers[index])) {
        boundFilter = filter
        boundBuffers = buffers
        injectGroup = device.createBindGroup({
          layout: inject.getBindGroupLayout(0),
          entries: [...buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
            { binding: 9, resource: filter }],
        })
      }

      // Fixed strata give a stationary image when the camera or the lens moves.
      // Keep enough samples on the narrow axis for thin, textured footprints.
      const aspect = Math.min(2, Math.max(0.5, Math.sqrt(lens.sizeX / lens.sizeY)))
      const strata = training ? trainingSamplesPerSide : samplesPerSide
      const nx = Math.round(strata * aspect), ny = Math.round(strata / aspect)
      const { min, max } = beamVolumeBounds(result, receiverZ)
      words.set([lens.columns, lens.rows, nx, ny], 0)
      floats.set([lens.sizeX, lens.sizeY], 4)
      words[6] = result.firstHits ? 1 : 0
      words[7] = Number(!!colorFilter)
      floats.set(min, 8)
      floats.set(max.map((upper, axis) => (upper - min[axis]) / resolution[axis]), 12)
      device.queue.writeBuffer(params, 0, data)
      encoder.clearBuffer(accumulator, 0, colorFilter ? 3 * planeBytes : planeBytes)
      const pass = encoder.beginComputePass({ label: 'Build refracted light field' })
      pass.setPipeline(inject)
      pass.setBindGroup(0, injectGroup)
      pass.dispatchWorkgroups(Math.ceil(nx / 8), Math.ceil(ny / 8))
      pass.setPipeline(filterRows)
      pass.setBindGroup(0, rowGroup)
      pass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16), depth)
      pass.setPipeline(resolve)
      pass.setBindGroup(0, resolveGroup)
      pass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16), depth)
      pass.end()
    },
    destroy() {
      params.destroy()
      emptyHits.destroy()
      whiteFilter.destroy()
      accumulator.destroy()
      filteredRows.destroy()
      texture.destroy()
    },
  }
}
