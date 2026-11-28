import type { Mat4 } from 'wgpu-matrix'
import type { TriangleBeamField } from '../contracts'
import apertureUV from './shaders/aperture-uv.wgsl?raw'
import common from './shaders/beam-common.wgsl?raw'
import hitTypes from './shaders/scene-hit.wgsl?raw'
import query from './shaders/bvh-query.wgsl?raw'
import coverage from './shaders/pixel-coverage.wgsl?raw'
import source from './shaders/export-source.wgsl?raw'
import trace from './shaders/export-trace.wgsl?raw'
import splat from './shaders/export-splat.wgsl?raw'
import atomic from './shaders/export-atomic.wgsl?raw'

const blockFaces = 128
const facesPerTrace = blockFaces * 256
const facesPerDraw = blockFaces * 128
const geometry = `
  @group(2) @binding(0) var<storage, read> triangles: array<Triangle>;
  @group(2) @binding(1) var<storage, read> nodes: array<Node>;
`

interface TileTargets {
  width: number
  height: number
  depth: GPUTextureView
  normal: GPUTextureView
  albedo: GPUTextureView
  object: GPUTextureView
  material: GPUTextureView
  direct: GPUTextureView
}

// AABB/frustum test in optical coordinates. All source faces in a block form
// one draw range. Consecutive visible blocks share a draw call.
function visibleRanges(bounds: Float32Array, vp: Mat4, count: number) {
  const planes: number[][] = []
  for (const [row, sign, w] of [[0, 1, 1], [0, -1, 1], [1, 1, 1], [1, -1, 1], [2, 1, 0], [2, -1, 1]]) {
    planes.push([0, 4, 8, 12].map(i => w * vp[i + 3] + sign * vp[i + row]))
  }
  const ranges: { first: number; count: number }[] = []
  for (let i = 0, first = 0; i < bounds.length; i += 8, first += blockFaces) {
    if (bounds[i] > bounds[i + 4]) continue
    const outside = planes.some(([x, y, z, w]) => {
      const dx = x * bounds[i + (x >= 0 ? 4 : 0)]
      const dy = y * bounds[i + (y >= 0 ? 5 : 1)]
      const dz = z * bounds[i + (z >= 0 ? 6 : 2)]
      // Retain boundary polygons under float32 round-off.
      return dx + dy + dz + w < -1e-6 * (1 + Math.abs(dx) + Math.abs(dy) + Math.abs(dz) + Math.abs(w))
    })
    if (outside) continue
    const n = Math.min(blockFaces, count - first)
    const previous = ranges[ranges.length - 1]
    if (previous && previous.first + previous.count === first) previous.count += n
    else ranges.push({ first, count: n })
  }
  return ranges
}

function storage(binding: number, visibility: GPUShaderStageFlags, writable = false): GPUBindGroupLayoutEntry {
  return { binding, visibility, buffer: { type: writable ? 'storage' : 'read-only-storage' } }
}
function buffer(binding: number, resource: GPUBuffer): GPUBindGroupEntry {
  return { binding, resource: { buffer: resource } }
}

export async function createExportTransport(
  device: GPUDevice, field: TriangleBeamField, triangles: GPUBuffer, nodes: GPUBuffer,
  colorFilter: GPUTextureView | undefined, signal: AbortSignal, progress?: (fraction: number) => void,
) {
  signal.throwIfAborted()
  const { lens } = field
  const count = 2 * (lens.columns - 1) * (lens.rows - 1)
  const blocks = Math.ceil(count / blockFaces)
  const owned: (GPUBuffer | GPUTexture)[] = []
  const temporary: GPUBuffer[] = []
  function allocate(label: string, size: number, usage: GPUBufferUsageFlags, transient = false) {
    const b = device.createBuffer({ label, size, usage })
    if (transient) temporary.push(b)
    else owned.push(b)
    return b
  }
  const release = (resources: (GPUBuffer | GPUTexture)[]) => {
    for (const resource of resources.splice(0)) resource.destroy()
  }
  const destroy = () => release(owned)
  device.pushErrorScope('out-of-memory')
  let allocationScopeOpen = true
  try {
    const C = GPUShaderStage.COMPUTE, V = GPUShaderStage.VERTEX, F = GPUShaderStage.FRAGMENT
    const S = GPUBufferUsage.STORAGE
    if (count < 1 || count * 16 > device.limits.maxStorageBufferBindingSize) {
      throw new Error('The lens exceeds this GPU’s export transport storage limit.')
    }
    const ids = allocate('Export child first-hit triangle IDs', count * 16, S)
    const gpuBounds = allocate('Export receiver bounds', blocks * 32, S | GPUBufferUsage.COPY_SRC, true)
    const readback = allocate('Export bounds readback', gpuBounds.size, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, true)
    const uniform = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    const lensParams = allocate('Export tracing lens', 176, uniform, true)
    const batch = allocate('Export tracing batch', 16, uniform, true)
    const offsets = lens.offsetBuffer ?? allocate('Zero export lens offsets', lens.columns * lens.rows * 8, S)
    let filter = colorFilter
    if (!filter) {
      const texture = device.createTexture({ label: 'Unused export filter', size: [1, 1], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING })
      owned.push(texture)
      filter = texture.createView()
    }
    // Check the large hit cache before submitting the first tracing batch.
    allocationScopeOpen = false
    const allocationError = await device.popErrorScope()
    if (allocationError) throw new Error('Insufficient GPU memory for the export light transport.')
    const data = new Float32Array(44)
    const words = new Uint32Array(data.buffer)
    words.set([lens.columns, lens.rows], 32)
    data.set([lens.sizeX, lens.sizeY], 34)
    words[42] = Number(!!colorFilter)
    device.queue.writeBuffer(lensParams, 0, data)

    const fieldLayout = device.createBindGroupLayout({ entries: [
      storage(0, C | V), storage(1, C | V), storage(2, C | V),
      { binding: 5, visibility: C | V, texture: { sampleType: 'unfilterable-float' } },
    ] })
    const fieldGroup = device.createBindGroup({ layout: fieldLayout, entries: [
      buffer(0, lens.heightBuffer), buffer(1, field.buffer), buffer(2, offsets), { binding: 5, resource: filter },
    ] })
    const sharedCode = [apertureUV, common, hitTypes, geometry, query, source].join('\n')
    const traceLayout = device.createBindGroupLayout({ entries: [
      storage(0, C), storage(1, C), storage(2, C, true), storage(3, C, true),
      { binding: 4, visibility: C, buffer: { type: 'uniform' } },
    ] })
    const lensLayout = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: C, buffer: { type: 'uniform' } }] })
    const tracePipeline = await device.createComputePipelineAsync({
      label: 'Stratified finite-beam receiver tracing',
      layout: device.createPipelineLayout({ bindGroupLayouts: [lensLayout, fieldLayout, traceLayout] }),
      compute: { module: device.createShaderModule({ code: sharedCode + '\n' + trace }), entryPoint: 'traceChildren' },
    })
    const traceLens = device.createBindGroup({ layout: lensLayout, entries: [buffer(0, lensParams)] })
    const traceScene = device.createBindGroup({ layout: traceLayout, entries: [
      buffer(0, triangles), buffer(1, nodes), buffer(2, ids), buffer(3, gpuBounds), buffer(4, batch),
    ] })
    for (let first = 0; first < count; first += facesPerTrace) {
      signal.throwIfAborted()
      device.queue.writeBuffer(batch, 0, new Uint32Array([first]))
      const encoder = device.createCommandEncoder()
      const pass = encoder.beginComputePass()
      pass.setPipeline(tracePipeline)
      pass.setBindGroup(0, traceLens)
      pass.setBindGroup(1, fieldGroup)
      pass.setBindGroup(2, traceScene)
      const n = Math.min(facesPerTrace, count - first)
      pass.dispatchWorkgroups(Math.ceil(n / blockFaces))
      pass.end()
      device.queue.submit([encoder.finish()])
      await device.queue.onSubmittedWorkDone()
      progress?.((first + n) / count)
    }
    signal.throwIfAborted()
    const copy = device.createCommandEncoder()
    copy.copyBufferToBuffer(gpuBounds, 0, readback, 0, gpuBounds.size)
    device.queue.submit([copy.finish()])
    await readback.mapAsync(GPUMapMode.READ)
    const bounds = new Float32Array(readback.getMappedRange().slice(0))
    readback.unmap()
    release(temporary)
    signal.throwIfAborted()

    const blend = device.features.has('float32-blendable')
    const sceneLayout = device.createBindGroupLayout({ entries: [
      storage(0, V | F), storage(1, F), storage(2, V),
      ...(!blend ? [storage(3, F, true)] : []),
    ] })
    const frameLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: V | F, buffer: { type: 'uniform' } },
      { binding: 1, visibility: F, texture: { sampleType: 'depth' } },
      { binding: 2, visibility: F, texture: { sampleType: 'unfilterable-float' } },
      { binding: 3, visibility: F, texture: { sampleType: 'unfilterable-float' } },
      { binding: 5, visibility: F, texture: { sampleType: 'uint' } },
      { binding: 6, visibility: F, texture: { sampleType: 'uint' } },
    ] })
    const layout = device.createPipelineLayout({ bindGroupLayouts: [frameLayout, fieldLayout, sceneLayout] })
    const native = `@fragment fn exportFragment(input: ExportSplat) -> @location(0) vec4f { return exportRadiance(input); }`
    const module = device.createShaderModule({
      label: 'Export finite beam BRDF and visibility',
      code: [sharedCode, '@group(2) @binding(2) var<storage, read> hitIds: array<vec4u>;',
        coverage, splat, blend ? native : atomic].join('\n'),
    })
    const format = 'rgba32float'
    const add: GPUBlendComponent = { operation: 'add', srcFactor: 'one', dstFactor: 'one' }
    const resolveSums = blend ? undefined : await device.createRenderPipelineAsync({
      label: 'Resolve atomic export radiance', layout,
      vertex: { module, entryPoint: 'resolveSumVertex' },
      fragment: { module, entryPoint: 'resolveSumFragment', targets: [{ format }] },
    })
    let sums: GPUBuffer | undefined
    const contention = blend ? undefined : allocate('Export contention readback', 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST)
    let sceneGroup: GPUBindGroup | undefined
    async function groupFor(width: number, height: number) {
      const bytes = width * height * 12 + 4 // final u32 is the contention flag
      if (!blend && (!sums || sums.size < bytes)) {
        if (bytes > device.limits.maxStorageBufferBindingSize) throw new Error('The export tile is too large for this GPU.')
        sums?.destroy()
        device.pushErrorScope('out-of-memory')
        try {
          sums = allocate('Export FP32 sample sums', bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC)
        } finally {
          if (await device.popErrorScope()) throw new Error('Insufficient GPU memory for export sample accumulation.')
        }
        sceneGroup = undefined
      }
      return sceneGroup ??= device.createBindGroup({ layout: sceneLayout, entries: [
        buffer(0, triangles), buffer(1, nodes), buffer(2, ids), ...(sums ? [buffer(3, sums)] : []),
      ] })
    }
    const pipelines = new Map<boolean, Promise<GPURenderPipeline>>()
    function getPipeline(materialLighting: boolean) {
      let result = pipelines.get(materialLighting)
      if (!result) {
        result = device.createRenderPipelineAsync({
          label: 'Export radiance per optical direction', layout,
          vertex: { module, entryPoint: 'exportVertex' },
          fragment: { module, entryPoint: 'exportFragment', constants: { materialLighting: Number(materialLighting) },
            targets: [blend ? { format, blend: { color: add, alpha: add } } : { format, writeMask: 0 }] },
          primitive: { topology: 'triangle-strip' },
        })
        pipelines.set(materialLighting, result)
      }
      return result
    }
    return {
      format, offsets, destroy,
      async render(
        frame: GPUBuffer, targets: TileTargets, localVP: Mat4, materialLighting: boolean,
        abort: AbortSignal, onProgress?: (fraction: number) => void,
      ) {
        abort.throwIfAborted()
        const pipeline = await getPipeline(materialLighting)
        const scene = await groupFor(targets.width, targets.height)
        const frameGroup = device.createBindGroup({ layout: frameLayout, entries: [
          buffer(0, frame), { binding: 1, resource: targets.depth }, { binding: 2, resource: targets.normal },
          { binding: 3, resource: targets.albedo }, { binding: 5, resource: targets.object }, { binding: 6, resource: targets.material },
        ] })
        const ranges = visibleRanges(bounds, localVP, count)
        const visibleFaces = ranges.reduce((sum, r) => sum + r.count, 0)
        // Smaller software-blend batches bound the number of fragments that
        // can contend for a single pixel in an exceptionally focused caustic.
        let batchFaces = blend ? facesPerDraw : facesPerDraw / 4
        let range = 0, offset = 0, clear = true
        let drawnFaces = 0, reportedFaces = 0
        do {
          abort.throwIfAborted()
          const encoder = device.createCommandEncoder()
          if (clear && sums) encoder.clearBuffer(sums)
          const pass = encoder.beginRenderPass({ colorAttachments: [{
            view: targets.direct, storeOp: blend ? 'store' : 'discard',
            loadOp: clear || !blend ? 'clear' : 'load', clearValue: [0, 0, 0, 0],
          }] })
          pass.setPipeline(pipeline)
          pass.setBindGroup(0, frameGroup)
          pass.setBindGroup(1, fieldGroup)
          pass.setBindGroup(2, scene)
          let remaining = batchFaces
          while (range < ranges.length && remaining > 0) {
            const run = ranges[range]
            const n = Math.min(remaining, run.count - offset)
            pass.draw(4, n * 4, 0, (run.first + offset) * 4)
            offset += n
            remaining -= n
            if (offset === run.count) { range++; offset = 0 }
          }
          pass.end()
          if (resolveSums && sums && contention && range === ranges.length) {
            const resolve = encoder.beginRenderPass({ colorAttachments: [{
              view: targets.direct, loadOp: 'clear', clearValue: [0, 0, 0, 0], storeOp: 'store',
            }] })
            resolve.setPipeline(resolveSums)
            resolve.setBindGroup(0, frameGroup)
            resolve.setBindGroup(1, fieldGroup)
            resolve.setBindGroup(2, scene)
            resolve.draw(3)
            resolve.end()
            encoder.copyBufferToBuffer(sums, sums.size - 4, contention, 0, 4)
          }
          device.queue.submit([encoder.finish()])
          await device.queue.onSubmittedWorkDone()
          drawnFaces += batchFaces - remaining
          reportedFaces = Math.max(drawnFaces, reportedFaces)
          onProgress?.(visibleFaces ? reportedFaces / visibleFaces : 1)
          clear = false
          if (contention && range === ranges.length) {
            await contention.mapAsync(GPUMapMode.READ)
            const retry = new Uint32Array(contention.getMappedRange())[0] !== 0
            contention.unmap()
            if (retry) {
              if (batchFaces === 1) throw new Error('This GPU could not accumulate export radiance reliably.')
              // Replay from zero so a racing fragment can never drop energy.
              batchFaces = Math.max(1, Math.floor(batchFaces / 8))
              range = 0; offset = 0; drawnFaces = 0; clear = true
            }
          }
        } while (clear || range < ranges.length)
        abort.throwIfAborted()
      },
    } as const
  } catch (error) {
    destroy()
    throw error
  } finally {
    release(temporary)
    if (allocationScopeOpen) await device.popErrorScope()
  }
}
