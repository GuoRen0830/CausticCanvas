import { mat4 } from 'wgpu-matrix'
import type { Mat4 } from 'wgpu-matrix'
import type { LensState } from '../contracts'
import { buildLensIndices } from '../geometry/lens-mesh'
import lensShaderSource from './shaders/lens.wgsl?raw'
import boundsShader from './shaders/lens-bounds.wgsl?raw'

export interface LensRenderer {
  update(lens: LensState, viewProjection: Mat4, relief?: number): void
  prepare(encoder: GPUCommandEncoder): void
  draw(pass: GPURenderPassEncoder): void
  destroy(): void
}

function createLensIndexBuffer(device: GPUDevice, lens: LensState) {
  const indices = buildLensIndices(lens)

  // Mesh connectivity stays fixed while GPU heights change.
  const buffer = device.createBuffer({
    label: 'Lens indices',
    size: indices.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  })

  device.queue.writeBuffer(buffer, 0, indices)

  return { buffer, indexCount: indices.length }
}

function createLensUniforms(
  device: GPUDevice,
  lens: LensState,
  model: Mat4,
  referenceHeight: number,
) {
  // Store MVP, local-to-world, and grid parameters in one uniform block.
  const buffer = device.createBuffer({
    label: 'Lens uniforms',
    size: 160,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })

  const data = new ArrayBuffer(160)
  const mvp = new Float32Array(data, 0, 16)
  new Float32Array(data, 64, 16).set(model)
  const parameters = new DataView(data)

  // Match the WGSL field types and byte offsets.
  parameters.setUint32(128, lens.columns, true)
  parameters.setUint32(132, lens.rows, true)
  parameters.setFloat32(136, lens.sizeX, true)
  parameters.setFloat32(140, lens.sizeY, true)
  parameters.setFloat32(148, referenceHeight, true)

  const update = (viewProjection: Mat4, relief: number) => {
    parameters.setFloat32(144, relief, true)
    mat4.multiply(viewProjection, model, mvp)
    device.queue.writeBuffer(buffer, 0, data)
  }

  return { buffer, update }
}

function createLensPipeline(
  device: GPUDevice,
  format: GPUTextureFormat,
) {
  const module = device.createShaderModule({
    label: 'Lens shader',
    code: lensShaderSource,
  })

  return device.createRenderPipeline({
    label: 'Lens pipeline',
    layout: 'auto',
    vertex: {
      module,
      entryPoint: 'vertexMain',
    },
    fragment: {
      module,
      entryPoint: 'fragmentMain',
      targets: [{ format }],
    },
    primitive: {
      topology: 'triangle-list',
      frontFace: 'ccw',
      // The preview is an open exit surface; allow inspection from either side.
      cullMode: 'none',
    },
    depthStencil: {
      format: 'depth24plus',
      depthWriteEnabled: true,
      depthCompare: 'less',
    },
  })
}

export function createLensRenderer(
  device: GPUDevice,
  format: GPUTextureFormat,
  initialLens: LensState,
  model: Mat4,
  referenceHeight = 0,
): LensRenderer {
  const { columns, rows, sizeX, sizeY } = initialLens
  const indices = createLensIndexBuffer(device, initialLens)
  const uniforms = createLensUniforms(device, initialLens, model, referenceHeight)
  const pipeline = createLensPipeline(device, format)
  const layout = pipeline.getBindGroupLayout(0)
  const bindings = new WeakMap<GPUBuffer, WeakMap<GPUBuffer, GPUBindGroup>>()
  const scans = new WeakMap<GPUBuffer, GPUBindGroup>()
  // Lenses without vertex offsets read zeros.
  const zeroOffsets = device.createBuffer({ label: 'Zero lens offsets', size: columns * rows * 8, usage: GPUBufferUsage.STORAGE })
  const partials = device.createBuffer({ label: 'Lens preview partial bounds', size: 256 * 8, usage: GPUBufferUsage.STORAGE })
  const bounds = device.createBuffer({ label: 'Lens preview bounds', size: 8, usage: GPUBufferUsage.STORAGE })
  const boundsModule = device.createShaderModule({ label: 'Lens preview framing', code: boundsShader })
  const scan = device.createComputePipeline({ layout: 'auto', compute: { module: boundsModule, entryPoint: 'scan' } })
  const finish = device.createComputePipeline({ layout: 'auto', compute: { module: boundsModule, entryPoint: 'finish' } })
  const finishGroup = device.createBindGroup({ layout: finish.getBindGroupLayout(0), entries: [
    { binding: 1, resource: { buffer: partials } }, { binding: 2, resource: { buffer: bounds } },
  ] })

  const bindingsFor = (heightBuffer: GPUBuffer, offsetBuffer = zeroOffsets) => {
    let byOffsets = bindings.get(heightBuffer)
    if (!byOffsets) bindings.set(heightBuffer, byOffsets = new WeakMap())
    let group = byOffsets.get(offsetBuffer)

    if (!group) {
      // Reuse bindings when the optimizer alternates height buffers.
      group = device.createBindGroup({
        label: 'Lens bindings',
        layout,
        entries: [
          {
            binding: 0,
            resource: { buffer: uniforms.buffer },
          },
          {
            binding: 1,
            resource: { buffer: heightBuffer, size: columns * rows * 4 },
          },
          { binding: 2, resource: { buffer: bounds } },
          { binding: 3, resource: { buffer: offsetBuffer, size: columns * rows * 8 } },
        ],
      })
      byOffsets.set(offsetBuffer, group)
      if (!scans.has(heightBuffer)) scans.set(heightBuffer, device.createBindGroup({ layout: scan.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: heightBuffer, size: columns * rows * 4 } },
        { binding: 1, resource: { buffer: partials } },
      ] }))
    }

    return group
  }

  let bindGroup = bindingsFor(initialLens.heightBuffer, initialLens.offsetBuffer)
  let scanGroup = scans.get(initialLens.heightBuffer)!

  return {
    update(lens, viewProjection, relief = 1) {
      if (!Number.isFinite(relief) || relief < 1) throw new Error('Lens preview relief must be at least 1.')
      if (
        lens.columns !== columns ||
        lens.rows !== rows ||
        lens.sizeX !== sizeX ||
        lens.sizeY !== sizeY
      ) {
        throw new Error('Lens grid or aperture changes require a new renderer.')
      }

      // Bind the current GPU resource without reading back its contents.
      bindGroup = bindingsFor(lens.heightBuffer, lens.offsetBuffer)
      scanGroup = scans.get(lens.heightBuffer)!
      uniforms.update(viewProjection, relief)
    },

    prepare(encoder) {
      const pass = encoder.beginComputePass({ label: 'Fit lens preview' })
      pass.setPipeline(scan)
      pass.setBindGroup(0, scanGroup)
      pass.dispatchWorkgroups(256)
      pass.setPipeline(finish)
      pass.setBindGroup(0, finishGroup)
      pass.dispatchWorkgroups(1)
      pass.end()
    },

    draw(pass) {
      // Vertex positions come from the shared height buffer.
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bindGroup)
      pass.setIndexBuffer(indices.buffer, 'uint32')
      pass.drawIndexed(indices.indexCount)
    },

    destroy() {
      // Shared heights remain owned by the producer.
      indices.buffer.destroy()
      uniforms.buffer.destroy()
      partials.destroy()
      bounds.destroy()
      zeroOffsets.destroy()
    },
  }
}
