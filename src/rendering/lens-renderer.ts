import { mat4 } from 'wgpu-matrix'
import type { LensState } from '../contracts'
import { buildLensIndices } from '../geometry/lens-mesh'
import lensShaderSource from './shaders/lens.wgsl?raw'

export interface LensRenderer {
  update(lens: LensState, aspect: number): void
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
) {
  // Store the MVP matrix and grid parameters in one uniform block.
  const buffer = device.createBuffer({
    label: 'Lens uniforms',
    size: 80,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })

  const span = Math.max(lens.sizeX, lens.sizeY)

  // Keep lens-local coordinates unchanged in the preview world.
  const model = mat4.identity()

  // View the exit surface from an oblique angle.
  const view = mat4.lookAt(
    [span * 1.3, span * 0.9, span * 1.8],
    [0, 0, 0],
    [0, 1, 0],
  )

  const modelView = mat4.multiply(view, model)
  const data = new ArrayBuffer(80)
  const mvp = new Float32Array(data, 0, 16)
  const parameters = new DataView(data)

  // Match the WGSL field types and byte offsets.
  parameters.setUint32(64, lens.columns, true)
  parameters.setUint32(68, lens.rows, true)
  parameters.setFloat32(72, lens.sizeX, true)
  parameters.setFloat32(76, lens.sizeY, true)

  const update = (aspect: number) => {
    const projection = mat4.perspective(
      Math.PI / 4,
      aspect,
      span * 0.01,
      span * 20,
    )

    // Column vectors apply model, view, then projection.
    mat4.multiply(projection, modelView, mvp)
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
      cullMode: 'back',
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
): LensRenderer {
  const { columns, rows, sizeX, sizeY } = initialLens
  const indices = createLensIndexBuffer(device, initialLens)
  const uniforms = createLensUniforms(device, initialLens)
  const pipeline = createLensPipeline(device, format)
  const layout = pipeline.getBindGroupLayout(0)
  const bindings = new WeakMap<GPUBuffer, GPUBindGroup>()

  const bindingsFor = (heightBuffer: GPUBuffer) => {
    let group = bindings.get(heightBuffer)

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
        ],
      })
      bindings.set(heightBuffer, group)
    }

    return group
  }

  let bindGroup = bindingsFor(initialLens.heightBuffer)

  return {
    update(lens, aspect) {
      if (
        lens.columns !== columns ||
        lens.rows !== rows ||
        lens.sizeX !== sizeX ||
        lens.sizeY !== sizeY
      ) {
        throw new Error('Lens grid or aperture changes require a new renderer.')
      }

      // Bind the current GPU resource without reading back its contents.
      bindGroup = bindingsFor(lens.heightBuffer)
      uniforms.update(aspect)
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
    },
  }
}
