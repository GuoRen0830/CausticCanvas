import { mat4 } from 'wgpu-matrix'
import type { LensState } from '../contracts/lens'
import {
  buildLensPositions,
  buildLensIndices,
} from '../geometry/lens-mesh'
import lensShaderSource from './shaders/lens.wgsl?raw'

// Each vertex contains three consecutive float32 position components.
export const lensVertexLayout: GPUVertexBufferLayout = {
  arrayStride: 12,
  stepMode: 'vertex',
  attributes: [
    {
      shaderLocation: 0,
      offset: 0,
      format: 'float32x3',
    },
  ],
}

export interface LensBuffers {
  vertexBuffer: GPUBuffer
  indexBuffer: GPUBuffer
  indexCount: number
}

export function createLensBuffers(
  device: GPUDevice,
  lens: LensState,
): LensBuffers {
  const positions = buildLensPositions(lens)
  const indices = buildLensIndices(lens)

  // Vertex positions can be uploaded again when the lens changes.
  const vertexBuffer = device.createBuffer({
    label: 'Lens vertices',
    size: positions.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  })

  // Mesh connectivity stays fixed while only heights change.
  const indexBuffer = device.createBuffer({
    label: 'Lens indices',
    size: indices.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  })

  // Upload the initial CPU mesh into GPU buffers.
  device.queue.writeBuffer(vertexBuffer, 0, positions)
  device.queue.writeBuffer(indexBuffer, 0, indices)

  return {
    vertexBuffer,
    indexBuffer,
    indexCount: indices.length,
  }
}

export function createLensTransforms(
  device: GPUDevice,
  lens: LensState,
) {
  // One mat4x4<f32> occupies 64 bytes.
  const buffer = device.createBuffer({
    label: 'Lens transforms',
    size: 64,
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
  const mvp = new Float32Array(16)

  const update = (aspect: number) => {
    const projection = mat4.perspective(
      Math.PI / 4,
      aspect,
      span * 0.01,
      span * 20,
    )

    // Column vectors apply model, view, then projection.
    mat4.multiply(projection, modelView, mvp)
    device.queue.writeBuffer(buffer, 0, mvp)
  }

  return { buffer, update }
}

export function createLensPipeline(
  device: GPUDevice,
  format: GPUTextureFormat,
  transformBuffer: GPUBuffer,
) {
  const module = device.createShaderModule({
    label: 'Lens shader',
    code: lensShaderSource,
  })

  const pipeline = device.createRenderPipeline({
    label: 'Lens pipeline',
    layout: 'auto',
    vertex: {
      module,
      entryPoint: 'vertexMain',
      buffers: [lensVertexLayout],
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

  // Connect group 0, binding 0 to the MVP uniform buffer.
  const bindGroup = device.createBindGroup({
    label: 'Lens transform bindings',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: { buffer: transformBuffer },
      },
    ],
  })

  return { pipeline, bindGroup }
}