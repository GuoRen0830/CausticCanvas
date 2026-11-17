import { mat4 } from 'wgpu-matrix'
import type { Mat4 } from 'wgpu-matrix'
import sceneShaderSource from './shaders/scene.wgsl?raw'
import { sceneVertexStride, type SceneMesh } from './scene-mesh'

export function createSceneRenderer(
  device: GPUDevice,
  mesh: SceneMesh,
  model: Mat4,
) {
  const vertexBuffer = device.createBuffer({
    label: 'Scene vertices',
    size: mesh.vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  })
  const indexBuffer = device.createBuffer({
    label: 'Scene indices',
    size: mesh.indices.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(vertexBuffer, 0, mesh.vertices)
  device.queue.writeBuffer(indexBuffer, 0, mesh.indices)

  const mvp = mat4.identity()
  const uniformBuffer = device.createBuffer({
    label: 'Scene uniforms',
    size: mvp.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  const module = device.createShaderModule({
    label: 'Scene shader',
    code: sceneShaderSource,
  })
  const pipeline = device.createRenderPipeline({
    label: 'Scene pipeline',
    layout: 'auto',
    vertex: {
      module,
      entryPoint: 'vertexMain',
      buffers: [{
        arrayStride: sceneVertexStride * 4,
        attributes: [
          { shaderLocation: 0, offset: 0, format: 'float32x3' },
          { shaderLocation: 1, offset: 12, format: 'float32x3' },
          { shaderLocation: 2, offset: 24, format: 'float32x3' },
          { shaderLocation: 3, offset: 36, format: 'float32' },
          { shaderLocation: 4, offset: 40, format: 'float32' },
        ],
      }],
    },
    fragment: {
      module,
      entryPoint: 'fragmentMain',
      // Albedo, shading normal + floor flag, object ID + packed geometric normal.
      targets: [{ format: 'rgba8unorm' }, { format: 'rgba8unorm' }, { format: 'rg32uint' }],
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
  const bindGroup = device.createBindGroup({
    label: 'Scene bindings',
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  })

  return {
    update(viewProjection: Mat4) {
      mat4.multiply(viewProjection, model, mvp)
      device.queue.writeBuffer(uniformBuffer, 0, mvp)
    },

    draw(pass: GPURenderPassEncoder) {
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bindGroup)
      pass.setVertexBuffer(0, vertexBuffer)
      pass.setIndexBuffer(indexBuffer, 'uint32')
      pass.drawIndexed(mesh.indices.length)
    },

    destroy() {
      vertexBuffer.destroy()
      indexBuffer.destroy()
      uniformBuffer.destroy()
    },
  }
}
