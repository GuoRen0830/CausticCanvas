import { mat4 } from 'wgpu-matrix'
import type { Mat4 } from 'wgpu-matrix'
import sceneShaderSource from './shaders/scene.wgsl?raw'
import { sceneAlbedoFormat, sceneVertexStride, type BaseColorMaterial, type SceneMesh } from './scene-mesh'
import { createBaseColorBindings } from './base-color'

export async function createSceneRenderer(
  device: GPUDevice,
  mesh: SceneMesh,
  model: Mat4,
  materials: ReadonlyMap<string, BaseColorMaterial> = new Map(),
  warn: (message: string) => void = console.warn,
) {
  const mvp = mat4.identity()
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
      }, {
        arrayStride: 8, stepMode: mesh.uvs ? 'vertex' : 'instance',
        attributes: [{ shaderLocation: 5, offset: 0, format: 'float32x2' }],
      }],
    },
    fragment: {
      module,
      entryPoint: 'fragmentMain',
      // Albedo, shading normal + floor flag, object ID + packed geometric normal.
      targets: [{ format: sceneAlbedoFormat(materials) }, { format: 'rgba8unorm' }, { format: 'rg32uint' }],
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
  const draws = mesh.draws ?? [{ firstIndex: 0, indexCount: mesh.indices.length }]
  const baseColor = await createBaseColorBindings(device, pipeline.getBindGroupLayout(1), draws, materials, warn)
  const upload = (label: string, data: Float32Array<ArrayBuffer> | Uint32Array<ArrayBuffer>, usage: GPUBufferUsageFlags) => {
    const buffer = device.createBuffer({ label, size: data.byteLength, usage: usage | GPUBufferUsage.COPY_DST })
    device.queue.writeBuffer(buffer, 0, data)
    return buffer
  }
  const vertexBuffer = upload('Scene vertices', mesh.vertices, GPUBufferUsage.VERTEX)
  const uvBuffer = upload('Scene UVs', mesh.uvs ?? new Float32Array(2), GPUBufferUsage.VERTEX)
  const indexBuffer = upload('Scene indices', mesh.indices, GPUBufferUsage.INDEX)
  const uniformBuffer = device.createBuffer({
    label: 'Scene uniforms', size: mvp.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
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
      pass.setVertexBuffer(1, uvBuffer)
      pass.setIndexBuffer(indexBuffer, 'uint32')
      for (const draw of baseColor.bindings) {
        pass.setBindGroup(1, draw.group)
        pass.drawIndexed(draw.indexCount, 1, draw.firstIndex)
      }
    },

    destroy() {
      vertexBuffer.destroy()
      uvBuffer.destroy()
      baseColor.destroy()
      indexBuffer.destroy()
      uniformBuffer.destroy()
    },
  }
}
