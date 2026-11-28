import { mat4 } from 'wgpu-matrix'
import type { Mat4 } from 'wgpu-matrix'
import sceneShaderSource from './shaders/scene.wgsl?raw'
import { sceneAlbedoFormat, sceneVertexStride, type SceneMaterial, type SceneMapSlot, type SceneMesh } from './scene-mesh'
import { createMaterialBindings } from './scene-materials'

export type SceneSurfaceMode = 'material' | 'base-color' | 'map-inspection'

export async function createSceneRenderer(
  device: GPUDevice,
  mesh: SceneMesh,
  model: Mat4,
  materials: ReadonlyMap<string, SceneMaterial> = new Map(),
  warn: (message: string) => void = console.warn,
) {
  const mvp = mat4.identity()
  const offscreenMvp = mat4.identity()
  const draws = mesh.draws ?? [{ firstIndex: 0, indexCount: mesh.indices.length }]
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
      }, {
        arrayStride: 8, stepMode: mesh.uvs1 ? 'vertex' : 'instance',
        attributes: [{ shaderLocation: 6, offset: 0, format: 'float32x2' }],
      }, {
        arrayStride: 16, stepMode: mesh.tangents ? 'vertex' : 'instance',
        attributes: [{ shaderLocation: 7, offset: 0, format: 'float32x4' }],
      }],
    },
    fragment: {
      module,
      entryPoint: 'fragmentMain',
      // The last attachment holds packed surface parameters and HDR emission.
      targets: [{ format: sceneAlbedoFormat(materials) }, { format: 'rgba8unorm' },
        { format: 'rg32uint' }, { format: 'rg32uint' }],
    },
    primitive: {
      topology: 'triangle-list',
      frontFace: 'ccw',
      cullMode: draws.some(draw => draw.doubleSided) ? 'none' : 'back',
    },
    depthStencil: {
      format: 'depth24plus',
      depthWriteEnabled: true,
      depthCompare: 'less',
    },
  })
  const materialLayout = pipeline.getBindGroupLayout(1)
  let materialBindings = await createMaterialBindings(device, materialLayout, draws, materials, warn)
  let materialRequest = 0, disposed = false
  const upload = (label: string, data: Float32Array<ArrayBuffer> | Uint32Array<ArrayBuffer>, usage: GPUBufferUsageFlags) => {
    const buffer = device.createBuffer({ label, size: data.byteLength, usage: usage | GPUBufferUsage.COPY_DST })
    device.queue.writeBuffer(buffer, 0, data)
    return buffer
  }
  const vertexBuffer = upload('Scene vertices', mesh.vertices, GPUBufferUsage.VERTEX)
  const uvBuffer = upload('Scene UVs', mesh.uvs ?? new Float32Array(2), GPUBufferUsage.VERTEX)
  const secondaryUV = upload('Scene secondary UVs', mesh.uvs1 ?? new Float32Array(2), GPUBufferUsage.VERTEX)
  const tangentBuffer = upload('Scene tangents', mesh.tangents ?? new Float32Array(4), GPUBufferUsage.VERTEX)
  const indexBuffer = upload('Scene indices', mesh.indices, GPUBufferUsage.INDEX)
  // Keep draw modes independent even when several draws share one submission.
  function frameBinding(mode: SceneSurfaceMode) {
    const buffer = device.createBuffer({
      label: 'Scene uniforms: ' + mode, size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(buffer, mvp.byteLength, Uint32Array.of(Number(mode === 'map-inspection'), Number(mode === 'base-color')))
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer } }] })
    return { buffer, group }
  }
  const frameBindings = {
    material: frameBinding('material'),
    'base-color': frameBinding('base-color'),
    'map-inspection': frameBinding('map-inspection'),
  }

  return {
    mapCoverage: (channels: readonly SceneMapSlot[]) => materialBindings.mapCoverage(channels),
    async replaceMaterials(next: ReadonlyMap<string, SceneMaterial>, signal: AbortSignal) {
      signal.throwIfAborted()
      if (disposed) throw new DOMException('Scene has been closed.', 'AbortError')
      if (sceneAlbedoFormat(next) !== sceneAlbedoFormat(materials)) {
        throw new Error('Replacement materials must use the same G-buffer color format.')
      }
      const generation = ++materialRequest
      const replacement = await createMaterialBindings(device, materialLayout, draws, next,
        message => { throw new Error(message) }, signal)
      if (generation !== materialRequest || disposed || signal.aborted) {
        replacement.destroy()
        signal.throwIfAborted()
        throw new DOMException('A newer material set or scene was selected.', 'AbortError')
      }
      const previous = materialBindings
      materialBindings = replacement
      previous.destroy()
    },

    update(viewProjection: Mat4) {
      mat4.multiply(viewProjection, model, mvp)
    },

    draw(pass: GPURenderPassEncoder, mode: SceneSurfaceMode = 'material', viewProjection?: Mat4) {
      const frame = frameBindings[mode]
      device.queue.writeBuffer(frame.buffer, 0, viewProjection ? mat4.multiply(viewProjection, model, offscreenMvp) : mvp)
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, frame.group)
      pass.setVertexBuffer(0, vertexBuffer)
      pass.setVertexBuffer(1, uvBuffer)
      pass.setVertexBuffer(2, secondaryUV)
      pass.setVertexBuffer(3, tangentBuffer)
      pass.setIndexBuffer(indexBuffer, 'uint32')
      for (const draw of materialBindings.bindings) {
        pass.setBindGroup(1, draw.group)
        pass.drawIndexed(draw.indexCount, 1, draw.firstIndex)
      }
    },

    destroy() {
      disposed = true
      ++materialRequest
      vertexBuffer.destroy()
      uvBuffer.destroy()
      secondaryUV.destroy()
      tangentBuffer.destroy()
      materialBindings.destroy()
      indexBuffer.destroy()
      for (const frame of Object.values(frameBindings)) frame.buffer.destroy()
    },
  }
}
