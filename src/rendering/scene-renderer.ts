import { mat4, vec3 } from 'wgpu-matrix'
import type { Mat4 } from 'wgpu-matrix'
import type { ReceiverPlane } from '../contracts'
import sceneShaderSource from './shaders/scene.wgsl?raw'

type Triple = readonly [number, number, number]

interface SceneBox {
  // Lens-local coordinates and full dimensions, in meters.
  center: Triple
  size: Triple
}

function buildSceneMesh(receiver: ReceiverPlane, box: SceneBox) {
  const dimensions = [receiver.sizeX, receiver.sizeY, ...box.size]
  if (dimensions.some((value) => !Number.isFinite(value) || value <= 0)) {
    throw new Error('Scene dimensions must be finite and positive.')
  }
  if (![receiver.z, ...box.center].every(Number.isFinite)) {
    throw new Error('Scene positions must be finite.')
  }

  // Interleaved position, flat normal, and base color: 9 floats per vertex.
  const vertices: number[] = []
  const indices: number[] = []

  const addQuad = (center: Triple, u: Triple, v: Triple, color: Triple) => {
    const first = vertices.length / 9
    const normal = vec3.normalize(vec3.cross(u, v))
    // u and v are half-edge vectors; u cross v determines the front face.
    for (const [s, t] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      vertices.push(
        center[0] + s * u[0] + t * v[0],
        center[1] + s * u[1] + t * v[1],
        center[2] + s * u[2] + t * v[2],
        normal[0], normal[1], normal[2],
        ...color,
      )
    }
    indices.push(first, first + 1, first + 2, first, first + 2, first + 3)
  }

  // The receiver faces the lens, toward local -Z.
  addQuad(
    [0, 0, receiver.z],
    [receiver.sizeX / 2, 0, 0],
    [0, -receiver.sizeY / 2, 0],
    [0.48, 0.52, 0.58],
  )

  const [x, y, z] = box.center
  const [hx, hy, hz] = box.size.map((value) => value / 2)
  const color: Triple = [0.85, 0.43, 0.16]

  // Each face owns four vertices so the box keeps sharp normal discontinuities.
  addQuad([x + hx, y, z], [0, hy, 0], [0, 0, hz], color)
  addQuad([x - hx, y, z], [0, 0, hz], [0, hy, 0], color)
  addQuad([x, y + hy, z], [0, 0, hz], [hx, 0, 0], color)
  addQuad([x, y - hy, z], [hx, 0, 0], [0, 0, hz], color)
  addQuad([x, y, z + hz], [hx, 0, 0], [0, hy, 0], color)
  addQuad([x, y, z - hz], [0, hy, 0], [hx, 0, 0], color)

  return {
    vertices: new Float32Array(vertices),
    indices: new Uint16Array(indices),
  }
}

export function createSceneRenderer(
  device: GPUDevice,
  format: GPUTextureFormat,
  receiver: ReceiverPlane,
  box: SceneBox,
  model: Mat4,
) {
  const mesh = buildSceneMesh(receiver, box)
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

  // Two mat4 values: MVP at byte 0, local-to-world at byte 64.
  const data = new Float32Array(32)
  const mvp = data.subarray(0, 16)
  data.set(model, 16)
  const uniformBuffer = device.createBuffer({
    label: 'Scene uniforms',
    size: data.byteLength,
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
        arrayStride: 36,
        attributes: [
          { shaderLocation: 0, offset: 0, format: 'float32x3' },
          { shaderLocation: 1, offset: 12, format: 'float32x3' },
          { shaderLocation: 2, offset: 24, format: 'float32x3' },
        ],
      }],
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
  const bindGroup = device.createBindGroup({
    label: 'Scene bindings',
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  })

  return {
    update(viewProjection: Mat4) {
      mat4.multiply(viewProjection, model, mvp)
      device.queue.writeBuffer(uniformBuffer, 0, data)
    },

    draw(pass: GPURenderPassEncoder) {
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bindGroup)
      pass.setVertexBuffer(0, vertexBuffer)
      pass.setIndexBuffer(indexBuffer, 'uint16')
      pass.drawIndexed(mesh.indices.length)
    },

    destroy() {
      vertexBuffer.destroy()
      indexBuffer.destroy()
      uniformBuffer.destroy()
    },
  }
}
