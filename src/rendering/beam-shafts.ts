import type { Mat4 } from 'wgpu-matrix'
import type { ForwardResult } from '../contracts'
import source from './shaders/beam-shafts.wgsl?raw'

// Draw a few representative paths into the same HDR overlay as the glass.
// Both their exit points and their endpoints belong to the current optical evaluation.
export function createBeamShafts(device: GPUDevice) {
  const module = device.createShaderModule({ label: 'Representative refracted paths', code: source })
  const add: GPUBlendComponent = { srcFactor: 'one', dstFactor: 'one', operation: 'add' }
  const pipeline = device.createRenderPipeline({
    label: 'Draw refracted light shafts', layout: 'auto',
    vertex: { module, entryPoint: 'shaftVS' },
    fragment: { module, entryPoint: 'shaftFS', targets: [
      { format: 'rgba16float', blend: { color: add, alpha: add } },
    ] },
    primitive: { topology: 'triangle-strip' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less-equal' },
  })
  const data = new Float32Array(32), words = new Uint32Array(data.buffer)
  const uniform = device.createBuffer({ label: 'Light shaft frame', size: data.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  const noImage = device.createBuffer({ label: 'Unused shaft endpoint irradiance', size: 4, usage: GPUBufferUsage.STORAGE })
  let buffers: GPUBuffer[] = []
  let group: GPUBindGroup

  return {
    draw(pass: GPURenderPassEncoder, vp: Mat4,
      width: number, height: number, result: ForwardResult, offsets: GPUBuffer,
      incidentIrradiance: number, hasDrawing: boolean) {
      const { lens, buffer: rays } = result.beams
      const hits = result.firstHits
      const receiver = result.receiver
      if (!hits || incidentIrradiance <= 0) return
      const image = result.caustic.kind === 'buffer' ? result.caustic.buffer : noImage
      const next = [uniform, lens.heightBuffer, offsets, rays, hits.buffer, image]
      if (next.some((buffer, i) => buffer !== buffers[i])) {
        buffers = next
        group = device.createBindGroup({ label: 'Current refracted paths', layout: pipeline.getBindGroupLayout(0),
          entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) })
      }
      data.set(vp)
      data.set([width, height], 16)
      words.set([lens.columns, lens.rows], 18)
      data.set([lens.sizeX, lens.sizeY, receiver.sizeX, receiver.sizeY], 20)
      words.set([receiver.width, receiver.height, Number(image !== noImage), Number(hasDrawing)], 24)
      data[28] = incidentIrradiance * lens.sizeX * lens.sizeY / (receiver.sizeX * receiver.sizeY)
      const strataX = Math.min(4, lens.columns - 1), strataY = Math.min(3, lens.rows - 1)
      words.set([strataX, strataY], 30)
      device.queue.writeBuffer(uniform, 0, data)
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, group)
      pass.draw(4, strataX * strataY * (hasDrawing ? 2 : 1))
    },
    destroy() { uniform.destroy(); noImage.destroy() },
  }
}
