import { mat4 } from 'wgpu-matrix'
import type { Mat4 } from 'wgpu-matrix'
import type { LensState } from '../contracts'
import source from './shaders/room-glass.wgsl?raw'

// The visible shell can use a coarser mesh than the optical simulation.
const maxSurfaceCells = 96

export function createRoomGlass(device: GPUDevice) {
  const data = new Float32Array(28)
  const words = new Uint32Array(data.buffer)
  const inverse = mat4.identity()
  const uniform = device.createBuffer({
    label: 'Room glass framing', size: data.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  const layout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
  ] })
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] })
  const module = device.createShaderModule({ label: 'Room glass shell', code: source })
  const over: GPUBlendComponent = { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }
  const makePipeline = (cullMode: GPUCullMode) => device.createRenderPipeline({
    label: 'Room glass ' + cullMode, layout: pipelineLayout,
    vertex: { module, entryPoint: 'shellVS' },
    fragment: { module, entryPoint: 'shellFS', targets: [{ format: 'rgba16float', blend: { color: over, alpha: over } }] },
    primitive: { topology: 'triangle-list', cullMode },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less-equal' },
  })
  const backFaces = makePipeline('front')
  const frontFaces = makePipeline('back')
  let bindings: { heights: GPUBuffer; offsets: GPUBuffer; group: GPUBindGroup } | undefined

  return {
    draw(pass: GPURenderPassEncoder, vpLocal: Mat4, lens: LensState, offsets: GPUBuffer, relativeIOR: number) {
      if (relativeIOR === 1) return
      const cellsX = Math.min(maxSurfaceCells, lens.columns - 1)
      const cellsY = Math.min(maxSurfaceCells, lens.rows - 1)
      data.set(vpLocal, 0)
      mat4.inverse(vpLocal, inverse)
      // Inverse-project the homogeneous eye. Orthographic projections put it at infinity.
      const w = inverse[11]
      const perspective = Math.abs(w) > 1e-7
      const scale = perspective ? 1 / w : -1
      data.set([inverse[8] * scale, inverse[9] * scale, inverse[10] * scale, Number(perspective)], 16)
      words.set([lens.columns, lens.rows, cellsX, cellsY], 20)
      data.set([lens.sizeX, lens.sizeY], 24)
      data[26] = ((relativeIOR - 1) / (relativeIOR + 1)) ** 2
      device.queue.writeBuffer(uniform, 0, data)
      if (bindings?.heights !== lens.heightBuffer || bindings.offsets !== offsets) {
        bindings = { heights: lens.heightBuffer, offsets, group: device.createBindGroup({
          layout, entries: [
            { binding: 0, resource: { buffer: uniform } },
            { binding: 1, resource: { buffer: lens.heightBuffer } },
            { binding: 2, resource: { buffer: offsets } },
          ],
        }) }
      }
      const quads = cellsX * cellsY + 1 + 2 * (cellsX + cellsY)
      pass.setBindGroup(0, bindings.group)
      // Composite the far side first so the near surface transmits its highlights.
      pass.setPipeline(backFaces)
      pass.draw(6 * quads)
      pass.setPipeline(frontFaces)
      pass.draw(6 * quads)
    },
    destroy() { uniform.destroy() },
  }
}
