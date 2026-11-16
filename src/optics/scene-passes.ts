import type { DesignView } from './design-view'
import opticsShader from './optics.wgsl?raw'
import bvhShader from '../rendering/shaders/bvh-query.wgsl?raw'
import sceneShader from './scene-optics.wgsl?raw'

// Each entry point reflects only the bindings it actually uses. This keeps even
// the backward pass within the standard eight storage bindings per shader stage.
export async function createScenePasses(
  device: GPUDevice, view: DesignView, faceCount: number, faceGrid: [number, number], vertexGrid: [number, number],
  beams: GPUBuffer, image: GPUBuffer, triangleGradients: GPUBuffer,
  allocate: (label: string, size: number, usage: GPUBufferUsageFlags) => GPUBuffer,
  timestamps?: (stage: string) => GPUComputePassTimestampWrites | undefined,
) {
  const module = device.createShaderModule({ label: 'Differentiable scene receiver',
    code: [opticsShader, bvhShader, sceneShader].join('\n') })
  const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === 'error')
  if (errors.length) throw new Error(errors.map(m => `Scene optics line ${m.lineNum}: ${m.message}`).join('\n'))
  const [trace, forward, backward, gather] = await Promise.all(
    ['sceneTrace', 'sceneForward', 'sceneBackward', 'gather'].map(entryPoint =>
      device.createComputePipelineAsync({ label: entryPoint, layout: 'auto', compute: { module, entryPoint } })),
  )
  function upload(label: string, data: Uint32Array<ArrayBuffer>) {
    const buffer = allocate(label, data.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST)
    device.queue.writeBuffer(buffer, 0, data)
    return buffer
  }
  const triangles = upload('Training triangles', view.bvh.triangles)
  const nodes = upload('Training BVH nodes', view.bvh.nodes)
  const pixels = upload('Light-view visible surfaces', view.samples)
  const hits = allocate('Solver first hits', faceCount * 8, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC)
  const firstHits = { mesh: view.mesh, buffer: hits, triangles }
  // Each step binds the same buffers, so keep the last bind group per pipeline
  // instead of creating four new ones per training step.
  const cached = new Map<GPUComputePipeline, { buffers: GPUBuffer[], group: GPUBindGroup }>()
  function run(encoder: GPUCommandEncoder, pipeline: GPUComputePipeline, grid: [number, number],
    bindings: [number, GPUBuffer][]) {
    const buffers = bindings.map(([, buffer]) => buffer)
    let entry = cached.get(pipeline)
    if (!entry || entry.buffers.some((buffer, i) => buffer !== buffers[i])) {
      entry = { buffers, group: device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
        entries: bindings.map(([binding, buffer]) => ({ binding, resource: { buffer } })) }) }
      cached.set(pipeline, entry)
    }
    const group = entry.group
    const pass = encoder.beginComputePass({ label: pipeline.label, timestampWrites: timestamps?.(pipeline.label) })
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, group)
    pass.dispatchWorkgroups(...grid)
    pass.end()
  }
  return {
    firstHits,
    encode(encoder: GPUCommandEncoder, params: GPUBuffer, heights: GPUBuffer) {
      run(encoder, trace, faceGrid, [[0,params],[1,heights],[2,beams],[8,triangles],[9,nodes],[10,hits]])
      run(encoder, forward, faceGrid, [[0,params],[1,heights],[2,beams],[3,image],[8,triangles],[10,hits],[11,pixels]])
    },
    backward(encoder: GPUCommandEncoder, params: GPUBuffer, heights: GPUBuffer, upstream: GPUBuffer, output: GPUBuffer) {
      run(encoder, backward, faceGrid, [[0,params],[1,heights],[2,beams],[4,upstream],[5,triangleGradients],[8,triangles],[10,hits],[11,pixels]])
      run(encoder, gather, vertexGrid, [[0,params],[5,triangleGradients],[6,output]])
    },
  }
}
