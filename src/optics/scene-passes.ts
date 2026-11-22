import type { DesignView } from './design-view'
import opticsShader from './optics.wgsl?raw'
import hitTypes from '../rendering/shaders/scene-hit.wgsl?raw'
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
    code: [opticsShader, hitTypes, bvhShader, sceneShader].join('\n') })
  const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === 'error')
  if (errors.length) throw new Error(errors.map(m => `Scene optics line ${m.lineNum}: ${m.message}`).join('\n'))
  // Backward and gather run once per differentiated coordinate: z, then x and y
  // when the caller asks for vertex-motion gradients.
  const stage = (entryPoint: string, derivativeMode = 0) => device.createComputePipelineAsync({
    label: derivativeMode ? `${entryPoint} ${'zxy'[derivativeMode]}` : entryPoint, layout: 'auto',
    compute: { module, entryPoint, constants: { derivativeMode } } })
  const [trace, forward, ...modes] = await Promise.all([stage('sceneTrace'), stage('sceneForward'),
    ...[0, 1, 2].flatMap(mode => [stage('sceneBackward', mode), stage('gather', mode)])])
  const backwards = [modes[0], modes[2], modes[4]], gathers = [modes[1], modes[3], modes[5]]
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
    encode(encoder: GPUCommandEncoder, params: GPUBuffer, heights: GPUBuffer, offsets: GPUBuffer) {
      run(encoder, trace, faceGrid, [[0,params],[1,heights],[2,beams],[8,triangles],[9,nodes],[10,hits],[12,offsets]])
      run(encoder, forward, faceGrid, [[0,params],[1,heights],[2,beams],[3,image],[8,triangles],[10,hits],[11,pixels],[12,offsets]])
    },
    backward(encoder: GPUCommandEncoder, params: GPUBuffer, heights: GPUBuffer, offsets: GPUBuffer,
      upstream: GPUBuffer, output: GPUBuffer, offsetOutput?: GPUBuffer) {
      for (let mode = 0; mode < (offsetOutput ? 3 : 1); mode++) {
        run(encoder, backwards[mode], faceGrid, [[0,params],[1,heights],[2,beams],[4,upstream],[5,triangleGradients],
          [8,triangles],[10,hits],[11,pixels],[12,offsets]])
        run(encoder, gathers[mode], vertexGrid, [[0,params],[5,triangleGradients],[6,mode ? offsetOutput! : output]])
      }
    },
  }
}
