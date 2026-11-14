import { buildSceneBvh } from './scene-bvh'
import type { SceneMesh } from './scene-mesh'
import type { TriangleBeamField } from '../contracts'

// The scene is immutable for this renderer's lifetime. Optical results can
// reuse the very same storage buffers, so the evaluation token is separate.
export function createHitTracer(
  device: GPUDevice, module: GPUShaderModule, uniform: GPUBuffer,
  mesh: SceneMesh,
) {
  const bvh = buildSceneBvh(mesh)
  function upload(label: string, data: Uint32Array<ArrayBuffer>) {
    if (data.byteLength > device.limits.maxStorageBufferBindingSize) throw new Error(label + ' exceeds storage limit.')
    const buffer = device.createBuffer({ label, size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
    device.queue.writeBuffer(buffer, 0, data)
    return buffer
  }
  const nodes = upload('Scene BVH nodes', bvh.nodes)
  const triangles = upload('Scene BVH leaf triangles', bvh.triangles)
  const trace = device.createComputePipeline({
    label: 'Closest scene hit per source triangle', layout: 'auto',
    compute: { module, entryPoint: 'traceFirstHit' },
  })
  const frameGroup = device.createBindGroup({ layout: trace.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: uniform } },
  ] })
  // Explicit vertex-only layout keeps the cached hits read-only while drawing.
  const splatLayout = device.createBindGroupLayout({ entries: [0, 3].map(binding => ({
    binding, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' },
  })) })
  let hits: GPUBuffer | undefined
  let sceneGroup: GPUBindGroup
  let splatGroup: GPUBindGroup
  let fieldGroup: GPUBindGroup
  let lastField: TriangleBeamField | undefined
  let lastToken: object | undefined
  let lastCount = 0
  return {
    splatLayout,
    update(encoder: GPUCommandEncoder, field: TriangleBeamField, count: number, token: object) {
      if (!hits || hits.size < 8 * count) {
        if (8 * count > device.limits.maxStorageBufferBindingSize) throw new Error('Hit buffer exceeds storage limit.')
        hits?.destroy()
        hits = device.createBuffer({
          label: 'Cached first hits', size: 8 * count,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        })
        sceneGroup = device.createBindGroup({ layout: trace.getBindGroupLayout(2), entries: [
          { binding: 0, resource: { buffer: triangles } }, { binding: 1, resource: { buffer: nodes } },
          { binding: 2, resource: { buffer: hits } },
        ] })
        splatGroup = device.createBindGroup({ layout: splatLayout, entries: [
          { binding: 0, resource: { buffer: triangles } }, { binding: 3, resource: { buffer: hits } },
        ] })
        lastToken = undefined
      }
      if (lastField?.buffer !== field.buffer || lastField?.lens.heightBuffer !== field.lens.heightBuffer) {
        fieldGroup = device.createBindGroup({ layout: trace.getBindGroupLayout(1), entries: [
          { binding: 0, resource: { buffer: field.lens.heightBuffer } }, { binding: 1, resource: { buffer: field.buffer } },
        ] })
        lastToken = undefined
      }
      lastField = field
      if (lastToken !== token || lastCount !== count) {
        const pass = encoder.beginComputePass({ label: 'Trace first hits' })
        pass.setPipeline(trace)
        pass.setBindGroup(0, frameGroup)
        pass.setBindGroup(1, fieldGroup)
        pass.setBindGroup(2, sceneGroup)
        const groups = Math.ceil(count / 64)
        const width = Math.min(groups, device.limits.maxComputeWorkgroupsPerDimension)
        pass.dispatchWorkgroups(width, Math.ceil(groups / width))
        pass.end()
        lastToken = token
        lastCount = count
      }
      return { group: splatGroup, buffer: hits }
    },
    destroy() { hits?.destroy(); nodes.destroy(); triangles.destroy() },
  }
}
