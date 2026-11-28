import type { Mat4 } from 'wgpu-matrix'
import { config, incidentColumnHeight } from '../config'
import marchShader from './shaders/volume-march.wgsl?raw'

// RGB stores linear scattered radiance. A stores the opaque-surface distance
// from the near plane for depth-aware upsampling.
export function createVolumeMarch(device: GPUDevice, fullResolution = false) {
  const format: GPUTextureFormat = 'rgba16float'
  const module = device.createShaderModule({ label: 'Room single scattering', code: marchShader })
  const pipeline = device.createRenderPipeline({
    label: 'Integrate visible light volume',
    layout: 'auto',
    vertex: { module, entryPoint: 'fullscreen' },
    fragment: { module, entryPoint: 'integrate', targets: [{ format }],
      constants: { scatteringPerMeter: config.rendering.roomScattering } },
  })
  const data = new Float32Array(32)
  const uniform = device.createBuffer({
    label: 'Light volume camera', size: data.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })

  let target: GPUTexture | undefined
  let outputView: GPUTextureView
  let bound: { depth: GPUTextureView; field: GPUTextureView; sampler: GPUSampler; group: GPUBindGroup } | undefined

  return {
    render(encoder: GPUCommandEncoder, width: number, height: number, depth: GPUTextureView,
      field: GPUTextureView, sampler: GPUSampler, inverseLocalVP: Mat4,
      bounds: { min: [number, number, number]; max: [number, number, number] },
      aperture: [number, number], receiverZ: number, incidentIrradiance: number, relativeIOR?: number,
      receiverSize: [number, number] = aperture) {
      const marchWidth = fullResolution ? width : Math.ceil(width / 2)
      const marchHeight = fullResolution ? height : Math.ceil(height / 2)
      if (!target || target.width !== marchWidth || target.height !== marchHeight) {
        target?.destroy()
        target = device.createTexture({
          label: 'Integrated fog and surface distance', size: [marchWidth, marchHeight], format,
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        })
        outputView = target.createView()
      }
      if (!bound || bound.depth !== depth || bound.field !== field || bound.sampler !== sampler) {
        const group = device.createBindGroup({
          label: 'Light volume sampling', layout: pipeline.getBindGroupLayout(0), entries: [
            { binding: 0, resource: { buffer: uniform } }, { binding: 1, resource: depth },
            { binding: 2, resource: field }, { binding: 3, resource: sampler },
          ],
        })
        bound = { depth, field, sampler, group }
      }

      data.set(inverseLocalVP, 0)
      data.set(bounds.min, 16)
      data[19] = incidentColumnHeight(receiverZ, Math.max(...aperture))
      data.set(bounds.max, 20)
      data[23] = incidentIrradiance
      data.set(aperture, 24)
      data[26] = marchWidth
      data[27] = marchHeight
      // Through a flat entrance the incident ray is axial. At the steepest
      // transmitting exit, tan(max deflection) = sqrt(max(eta, 1/eta)^2 - 1).
      data[28] = relativeIOR === undefined ? -1 : Math.sqrt(Math.max(relativeIOR, 1 / relativeIOR) ** 2 - 1)
      data[30] = receiverSize[0] / aperture[0]
      data[31] = receiverSize[1] / aperture[1]
      device.queue.writeBuffer(uniform, 0, data)
      const pass = encoder.beginRenderPass({
        label: 'March visible light volume',
        colorAttachments: [{ view: outputView, loadOp: 'clear', clearValue: [0, 0, 0, 60000], storeOp: 'store' }],
      })
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bound.group)
      pass.draw(3)
      pass.end()
      return outputView
    },
    destroy() {
      target?.destroy()
      uniform.destroy()
    },
  }
}
