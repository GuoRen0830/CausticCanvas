import { mat4, type Mat4 } from 'wgpu-matrix'
import type { LensState } from '../contracts'

interface FrameLighting {
  environmentLight: number
  incidentIrradiance: number
  exposure: number
  colorFilter?: GPUTextureView
  materialsEnabled?: boolean
}

// Shared binary layout with Frame in beam-common.wgsl. Each renderer owns its
// uniform and scratch matrices so an export can use a different projection.
export function createBeamFrame(device: GPUDevice, opticalToWorld: Mat4, label: string) {
  const data = new Float32Array(44)
  const words = new Uint32Array(data.buffer)
  const localVP = data.subarray(0, 16), inverseLocalVP = data.subarray(16, 32)
  const uniform = device.createBuffer({ label, size: data.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  return {
    uniform, localVP, inverseLocalVP,
    update(viewProjection: Mat4, lens: Readonly<LensState>, width: number, height: number, mode: number,
      lighting: FrameLighting) {
      mat4.multiply(viewProjection, opticalToWorld, localVP)
      mat4.inverse(localVP, inverseLocalVP)
      words.set([lens.columns, lens.rows], 32)
      data.set([lens.sizeX, lens.sizeY, lighting.environmentLight, Math.max(lighting.incidentIrradiance, 1e-6),
        width, height], 34)
      words[40] = mode
      data[41] = lighting.exposure
      words[42] = Number(!!lighting.colorFilter) | (lighting.materialsEnabled === false ? 2 : 0)
      data[43] = lens.flatThickness ?? 0
      device.queue.writeBuffer(uniform, 0, data)
    },
    destroy() { uniform.destroy() },
  }
}
