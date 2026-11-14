import { vec3 } from 'wgpu-matrix'
import type { TriangleBeamField } from '../contracts'
import { createFlatLens } from '../geometry/lens-mesh'

// Two faces of a small flat grid use the same GPU layout as the optical solver.
// Drawing just face 0 reproduces the original single-beam fixture.
export function createTestBeams(
  device: GPUDevice, aperture: { sizeX: number; sizeY: number; thickness: number },
) {
  const lens = createFlatLens(device, {
    columns: 2, rows: 2, sizeX: aperture.sizeX / 2, sizeY: aperture.sizeY / 2,
    thickness: aperture.thickness,
  })
  const values = new Float32Array(8)
  values.set(vec3.normalize([-0.18, 0.15, 1]))
  values.set(vec3.normalize([-0.24, 0.09, 1]), 4)
  values[3] = values[7] = 0.012
  const buffer = device.createBuffer({
    label: 'Two test beams', size: values.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(buffer, 0, values)
  const field: TriangleBeamField = { lens, buffer }
  return {
    field, offset: [-0.125 * aperture.sizeX, -0.075 * aperture.sizeY],
    destroy() { buffer.destroy(); lens.heightBuffer.destroy() },
  }
}
