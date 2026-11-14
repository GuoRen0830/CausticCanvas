import type { ReceiverPlane } from '../contracts'
import { buildSceneBvh } from '../rendering/scene-bvh'
import type { SceneMesh } from '../rendering/scene-mesh'

// One sample per pixel in the parallel incident-light view. Row zero is -Y,
// exactly like the target and caustic image. Triangle indices are BVH indices.
export function buildDesignView(mesh: SceneMesh, receiver: ReceiverPlane) {
  const { width, height, sizeX, sizeY } = receiver
  const bvh = buildSceneBvh(mesh)
  const vertices = new Float32Array(bvh.triangles.buffer)
  const samples = new Uint32Array(width * height * 2)
  const depths = new Float32Array(samples.buffer)
  const surfaceAreas = new Float32Array(width * height)
  const pixelArea = (sizeX / width) * (sizeY / height)
  const col = (x: number) => (x / sizeX + 0.5) * width - 0.5
  const row = (y: number) => (y / sizeY + 0.5) * height - 0.5
  for (let base = 0; base < vertices.length; base += 12) {
    const ax = vertices[base], ay = vertices[base + 1], az = vertices[base + 2]
    const ux = vertices[base + 4] - ax, uy = vertices[base + 5] - ay, uz = vertices[base + 6] - az
    const vx = vertices[base + 8] - ax, vy = vertices[base + 9] - ay, vz = vertices[base + 10] - az
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx
    // Same single-sided convention as the scene G-buffer.
    if (nz >= 0) continue
    const area = pixelArea * Math.hypot(nx, ny, nz) / -nz
    const x0 = Math.max(0, Math.ceil(col(ax + Math.min(0, ux, vx))))
    const x1 = Math.min(width - 1, Math.floor(col(ax + Math.max(0, ux, vx))))
    const y0 = Math.max(0, Math.ceil(row(ay + Math.min(0, uy, vy))))
    const y1 = Math.min(height - 1, Math.floor(row(ay + Math.max(0, uy, vy))))
    for (let y = y0; y <= y1; y++) {
      const dy = ((y + 0.5) / height - 0.5) * sizeY - ay
      for (let x = x0; x <= x1; x++) {
        const dx = ((x + 0.5) / width - 0.5) * sizeX - ax
        const u = (dx * vy - dy * vx) / nz, v = (ux * dy - uy * dx) / nz
        if (u < -1e-10 || v < -1e-10 || u + v > 1 + 1e-10) continue
        const z = Math.fround(az + u * uz + v * vz)
        const i = y * width + x
        if (z <= 0 || (samples[2 * i] && z >= depths[2 * i + 1])) continue
        samples[2 * i] = base / 12 + 1
        depths[2 * i + 1] = z
        surfaceAreas[i] = area
      }
    }
  }
  return { mesh, bvh, samples, surfaceAreas, width, height, sizeX, sizeY }
}

export type DesignView = ReturnType<typeof buildDesignView>
