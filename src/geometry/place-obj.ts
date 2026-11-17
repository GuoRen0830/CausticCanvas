import type { ReceiverPlane } from '../contracts'
import { sceneVertexStride, type SceneMesh } from '../rendering/scene-mesh'

// Fit the freshly parsed Y-up model into the illuminated part of the floor.
// Bake one rigid rotation and uniform scale into every object in the file.
export function placeObjOnReceiver(
  mesh: SceneMesh, lens: { sizeX: number; sizeY: number; thickness: number }, receiver: ReceiverPlane,
) {
  const v = mesh.vertices
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity]
  for (let i = 0; i < v.length; i += sceneVertexStride) {
    for (let axis = 0; axis < 3; axis++) {
      lo[axis] = Math.min(lo[axis], v[i + axis])
      hi[axis] = Math.max(hi[axis], v[i + axis])
    }
  }
  const headroom = receiver.z - lens.thickness
  // A small contact offset keeps a horizontal sheet distinguishable from the floor.
  const floorZ = receiver.z - headroom * 1e-4
  const scale = 0.8 * Math.min(
    Math.min(lens.sizeX, receiver.sizeX) / (hi[0] - lo[0]),
    Math.min(lens.sizeY, receiver.sizeY) / (hi[2] - lo[2]),
    (floorZ - lens.thickness) / (hi[1] - lo[1]),
  )
  if (!Number.isFinite(scale) || scale <= 0) throw new Error('Cannot fit this model between the lens and floor.')
  const centerX = 0.5 * (lo[0] + hi[0]), centerZ = 0.5 * (lo[2] + hi[2])
  for (let i = 0; i < v.length; i += sceneVertexStride) {
    const y = v[i + 1], ny = v[i + 4]
    v[i] = (v[i] - centerX) * scale
    v[i + 1] = (v[i + 2] - centerZ) * scale
    v[i + 2] = floorZ - (y - lo[1]) * scale
    // Maya (X,Y,Z) -> optical (X,Z,-Y) preserves winding.
    v[i + 4] = v[i + 5]
    v[i + 5] = -ny
  }
  return { width: (hi[0] - lo[0]) * scale, height: (hi[1] - lo[1]) * scale, depth: (hi[2] - lo[2]) * scale }
}
