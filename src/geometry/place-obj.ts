import type { ReceiverPlane } from '../contracts'
import { sceneVertexStride, type SceneMesh } from '../rendering/scene-mesh'

// One uniform scale preserves the Y-up mesh. Its top-down bounds determine the
// aperture, receiver and target raster in the same optical coordinate system.
export function placeObjOnReceiver(
  mesh: SceneMesh, lens: { sizeX: number; sizeY: number; thickness: number }, limits: ReceiverPlane,
) {
  const v = mesh.vertices
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity]
  for (let i = 0; i < v.length; i += sceneVertexStride) {
    for (let axis = 0; axis < 3; axis++) {
      lo[axis] = Math.min(lo[axis], v[i + axis])
      hi[axis] = Math.max(hi[axis], v[i + axis])
    }
  }
  const headroom = limits.z - lens.thickness
  // A small contact offset keeps a horizontal sheet distinguishable from the floor.
  const floorZ = limits.z - headroom * 1e-4
  const scale = Math.min(
    Math.min(lens.sizeX, limits.sizeX) / (hi[0] - lo[0]),
    Math.min(lens.sizeY, limits.sizeY) / (hi[2] - lo[2]),
    0.9 * headroom / (hi[1] - lo[1]),
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
  const width = (hi[0] - lo[0]) * scale, height = (hi[1] - lo[1]) * scale, depth = (hi[2] - lo[2]) * scale
  const pixelSize = Math.max(width / limits.width, depth / limits.height)
  if (!(pixelSize > 0)) throw new Error('Model has no top-down extent.')
  const receiver: ReceiverPlane = {
    z: limits.z,
    sizeX: Math.max(width, pixelSize), sizeY: Math.max(depth, pixelSize),
    width: Math.max(1, Math.round(width / pixelSize)), height: Math.max(1, Math.round(depth / pixelSize)),
  }
  return { width, height, depth, receiver }
}
