import type { ReceiverPlane } from '../contracts'
import { config, sceneThrowPerRadius } from '../config'
import { sceneVertexStride, type SceneMesh } from '../rendering/scene-mesh'

// One uniform scale preserves the Y-up mesh. Its top-down bounds determine the
// receiver and target raster in the same optical coordinate system. The
// scene lamp is sized afterwards, independently of this mesh-fitting envelope.
export function placeObjOnReceiver(
  mesh: SceneMesh, envelope: { sizeX: number; sizeY: number; thickness: number }, limits: ReceiverPlane,
) {
  const v = mesh.vertices
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity]
  for (let i = 0; i < v.length; i += sceneVertexStride) {
    for (let axis = 0; axis < 3; axis++) {
      lo[axis] = Math.min(lo[axis], v[i + axis])
      hi[axis] = Math.max(hi[axis], v[i + axis])
    }
  }
  const headroom = limits.z - envelope.thickness
  const scale = Math.min(
    Math.min(envelope.sizeX, limits.sizeX) / (hi[0] - lo[0]),
    Math.min(envelope.sizeY, limits.sizeY) / (hi[2] - lo[2]),
    0.9 * headroom / (hi[1] - lo[1]),
  )
  if (!Number.isFinite(scale) || scale <= 0) throw new Error('Cannot fit this model between the lens and floor.')
  const width = (hi[0] - lo[0]) * scale, height = (hi[1] - lo[1]) * scale, depth = (hi[2] - lo[2]) * scale
  const flatThickness = config.sceneApertureFraction * envelope.thickness
  const throwPerRadius = sceneThrowPerRadius()
  // Cover the floor corners and keep the lamp above any nearly on-axis peak.
  let throwDistance = Math.max(0.5 * Math.hypot(width, depth) * throwPerRadius,
    height + Math.max(2 * flatThickness, 0.18 * Math.max(width, depth)))
  const centerX = 0.5 * (lo[0] + hi[0]), centerZ = 0.5 * (lo[2] + hi[2])
  for (let i = 0; i < v.length; i += sceneVertexStride) {
    const elevation = (v[i + 1] - lo[1]) * scale, ny = v[i + 4]
    v[i] = (v[i] - centerX) * scale
    v[i + 1] = (v[i + 2] - centerZ) * scale
    v[i + 2] = -elevation
    // A high off-axis vertex needs a longer throw than a peak near the center.
    throwDistance = Math.max(throwDistance, elevation + Math.hypot(v[i], v[i + 1]) * throwPerRadius)
    // Maya (X,Y,Z) -> optical (X,Z,-Y) preserves winding.
    v[i + 4] = v[i + 5]
    v[i + 5] = -ny
  }
  const receiverZ = Math.min(limits.z, flatThickness + throwDistance)
  // A small contact offset keeps a horizontal sheet distinguishable from the floor.
  const floorZ = receiverZ - (receiverZ - flatThickness) * 1e-4
  for (let i = 2; i < v.length; i += sceneVertexStride) v[i] += floorZ
  const pixelSize = Math.max(width / limits.width, depth / limits.height)
  if (!(pixelSize > 0)) throw new Error('Model has no top-down extent.')
  const receiver: ReceiverPlane = {
    z: receiverZ,
    sizeX: Math.max(width, pixelSize), sizeY: Math.max(depth, pixelSize),
    width: Math.max(1, Math.round(width / pixelSize)), height: Math.max(1, Math.round(depth / pixelSize)),
  }
  return { width, height, depth, receiver }
}
