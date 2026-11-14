import { vec3 } from 'wgpu-matrix'
import type { ReceiverPlane } from '../contracts'

type Triple = readonly [number, number, number]

export interface SceneBox {
  // Lens-local coordinates and full dimensions, in meters.
  center: Triple
  size: Triple
}

// Vertex layout: position(3), normal(3), albedo(3), floor flag, object ID.
export interface SceneMesh {
  vertices: Float32Array<ArrayBuffer>
  indices: Uint32Array<ArrayBuffer>
}

export const sceneVertexStride = 11

export function buildSceneMesh(receiver: ReceiverPlane, box: SceneBox): SceneMesh {
  const dimensions = [receiver.sizeX, receiver.sizeY, ...box.size]
  if (dimensions.some((value) => !Number.isFinite(value) || value <= 0)) {
    throw new Error('Scene dimensions must be finite and positive.')
  }
  if (![receiver.z, ...box.center].every(Number.isFinite)) {
    throw new Error('Scene positions must be finite.')
  }

  const vertices: number[] = []
  const indices: number[] = []

  const addQuad = (center: Triple, u: Triple, v: Triple, color: Triple, isFloor = false) => {
    const first = vertices.length / sceneVertexStride
    const normal = vec3.normalize(vec3.cross(u, v))
    // u and v are half-edge vectors; u cross v determines the front face.
    for (const [s, t] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      vertices.push(
        center[0] + s * u[0] + t * v[0],
        center[1] + s * u[1] + t * v[1],
        center[2] + s * u[2] + t * v[2],
        normal[0], normal[1], normal[2],
        ...color, Number(isFloor), isFloor ? 1 : 2,
      )
    }
    indices.push(first, first + 1, first + 2, first, first + 2, first + 3)
  }

  // The receiver faces the lens, toward local -Z.
  addQuad(
    [0, 0, receiver.z],
    [receiver.sizeX / 2, 0, 0],
    [0, -receiver.sizeY / 2, 0],
    [0.48, 0.52, 0.58],
    true,
  )

  const [x, y, z] = box.center
  const [hx, hy, hz] = box.size.map((value) => value / 2)
  const color: Triple = [0.85, 0.43, 0.16]

  // Each face owns four vertices so the box keeps sharp normal discontinuities.
  addQuad([x + hx, y, z], [0, hy, 0], [0, 0, hz], color)
  addQuad([x - hx, y, z], [0, 0, hz], [0, hy, 0], color)
  addQuad([x, y + hy, z], [0, 0, hz], [hx, 0, 0], color)
  addQuad([x, y - hy, z], [hx, 0, 0], [0, 0, hz], color)
  addQuad([x, y, z + hz], [hx, 0, 0], [0, hy, 0], color)
  addQuad([x, y, z - hz], [0, hy, 0], [hx, 0, 0], color)

  return {
    vertices: new Float32Array(vertices),
    indices: new Uint32Array(indices),
  }
}
