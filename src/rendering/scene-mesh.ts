import { vec3 } from 'wgpu-matrix'
import type { ReceiverPlane } from '../contracts'

type Triple = readonly [number, number, number]
type Quad = readonly [center: Triple, u: Triple, v: Triple]

export interface SceneBox {
  // Lens-local coordinates and full dimensions, in meters.
  center: Triple
  size: Triple
}

// Vertex layout: position(3), normal(3), albedo(3), floor flag, object ID.
// A model uses IDs >= 2. The scene reserves ID 1 for the receiver floor.
export interface SceneMesh {
  vertices: Float32Array<ArrayBuffer>
  indices: Uint32Array<ArrayBuffer>
}

export const sceneVertexStride = 11

export interface SceneGeometry {
  mesh: SceneMesh
  // Only the built-in box has an exact analytic occlusion reference.
  referenceBox?: SceneBox
}

function meshFromQuads(quads: Quad[], color: Triple, objectId: number): SceneMesh {
  const vertices: number[] = []
  const indices: number[] = []
  for (const [center, u, v] of quads) {
    const first = vertices.length / sceneVertexStride
    const normal = vec3.normalize(vec3.cross(u, v))
    // Each face owns its vertices to preserve sharp edges.
    for (const [s, t] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      vertices.push(
        center[0] + s * u[0] + t * v[0],
        center[1] + s * u[1] + t * v[1],
        center[2] + s * u[2] + t * v[2],
        normal[0], normal[1], normal[2],
        ...color, Number(objectId === 1), objectId,
      )
    }
    indices.push(first, first + 1, first + 2, first, first + 2, first + 3)
  }
  return { vertices: new Float32Array(vertices), indices: new Uint32Array(indices) }
}

function buildFloorMesh(receiver: ReceiverPlane): SceneMesh {
  if (![receiver.sizeX, receiver.sizeY].every(value => Number.isFinite(value) && value > 0) ||
      !Number.isFinite(receiver.z)) {
    throw new Error('Invalid receiver plane dimensions or position.')
  }
  // The receiver faces the lens, toward local -Z.
  return meshFromQuads([
    [[0, 0, receiver.z], [receiver.sizeX / 2, 0, 0], [0, -receiver.sizeY / 2, 0]],
  ], [0.48, 0.52, 0.58], 1)
}

function buildBoxMesh(box: SceneBox): SceneMesh {
  if (!box.size.every(value => Number.isFinite(value) && value > 0) || !box.center.every(Number.isFinite)) {
    throw new Error('Invalid box dimensions or position.')
  }
  const [x, y, z] = box.center
  const [hx, hy, hz] = box.size.map(value => value / 2)
  return meshFromQuads([
    [[x + hx, y, z], [0, hy, 0], [0, 0, hz]],
    [[x - hx, y, z], [0, 0, hz], [0, hy, 0]],
    [[x, y + hy, z], [0, 0, hz], [hx, 0, 0]],
    [[x, y - hy, z], [hx, 0, 0], [0, 0, hz]],
    [[x, y, z + hz], [hx, 0, 0], [0, hy, 0]],
    [[x, y, z - hz], [0, hy, 0], [hx, 0, 0]],
  ], [0.85, 0.43, 0.16], 2)
}

// Assemble once and share the resulting mesh with the design view and both renderers.
// Custom model positions must already be in lens-local meters.
export function buildSceneGeometry(receiver: ReceiverPlane, source: { box: SceneBox } | { mesh: SceneMesh }): SceneGeometry {
  const floor = buildFloorMesh(receiver)
  const model = 'box' in source ? buildBoxMesh(source.box) : source.mesh
  const vertexCount = model.vertices.length / sceneVertexStride
  if (!Number.isInteger(vertexCount) || !model.indices.length || model.indices.length % 3 !== 0) {
    throw new Error('Expected an indexed triangle model.')
  }
  for (let start = 0; start < model.vertices.length; start += sceneVertexStride) {
    const objectId = model.vertices[start + 10]
    if (model.vertices[start + 9] !== 0 || !Number.isInteger(objectId) || objectId < 2 || objectId > 0xffffff) {
      throw new Error('Model vertices must be non-floor vertices with object IDs from 2 to 0xffffff.')
    }
  }
  const vertices = new Float32Array(floor.vertices.length + model.vertices.length)
  vertices.set(floor.vertices)
  vertices.set(model.vertices, floor.vertices.length)
  const indices = new Uint32Array(floor.indices.length + model.indices.length)
  indices.set(floor.indices)
  const offset = floor.vertices.length / sceneVertexStride
  for (let i = 0; i < model.indices.length; i++) {
    const index = model.indices[i]
    if (index >= vertexCount) throw new Error('Model vertex index out of range.')
    indices[floor.indices.length + i] = offset + index
  }
  const mesh = { vertices, indices }
  return 'box' in source ? { mesh, referenceBox: source.box } : { mesh }
}
