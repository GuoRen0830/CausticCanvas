import type { ReceiverPlane } from '../contracts'

// Vertex layout: position(3), normal(3), albedo(3), floor flag, object ID.
// A model uses IDs >= 2. The scene reserves ID 1 for the receiver floor.
export interface SceneMesh {
  vertices: Float32Array<ArrayBuffer>
  indices: Uint32Array<ArrayBuffer>
  // UVs are a separate render-only stream. BVH triangles use the original layout.
  uvs?: Float32Array<ArrayBuffer>
  uvs1?: Float32Array<ArrayBuffer>
  // A zero W means that the shader should reconstruct a tangent from the UVs.
  tangents?: Float32Array<ArrayBuffer>
  draws?: SceneDraw[]
}

export interface SceneDraw {
  firstIndex: number
  indexCount: number
  material?: string
  hasUV?: boolean
  uvMask?: number // Bit 0/1: this draw has complete data for the corresponding UV stream.
  doubleSided?: boolean // Copied from the material so ray hits and rasterization agree.
}

export interface SceneMaterial {
  color: [number, number, number]
  doubleSided?: boolean
  vertexColor?: boolean // glTF COLOR_0 multiplies the base color factor and texture.
  // Optional water film on vertex-tinted surfaces. Red goes from 1 (dry) to
  // saturatedRed (fully wet). Ordinary vertex colors affect only base color.
  vertexWetness?: { saturatedRed: number; saturatedRoughness: number }
  map?: File
  uvTransform?: MaterialUVTransform
  clamp?: boolean
  roughness?: number
  metallic?: number
  metallicRoughnessMap?: File // glTF: green roughness, blue metallic, linear data
  metallicRoughnessUVTransform?: MaterialUVTransform
  metallicRoughnessClamp?: boolean
  roughnessMap?: File // standalone scalar maps use red
  roughnessUVTransform?: MaterialUVTransform
  roughnessClamp?: boolean
  metallicMap?: File
  metallicUVTransform?: MaterialUVTransform
  metallicClamp?: boolean
  normalMap?: File
  normalUVTransform?: MaterialUVTransform
  normalClamp?: boolean
  normalScale?: number
  normalYSign?: 1 | -1 // Tangent basis uses OBJ V-up. glTF's V axis is reversed.
  occlusionMap?: File // Red channel, linear data. Modulates ambient illumination.
  occlusionStrength?: number
  occlusionUVTransform?: MaterialUVTransform
  occlusionClamp?: boolean
  emissive?: [number, number, number] // Linear radiance, including emissive strength.
  emissiveMap?: File // RGB uses the sRGB transfer function.
  emissiveUVTransform?: MaterialUVTransform
  emissiveClamp?: boolean
  sampling?: Partial<Record<SceneMapSlot, SceneMapSampling>>
}

export type SceneMapSlot = 'map' | 'metallicRoughnessMap' | 'normalMap' | 'roughnessMap' |
  'metallicMap' | 'occlusionMap' | 'emissiveMap'
export interface SceneMapSampling {
  uvSet?: 0 | 1
  rotation?: number // Radians in glTF top-left UV space, rotated after scaling and before offset.
  sampler?: GPUSamplerDescriptor
}

// Scale U/V and then offset U/V in the mesh's bottom-left coordinate system.
export type MaterialUVTransform = [number, number, number, number]

export function sceneAlbedoFormat(materials?: ReadonlyMap<string, SceneMaterial>): GPUTextureFormat {
  return materials?.size ? 'rgba8unorm-srgb' : 'rgba8unorm'
}

export const sceneVertexStride = 11

export interface SceneGeometry {
  mesh: SceneMesh
  materials?: Map<string, SceneMaterial>
}

function buildFloorMesh({ sizeX, sizeY, z }: ReceiverPlane): SceneMesh {
  const x = sizeX / 2, y = sizeY / 2
  // Counterclockwise as seen from the lens, with floor object ID 1.
  const vertices = [[-x, y], [x, y], [x, -y], [-x, -y]].flatMap(([px, py]) =>
    [px, py, z, 0, 0, -1, 0.48, 0.52, 0.58, 1, 1])
  return { vertices: new Float32Array(vertices), indices: new Uint32Array([0, 1, 2, 0, 2, 3]) }
}

// Assemble once and share the resulting mesh with the design view and both renderers.
// Custom model positions must already be in lens-local meters.
export function buildSceneGeometry(receiver: ReceiverPlane,
  { mesh: model, receiverFloor = true }: { mesh: SceneMesh; receiverFloor?: boolean }): SceneGeometry {
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
  if (model.indices.some(index => index >= vertexCount)) throw new Error('Model vertex index out of range.')
  if (model.uvs && model.uvs.length !== 2 * vertexCount) throw new Error('Expected one UV pair per model vertex.')
  if (model.uvs1 && model.uvs1.length !== 2 * vertexCount) throw new Error('Expected one secondary UV pair per model vertex.')
  if (model.tangents && model.tangents.length !== 4 * vertexCount) throw new Error('Expected one tangent and sign per model vertex.')
  // Terrain and street scenes already contain the receiving surface. Use it
  // for both drawing and ray hits, leaving areas outside its outline empty.
  if (!receiverFloor) return { mesh: model }
  const floor = buildFloorMesh(receiver)
  const vertices = new Float32Array(floor.vertices.length + model.vertices.length)
  vertices.set(floor.vertices)
  vertices.set(model.vertices, floor.vertices.length)
  const indices = new Uint32Array(floor.indices.length + model.indices.length)
  indices.set(floor.indices)
  const offset = floor.vertices.length / sceneVertexStride
  for (let i = 0; i < model.indices.length; i++) {
    indices[floor.indices.length + i] = offset + model.indices[i]
  }
  const mesh: SceneMesh = { vertices, indices }
  for (const [key, width] of [['uvs', 2], ['uvs1', 2], ['tangents', 4]] as const) {
    const stream = model[key]
    if (!stream) continue
    const fitted = new Float32Array(width * offset + stream.length)
    fitted.set(stream, width * offset)
    mesh[key] = fitted
  }
  if (model.draws) {
    mesh.draws = [
      { firstIndex: 0, indexCount: floor.indices.length },
      ...model.draws.map(draw => ({ ...draw, firstIndex: floor.indices.length + draw.firstIndex })),
    ]
  }
  return { mesh }
}
