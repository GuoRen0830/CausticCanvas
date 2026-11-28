import { mat4, type Mat4 } from 'wgpu-matrix'
import { sceneVertexStride, type SceneMaterial, type SceneDraw, type SceneMesh, type SceneMapSampling } from '../rendering/scene-mesh'
import { clusterPreviewPrimitive } from './preview-lod'

type TextureInfo = {
  index: number; texCoord?: number
  extensions?: { KHR_texture_transform?: { offset?: number[]; scale?: number[]; rotation?: number; texCoord?: number } }
}
type GltfMaterial = {
  name?: string; alphaMode?: 'OPAQUE' | 'MASK' | 'BLEND'; alphaCutoff?: number; doubleSided?: boolean
  pbrMetallicRoughness?: { baseColorFactor?: number[]; baseColorTexture?: TextureInfo
    metallicFactor?: number; roughnessFactor?: number; metallicRoughnessTexture?: TextureInfo }
  normalTexture?: TextureInfo & { scale?: number }
  occlusionTexture?: TextureInfo & { strength?: number }; emissiveTexture?: TextureInfo; emissiveFactor?: number[]
  extensions?: Record<string, unknown> & { KHR_materials_emissive_strength?: { emissiveStrength?: number } }
}
type Primitive = { attributes: Record<string, number>; indices?: number; material?: number; mode?: number; targets?: unknown[] }
type SceneNode = { name?: string; mesh?: number; children?: number[]; matrix?: number[]
  translation?: number[]; rotation?: number[]; scale?: number[] }
export type PreviewGltfDocument = {
  asset: { version: string }
  extensionsRequired?: string[]; skins?: unknown[]; animations?: unknown[]
  scene?: number; scenes: { nodes?: number[] }[]; nodes: SceneNode[]
  meshes: { name?: string; primitives: Primitive[] }[]
  accessors: { bufferView?: number; byteOffset?: number; componentType: number; count: number; type: string
    normalized?: boolean; min?: number[]; max?: number[]
    sparse?: { count: number; indices: { bufferView: number; byteOffset?: number; componentType: number }
      values: { bufferView: number; byteOffset?: number } } }[]
  buffers: { uri?: string; byteLength: number }[]
  bufferViews: { buffer: number; byteOffset?: number; byteLength: number; byteStride?: number }[]
  materials?: GltfMaterial[]
  textures?: { source?: number; sampler?: number; extensions?: { EXT_texture_avif?: { source: number }; EXT_texture_webp?: { source: number } } }[]
  images?: { uri?: string; bufferView?: number; mimeType?: string; name?: string }[]
  samplers?: { wrapS?: number; wrapT?: number; magFilter?: number; minFilter?: number }[]
}

export interface PreviewBounds { min: readonly number[]; max: readonly number[] }
export interface PreviewNodeBounds extends PreviewBounds { nodeIndex: number; name: string; meshName: string; path: string }
export interface PreviewGltfOptions {
  resolveUri?: (uri: string, documentUrl: URL) => URL | string
  readUri?: (uri: string) => Promise<Blob>
  pbr?: boolean
  crop?: (nodes: readonly PreviewNodeBounds[]) => PreviewBounds
  preserveSurface?: (node: PreviewNodeBounds, materialName: string, primitiveBounds: PreviewBounds) => boolean
  maxTriangles?: number
  simplifyToBudget?: boolean
  onSimplified?: (sourceTriangles: number, outputTriangles: number) => void
}

function worldBounds(world: Mat4, min: number[], max: number[]) {
  const lower: number[] = [], upper: number[] = []
  for (let axis = 0; axis < 3; axis++) {
    let low = world[12 + axis], high = low
    for (let j = 0; j < 3; j++) {
      const a = world[4 * j + axis] * min[j], b = world[4 * j + axis] * max[j]
      low += Math.min(a, b)
      high += Math.max(a, b)
    }
    lower.push(low)
    upper.push(high)
  }
  return { min: lower, max: upper }
}

// Clip a boundary triangle and its interpolants to the same world-space box.
// A large pavement triangle must not make the camera frame the entire street.
function clipPolygon(polygon: number[][], box: PreviewBounds) {
  for (let axis = 0; axis < 3; axis++) {
    for (const side of [1, -1]) {
      if (polygon.length < 3) return []
      const boundary = side === 1 ? box.min[axis] : box.max[axis]
      const output: number[][] = []
      let previous = polygon.at(-1)!, d0 = (previous[axis] - boundary) * side
      for (const current of polygon) {
        const d1 = (current[axis] - boundary) * side
        if ((d0 < 0 && d1 > 0) || (d0 > 0 && d1 < 0)) {
          const t = d0 / (d0 - d1)
          const point = previous.map((value, i) => value + t * (current[i] - value))
          point[axis] = boundary
          output.push(point)
        }
        if (d1 >= 0) output.push(current)
        previous = current
        d0 = d1
      }
      polygon = output
    }
  }
  for (const point of polygon) {
    for (let axis = 0; axis < 3; axis++) point[axis] = Math.fround(point[axis])
  }
  return polygon
}

function hasArea(a: ArrayLike<number>, b: ArrayLike<number>, c: ArrayLike<number>, i = 0, j = 0, k = 0) {
  const ux = b[j] - a[i], uy = b[j + 1] - a[i + 1], uz = b[j + 2] - a[i + 2]
  const vx = c[k] - a[i], vy = c[k + 1] - a[i + 1], vz = c[k + 2] - a[i + 2]
  return uy * vz !== uz * vy || uz * vx !== ux * vz || ux * vy !== uy * vx
}

function worldPoints(position: { count: number; get(i: number, k: number): number }, world: Mat4) {
  const points = new Float32Array(position.count * 3)
  for (let i = 0; i < position.count; i++) {
    const x = position.get(i, 0), y = position.get(i, 1), z = position.get(i, 2)
    for (let k = 0; k < 3; k++) {
      points[i * 3 + k] = world[k] * x + world[k + 4] * y + world[k + 8] * z + world[k + 12]
    }
  }
  return points
}

function outsidePlanes(points: Float32Array, box: PreviewBounds) {
  const masks = new Uint8Array(points.length / 3)
  for (let i = 0; i < masks.length; i++) for (let k = 0; k < 3; k++) {
    const p = points[i * 3 + k]
    if (p < box.min[k]) masks[i] |= 1 << (2 * k)
    if (p > box.max[k]) masks[i] |= 2 << (2 * k)
  }
  return masks
}

function unpack(data: ArrayBuffer) {
  const view = new DataView(data)
  const decode = (bytes: ArrayBuffer | Uint8Array) => JSON.parse(new TextDecoder().decode(bytes)) as PreviewGltfDocument
  if (data.byteLength < 12 || view.getUint32(0, true) !== 0x46546c67) return { gltf: decode(data) }
  if (view.getUint32(4, true) !== 2 || view.getUint32(8, true) !== data.byteLength) throw new Error('Invalid GLB header.')
  let gltf: PreviewGltfDocument | undefined, binary: ArrayBuffer | undefined
  for (let offset = 12; offset < data.byteLength;) {
    if (data.byteLength - offset < 8) throw new Error('Truncated GLB chunk header.')
    const length = view.getUint32(offset, true), type = view.getUint32(offset + 4, true)
    offset += 8
    if (offset + length > data.byteLength) throw new Error('Truncated GLB chunk.')
    if (type === 0x4e4f534a) gltf = decode(new Uint8Array(data, offset, length))
    if (type === 0x004e4942) binary = data.slice(offset, offset + length)
    offset += length
  }
  if (!gltf) throw new Error('Missing GLB JSON chunk.')
  return { gltf, binary }
}

const components: Record<number, Int8ArrayConstructor | Uint8ArrayConstructor | Int16ArrayConstructor |
  Uint16ArrayConstructor | Uint32ArrayConstructor | Float32ArrayConstructor> = {
  5120: Int8Array, 5121: Uint8Array, 5122: Int16Array, 5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array,
}
const dimensions: Record<string, number> = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 }

type GltfSource = URL | string | { document: PreviewGltfDocument; base: URL } | { data: ArrayBuffer; base: URL }

type TextureSlot = 'map' | 'metallicRoughnessMap' | 'normalMap' | 'occlusionMap' | 'emissiveMap'
type UvTransform = [number, number, number, number]
const textureProperties = {
  map: ['clamp','uvTransform','base color'], normalMap: ['normalClamp','normalUVTransform','normal'],
  metallicRoughnessMap: ['metallicRoughnessClamp','metallicRoughnessUVTransform','metallic-roughness'],
  occlusionMap: ['occlusionClamp','occlusionUVTransform','occlusion'], emissiveMap: ['emissiveClamp','emissiveUVTransform','emissive'],
} as const

function materialCoordinates(source: GltfMaterial | undefined, pbr: boolean, warnings: Set<string>) {
  const mr = source?.pbrMetallicRoughness
  const slots: [TextureSlot, TextureInfo | undefined][] = [['map', mr?.baseColorTexture]]
  if (pbr) slots.push(['normalMap', source?.normalTexture], ['metallicRoughnessMap', mr?.metallicRoughnessTexture],
    ['occlusionMap', source?.occlusionTexture], ['emissiveMap', source?.emissiveTexture])
  const base = mr?.baseColorTexture, baked = pbr ? undefined : base?.extensions?.KHR_texture_transform
  const sourceUV = (info: TextureInfo) => info.extensions?.KHR_texture_transform?.texCoord ?? info.texCoord ?? 0
  // Each material can bind two distinct streams. A source is free to use
  // TEXCOORD_3 for its normal map even if neither of the first two is present.
  const sourceSets = [...new Set(slots.flatMap(([, info]) => info ? [sourceUV(info)] : []))].slice(0, 2)
  const streams: [number | undefined, number | undefined] = [sourceSets.includes(0) ? 0 : undefined,
    sourceSets.includes(1) ? 1 : undefined]
  for (const set of sourceSets) {
    const available = streams.indexOf(undefined)
    if (!streams.includes(set) && available >= 0) streams[available] = set
  }
  const channels: { slot: TextureSlot; info: TextureInfo; set: 0 | 1; rotation: number; uvTransform?: UvTransform }[] = []
  for (const [slot, info] of slots) {
    if (!info) continue
    const t = info.extensions?.KHR_texture_transform
    const set = pbr ? streams.indexOf(sourceUV(info)) : 0
    if (set !== 0 && set !== 1) {
      warnings.add(`${source?.name || 'glTF material'} uses more than two UV sets. Using its factor only for ${slot} (TEXCOORD_${sourceUV(info)}).`)
      continue
    }
    const [sx,sy] = t?.scale ?? [1,1], [ox,oy] = t?.offset ?? [0,0]
    // The GPU uses bottom-left vertex UVs and reconstructs the original glTF
    // offset before applying scale, rotation, then translation in image space.
    const uvTransform: UvTransform | undefined = pbr && (sx !== 1 || sy !== 1 || ox !== 0 || oy !== 0)
      ? [sx, sy, ox, 1-sy-oy] : undefined
    channels.push({ slot, info, set, rotation: pbr ? (t?.rotation ?? 0) : 0, uvTransform })
  }
  return { primarySet: pbr ? streams[0] ?? 0 : (base ? sourceUV(base) : 0),
    secondarySet: streams[1] ?? 1, baked, channels }
}

function textureSampler(source: NonNullable<PreviewGltfDocument['samplers']>[number]): GPUSamplerDescriptor {
  function address(mode?: number): GPUAddressMode {
    if (mode === 33071) return 'clamp-to-edge'
    if (mode === 33648) return 'mirror-repeat'
    return 'repeat'
  }
  const min = source.minFilter
  return { addressModeU: address(source.wrapS), addressModeV: address(source.wrapT),
    magFilter: source.magFilter === 9728 ? 'nearest' : 'linear',
    minFilter: min === 9728 || min === 9984 || min === 9986 ? 'nearest' : 'linear',
    mipmapFilter: min === 9984 || min === 9985 ? 'nearest' : 'linear',
    ...(min === 9728 || min === 9729 ? { lodMaxClamp: 0 } : {}) }
}

export async function readPreviewGltf(file: GltfSource, signal?: AbortSignal, options: PreviewGltfOptions = {}) {
  const fetchOk = async (location: string | URL) => {
    const response = await fetch(location, { signal })
    if (!response.ok) throw new Error(`Could not load ${location} (HTTP ${response.status}).`)
    return response
  }
  async function loadDocument() {
    if (typeof file === 'object' && 'document' in file) return { gltf: file.document, base: file.base, binary: undefined }
    if (typeof file === 'object' && 'data' in file) return { ...unpack(file.data), base: file.base }
    const url = new URL(file, globalThis.document?.baseURI)
    const response = await fetchOk(url)
    return { ...unpack(await response.arrayBuffer()), base: new URL(response.url || url.href) }
  }
  const { gltf, binary, base } = await loadDocument()
  const resource = (uri: string) => options.readUri?.(uri) ?? fetchOk(options.resolveUri?.(uri, base) ?? new URL(uri, base))
  if (gltf.asset?.version !== '2.0') throw new Error('Expected a glTF 2.0 asset.')
  if (options.pbr) {
    const supported = ['KHR_mesh_quantization', 'KHR_texture_transform', 'KHR_materials_emissive_strength', 'EXT_texture_avif', 'EXT_texture_webp']
    const missing = gltf.extensionsRequired?.filter(name => !supported.includes(name))
    if (missing?.length) throw new Error('Unsupported required glTF extension: ' + missing.join(', '))
  }

  const buffers = new Map<number, Promise<ArrayBuffer>>()
  function loadBuffer(index: number) {
    let pending = buffers.get(index)
    if (!pending) {
      const source = gltf.buffers[index]
      if (!source) throw new Error('Missing glTF buffer.')
      if (source.uri) pending = resource(source.uri).then(body => body.arrayBuffer())
      else if (index === 0 && binary) pending = Promise.resolve(binary)
      else throw new Error('Missing BIN data in the scene preview.')
      buffers.set(index, pending)
    }
    return pending
  }

  async function readElements(viewIndex: number, offset: number, componentType: number, size: number, count: number) {
    const view = gltf.bufferViews[viewIndex], ArrayType = components[componentType]
    if (!view || !ArrayType) throw new Error('Invalid glTF accessor layout.')
    const bytes = ArrayType.BYTES_PER_ELEMENT
    const stride = (view.byteStride ?? size * bytes) / bytes
    const length = count === 0 ? 0 : (count - 1) * stride + size
    if (!Number.isInteger(stride) || stride < size || offset < 0 || offset + length * bytes > view.byteLength) {
      throw new Error('glTF accessor extends beyond its buffer view.')
    }
    const data = new ArrayType(await loadBuffer(view.buffer), (view.byteOffset ?? 0) + offset, length)
    return { data, stride }
  }

  async function readAccessor(attr: PreviewGltfDocument['accessors'][number], size: number) {
    const ArrayType = components[attr.componentType]
    if (!ArrayType || !Number.isSafeInteger(attr.count) || attr.count < 0) throw new Error('Invalid glTF accessor layout.')
    const base = attr.bufferView === undefined ? undefined
      : await readElements(attr.bufferView, attr.byteOffset ?? 0, attr.componentType, size, attr.count)
    let data = base?.data, stride = base?.stride ?? size
    if (attr.sparse) {
      const { indices, values, count } = attr.sparse
      if (!Number.isSafeInteger(count) || count < 1 || count > attr.count || ![5121,5123,5125].includes(indices.componentType)) {
        throw new Error('Invalid sparse glTF accessor.')
      }
      const [keys, patches] = await Promise.all([
        readElements(indices.bufferView, indices.byteOffset ?? 0, indices.componentType, 1, count),
        readElements(values.bufferView, values.byteOffset ?? 0, attr.componentType, size, count),
      ])
      // Sparse values replace complete elements in the original component type.
      // A missing base bufferView represents an array initially filled with zero.
      const dense = new ArrayType(attr.count * size)
      if (base) {
        if (base.stride === size) dense.set(base.data)
        else for (let i = 0; i < attr.count; i++) dense.set(base.data.subarray(i * base.stride, i * base.stride + size), i * size)
      }
      let previous = -1
      for (let i = 0; i < count; i++) {
        const key = keys.data[i * keys.stride]
        if (key <= previous || key >= attr.count) throw new Error('Invalid sparse glTF index.')
        const first = i * patches.stride
        dense.set(patches.data.subarray(first, first + size), key * size)
        previous = key
      }
      data = dense
      stride = size
    }
    const bytes = ArrayType.BYTES_PER_ELEMENT
    const signed = attr.componentType === 5120 || attr.componentType === 5122
    const scale = attr.normalized ? 1 / (2 ** (8 * bytes - Number(signed)) - 1) : 1
    return { count: attr.count, get(vertex: number, component = 0) {
      const value = data ? data[vertex * stride + component] : 0
      return attr.normalized ? Math.max(-1, value * scale) : value
    } }
  }

  const accessors = new Map<number, ReturnType<typeof readAccessor>>()
  function accessor(index: number, size: number) {
    const attr = gltf.accessors[index]
    if (!attr || dimensions[attr.type] !== size) throw new Error('Unsupported or missing glTF accessor in the scene preview.')
    let existing = accessors.get(index)
    if (!existing) { existing = readAccessor(attr, size); accessors.set(index, existing) }
    return existing
  }

  const images = new Map<number, Promise<File>>()
  async function image(index: number) {
    const source = gltf.images?.[index]
    if (!source) throw new Error('Missing glTF image.')
    let blob: Blob
    if (source.uri) {
      const body = await resource(source.uri)
      blob = body instanceof Blob ? body : await body.blob()
    }
    else {
      const view = gltf.bufferViews[source.bufferView!]
      if (!view) throw new Error('Missing glTF image buffer view.')
      const start = view.byteOffset ?? 0, buffer = await loadBuffer(view.buffer)
      blob = new Blob([new Uint8Array(buffer, start, view.byteLength)], { type: source.mimeType })
    }
    return new File([blob], source.name || `glTF-image-${index}`, { type: source.mimeType || blob.type })
  }

  const vertices: number[] = [], uvs: number[] = [], indices: number[] = [], draws: SceneDraw[] = []
  let secondaryUVs: number[] | undefined, tangents: number[] | undefined
  const materials = new Map<string, SceneMaterial>(), warnings = new Set<string>(), used = new Set<number>()
  const vertexColorMaterials = new Set<number>()
  const materialUvMasks = new Map<number, number>()
  const coordinateCache = new Map<number, ReturnType<typeof materialCoordinates>>()
  function coordinates(id: number) {
    let layout = coordinateCache.get(id)
    if (!layout) {
      layout = materialCoordinates(gltf.materials?.[id], !!options.pbr, warnings)
      coordinateCache.set(id, layout)
    }
    return layout
  }
  const scene = gltf.scenes?.[gltf.scene ?? 0]
  if (!scene) throw new Error('Missing glTF scene.')
  if (options.pbr && (gltf.animations?.length || gltf.skins?.length)) {
    warnings.add('Animated or skinned meshes are shown in their undeformed static pose.')
  }

  // A source primitive may contain the whole district's paving. When the
  // bounds alone cannot prove the crop fits, count the actual cut triangles
  // before reducing any detail. Only source face offsets survive this pass.
  async function measureCrop(primitive: Primitive, world: Mat4, box: PreviewBounds) {
    const [position, corners] = await Promise.all([accessor(primitive.attributes.POSITION, 3),
      primitive.indices === undefined ? undefined : accessor(primitive.indices, 1)])
    const points = worldPoints(position, world), masks = outsidePlanes(points, box)
    const count = corners?.count ?? position.count
    if (count % 3) throw new Error('Invalid triangle count in glTF preview.')
    const faceOffsets: number[] = []
    const flipped = mat4.determinant(world) < 0
    let triangles = 0
    for (let i = 0; i < count; i += 3) {
      const a = corners ? corners.get(i) : i, b = corners ? corners.get(i + 1) : i + 1
      const c = corners ? corners.get(i + 2) : i + 2
      if (Math.min(a, b, c) < 0 || Math.max(a, b, c) >= position.count) throw new Error('glTF index outside vertex array.')
      if (masks[a] & masks[b] & masks[c]) continue
      if (!hasArea(points, points, points, a * 3, b * 3, c * 3)) continue
      if (!(masks[a] | masks[b] | masks[c])) triangles++
      else {
        const face = flipped ? [a, c, b] : [a, b, c]
        const polygon = clipPolygon(face.map(v => [points[3 * v], points[3 * v + 1], points[3 * v + 2]]), box)
        let faces = 0
        for (let j = 2; j < polygon.length; j++) if (hasArea(polygon[0], polygon[j - 1], polygon[j])) faces++
        if (!faces) continue
        triangles += faces
      }
      faceOffsets.push(i)
    }
    return { triangles, offsets: Uint32Array.from(faceOffsets) }
  }

  async function addPrimitive(primitive: Primitive, world: Mat4, objectId: number, box?: PreviewBounds,
    triangleTarget?: number, offsets?: Uint32Array, preserveSurface = false) {
    if ((primitive.mode ?? 4) !== 4) return
    const matId = primitive.material ?? -1
    const layout = coordinates(matId), transform = layout.baked
    const attributes = primitive.attributes
    if (options.pbr && primitive.targets?.length) warnings.add('Morph targets are shown in their undeformed pose.')
    const uvId = attributes[`TEXCOORD_${layout.primarySet}`]
    const uv1Id = options.pbr ? attributes[`TEXCOORD_${layout.secondarySet}`] : undefined
    const uvMask = Number(uvId !== undefined) | (Number(uv1Id !== undefined) << 1)
    if (options.pbr) for (const { set, slot } of layout.channels) {
      const sourceSet = set === 0 ? layout.primarySet : layout.secondarySet
      if (!(uvMask & (1 << set))) warnings.add(`Faces without TEXCOORD_${sourceSet} use the material factor for ${slot}.`)
    }
    const colorId = options.pbr ? attributes.COLOR_0 : undefined
    const tangentId = options.pbr && attributes.NORMAL !== undefined ? attributes.TANGENT : undefined
    const [position, normal, uv, corners, color, uv1, tangent] = await Promise.all([
      accessor(attributes.POSITION, 3),
      attributes.NORMAL === undefined ? undefined : accessor(attributes.NORMAL, 3),
      uvId === undefined ? undefined : accessor(uvId, 2),
      primitive.indices === undefined ? undefined : accessor(primitive.indices, 1),
      colorId === undefined ? undefined : accessor(colorId, gltf.accessors[colorId]?.type === 'VEC4' ? 4 : 3),
      uv1Id === undefined ? undefined : accessor(uv1Id, 2),
      tangentId === undefined ? undefined : accessor(tangentId, 4),
    ])
    if ([normal, uv, color, uv1, tangent].some(attr => attr && attr.count !== position.count)) {
      throw new Error('glTF vertex attributes have inconsistent lengths.')
    }
    if (uv1 && !secondaryUVs) secondaryUVs = Array(uvs.length).fill(0)
    if (tangent && !tangents) tangents = Array(2 * uvs.length).fill(0)
    const firstIndex = indices.length
    const flipped = mat4.determinant(world) < 0
    const normalMatrix = normal ? mat4.transpose(mat4.inverse(world)) : undefined
    const points = worldPoints(position, world)
    const sourceCount = offsets ? offsets.length * 3 : corners?.count ?? position.count
    const sourceCorner = (i: number) => {
      const slot = offsets ? offsets[Math.floor(i / 3)] + i % 3 : i
      return corners ? corners.get(slot) : slot
    }
    if (sourceCount % 3) throw new Error('Invalid triangle count in glTF preview.')
    let simplified: Uint32Array | undefined
    if (triangleTarget !== undefined && sourceCount / 3 > triangleTarget) {
      const source = new Uint32Array(sourceCount)
      for (let i = 0; i < sourceCount; i++) {
        const vertex = sourceCorner(i)
        if (!Number.isInteger(vertex) || vertex < 0 || vertex >= position.count) throw new Error('glTF index outside vertex array.')
        source[i] = vertex
      }
      simplified = clusterPreviewPrimitive(points, source, triangleTarget, preserveSurface)
    }
    const outside = box ? outsidePlanes(points, box) : undefined
    const remap = normal ? new Int32Array(position.count).fill(-1) : undefined
    const angle = transform?.rotation ?? 0, cos = Math.cos(angle), sin = Math.sin(angle)
    const getVertex = (i: number, nx: number, ny: number, nz: number) => {
      if (normal && normalMatrix) {
        const x = normal.get(i, 0), y = normal.get(i, 1), z = normal.get(i, 2), m = normalMatrix
        const dx = m[0] * x + m[4] * y + m[8] * z
        const dy = m[1] * x + m[5] * y + m[9] * z
        const dz = m[2] * x + m[6] * y + m[10] * z
        const length = Math.hypot(dx, dy, dz)
        if (Number.isFinite(length) && length > 0) { nx = dx / length; ny = dy / length; nz = dz / length }
      }
      const u = (uv?.get(i, 0) ?? 0) * (transform?.scale?.[0] ?? 1)
      const v = (uv?.get(i, 1) ?? 0) * (transform?.scale?.[1] ?? 1)
      // The scene shader converts bottom-left OBJ UVs to top-left texture UVs.
      const positionOffset = i * 3
      const textureU = (transform?.offset?.[0] ?? 0) + cos * u - sin * v
      const textureV = (transform?.offset?.[1] ?? 0) + sin * u + cos * v
      const result = [points[positionOffset], points[positionOffset + 1], points[positionOffset + 2], nx, ny, nz, textureU, 1 - textureV,
        color?.get(i,0) ?? 1, color?.get(i,1) ?? 1, color?.get(i,2) ?? 1]
      if (secondaryUVs || tangents) result.push(uv1?.get(i,0) ?? 0, uv1 ? 1 - uv1.get(i,1) : 0)
      if (tangent) {
        const x = tangent.get(i,0), y = tangent.get(i,1), z = tangent.get(i,2)
        const tx = world[0]*x + world[4]*y + world[8]*z, ty = world[1]*x + world[5]*y + world[9]*z
        const tz = world[2]*x + world[6]*y + world[10]*z, length = Math.hypot(tx,ty,tz) || 1
        // Mirroring the node and converting its glTF V-down frame to V-up
        // each reverse handedness once. Tangents transform with model space.
        result.push(tx/length,ty/length,tz/length,(flipped ? 1 : -1)*tangent.get(i,3))
      }
      return result
    }
    const write = (vertex: number[]) => {
      const [px, py, pz, nx, ny, nz, u, v, red, green, blue] = vertex, index = vertices.length / sceneVertexStride
      const n = Math.hypot(nx, ny, nz) || 1
      vertices.push(px, py, pz, nx / n, ny / n, nz / n, red, green, blue, 0, objectId)
      uvs.push(u, v)
      if (secondaryUVs) secondaryUVs.push(vertex[11],vertex[12])
      if (tangents) tangents.push(vertex[13] ?? 0,vertex[14] ?? 0,vertex[15] ?? 0,vertex[16] ?? 0)
      return index
    }
    const count = simplified?.length ?? sourceCount
    const cornerAt = (i: number) => simplified ? simplified[i] : sourceCorner(i)
    for (let i = 0; i < count; i += 3) {
      const a = cornerAt(i)
      let b = cornerAt(i + 1), c = cornerAt(i + 2)
      if (Math.min(a, b, c) < 0 || Math.max(a, b, c) >= position.count) throw new Error('glTF index outside vertex array.')
      if (outside && (outside[a] & outside[b] & outside[c])) continue
      if (flipped) [b, c] = [c, b]
      const u = b * 3, v = c * 3, o = a * 3
      const abx = points[u] - points[o], aby = points[u + 1] - points[o + 1], abz = points[u + 2] - points[o + 2]
      const acx = points[v] - points[o], acy = points[v + 1] - points[o + 1], acz = points[v + 2] - points[o + 2]
      const x = aby * acz - abz * acy, y = abz * acx - abx * acz, z = abx * acy - aby * acx
      const length = Math.hypot(x, y, z)
      if (!Number.isFinite(length)) throw new Error('Non-finite glTF triangle.')
      if (!length) continue
      if (box && outside && (outside[a] | outside[b] | outside[c])) {
        const polygon = clipPolygon([a, b, c].map(corner => getVertex(corner, x / length, y / length, z / length)), box)
        if (polygon.length < 3) continue
        const polygonIndices: number[] = []
        for (let j = 2; j < polygon.length; j++) {
          if (!hasArea(polygon[0], polygon[j - 1], polygon[j])) continue
          for (const corner of [0, j - 1, j]) indices.push(polygonIndices[corner] ??= write(polygon[corner]))
        }
      } else {
        for (const corner of [a, b, c]) {
          let index = remap?.[corner] ?? -1
          if (index < 0) {
            index = write(getVertex(corner, x / length, y / length, z / length))
            if (remap) remap[corner] = index
          }
          indices.push(index)
        }
      }
      if (options.maxTriangles !== undefined && indices.length > options.maxTriangles * 3) {
        throw new Error('Scene crop exceeds the triangle budget.')
      }
    }
    if (indices.length > firstIndex) {
      const draw: SceneDraw = { firstIndex, indexCount: indices.length - firstIndex, material: String(matId),
        hasUV: uvMask !== 0, uvMask }
      if (options.pbr && gltf.materials?.[matId]?.doubleSided) draw.doubleSided = true
      draws.push(draw)
      used.add(matId)
      materialUvMasks.set(matId, (materialUvMasks.get(matId) ?? 0) | uvMask)
      if (color) vertexColorMaterials.add(matId)
    }
  }

  const activeNodes = new Set<number>()
  const instances: { primitive: Primitive; world: Mat4; objectId: number; bounds?: PreviewBounds }[] = []
  const nodes: PreviewNodeBounds[] = []
  function visit(index: number, parent: Mat4, parentPath: string) {
    signal?.throwIfAborted()
    const node = gltf.nodes[index]
    if (!node || activeNodes.has(index)) throw new Error('Invalid glTF scene hierarchy.')
    activeNodes.add(index)
    let local: Mat4
    if (node.matrix) local = new Float32Array(node.matrix)
    else {
      local = node.rotation ? mat4.fromQuat(node.rotation) : mat4.identity()
      if (node.scale) mat4.scale(local, node.scale, local)
      if (node.translation) mat4.setTranslation(local, node.translation, local)
    }
    const world = mat4.multiply(parent, local)
    const mesh = node.mesh === undefined ? undefined : gltf.meshes[node.mesh]
    const name = node.name ?? '', meshName = mesh?.name ?? ''
    const path = parentPath + '/' + name
    if (mesh && mat4.determinant(world) !== 0) {
      const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity]
      for (const primitive of mesh.primitives) {
        const position = gltf.accessors[primitive.attributes.POSITION]
        const bounds = position?.min && position.max ? worldBounds(world, position.min, position.max) : undefined
        instances.push({ primitive, world, objectId: index + 2, bounds })
        if (bounds) for (let k = 0; k < 3; k++) {
          min[k] = Math.min(min[k], bounds.min[k])
          max[k] = Math.max(max[k], bounds.max[k])
        }
      }
      if (min[0] !== Infinity) nodes.push({ nodeIndex: index, name, meshName, path, min, max })
    }
    for (const child of node.children ?? []) visit(child, world, path)
    activeNodes.delete(index)
  }
  for (const node of scene.nodes ?? []) visit(node, mat4.identity(), '')
  const box = options.crop?.(nodes)
  if (box && [0, 1, 2].some(k => !(box.min[k] < box.max[k]))) throw new Error('Invalid glTF preview crop.')
  const selected = instances.filter(({ primitive, bounds }) => (primitive.mode ?? 4) === 4
    && !(box && bounds && [0, 1, 2].some(k => bounds.max[k] < box.min[k] || bounds.min[k] > box.max[k])))
  if (options.pbr && instances.some(({ primitive }) => (primitive.mode ?? 4) !== 4)) {
    warnings.add('Only TRIANGLES primitives are imported; other primitive modes were skipped.')
  }
  const targets: (number | undefined)[] = []
  const offsets: (Uint32Array | undefined)[] = []
  const protectedSurfaces: boolean[] = []
  let triangleCountBeforeSimplifying: number | undefined
  if (options.simplifyToBudget && options.maxTriangles !== undefined) {
    const limit = options.maxTriangles
    const counts = selected.map(({ primitive }) =>
      gltf.accessors[primitive.indices ?? primitive.attributes.POSITION].count / 3)
    // Each intersecting clip plane can add at most one triangle to a face.
    // Include all instances, including copies of the same primitive, in the cap.
    const weights = selected.map(({ bounds }) => {
      if (!box) return 1
      if (!bounds) return 7
      let weight = 1
      for (let k = 0; k < 3; k++) {
        if (bounds.min[k] < box.min[k]) weight++
        if (bounds.max[k] > box.max[k]) weight++
      }
      return weight
    })
    const upperBound = counts.reduce((sum, n, i) => sum + n * weights[i], 0)
    if (box && upperBound > limit) {
      for (const [i, instance] of selected.entries()) {
        if (weights[i] === 1) continue
        signal?.throwIfAborted()
        const actual = await measureCrop(instance.primitive, instance.world, box)
        counts[i] = actual.triangles
        offsets[i] = actual.offsets
      }
    }
    const full = counts.reduce((sum, n) => sum + n, 0)
    if (full > limit) {
      triangleCountBeforeSimplifying = full
      const reserve = Math.min(32, Math.floor(limit / weights.reduce((sum, w, i) => sum + (counts[i] ? w : 0), 0)))
      if (reserve < 1) throw new Error('Too many separate parts for the scene triangle budget.')
      const base = counts.map((n, i) => Math.min(n, reserve * weights[i]))
      let baseCount = base.reduce((sum, n) => sum + n, 0)
      if (options.preserveSurface) {
        const byIndex = new Map(nodes.map(node => [node.nodeIndex, node]))
        let cost = 0
        for (const [i, { primitive, bounds, objectId }] of selected.entries()) {
          const node = byIndex.get(objectId - 2)
          const name = gltf.materials?.[primitive.material ?? -1]?.name ?? ''
          if (counts[i] && node && bounds && options.preserveSurface(node, name, bounds)) {
            protectedSurfaces[i] = true
            cost += counts[i] - base[i]
          }
        }
        // Keep true road/curb topology across independent materials whenever
        // it fits after allowing every other visible part a representation.
        if (baseCount + cost <= limit) {
          for (const [i, keep] of protectedSurfaces.entries()) if (keep) base[i] = counts[i]
          baseCount += cost
        }
      }
      const ratio = (limit - baseCount) / (full - baseCount)
      for (const [i, n] of counts.entries()) {
        const allowance = base[i] + Math.floor((n - base[i]) * ratio)
        if (allowance < n) targets[i] = Math.floor(allowance / weights[i])
      }
    }
  }
  for (const [i, { primitive, world, objectId }] of selected.entries()) {
    signal?.throwIfAborted()
    if (offsets[i]?.length !== 0) await addPrimitive(primitive, world, objectId, box, targets[i], offsets[i], protectedSurfaces[i])
  }
  if (!indices.length) throw new Error('No triangles were found for this glTF preview.')
  if (triangleCountBeforeSimplifying !== undefined && targets.some(target => target !== undefined)) {
    options.onSimplified?.(triangleCountBeforeSimplifying, indices.length / 3)
  }
  await Promise.all([...used].map(async id => {
    const source = gltf.materials?.[id], pbr = source?.pbrMetallicRoughness, rgb = pbr?.baseColorFactor ?? [1, 1, 1]
    const material: SceneMaterial = { color: [rgb[0], rgb[1], rgb[2]] }
    if (options.pbr) {
      material.metallic = pbr?.metallicFactor ?? 1
      material.roughness = pbr?.roughnessFactor ?? 1
      if (vertexColorMaterials.has(id)) material.vertexColor = true
      if (source?.normalTexture) {
        material.normalScale = source.normalTexture.scale ?? 1
        material.normalYSign = -1
      }
      if (source?.occlusionTexture) material.occlusionStrength = source.occlusionTexture.strength ?? 1
      if (source?.emissiveFactor || source?.emissiveTexture) {
        const strength = source.extensions?.KHR_materials_emissive_strength?.emissiveStrength ?? 1
        const [r,g,b] = source.emissiveFactor ?? [0,0,0]
        material.emissive = [r*strength,g*strength,b*strength]
      }
      if (source?.alphaMode && source.alphaMode !== 'OPAQUE') warnings.add('Alpha-masked and blended materials are previewed as opaque.')
      if (source?.doubleSided) material.doubleSided = true
      for (const name of Object.keys(source?.extensions ?? {})) {
        if (name !== 'KHR_materials_emissive_strength') warnings.add(`Optional ${name} material uses its core metallic-roughness fallback.`)
      }
    }
    await Promise.all(coordinates(id).channels.map(async ({ slot, info, set, rotation, uvTransform }) => {
      if (options.pbr && !((materialUvMasks.get(id) ?? 0) & (1 << set))) return
      const tex = gltf.textures?.[info.index], sampler = gltf.samplers?.[tex?.sampler ?? -1]
      const clamp = sampler?.wrapS === 33071 && sampler?.wrapT === 33071
      const [clampKey, transformKey, label] = textureProperties[slot]
      material[clampKey] = clamp
      if (uvTransform) material[transformKey] = uvTransform
      if (options.pbr && (set || rotation || sampler)) {
        const sampling: SceneMapSampling = { uvSet: set, rotation }
        if (sampler) sampling.sampler = textureSampler(sampler)
        material.sampling ??= {}
        material.sampling[slot] = sampling
      }
      const sourceIndex = tex?.extensions?.EXT_texture_avif?.source ?? tex?.extensions?.EXT_texture_webp?.source ?? tex?.source
      if (sourceIndex === undefined) { warnings.add(`Unsupported ${label} texture. Using material factors.`); return }
      const pending = images.get(sourceIndex) ?? image(sourceIndex)
      images.set(sourceIndex, pending)
      try {
        material[slot] = await pending
      } catch (error) {
        if (signal?.aborted) throw error
        const name = gltf.images?.[sourceIndex]?.name || sourceIndex
        const detail = error instanceof Error ? ` (${error.message})` : ''
        warnings.add(`Could not load ${source?.name || 'glTF material'} ${label} ${name}${detail}. Using the material ${slot === 'map' ? 'color' : 'factor'}.`)
      }
    }))
    materials.set(String(id), material)
  }))
  const mesh: SceneMesh = { vertices: new Float32Array(vertices), indices: new Uint32Array(indices),
    uvs: new Float32Array(uvs), draws }
  if (secondaryUVs) mesh.uvs1 = new Float32Array(secondaryUVs)
  if (tangents) mesh.tangents = new Float32Array(tangents)
  return { mesh, materials, warnings }
}
