import { sceneVertexStride, type SceneDraw, type SceneMesh } from '../rendering/scene-mesh'

type Triple = [number, number, number]
const defaultAlbedo = [0.7, 0.7, 0.7] as const
const decimal = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i

function invalid(line: number, message: string): never {
  throw new Error(`OBJ line ${line}: ${message}`)
}

function coordinate(token: string, line: number): number {
  const value = Number(token)
  if (!decimal.test(token) || !Number.isFinite(Math.fround(value))) {
    invalid(line, `Invalid coordinate "${token}".`)
  }
  return value
}

function objIndex(token: string, count: number, kind: string, line: number): number {
  const index = Number(token)
  if (!/^[+-]?\d+$/.test(token) || !Number.isSafeInteger(index) || index === 0) {
    invalid(line, `Invalid ${kind} index "${token}".`)
  }
  const resolved = index > 0 ? index - 1 : count + index
  if (resolved < 0 || resolved >= count) invalid(line, `${kind} index ${index} is out of range.`)
  return resolved
}

function faceCorner(token: string, positionCount: number, texcoordCount: number, normalCount: number, line: number) {
  const parts = token.split('/')
  if (parts.length > 3 || parts.at(-1) === '') invalid(line, `Invalid face corner "${token}".`)
  const position = objIndex(parts[0], positionCount, 'position', line)
  const uv = parts[1] ? objIndex(parts[1], texcoordCount, 'texture coordinate', line) : undefined
  const normal = parts[2] === undefined ? undefined : objIndex(parts[2], normalCount, 'normal', line)
  return { position, normal, uv }
}

function unit(x: number, y: number, z: number, line: number, error: string): Triple {
  const length = Math.hypot(x, y, z)
  if (length === 0) invalid(line, error)
  return [x / length, y / length, z / length]
}

function faceNormal(a: Triple, b: Triple, c: Triple, line: number): Triple {
  const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2]
  const vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2]
  return unit(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx, line, 'Degenerate triangle.')
}

// Read a triangulated polygon OBJ in its original coordinate system and units.
// Explicit vertex normals preserve hard/smooth edges. Missing normals use the face winding.
export function parseObjMesh(text: string): SceneMesh & { materialLibraries: string[]; vertexColorMaterials: Set<string> } {
  const positions: Triple[] = []
  const colors = new Map<number, Triple>(), vertexColorMaterials = new Set<string>()
  const coloredVertices = new Set<number>()
  const secondaryUVs: number[][] = [], sourceTangents: number[][] = []
  const outputUV1: number[] = [], outputTangents: number[] = []
  const normals: Triple[] = []
  const texcoords: [number, number][] = []
  const uvs: number[] = []
  const materialLibraries: string[] = [], draws: SceneDraw[] = []
  let material: string | undefined
  let uvMask: number | undefined
  const vertices: number[] = []
  const indices: number[] = []
  const shared = new Map<string, number>()
  const objects = new Map<string, number>()
  let objectName = '', groupName = ''

  const lines = text.split(/\r\n?|\n/)
  for (let i = 0; i < lines.length; i++) {
    const line = i + 1
    const raw = lines[i].trim()
    // Optional attributes from a Caustic Canvas preset are indexed by the
    // preceding v record, not by the eventual reindexed GPU vertex.
    if (raw.startsWith('# caustic-')) {
      const [tag, ...values] = raw.slice(2).split(/\s+/)
      if (tag === 'caustic-uv1' || tag === 'caustic-tangent') {
        if (!positions.length || values.length !== (tag === 'caustic-uv1' ? 2 : 4)) invalid(line, 'Invalid preset vertex attributes.')
        const isUV = tag === 'caustic-uv1'
        const stream = isUV ? secondaryUVs : sourceTangents
        if (!stream.length && vertices.length) {
          const output = isUV ? outputUV1 : outputTangents
          output.length = vertices.length / sceneVertexStride * (isUV ? 2 : 4)
          output.fill(0)
        }
        stream[positions.length - 1] = values.map(n => coordinate(n, line))
      } else if (tag === 'caustic-uv-mask') {
        if (values.length !== 1 || !/^[0-3]$/.test(values[0])) invalid(line, 'Invalid texture coordinate mask.')
        uvMask = Number(values[0])
      }
      continue
    }
    const record = raw.split('#', 1)[0].trim()
    if (!record) continue
    const [op, ...args] = record.split(/\s+/)
    switch (op) {
      case 'v': {
        const hasColor = args.length === 6 || args.length === 7
        const hasWeight = args.length === 4 || args.length === 7
        if ((args.length !== 3 && !hasColor && !hasWeight) || (hasWeight && coordinate(args[3], line) !== 1)) {
          invalid(line, 'Expected v x y z with optional unit weight and vertex RGB.')
        }
        // Validate geometry at the precision used by both the BVH and the GPU.
        const [x, y, z] = args.slice(0, 3).map(value => Math.fround(coordinate(value, line)))
        if (hasColor) {
          const rgb = args.slice(-3).map(value => coordinate(value, line))
          if (rgb.some(n => n < 0 || n > 1)) invalid(line, 'OBJ vertex colors must be in [0, 1].')
          colors.set(positions.length, [rgb[0], rgb[1], rgb[2]])
        }
        positions.push([x, y, z])
        break
      }
      case 'vt': {
        if (args.length < 1 || args.length > 3) invalid(line, 'Expected vt with one to three coordinates.')
        const [u, v = 0] = args.map(value => coordinate(value, line))
        texcoords.push([u, v])
        break
      }
      case 'mtllib':
        materialLibraries.push(args.join(' '))
        break
      case 'usemtl':
        material = args.join(' ').replace(/^"(.*)"$/, '$1') || undefined
        uvMask = undefined
        break
      case 'vn': {
        if (args.length !== 3) invalid(line, 'Expected vn x y z.')
        const [x, y, z] = args.map(value => coordinate(value, line))
        normals.push(unit(x, y, z, line, 'Zero-length vertex normal.'))
        break
      }
      case 'o':
        objectName = args.join(' ')
        groupName = ''
        break
      case 'g':
        groupName = args.join(' ')
        break
      case 'f': {
        if (args.length !== 3) invalid(line, `Expected a triangle, got ${args.length} corners. Triangulate the mesh before export.`)
        const corners = args.map(token => faceCorner(token, positions.length, texcoords.length, normals.length, line))
        const [a, b, c] = corners
        const flat = faceNormal(positions[a.position], positions[b.position], positions[c.position], line)
        // Maya can export groups without named objects. Keep object IDs stable across material changes.
        const name = objectName ? 'o:' + objectName : 'g:' + groupName
        let objectId = objects.get(name)
        if (objectId === undefined) {
          objectId = objects.size + 2
          if (objectId > 0xffffff) invalid(line, 'Too many scene objects.')
          objects.set(name, objectId)
        }
        const hasUV = uvMask === undefined ? corners.every(corner => corner.uv !== undefined) : uvMask !== 0
        if (material && colors.size && corners.some(c => colors.has(c.position))) vertexColorMaterials.add(material)
        const last = draws.at(-1)
        if (last && last.material === material && last.hasUV === hasUV && last.uvMask === uvMask) last.indexCount += 3
        else draws.push({ firstIndex: indices.length, indexCount: 3, material, hasUV, uvMask })
        for (const { position, normal, uv } of corners) {
          const key = normal === undefined ? undefined : `${objectId}/${position}/${normal}/${uv ?? ''}`
          let index = key === undefined ? undefined : shared.get(key)
          if (index === undefined) {
            index = vertices.length / sceneVertexStride
            const rgb = colors.get(position)
            if (rgb) coloredVertices.add(index)
            vertices.push(...positions[position], ...(normal === undefined ? flat : normals[normal]),
              ...(rgb ?? defaultAlbedo), 0, objectId)
            if (uv !== undefined) uvs.push(...texcoords[uv])
            else uvs.push(0, 0)
            if (secondaryUVs.length) outputUV1.push(...(secondaryUVs[position] ?? [0, 0]))
            if (sourceTangents.length) outputTangents.push(...(sourceTangents[position] ?? [0, 0, 0, 0]))
            if (key !== undefined) shared.set(key, index)
          }
          indices.push(index)
        }
        break
      }
      // Other non-polygon records do not add triangles.
    }
  }
  if (!indices.length) throw new Error('OBJ contains no triangle faces.')
  // RGB is a multiplier. An uncolored corner of a colored material is neutral.
  for (const draw of draws) {
    if (!draw.material || !vertexColorMaterials.has(draw.material)) continue
    for (let j = draw.firstIndex; j < draw.firstIndex + draw.indexCount; j++) {
      const vertex = indices[j]
      if (!coloredVertices.has(vertex)) {
        const rgb = vertex * sceneVertexStride + 6
        vertices[rgb] = vertices[rgb + 1] = vertices[rgb + 2] = 1
      }
    }
  }
  return { vertices: new Float32Array(vertices), indices: new Uint32Array(indices), draws, materialLibraries, vertexColorMaterials,
    uvs1: outputUV1.length ? new Float32Array(outputUV1) : undefined,
    tangents: outputTangents.length ? new Float32Array(outputTangents) : undefined,
    uvs: texcoords.length ? new Float32Array(uvs) : undefined }
}

// File extends Blob, so the file picker can use this entry point directly.
export async function readObjMesh(file: Blob) {
  return parseObjMesh(await file.text())
}
