import { readPreviewGltf, type PreviewGltfDocument } from '../src/geometry/preview-gltf'
import { sceneVertexStride } from '../src/rendering/scene-mesh'

export async function checkPreviewLod(report: (line: string) => void) {
  // Two adjacent UV charts meet at x=1, with U=1 on one side and U=0 on the other.
  const positions: number[] = [], uvs: number[] = [], indices: number[] = []
  const steps = 30
  for (let chart = 0; chart < 2; chart++) {
    const first = positions.length / 3
    for (let z = 0; z <= steps; z++) for (let x = 0; x <= steps; x++) {
      positions.push(chart + x / steps, 0, z / steps)
      uvs.push(x / steps, z / steps)
    }
    for (let z = 0; z < steps; z++) for (let x = 0; x < steps; x++) {
      const a = first + z * (steps + 1) + x, b = a + steps + 1
      indices.push(a, b, a + 1, a + 1, b, b + 1)
    }
  }
  const buffers = [new Float32Array(positions), new Float32Array(uvs), new Uint32Array(indices)]
  const urls = buffers.map(data => URL.createObjectURL(new Blob([data])))
  const doc: PreviewGltfDocument = {
    asset: { version: '2.0' }, scenes: [{ nodes: [0, 1] }],
    nodes: [{ mesh: 0 }, { mesh: 0, translation: [0, 0, 1] }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0, TEXCOORD_0: 1 }, indices: 2, material: 0 }] }],
    buffers: buffers.map((data, i) => ({ uri: urls[i], byteLength: data.byteLength })),
    bufferViews: buffers.map((data, i) => ({ buffer: i, byteLength: data.byteLength })),
    accessors: [
      { bufferView: 0, type: 'VEC3', componentType: 5126, count: positions.length / 3, min: [0, 0, 0], max: [2, 0, 1] },
      { bufferView: 1, type: 'VEC2', componentType: 5126, count: uvs.length / 2 },
      { bufferView: 2, type: 'SCALAR', componentType: 5125, count: indices.length },
    ],
    materials: [{ pbrMetallicRoughness: { baseColorFactor: [0.8, 0.6, 0.4, 1] } }],
  }
  try {
    let sourceTriangles = 0, outputTriangles = 0
    const { mesh, materials } = await readPreviewGltf({ document: doc, base: new URL('https://fixture.invalid/') }, undefined, {
      maxTriangles: 900, simplifyToBudget: true,
      crop: () => ({ min: [0.2, -1, 0.25], max: [1.8, 1, 1.75] }),
      onSimplified(source, result) { sourceTriangles = source; outputTriangles = result },
    })
    if (mesh.indices.length > 900 * 3 || mesh.draws?.length !== 2 || !materials.has('0')) {
      throw new Error('LOD must preserve both material instances and stay within the final clipped triangle budget.')
    }
    if (sourceTriangles <= 900 || outputTriangles !== mesh.indices.length / 3) {
      throw new Error('Source LOD reporting must count measured input faces and the final clipped output.')
    }
    let area = 0, seam = false
    const byPosition = new Map<string, number>()
    for (let i = 0; i < mesh.indices.length; i += 3) {
      const a = mesh.indices[i] * sceneVertexStride, b = mesh.indices[i + 1] * sceneVertexStride
      const c = mesh.indices[i + 2] * sceneVertexStride, p = mesh.vertices
      area += 0.5 * ((p[b + 2] - p[a + 2]) * (p[c] - p[a]) - (p[b] - p[a]) * (p[c + 2] - p[a + 2]))
      for (let j = 0; j < 3; j++) {
        const index = mesh.indices[i + j], v = index * sceneVertexStride
        const key = p[v] + '/' + p[v + 2] + '/' + p[v + 10], u = mesh.uvs![2 * index]
        const previous = byPosition.get(key)
        if (previous !== undefined && Math.abs(previous - u) > 0.6) seam = true
        byPosition.set(key, u)
      }
    }
    if (Math.abs(area - 1.6 * 1.5) > 0.001 || !seam) throw new Error('LOD lost surface coverage or a UV seam.')
    report('glTF preview LOD preserves coverage, UV seams, instances and the clipped triangle budget.')

    // One material primitive can hold two pavement patches a district apart.
    // The remote patch must not consume the visible patch's geometric budget.
    const district = new Float32Array(positions)
    for (let i = 3 * (steps + 1) ** 2; i < district.length; i += 3) {
      district[i] += 1000
      district[i + 2] += 1000
    }
    const districtUrl = URL.createObjectURL(new Blob([district]))
    urls.push(districtUrl)
    const document: PreviewGltfDocument = { ...doc, scenes: [{ nodes: [0] }],
      buffers: [{ uri: districtUrl, byteLength: district.byteLength }, ...doc.buffers.slice(1)],
      accessors: [{ ...doc.accessors[0], max: [1002, 0, 1001] }, ...doc.accessors.slice(1)] }
    const input = { document, base: new URL('https://fixture.invalid/') }
    const crop = () => ({ min: [-0.1, -1, -0.1], max: [1.1, 1, 1.1] })
    const intact = await readPreviewGltf(input, undefined, { crop, simplifyToBudget: true, maxTriangles: 1800 })
    if (intact.mesh.indices.length !== 1800 * 3) throw new Error('Off-screen paving consumed the crop budget.')
    const reduced = await readPreviewGltf(input, undefined, { crop, simplifyToBudget: true, maxTriangles: 100 })
    let visibleArea = 0
    const { vertices: p, indices: corners } = reduced.mesh
    for (let i = 0; i < corners.length; i += 3) {
      const a = corners[i] * sceneVertexStride, b = corners[i + 1] * sceneVertexStride, c = corners[i + 2] * sceneVertexStride
      visibleArea += 0.5 * ((p[b + 2] - p[a + 2]) * (p[c] - p[a]) - (p[b] - p[a]) * (p[c + 2] - p[a + 2]))
    }
    if (Math.abs(visibleArea - 1) > 0.001 || corners.length > 100 * 3) throw new Error('Distant paving distorted the visible LOD.')
    report('A distant mesh patch neither spends the crop budget nor distorts the local simplification grid.')

    // Two road materials join on a curved curb. Their grids have different
    // world origins/extents, so independent grid LODs can leave a zigzag gap.
    const road = new Float32Array(positions), seamRow = new Map<string, number>()
    for (let i = 0; i < road.length; i += 3) {
      const x = road[i], z = road[i + 2]
      road[i] += 0.12 * Math.sin(z * Math.PI * 3) * (1 - Math.abs(1 - x))
      if (x === 1) seamRow.set(road[i] + '/' + z, Math.round(z * steps))
    }
    const roadUrl = URL.createObjectURL(new Blob([road]))
    urls.push(roadUrl)
    const half = indices.length / 2
    const street: PreviewGltfDocument = { ...doc,
      nodes: [{ mesh: 0, name: 'Street' }, { mesh: 1, name: 'Decorations', translation: [0, 1, 0] }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0, TEXCOORD_0: 1 }, indices: 3, material: 0 },
        { attributes: { POSITION: 0, TEXCOORD_0: 1 }, indices: 4, material: 1 }] }, doc.meshes[0]],
      buffers: [{ byteLength: road.byteLength, uri: roadUrl }, ...doc.buffers.slice(1)],
      accessors: [...doc.accessors, { ...doc.accessors[2], count: half },
        { ...doc.accessors[2], count: half, byteOffset: half * Uint32Array.BYTES_PER_ELEMENT }],
      materials: [{ name: 'Cobbles' }, { name: 'Paving' }],
    }
    for (const limit of [4200, 1800, 120]) {
      const { mesh } = await readPreviewGltf({ document: street, base: input.base }, undefined, {
        maxTriangles: limit, simplifyToBudget: true,
        preserveSurface: (node, material, bounds) => node.path === '/Street' && bounds.max[1] === 0
          && (material === 'Cobbles' || material === 'Paving'),
      })
      if (mesh.indices.length > limit * 3 || mesh.draws?.length !== 3) throw new Error('Street LOD lost a part or exceeded its cap.')
      for (const draw of mesh.draws.slice(0, 2)) {
        if (limit === 4200 && draw.indexCount !== half) throw new Error('An affordable ground primitive was decimated.')
        if (limit !== 1800) continue
        if (draw.indexCount === half) throw new Error('Reduced-budget fixture failed to exercise curb LOD.')
        const edges = new Set<number>()
        for (let i = draw.firstIndex; i < draw.firstIndex + draw.indexCount; i += 3) {
          const rows = [0, 1, 2].map(j => {
            const k = mesh.indices[i + j] * sceneVertexStride
            return seamRow.get(mesh.vertices[k] + '/' + mesh.vertices[k + 2])
          })
          rows.forEach((a, j) => {
            const b = rows[(j + 1) % 3]
            if (a !== undefined && b !== undefined && Math.abs(a - b) === 1) edges.add(Math.min(a, b))
          })
        }
        if (edges.size !== steps) throw new Error('The two material LODs separated along their curved curb.')
      }
    }
    report('Street materials preserve original paving when affordable and matching curb edges at reduced detail; tiny caps stay bounded.')
  } finally {
    for (const url of urls) URL.revokeObjectURL(url)
  }
}
