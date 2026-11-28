// Give exterior vertices a stable cell of their own. Use positions to find
// edges: glTF often duplicates vertices at UV seams or even at every face.
function openBoundaryCells(points: Float32Array, source: Uint32Array) {
  const count = points.length / 3, positionIds = new Uint32Array(count)
  const byPosition = new Map<string, number>(), edges = new Set<number>()
  const canonical = (v: number) => {
    if (!positionIds[v]) {
      const p = v * 3, key = points[p] + '/' + points[p + 1] + '/' + points[p + 2]
      const id = byPosition.get(key) ?? v
      positionIds[v] = id + 1
      byPosition.set(key, id)
    }
    return positionIds[v] - 1
  }
  const toggleEdge = (a: number, b: number) => {
    const key = Math.min(a, b) * count + Math.max(a, b)
    if (!edges.delete(key)) edges.add(key)
  }
  for (let i = 0; i < source.length; i += 3) {
    const a = canonical(source[i]), b = canonical(source[i + 1]), c = canonical(source[i + 2])
    if (a === b || a === c || b === c) continue
    toggleEdge(a, b)
    toggleEdge(b, c)
    toggleEdge(c, a)
  }
  const boundary = new Uint8Array(count)
  for (const key of edges) { boundary[Math.floor(key / count)] = 1; boundary[key % count] = 1 }
  // Regular cell IDs fit in 30 bits at the maximum grid resolution below.
  for (let v = 0; v < count; v++) {
    const id = positionIds[v] - 1
    positionIds[v] = boundary[id] ? 0x80000000 + id : 0
  }
  return positionIds
}

// Vertex clustering for a source primitive that has no authored LOD small
// enough for the preview. Collapse positions first; keep attribute seams split.
export function clusterPreviewPrimitive(points: Float32Array, source: Uint32Array, budget: number, preserveBoundary = false) {
  const vertexCount = points.length / 3
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity]
  for (const v of source) {
    if (v >= vertexCount) throw new Error('glTF index outside vertex array.')
    for (let axis = 0; axis < 3; axis++) {
      min[axis] = Math.min(min[axis], points[3 * v + axis])
      max[axis] = Math.max(max[axis], points[3 * v + axis])
    }
  }
  const extent = Math.max(...max.map((upper, axis) => upper - min[axis]))
  if (!Number.isFinite(extent)) throw new Error('Non-finite glTF geometry.')
  if (extent === 0) return new Uint32Array()
  const cells = new Uint32Array(vertexCount)
  const distances = new Float32Array(vertexCount)
  const initialDivisions = Math.min(1024, Math.max(2, Math.ceil(Math.sqrt(budget))))
  let divisions = initialDivisions
  let boundary = preserveBoundary ? openBoundaryCells(points, source) : undefined
  const survivors: number[] = []
  for (;;) {
    const scale = min.map((lower, axis) => max[axis] > lower
      ? Math.max((divisions - 1) / extent, 1 / (max[axis] - lower)) : 0)
    for (let v = 0; v < vertexCount; v++) {
      if (boundary?.[v]) {
        cells[v] = boundary[v]
        distances[v] = 0
        continue
      }
      // Round to grid points so the outermost positions on a planar scan stay
      // on its boundary. Only referenced vertices participate in the output.
      const x = (points[3 * v] - min[0]) * scale[0]
      const y = (points[3 * v + 1] - min[1]) * scale[1]
      const z = (points[3 * v + 2] - min[2]) * scale[2]
      const i = Math.round(x), j = Math.round(y), k = Math.round(z)
      cells[v] = (i * divisions + j) * divisions + k
      distances[v] = (x - i) ** 2 + (y - j) ** 2 + (z - k) ** 2
    }
    const faces = new Set<string>()
    survivors.length = 0
    for (let offset = 0; offset < source.length; offset += 3) {
      const a = cells[source[offset]], b = cells[source[offset + 1]], c = cells[source[offset + 2]]
      if (a === b || b === c || c === a) continue
      // Preserve opposite winding, but discard triangles that collapse onto
      // the same oriented face in this material.
      const key = a < b && a < c ? a + '/' + b + '/' + c
        : b < c ? b + '/' + c + '/' + a : c + '/' + a + '/' + b
      if (faces.has(key)) continue
      faces.add(key)
      survivors.push(offset)
      if (survivors.length > budget) break
    }
    if (survivors.length <= budget) break
    if (divisions === 2) {
      if (!boundary) throw new Error('Triangle budget is too small to preserve the mesh.')
      // A crop can have more boundary faces than its entire allowance. The
      // usual bounded LOD is still usable when a literal border cannot fit.
      boundary = undefined
      divisions = initialDivisions
      continue
    }
    divisions = Math.max(2, Math.floor(divisions * 0.75))
  }

  // Only vertices connected by a real source edge may share UVs and normals.
  // Separate UV islands can therefore occupy the same welded geometric point.
  const parent = Uint32Array.from({ length: vertexCount }, (_, i) => i)
  function root(i: number) {
    while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i] }
    return i
  }
  function weld(a: number, b: number) {
    if (cells[a] !== cells[b]) return
    const ra = root(a), rb = root(b)
    if (ra !== rb) {
      if (distances[ra] <= distances[rb]) parent[rb] = ra
      else parent[ra] = rb
    }
  }
  const positionForCell = new Map<number, number>()
  for (let offset = 0; offset < source.length; offset += 3) {
    const a = source[offset], b = source[offset + 1], c = source[offset + 2]
    weld(a, b); weld(b, c); weld(c, a)
    for (let j = 0; j < 3; j++) {
      const v = source[offset + j], previous = positionForCell.get(cells[v])
      if (previous === undefined || distances[v] < distances[previous]) positionForCell.set(cells[v], v)
    }
  }
  const result = new Uint32Array(survivors.length * 3)
  for (let t = 0; t < survivors.length; t++) for (let j = 0; j < 3; j++) {
    const vertex = root(source[survivors[t] + j]), geometric = positionForCell.get(cells[vertex])!
    for (let axis = 0; axis < 3; axis++) points[3 * vertex + axis] = points[3 * geometric + axis]
    result[3 * t + j] = vertex
  }
  return result
}
