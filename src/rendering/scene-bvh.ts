import { sceneVertexStride } from './scene-mesh'
import type { SceneMesh } from './scene-mesh'

// Preorder nodes contain an escape index. A miss skips a whole subtree;
// a hit descends to the next node. Traversal needs no per-ray stack.
export function buildSceneBvh(mesh: SceneMesh) {
  const { vertices, indices } = mesh
  const count = indices.length / 3
  if (!Number.isInteger(count) || !count || vertices.length % sceneVertexStride) {
    throw new Error('Expected an indexed triangle scene.')
  }
  const bounds = new Float32Array(count * 6)
  const centers = new Float32Array(count * 3)
  const original = new ArrayBuffer(count * 48)
  const floats = new Float32Array(original)
  const words = new Uint32Array(original)
  const order = Array.from({ length: count }, (_, i) => i)
  for (const draw of mesh.draws ?? []) {
    if (draw.doubleSided) {
      const end = draw.firstIndex + draw.indexCount
      for (let i = draw.firstIndex; i < end; i += 3) words[i * 4 + 11] = 1
    }
  }
  for (let face = 0; face < count; face++) {
    const corners = indices.subarray(3 * face, 3 * face + 3)
    const object = vertices[corners[0] * sceneVertexStride + 10]
    if (!Number.isSafeInteger(object) || object < 1 || object > 0xffffff) throw new Error('Invalid scene object ID.')
    for (const index of corners) {
      if (index * sceneVertexStride >= vertices.length || vertices[index * sceneVertexStride + 10] !== object) {
        throw new Error('A triangle must reference three vertices of one object.')
      }
    }
    for (let axis = 0; axis < 3; axis++) {
      const a = vertices[corners[0] * sceneVertexStride + axis]
      const b = vertices[corners[1] * sceneVertexStride + axis]
      const c = vertices[corners[2] * sceneVertexStride + axis]
      if (![a, b, c].every(Number.isFinite)) throw new Error('Non-finite scene geometry.')
      bounds[6 * face + axis] = Math.min(a, b, c)
      bounds[6 * face + 3 + axis] = Math.max(a, b, c)
      centers[3 * face + axis] = (a + b + c) / 3
      floats[12 * face + axis] = a
      // Preserve identical shared vertices for the ray-space edge tests.
      floats[12 * face + 4 + axis] = b
      floats[12 * face + 8 + axis] = c
    }
    words[face * 12 + 3] = object
    words[face * 12 + 7] = face
  }
  const nodes: { lo: number[]; hi: number[]; first: number; size: number; escape: number }[] = []
  function split(first: number, end: number) {
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity]
    const cMin = lo.slice(), cMax = hi.slice()
    for (let i = first; i < end; i++) {
      const f = order[i]
      for (let a = 0; a < 3; a++) {
        lo[a] = Math.min(lo[a], bounds[f * 6 + a])
        hi[a] = Math.max(hi[a], bounds[f * 6 + 3 + a])
        cMin[a] = Math.min(cMin[a], centers[f * 3 + a])
        cMax[a] = Math.max(cMax[a], centers[f * 3 + a])
      }
    }
    const size = end - first
    const node = { lo, hi, first, size: size <= 4 ? size : 0, escape: 0 }
    nodes.push(node)
    if (size > 4) {
      const span = cMax.map((max, i) => max - cMin[i])
      const axis = span.indexOf(Math.max(...span))
      const sorted = order.slice(first, end).sort((a, b) => centers[3*a + axis] - centers[3*b + axis])
      for (let i = 0; i < size; i++) order[first + i] = sorted[i]
      const mid = first + Math.floor(size / 2)
      split(first, mid)
      split(mid, end)
    }
    node.escape = nodes.length
  }
  split(0, count)
  const nodeBuffer = new ArrayBuffer(nodes.length * 48)
  const nf = new Float32Array(nodeBuffer), nu = new Uint32Array(nodeBuffer)
  nodes.forEach((n, i) => {
    const k = 12 * i
    nf.set(n.lo, k); nf.set(n.hi, k+4)
    nu[k+3] = n.first; nu[k+7] = n.size; nu[k+8] = n.escape
  })
  const triangles = new Uint32Array(count * 12)
  order.forEach((f, i) => triangles.set(words.subarray(f*12, f*12+12), i*12))
  return { nodes: new Uint32Array(nodeBuffer), triangles }
}
