import tideColor from '../../scene/smugglers-cove/tide-color.png?inline'
import { sceneVertexStride, type BaseColorMaterial, type SceneDraw, type SceneMesh } from '../rendering/scene-mesh'

// The bathymetry atlas uses world x across this width and interpolates from
// coastline(x) to front along each column. Keep the beach in world units.
const front = 7.4, back = -4.5, halfWidth = 7.6, bottom = -0.55
export const shoreBounds = { halfWidth, back, front } as const

function coastline(x: number) {
  return 0.86 - 1.05 * Math.exp(-(((x + 0.6) / 2.75) ** 2)) +
    0.14 * Math.sin(1.4 * x) + 0.10 * Math.sin(0.5 * x + 0.8)
}

// The bed continues under the water so dock footings and rocks can be embedded.
export function shoreHeight(x: number, z: number) {
  const inland = coastline(x) - z
  if (inland < 0) return 0.49 * Math.tanh(0.46 * inland)
  const relief = 0.034 * Math.sin(0.87 * x + 0.61 * z) + 0.016 * Math.sin(2.0 * z - 0.48 * x)
  return 1.10 * Math.tanh(0.158 * inland) + Math.min(1, inland / 1.6) * relief
}

function visibleHeight(x: number, z: number) {
  const d = z - coastline(x)
  if (d <= 0) return shoreHeight(x, z)
  // A broad offshore swell meets shorter, skewed ripples in the shallows.
  // These shape normals so highlights respond to the artist's actual light.
  const phase = 1.30 * Math.sin(0.64 * x - 0.90 * z) + 0.61 * Math.sin(0.61 * z + 0.42 * x)
  const swell = Math.sin(6.1 * z + 2.2 * x + phase)
  const crosswave = Math.sin(10.1 * z - 3.4 * x + 1.04 * Math.sin(1.37 * x + 0.5 * z))
  const offshore = Math.min(1, d * 0.30)
  const envelope = 0.72 + 0.28 * Math.sin(1.18 * x - 0.67 * z + phase)
  const wave = (0.005 + 0.005 * offshore) * swell + (0.0035 - 0.0013 * offshore) * crosswave
  return Math.min(1, d * 2.5) * envelope * wave
}

// The shoreline-aligned atlas blends the photographed sand into green-blue
// water with depth, including shoals and patches of darker submerged growth.
function tideFile() {
  const data = atob(tideColor.slice(tideColor.indexOf(',') + 1))
  const bytes = Uint8Array.from(data, character => character.charCodeAt(0))
  return new File([bytes], 'tide-color.png', { type: 'image/png' })
}

export function makeCoveShore(sand: File) {
  const materials = new Map<string, BaseColorMaterial>([
    ['dry', { color: [1, 0.98, 0.94], map: sand }],
    ['damp', { color: [0.89, 0.90, 0.87], map: sand }],
    ['tidal', { color: [0.76, 0.80, 0.78], map: sand }],
    ['wet', { color: [0.65, 0.71, 0.69], map: sand }],
    ['water', { color: [1, 1, 1], map: tideFile(), clamp: true }],
    ['sediment', { color: [0.17, 0.16, 0.12], map: sand }],
    ['grass-olive', { color: [0.18, 0.23, 0.10] }],
    ['grass-sage', { color: [0.29, 0.33, 0.18] }],
    ['grass-straw', { color: [0.42, 0.36, 0.18] }],
  ])
  const vertices: number[] = [], texcoords: number[] = [], indices: number[] = [], draws: SceneDraw[] = []
  function vertex(x: number, y: number, z: number, nx: number, ny: number, nz: number, u: number, v: number, objectId = 2) {
    const index = vertices.length / sceneVertexStride, length = Math.hypot(nx, ny, nz)
    vertices.push(x, y, z, nx / length, ny / length, nz / length, 1, 1, 1, 0, objectId)
    texcoords.push(u, v)
    return index
  }
  const columns = 128
  // Each material boundary follows the same parametric line on both adjacent
  // grids. The surfaces share positions even where their texture coordinates differ.
  function rowZ(x: number, band: number) {
    const line = coastline(x), variation = 0.07 * Math.sin(1.17 * x + 0.3) + 0.046 * Math.sin(2.07 * x)
    return [back, line - 1.02 + variation, line - 0.61 + 0.78 * variation,
      line - 0.30 + 0.50 * variation, line, front][band]
  }
  const strips = [['dry', 15], ['damp', 3], ['tidal', 3], ['wet', 3], ['water', 64]] as const
  const left: number[] = [], right: number[] = [], inland: number[] = [], ocean: number[] = []
  for (const [band, [material, rows]] of strips.entries()) {
    const firstIndex = indices.length, firstVertex = vertices.length / sceneVertexStride
    for (let j = 0; j <= rows; j++) for (let i = 0; i <= columns; i++) {
      const x = -halfWidth + 2 * halfWidth * i / columns, t = j / rows
      const z = j === rows ? rowZ(x, band + 1) : (1 - t) * rowZ(x, band) + t * rowZ(x, band + 1)
      const h = visibleHeight(x, z), delta = 0.008
      const dx = visibleHeight(x - delta, z) - visibleHeight(x + delta, z)
      const dz = visibleHeight(x, z - delta) - visibleHeight(x, z + delta)
      const water = material === 'water'
      const v = vertex(x, h, z, dx, 2 * delta, dz, water ? i / columns : 0.54 * x, water ? 1 - t : -0.54 * z)
      if (band === 0 || j !== 0) {
        if (i === 0) left.push(v)
        if (i === columns) right.push(v)
      }
      if (band === 0 && j === 0) inland.push(v)
      if (water && j === rows) ocean.push(v)
      if (j < rows && i < columns) {
        const a = firstVertex + j * (columns + 1) + i, b = a + columns + 1
        indices.push(a, b, a + 1, a + 1, b, b + 1)
      }
    }
    draws.push({ firstIndex, indexCount: indices.length - firstIndex, hasUV: true, material })
  }

  // Close the cut edges down to the common base; looking across the bay never
  // reveals a gap between the water and the receiver underneath it.
  const firstIndex = indices.length
  function skirt(boundary: number[], nx: number, nz: number) {
    let previousTop = -1, previousBase = -1
    for (const id of boundary) {
      const p = id * sceneVertexStride, x = vertices[p], y = vertices[p + 1], z = vertices[p + 2]
      const u = (nx ? z : x) * 0.54
      const top = vertex(x, y, z, nx, 0, nz, u, y * 0.54)
      const base = vertex(x, bottom, z, nx, 0, nz, u, bottom * 0.54)
      if (previousTop >= 0) indices.push(previousTop, previousBase, top, top, previousBase, base)
      previousTop = top
      previousBase = base
    }
  }
  skirt(left, -1, 0)
  skirt(right.reverse(), 1, 0)
  skirt(inland.reverse(), 0, -1)
  skirt(ocean, 0, 1)
  const corners = [[-halfWidth, back], [halfWidth, back], [halfWidth, front], [-halfWidth, front]]
    .map(([x, z]) => vertex(x, bottom, z, 0, -1, 0, 0.54 * x, 0.54 * z))
  indices.push(corners[0], corners[1], corners[2], corners[0], corners[2], corners[3])
  draws.push({ firstIndex, indexCount: indices.length - firstIndex, hasUV: true, material: 'sediment' })
  const grass: number[][] = [[], [], []]
  let seed = 74271, tuftId = 2
  function random() {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0
    return seed / 4294967296
  }
  for (const [cx, cz] of [[-4.6, -2.7], [-1.7, -3.4], [4.6, -2.5], [-6.45, -3.55], [6.55, -2.46]]) {
    for (let tuft = 0; tuft < 5; tuft++) {
      const rootX = cx + (random() - 0.5) * 1.52, rootZ = cz + (random() - 0.5) * 0.80
      tuftId++
      for (let blade = 0; blade < 8; blade++) {
        const angle = random() * Math.PI * 2, dx = Math.cos(angle), dz = Math.sin(angle)
        const x = rootX + dx * random() * 0.08, z = rootZ + dz * random() * 0.08
        const y = shoreHeight(x, z) - 0.012, h = 0.19 + random() * 0.25
        const width = 0.009 + random() * 0.020, lean = h * (0.38 + random() * 0.68)
        const batch = grass[Math.floor(random() * grass.length)]
        const base = vertices.length / sceneVertexStride
        for (const t of [0, 0.58]) {
          const bx = x + t * t * lean * dx, bz = z + t * t * lean * dz, by = y + t * h
          const w = width * (1 - 0.45 * t)
          vertex(bx + dz * w, by, bz - dx * w, dz, 0.16, -dx, 0, 0, tuftId)
          vertex(bx - dz * w, by, bz + dx * w, -dz, 0.16, dx, 0, 0, tuftId)
          vertex(bx + dx * w * 0.34, by, bz + dz * w * 0.34, dx, 0.2, dz, 0, 0, tuftId)
        }
        const tip = vertex(x + dx * lean, y + h, z + dz * lean, -dx, 0.5, -dz, 0, 0, tuftId)
        for (let side = 0; side < 3; side++) {
          const a = base + side, b = base + (side + 1) % 3
          batch.push(a, b, a + 3, b, b + 3, a + 3, a + 3, b + 3, tip)
        }
      }
    }
  }
  for (const [i, material] of ['grass-olive', 'grass-sage', 'grass-straw'].entries()) {
    const start = indices.length
    indices.push(...grass[i])
    draws.push({ firstIndex: start, indexCount: indices.length - start, material })
  }
  const mesh: SceneMesh = { vertices: new Float32Array(vertices), indices: new Uint32Array(indices),
    uvs: new Float32Array(texcoords), draws }
  return { mesh, materials, warnings: new Set<string>() }
}
