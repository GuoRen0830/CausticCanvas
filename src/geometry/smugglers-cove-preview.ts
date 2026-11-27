import { makeCoveShore, shoreBounds, shoreHeight } from './cove-shore'
import { makeCoveSkiff } from './cove-skiff'
import { makeQuayside, makeDryingRack, makeLocalCargo, makeMoorings, pierApproach, quayLevel } from './cove-quayside'
import { makeCoveNetLoft } from './cove-net-loft'
import { makeCoveCargoGear } from './cove-cargo-gear'
import { makeCovePotMarkers } from './cove-pot-markers'
import { makeCoveTidePools } from './cove-tide-pools'
import { makeCoveLugger } from './cove-lugger'
import { makeCoveBreakwater } from './cove-breakwater'
import { makeCoveFishMarket } from './cove-fish-market'
import { supplyCoveRockColor } from './cove-rock-material'
import { sceneVertexStride, type BaseColorMaterial, type SceneDraw, type SceneMesh } from '../rendering/scene-mesh'
import { readPreviewGltf, type PreviewGltfDocument, type PreviewGltfOptions, type PreviewNodeBounds } from './preview-gltf'

type Download = { url: string; include?: Record<string, { url: string }> }
type FileList = Record<string, Record<string, Record<string, Download>>>
type Part = Awaited<ReturnType<typeof readPreviewGltf>>

async function fetchOk(url: string, signal?: AbortSignal, headers?: HeadersInit) {
  const response = await fetch(url, { signal, headers })
  if (!response.ok) throw new Error(`Could not load ${url} (HTTP ${response.status}).`)
  return response
}

// File names and companion locations can differ between Poly Haven asset releases.
async function getDownload(id: string, maps: string[], formats: string[], signal?: AbortSignal) {
  const url = 'https://api.polyhaven.com/files/' + id
  // Browsers identify the site with their Referer; Node has no document or Referer.
  const headers = typeof document === 'undefined' ? { 'User-Agent': 'CausticCanvas scene preview' } : undefined
  const files = await (await fetchOk(url, signal, headers)).json() as FileList
  for (const [map, resolutions] of Object.entries(files)) {
    if (!maps.includes(map.toLowerCase())) continue
    const sizes = Object.keys(resolutions).filter(key => /^\d+k$/i.test(key))
      .sort((a, b) => parseInt(a) - parseInt(b))
    for (const size of sizes) for (const format of formats) {
      const entry = resolutions[size][format]
      if (entry?.url) return entry
    }
  }
  throw new Error(`Poly Haven has no ${maps[0]} download for ${id}.`)
}

type LodNode = { name?: string; mesh?: number; children?: number[]; extensions?: { MSFT_lod?: { ids: number[] } } }

// A glTF export can contain the whole LOD hierarchy. Keep only the coarsest
// available copy of each named group; keep unlabelled geometry and children.
function selectLods(doc: { scenes: { nodes?: number[] }[]; nodes: LodNode[]; meshes: { name?: string }[] }) {
  function lod(index: number) {
    const node = doc.nodes[index]
    for (const name of [node?.name, node?.mesh !== undefined && doc.meshes[node.mesh]?.name]) {
      if (typeof name !== 'string') continue
      const match = /^(.*?)[_. -]?lod[_. -]?(\d+)(.*)$/i.exec(name)
      if (match) return { group: (match[1] + match[3]).toLowerCase(), level: Number(match[2]) }
    }
  }
  function filter(indices: number[]) {
    const selected = [...new Set(indices.map(i => doc.nodes[i]?.extensions?.MSFT_lod?.ids.at(-1) ?? i))]
    const choices = selected.map(lod), levels = new Map<string, number>()
    for (const item of choices) if (item) levels.set(item.group, Math.max(levels.get(item.group) ?? 0, item.level))
    return selected.filter((index, slot) => {
      const item = choices[slot]
      if (item) return item.level === levels.get(item.group)
      const node = doc.nodes[index]
      if (node?.mesh === undefined || node.children?.length) return true
      // Blender exports can leave the full-resolution mesh unsuffixed next to
      // Foo_LOD1, Foo_LOD2, etc.
      return ![node.name, doc.meshes[node.mesh].name].some(name => name && levels.has(name.toLowerCase()))
    })
  }
  for (const scene of doc.scenes) if (scene.nodes) scene.nodes = filter(scene.nodes)
  for (const node of doc.nodes) if (node.children) node.children = filter(node.children)
}

async function loadModel(id: string, maxTriangles: number, signal?: AbortSignal, crop?: PreviewGltfOptions['crop']) {
  const file = await getDownload(id, ['gltf'], ['gltf'], signal)
  const response = await fetchOk(file.url, signal)
  const documentUrl = response.url || file.url
  const document = await response.json() as PreviewGltfDocument
  selectLods(document)
  const resolve = (key: string) => new URL(key.replaceAll('\\', '/'), documentUrl).href
  const includes = new Map(Object.entries(file.include ?? {}).map(([key, item]) => [resolve(key), item.url]))
  return readPreviewGltf({ document, base: new URL(documentUrl) }, signal, {
    resolveUri(uri) { const url = resolve(uri); return includes.get(url) ?? url },
    crop, maxTriangles, simplifyToBudget: true,
  })
}

// Preserve supports underneath the selected module. Some exports put each deck
// in a separate mesh, others join the assembled pier into a single node.
function dockBounds(nodes: readonly PreviewNodeBounds[]) {
  if (!nodes.length) throw new Error('No bounds found for the wooden pier.')
  const footprint = (n: PreviewNodeBounds) => (n.max[0] - n.min[0]) * (n.max[2] - n.min[2])
  const decks = nodes.filter(n => /deck|section|platform/i.test(n.name + ' ' + n.meshName))
  const anchor = (decks.length ? decks : nodes).reduce((a, b) => footprint(a) >= footprint(b) ? a : b)
  const pad = Math.max(0.01, 0.08 * Math.max(anchor.max[0] - anchor.min[0], anchor.max[2] - anchor.min[2]))
  return { min: [anchor.min[0] - pad, -Infinity, anchor.min[2] - pad],
    max: [anchor.max[0] + pad, Infinity, anchor.max[2] + pad] }
}

function barrelBounds(nodes: readonly PreviewNodeBounds[]) {
  // The download is a kit with multiple barrels and detached staves. Take one
  // upright barrel and the hoops and lid inside its footprint.
  const barrels = nodes.filter(n => {
    const w = n.max[0] - n.min[0], d = n.max[2] - n.min[2], h = n.max[1] - n.min[1]
    return h > 0.85 * Math.max(w, d) && Math.min(w, d) > 0.55 * Math.max(w, d)
  })
  const volume = (n: PreviewNodeBounds) => (n.max[0] - n.min[0]) * (n.max[1] - n.min[1]) * (n.max[2] - n.min[2])
  const best = (barrels.length ? barrels : nodes).reduce((a, b) => volume(a) >= volume(b) ? a : b)
  const margin = 0.06 * Math.max(best.max[0] - best.min[0], best.max[2] - best.min[2], 0.01)
  return { min: best.min.map(x => x - margin), max: best.max.map(x => x + margin) }
}

async function loadShore(signal?: AbortSignal) {
  const sand = await getDownload('coast_sand_01', ['diffuse', 'diff', 'color', 'albedo'], ['jpg', 'png'], signal)
  const blob = await (await fetchOk(sand.url, signal)).blob()
  return makeCoveShore(new File([blob], 'coast-sand', { type: blob.type }))
}

function bounds(mesh: SceneMesh) {
  const v = mesh.vertices, min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity]
  for (let i = 0; i < v.length; i += sceneVertexStride) for (let k = 0; k < 3; k++) {
    min[k] = Math.min(min[k], v[i + k])
    max[k] = Math.max(max[k], v[i + k])
  }
  const width = max[0] - min[0], depth = max[2] - min[2]
  if (!(Math.max(width, depth) > 0)) throw new Error('Coastal model has no footprint.')
  return { min, max, width, depth, cx: (min[0] + max[0]) / 2, cz: (min[2] + max[2]) / 2 }
}

// Find the deck by horizontal surface area. Post tips can be much higher than
// the walking surface, so using the model's vertical bounds puts it on stilts.
function inspectDeck(dock: Part) {
  const { vertices: v, indices, draws } = dock.mesh, b = bounds(dock.mesh)
  const buckets = new Float64Array(64), heights = new Float64Array(64)
  const span = b.max[1] - b.min[1] || 1, step = sceneVertexStride
  const ranges = Array.from(buckets, () => ({ minX: Infinity, maxX: -Infinity, minZ: Infinity, maxZ: -Infinity }))
  const materials = Array.from(buckets, () => new Map<string, number>())
  for (const draw of draws ?? []) for (let i = draw.firstIndex; i < draw.firstIndex + draw.indexCount; i += 3) {
    const a = indices[i] * step, k = indices[i + 1] * step, c = indices[i + 2] * step
    const dx = v[k] - v[a], dz = v[k + 2] - v[a + 2], ex = v[c] - v[a], ez = v[c + 2] - v[a + 2]
    const area = dz * ex - dx * ez
    const heightRange = Math.max(v[a+1], v[k+1], v[c+1]) - Math.min(v[a+1], v[k+1], v[c+1])
    if (area <= 0 || heightRange > 0.18 * Math.max(Math.hypot(dx, dz), Math.hypot(ex, ez))) continue
    const h = (v[a+1] + v[k+1] + v[c+1]) / 3
    const bin = Math.min(63, Math.max(0, Math.floor(64 * (h - b.min[1]) / span)))
    buckets[bin] += area
    heights[bin] += h * area
    if (draw.material) materials[bin].set(draw.material, (materials[bin].get(draw.material) ?? 0) + area)
    const r = ranges[bin]
    for (const p of [a, k, c]) {
      r.minX = Math.min(r.minX, v[p]); r.maxX = Math.max(r.maxX, v[p])
      r.minZ = Math.min(r.minZ, v[p + 2]); r.maxZ = Math.max(r.maxZ, v[p + 2])
    }
  }
  const at = buckets.indexOf(Math.max(...buckets))
  if (!buckets[at]) return { ...b, level: b.max[1], planks: undefined }
  const level = heights[at] / buckets[at], r = ranges[at]
  const largest = [...materials[at]].sort((a, b) => b[1] - a[1])[0]
  return { ...b, level, width: r.maxX - r.minX, depth: r.maxZ - r.minZ,
    cx: (r.minX + r.maxX) / 2, cz: (r.minZ + r.maxZ) / 2,
    planks: largest ? dock.materials.get(largest[0]) : undefined }
}

type Pose = { x: number; y: number; z: number; angle: number; scale: number; cx: number; cy: number; cz: number
  below?: number; above?: number; rock?: { width: number; depth: number } }

function place(part: Part, pose: Pose): Part {
  const { cx, cy, cz, scale, x, y, z, rock } = pose, c = Math.cos(pose.angle), s = Math.sin(pose.angle)
  const v = part.mesh.vertices.slice()
  for (let i = 0; i < v.length; i += sceneVertexStride) {
    const dx = (v[i] - cx) * scale, dy = (v[i + 1] - cy) * scale, dz = (v[i + 2] - cz) * scale
    const px = x + c * dx + s * dz, pz = z - s * dx + c * dz
    const vertical = (dy < 0 ? pose.below : pose.above) ?? 1
    let py = y + dy * vertical
    let nx = c * v[i + 3] + s * v[i + 5], ny = v[i + 4] / vertical, nz = -s * v[i + 3] + c * v[i + 5]
    if (rock) {
      // A scan includes a collar of captured sand. Tuck its perimeter into the
      // same sloping bed as the beach so the outcrop grows out of the bank.
      const radius = Math.hypot(2 * dx / rock.width, 2 * dz / rock.depth)
      const sink = Math.max(0, radius - 0.43)
      const span = Math.max(rock.width, rock.depth)
      py += shoreHeight(px, pz) - span * (0.035 + 0.19 * sink * sink)
      const radialSlope = radius > 0 ? 1.52 * span * sink / radius : 0
      const slopeX = radialSlope * dx / (rock.width * rock.width), slopeZ = radialSlope * dz / (rock.depth * rock.depth)
      nx -= ny * ((shoreHeight(px + 0.02, pz) - shoreHeight(px - 0.02, pz)) / 0.04 - c * slopeX - s * slopeZ)
      nz -= ny * ((shoreHeight(px, pz + 0.02) - shoreHeight(px, pz - 0.02)) / 0.04 + s * slopeX - c * slopeZ)
      if (py < -0.39) {
        const depth = Math.tanh((-0.39 - py) / 0.13)
        py = -0.39 - 0.13 * depth
        ny /= Math.max(0.001, 1 - depth * depth)
      }
    }
    const length = Math.hypot(nx, ny, nz) || 1
    v[i] = px; v[i+1] = py; v[i+2] = pz
    v[i+3] = nx / length; v[i+4] = ny / length; v[i+5] = nz / length
  }
  return { ...part, mesh: { ...part.mesh, vertices: v } }
}

function groundProp(part: Part, x: number, z: number, width: number, angle: number, surface?: number) {
  const b = bounds(part.mesh), scale = width / Math.max(b.width, b.depth)
  if (surface === undefined) {
    // Seat the lowest part of the footprint in the dune, including its downhill
    // edge. A box centered at the mean sand level visibly floats on one side.
    const c = Math.cos(angle), s = Math.sin(angle), v = part.mesh.vertices
    surface = shoreHeight(x, z)
    for (let i = 0; i < v.length; i += sceneVertexStride) {
      if ((v[i + 1] - b.min[1]) * scale > 0.04) continue
      const dx = (v[i] - b.cx) * scale, dz = (v[i + 2] - b.cz) * scale
      surface = Math.min(surface, shoreHeight(x + c * dx + s * dz, z - s * dx + c * dz))
    }
  }
  return place(part, { x, z, y: surface - 0.012, cx: b.cx, cy: b.min[1], cz: b.cz, angle,
    scale })
}

function outcrop(part: Part, x: number, z: number, width: number, angle: number) {
  const b = bounds(part.mesh), scale = width / Math.max(b.width, b.depth)
  const w = b.width * scale, d = b.depth * scale
  const extentX = 0.5 * (Math.abs(Math.cos(angle)) * w + Math.abs(Math.sin(angle)) * d)
  const extentZ = 0.5 * (Math.abs(Math.sin(angle)) * w + Math.abs(Math.cos(angle)) * d)
  const inset = 0.04
  const fit = (center: number, extent: number, min: number, max: number) =>
    Math.max(min + extent + inset, Math.min(center, max - extent - inset))
  return place(part, { x: fit(x, extentX, -shoreBounds.halfWidth, shoreBounds.halfWidth), y: 0,
    z: fit(z, extentZ, shoreBounds.back, shoreBounds.front), cx: b.cx, cy: b.min[1], cz: b.cz,
    angle, scale, rock: { width: Math.max(0.001, w), depth: Math.max(0.001, d) } })
}

function combine(parts: Part[]): Part {
  const vertices = new Float32Array(parts.reduce((n, p) => n + p.mesh.vertices.length, 0))
  const indices = new Uint32Array(parts.reduce((n, p) => n + p.mesh.indices.length, 0))
  const uvs = new Float32Array(2 * vertices.length / sceneVertexStride), draws: SceneDraw[] = []
  const materials = new Map<string, BaseColorMaterial>(), warnings = new Set<string>()
  const catalogs = new Map<Map<string, BaseColorMaterial>, string>()
  let firstVertex = 0, firstIndex = 0, nextId = 2
  for (const part of parts) {
    const mesh = part.mesh, ids = new Map<number, number>()
    vertices.set(mesh.vertices, firstVertex * sceneVertexStride)
    if (mesh.uvs) uvs.set(mesh.uvs, 2 * firstVertex)
    for (let i = 10; i < mesh.vertices.length; i += sceneVertexStride) {
      const original = mesh.vertices[i]
      if (!ids.has(original)) ids.set(original, nextId++)
      vertices[firstVertex * sceneVertexStride + i] = ids.get(original)!
    }
    for (let i = 0; i < mesh.indices.length; i++) indices[firstIndex+i] = mesh.indices[i] + firstVertex
    let prefix = catalogs.get(part.materials)
    if (prefix === undefined) {
      prefix = catalogs.size + '/'
      catalogs.set(part.materials, prefix)
      for (const [key, material] of part.materials) materials.set(prefix + key, material)
    }
    for (const draw of mesh.draws ?? []) draws.push({ ...draw, firstIndex: draw.firstIndex + firstIndex,
      material: draw.material === undefined ? undefined : prefix + draw.material })
    for (const warning of part.warnings) warnings.add(warning)
    firstVertex += mesh.vertices.length / sceneVertexStride
    firstIndex += mesh.indices.length
  }
  return { mesh: { vertices, indices, uvs, draws }, materials, warnings }
}

export async function readCovePreview(signal?: AbortSignal): Promise<Part> {
  const warnings: string[] = []
  async function dressing(id: string, budget: number, crop?: PreviewGltfOptions['crop']) {
    try { return await loadModel(id, budget, signal, crop) }
    catch (error) {
      if (signal?.aborted) throw error
      warnings.push(id + ' could not load: ' + (error instanceof Error ? error.message : String(error)))
    }
  }
  async function cargo(id: string, kind: Parameters<typeof makeLocalCargo>[0], budget: number, crop?: PreviewGltfOptions['crop']) {
    try {
      const asset = await loadModel(id, budget, signal, crop)
      if (!asset.warnings.size) return asset
    } catch (error) {
      if (signal?.aborted) throw error
    }
    warnings.push(`Using the project-modeled ${kind} in place of ${id}.`)
    return makeLocalCargo(kind)
  }
  // Limit simultaneous high-resolution sources; each downloaded model can be
  // placed more than once without fetching or decoding another copy.
  const [shore, rocks, dock] = await Promise.all([
    loadShore(signal), loadModel('coast_land_rocks_02', 68_000, signal).then(supplyCoveRockColor),
    dressing('modular_wooden_pier', 42_000, dockBounds),
  ])
  const [chest, crate, barrels, lantern] = await Promise.all([
    loadModel('treasure_chest', 58_000, signal), cargo('wooden_crate_01', 'crate', 14_000),
    cargo('wooden_barrels_01', 'barrel', 16_000, barrelBounds), cargo('wooden_lantern_01', 'lantern', 8_000),
  ])
  const parts: Part[] = [shore,
    outcrop(rocks, -3.82, 0.22, 4.20, -0.32), outcrop(rocks, -6.46, -0.39, 2.86, 1.56),
    outcrop(rocks, 4.92, 1.30, 1.96, -1.22)]

  const approach = pierApproach, deckY = shoreHeight(approach.x, approach.z) + 0.12
  const quay = makeQuayside(deckY)
  parts.push(quay, makeCoveCargoGear(deckY), makeDryingRack(), makeCovePotMarkers(),
    makeCoveFishMarket(), makeCoveBreakwater(),
    place(makeCoveLugger(), { x: -2.35, y: 0, z: 5.17, cx: 0, cy: 0, cz: 0, scale: 1, angle: -0.16 }),
    makeCoveTidePools(quay.materials.get('weathered limestone')!, shore.materials.get('water')!),
    place(makeCoveNetLoft(), { x: 3.71, y: quayLevel, z: -2.93,
    cx: 0, cy: 0, cz: 0, scale: 1, angle: 0 }))
  let planks: BaseColorMaterial | undefined
  if (dock) {
    const deck = inspectDeck(dock), heading = 0.06
    const segment = 2.04, scale = segment / Math.max(deck.width, deck.depth)
    const angle = heading + (deck.width > deck.depth ? Math.PI / 2 : 0)
    planks = deck.planks
    for (const distance of [segment / 2, 1.49 * segment]) {
      parts.push(place(dock, { x: approach.x + Math.sin(heading) * distance,
        z: approach.z + Math.cos(heading) * distance, y: deckY, angle, scale,
        cx: deck.cx, cy: deck.level, cz: deck.cz,
        below: Math.min(1, (deckY + 0.49) / Math.max(0.001, (deck.level - deck.min[1]) * scale)),
        above: Math.min(1, 0.48 / Math.max(0.001, (deck.max[1] - deck.level) * scale)) }))
    }
    const portEdge = Math.max(0, Math.min(deck.width, deck.depth) * scale / 2 - 0.13)
    const cleat = (distance: number): [number, number, number] => [
      approach.x + Math.sin(heading)*distance - Math.cos(heading)*portEdge, deckY,
      approach.z + Math.cos(heading)*distance + Math.sin(heading)*portEdge]
    // The two lines reach the skiff's stem and transom. Both bollards sit on
    // the measured edge of this particular pier module, including narrow LODs.
    parts.push(makeMoorings([{ post: cleat(0.49), boat: [1.18,0.248,0.20] },
      { post: cleat(3.80), boat: [0.53,0.297,3.61] }]))
  }
  parts.push(place(makeCoveSkiff(planks), { x: 0.72, y: 0, z: 1.90,
    cx: 0, cy: 0, cz: 0, scale: 1, angle: -0.12 }))
  parts.push(groundProp(chest, 1.05, -2.03, 0.98, -0.17, quayLevel))
  const base = groundProp(crate, 1.05, -3.17, 0.88, 0.06, quayLevel)
  parts.push(base, groundProp(crate, 1.92, -3.11, 0.71, -0.12, quayLevel),
    groundProp(crate, 1.04, -3.18, 0.51, -0.10, bounds(base.mesh).max[1] + 0.013))
  const barrel = bounds(barrels.mesh)
  if (barrel.max[1] - barrel.min[1] < 0.55 * Math.max(barrel.width, barrel.depth)) {
    parts.push(groundProp(barrels, 4.90, -1.84, 1.14, 0.04, quayLevel))
  } else {
    parts.push(groundProp(barrels, 4.75, -1.80, 0.48, 0.48, quayLevel),
      groundProp(barrels, 5.37, -1.94, 0.57, -0.71, quayLevel))
    if (dock) parts.push(groundProp(barrels, 2.70, 0.30, 0.45, -0.38, deckY))
  }
  if (dock) parts.push(groundProp(lantern, 2.80, 3.26, 0.24, 0.37, deckY))
  parts.push(groundProp(lantern, 0.58, -2.56, 0.22, -0.42, quayLevel))
  const result = combine(parts)
  for (const warning of warnings) result.warnings.add(warning)
  return result
}

