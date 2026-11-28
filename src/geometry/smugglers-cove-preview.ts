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
import { makeCoveLiftNet } from './cove-lift-net'
import { coveScanBudgets } from './cove-scan-budgets'
import { supplyCoveRockColor } from './cove-rock-material'
import { addHarborDetail, addScanDetail, coveDetailFiles, type ReadCoveDetail } from './cove-material-detail'
import { sceneVertexStride, type SceneMaterial, type SceneDraw, type SceneMesh } from '../rendering/scene-mesh'
import { readPreviewGltf, type PreviewGltfDocument, type PreviewGltfOptions, type PreviewNodeBounds } from './preview-gltf'
import type { CoveSourceSlots, CoveSourceTextureRef } from './cove-source-textures'

type SourceFile = { url: string; size?: number; md5?: string }
type Download = SourceFile & { include?: Record<string, SourceFile> }
type FileList = Record<string, Record<string, Record<string, Download>>>
type Part = Awaited<ReturnType<typeof readPreviewGltf>>
type Variant = { size: number; file: Download }
type CoveOptions = { hasLocal?: (file: SourceFile) => Promise<boolean>; readDetailFile?: ReadCoveDetail }

async function fetchOk(url: string, signal?: AbortSignal, headers?: HeadersInit) {
  const response = await fetch(url, { signal, headers })
  if (!response.ok) throw new Error(`Could not load ${url} (HTTP ${response.status}).`)
  return response
}

async function fileList(id: string, signal?: AbortSignal): Promise<FileList> {
  const url = 'https://api.polyhaven.com/files/' + id
  // Browsers identify the site with their Referer; Node has no document or Referer.
  const headers = typeof document === 'undefined' ? { 'User-Agent': 'CausticCanvas scene preview' } : undefined
  return (await fetchOk(url, signal, headers)).json()
}

function versions(files: FileList, keys: string[], formats: string[]) {
  const choices: Variant[] = []
  for (const [key, sizes] of Object.entries(files)) {
    if (!keys.includes(key.toLowerCase())) continue
    for (const [size, entries] of Object.entries(sizes)) {
      if (!/^\d+k$/i.test(size)) continue
      for (const format of formats) if (entries[format]?.url) choices.push({ size: parseInt(size), file: entries[format] })
    }
  }
  return choices.sort((a, b) => b.size - a.size)
}

// Honor full-resolution local files, even if only one map was supplied. Prefer
// 2K over 1K for a network preview without pulling a multi-gigabyte 8K scene.
async function candidates(variants: Variant[], options: CoveOptions) {
  const limit = Math.max(2, variants.at(-1)?.size ?? 0)
  const available = await Promise.all(variants.map(v => v.size <= limit || (options.hasLocal?.(v.file) ?? false)))
  return variants.filter((_, i) => available[i]).map(v => v.file)
}

async function fetchFirst(choices: Download[], signal?: AbortSignal) {
  let lastError: unknown = new Error('No source file is listed.')
  const tried = new Set<string>()
  for (const choice of choices) {
    if (tried.has(choice.url)) continue
    tried.add(choice.url)
    try { return { response: await fetchOk(choice.url, signal), choice } }
    catch (error) { signal?.throwIfAborted(); lastError = error }
  }
  throw lastError
}

// These records travel with a baked fallback. Only an identified original can
// replace it later; any slot without a published signature stays local.
function rememberSource(material: SceneMaterial, slot: keyof CoveSourceSlots, file: SourceFile | undefined,
  settings: Partial<CoveSourceTextureRef>) {
  if (!file?.size || !file.md5 || !/^[a-f\d]{32}$/i.test(file.md5)) return
  material.recoverableMaps ??= {}
  material.recoverableMaps[slot] = { ...settings, url: file.url, bytes: file.size, md5: file.md5 }
}

type LodNode = { name?: string; mesh?: number; children?: number[]; extensions?: { MSFT_lod?: { ids: number[] } } }

// An MSFT_lod parent is the full-detail mesh. Some Blender kits instead place
// all numbered LODs and the unsuffixed original beside each other in the scene.
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
    const alternatives = new Set(indices.flatMap(i => doc.nodes[i]?.extensions?.MSFT_lod?.ids ?? []))
    const selected = [...new Set(indices)].filter(i => !alternatives.has(i))
    const choices = selected.map(lod), levels = new Map<string, number>()
    const originals = new Set<string>()
    for (const [slot, index] of selected.entries()) {
      const item = choices[slot], node = doc.nodes[index]
      if (item) levels.set(item.group, Math.min(levels.get(item.group) ?? Infinity, item.level))
      else if (node?.mesh !== undefined) {
        for (const name of [node.name, doc.meshes[node.mesh].name]) if (name) originals.add(name.toLowerCase())
      }
    }
    return selected.filter((_, slot) => {
      const item = choices[slot]
      return !item || (!originals.has(item.group) && item.level === levels.get(item.group))
    })
  }
  for (const scene of doc.scenes) if (scene.nodes) scene.nodes = filter(scene.nodes)
  for (const node of doc.nodes) if (node.children) node.children = filter(node.children)
}

async function loadModel(id: string, maxTriangles: number, options: CoveOptions, detail: ReturnType<typeof coveDetailFiles>,
  signal?: AbortSignal, crop?: PreviewGltfOptions['crop']) {
  const files = await fileList(id, signal), models = versions(files, ['gltf'], ['gltf'])
  const { choice: file, response } = await fetchFirst(await candidates(models, options), signal)
  const documentUrl = response.url || file.url
  const document = await response.json() as PreviewGltfDocument
  selectLods(document)
  if (document.extensionsRequired?.includes('MSFT_lod')) {
    document.extensionsRequired = document.extensionsRequired.filter(name => name !== 'MSFT_lod')
  }
  const keyFor = (path: string) => path.replaceAll('\\', '/').replace(/_\d+k(?=\.[^./]+$)/i, '')
  const images = new Map<string, Variant[]>(), originals = new Map<string, Download[]>(), loaded = new Map<string, Download>()
  for (const model of models) for (const [path, entry] of Object.entries(model.file.include ?? {})) {
    if (!/\.(?:jpe?g|png|webp|avif)$/i.test(path)) continue
    const key = keyFor(path)
    if (!images.has(key)) images.set(key, [])
    images.get(key)!.push({ size: model.size, file: entry })
  }
  let geometryNote: string | undefined
  const asset = await readPreviewGltf({ document, base: new URL(documentUrl) }, signal, {
    async readUri(uri) {
      // A .bin must match the accessor layout of the selected document.
      const exact = file.include?.[uri] ?? { url: new URL(uri.replaceAll('\\', '/'), documentUrl).href }
      const key = keyFor(uri)
      const alternatives = images.get(key)
      if (!alternatives) return (await fetchFirst([exact], signal)).response.blob()
      const choices = await candidates(alternatives, options)
      originals.set(key, choices)
      const result = await fetchFirst([...choices,exact], signal)
      loaded.set(key, result.choice)
      return result.response.blob()
    },
    crop, maxTriangles, simplifyToBudget: true, pbr: true,
    onSimplified(source, result) {
      geometryNote = `Source geometry ${id}: ${source} source triangles reduced to ${result} per copy for the interactive scene.`
    },
  })
  const scanRoughness: Record<string, number> = {
    coast_land_rocks_02: 0.77, modular_wooden_pier: 0.74, treasure_chest: 0.56,
    wooden_crate_01: 0.8, wooden_barrels_01: 0.68, wooden_lantern_01: 0.48,
  }
  for (const [key, material] of asset.materials) {
    const original = document.materials?.[Number(key)], source = original?.pbrMetallicRoughness
    const referencedImage = (info?: { index: number }) => {
      if (!info) return
      const texture = document.textures?.[info.index]
      const image = document.images?.[texture?.extensions?.EXT_texture_avif?.source ??
        texture?.extensions?.EXT_texture_webp?.source ?? texture?.source ?? -1]
      return image?.uri ? keyFor(image.uri) : undefined
    }
    const preferred = (info?: {index: number}) => originals.get(referencedImage(info) ?? '')?.[0]
    const base = referencedImage(source?.baseColorTexture), best = preferred(source?.baseColorTexture)
    if (base && best && (!material.map || loaded.get(base)?.url !== best.url)) rememberSource(material, 'map', best, {
      ...(material.map ? { upgrade: true } : {}), color: material.color, uvTransform: material.uvTransform,
      clamp: material.clamp, sampling: material.sampling?.map })
    if (!material.normalMap) rememberSource(material, 'normalMap', preferred(original?.normalTexture), {
      normalScale: material.normalScale, normalYSign: material.normalYSign, uvTransform: material.normalUVTransform,
      clamp: material.normalClamp, sampling: material.sampling?.normalMap })
    if (!material.occlusionMap) rememberSource(material, 'occlusionMap', preferred(original?.occlusionTexture), {
      occlusionStrength: material.occlusionStrength, uvTransform: material.occlusionUVTransform,
      clamp: material.occlusionClamp, sampling: material.sampling?.occlusionMap })
    // glTF's multiplier is 1 when a map exists. Use an estimated surface only
    // for a missing, referenced map; factors on a complete source pass through.
    if (source?.metallicRoughnessTexture && !material.metallicRoughnessMap) {
      rememberSource(material, 'metallicRoughnessMap', preferred(source.metallicRoughnessTexture), {
        roughness: material.roughness, metallic: material.metallic, uvTransform: material.metallicRoughnessUVTransform,
        clamp: material.metallicRoughnessClamp, sampling: material.sampling?.metallicRoughnessMap })
      material.roughness = source.roughnessFactor ?? scanRoughness[id] ?? 0.76
      material.metallic = source.metallicFactor ?? 0
    }
    await addScanDetail(material, original?.name, detail)
  }
  for (const warning of [...asset.warnings]) {
    // The reef's UV-matched local fallback specifically checks this source failure.
    if (id === 'coast_land_rocks_02' && /^Could not load .* base color /.test(warning)) continue
    asset.warnings.delete(warning)
    const missing = /^Could not load (.*?) (base color|metallic-roughness|normal|occlusion|emissive) ([^(]+?)(?: \([\s\S]*\))?\. Using the material (color|factor)\.$/.exec(warning)
    asset.warnings.add('Source material ' + id + ': ' + (missing
      ? `${missing[1]} / ${missing[2]} (${missing[3]}) unavailable.` : warning))
  }
  if (geometryNote) asset.warnings.add(geometryNote)
  return asset
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

async function loadShore(options: CoveOptions, signal?: AbortSignal) {
  const source = await fileList('coast_sand_01', signal)
  const channels = (keys: string[]) => candidates(versions(source, keys, ['jpg', 'png']), options)
  async function image(choices: Download[]) {
    const { response, choice } = await fetchFirst(choices, signal)
    const blob = await response.blob()
    return { file: new File([blob], new URL(choice.url).pathname.split('/').pop()!, { type: blob.type }), choice }
  }
  const colors = await channels(['diffuse', 'diff', 'color', 'albedo'])
  const { file: sand, choice: albedo } = await image(colors)
  const shore = makeCoveShore(sand)
  const surfaces = [...shore.materials].filter(([, m]) => m.map === sand)
  if (albedo.url !== colors[0].url) {
    for (const [, material] of surfaces) rememberSource(material, 'map', colors[0], { upgrade: true, color: material.color })
  }
  const packed = await channels(['arm'])
  // Some source packs contain only ARM, others contain the standalone maps.
  // Its R and G channels can supply authentic AO and roughness, respectively.
  let packedImage: Promise<File> | undefined
  const slots = [
    { key: 'normalMap', label: 'normal', inputs: ['nor_gl'] },
    { key: 'roughnessMap', label: 'roughness', inputs: ['rough', 'roughness'] },
    { key: 'occlusionMap', label: 'occlusion', inputs: ['ao', 'occlusion'] },
  ] as const
  await Promise.all(slots.map(async ({ key, label, inputs }) => {
    const choices = await channels([...inputs])
    let map: File | undefined, fromPacked = false
    if (choices.length) {
      try { map = (await image(choices)).file } catch { signal?.throwIfAborted() }
    }
    if (!map && key !== 'normalMap' && packed.length) {
      try { map = await (packedImage ??= image(packed).then(result => result.file)); fromPacked = true }
      catch { signal?.throwIfAborted() }
    }
    const reference = choices[0] ?? (key !== 'normalMap' ? packed[0] : undefined)
    if (!reference) return
    if (!map) shore.warnings.add(`Source material coast_sand_01: original ${label} unavailable.`)
    for (const [name, material] of surfaces) {
      const slot = key === 'roughnessMap' && (fromPacked || !choices.length) ? 'metallicRoughnessMap' : key
      const settings = key === 'roughnessMap' ? { roughness: name === 'dry' ? 1 : material.roughness, metallic: 0 }
        : key === 'normalMap' ? { normalScale: 1, normalYSign: 1 as const } : { occlusionStrength: 1 }
      if (map) Object.assign(material, { [slot]: map }, settings)
      else rememberSource(material, slot, reference, settings)
    }
  }))
  return shore
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
  const v = part.mesh.vertices.slice(), tangents = part.mesh.tangents?.slice()
  for (let i = 0; i < v.length; i += sceneVertexStride) {
    const dx = (v[i] - cx) * scale, dy = (v[i + 1] - cy) * scale, dz = (v[i + 2] - cz) * scale
    const px = x + c * dx + s * dz, pz = z - s * dx + c * dz
    const vertical = (dy < 0 ? pose.below : pose.above) ?? 1
    let py = y + dy * vertical
    let nx = c * v[i + 3] + s * v[i + 5], ny = v[i + 4] / vertical, nz = -s * v[i + 3] + c * v[i + 5]
    let warpX = 0, warpZ = 0, warpY = 1
    if (rock) {
      // A scan includes a collar of captured sand. Tuck its perimeter into the
      // same sloping bed as the beach so the outcrop grows out of the bank.
      const radius = Math.hypot(2 * dx / rock.width, 2 * dz / rock.depth)
      const sink = Math.max(0, radius - 0.43)
      const span = Math.max(rock.width, rock.depth)
      py += shoreHeight(px, pz) - span * (0.035 + 0.19 * sink * sink)
      const radialSlope = radius > 0 ? 1.52 * span * sink / radius : 0
      const slopeX = radialSlope * dx / (rock.width * rock.width), slopeZ = radialSlope * dz / (rock.depth * rock.depth)
      warpX = (shoreHeight(px + 0.02, pz) - shoreHeight(px - 0.02, pz)) / 0.04 - c * slopeX - s * slopeZ
      warpZ = (shoreHeight(px, pz + 0.02) - shoreHeight(px, pz - 0.02)) / 0.04 + s * slopeX - c * slopeZ
      nx -= ny * warpX
      nz -= ny * warpZ
      if (py < -0.39) {
        const depth = Math.tanh((-0.39 - py) / 0.13)
        py = -0.39 - 0.13 * depth
        warpY = Math.max(0.001, 1 - depth * depth)
        ny /= warpY
      }
    }
    const length = Math.hypot(nx, ny, nz) || 1
    if (tangents) {
      const at = 4 * i / sceneVertexStride
      const tx = c * tangents[at] + s * tangents[at+2], tz = -s * tangents[at] + c * tangents[at+2]
      const ty = warpY * (tangents[at+1] * vertical + warpX * tx + warpZ * tz)
      const length = Math.hypot(tx, ty, tz) || 1
      tangents[at] = tx / length; tangents[at+1] = ty / length; tangents[at+2] = tz / length
    }
    v[i] = px; v[i+1] = py; v[i+2] = pz
    v[i+3] = nx / length; v[i+4] = ny / length; v[i+5] = nz / length
  }
  return { ...part, mesh: { ...part.mesh, vertices: v, tangents } }
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
  const uvs1 = parts.some(p => p.mesh.uvs1) ? new Float32Array(uvs.length) : undefined
  const tangents = parts.some(p => p.mesh.tangents) ? new Float32Array(2 * uvs.length) : undefined
  const materials = new Map<string, SceneMaterial>(), warnings = new Set<string>()
  const catalogs = new Map<Map<string, SceneMaterial>, string>()
  let firstVertex = 0, firstIndex = 0, nextId = 2
  for (const part of parts) {
    const mesh = part.mesh, ids = new Map<number, number>()
    vertices.set(mesh.vertices, firstVertex * sceneVertexStride)
    if (mesh.uvs) uvs.set(mesh.uvs, 2 * firstVertex)
    if (mesh.uvs1) uvs1!.set(mesh.uvs1, 2 * firstVertex)
    if (mesh.tangents) tangents!.set(mesh.tangents, 4 * firstVertex)
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
  return { mesh: { vertices, indices, uvs, uvs1, tangents, draws }, materials, warnings }
}

export async function readCovePreview(signal?: AbortSignal, options: CoveOptions = {}): Promise<Part> {
  const warnings: string[] = []
  const detail = coveDetailFiles(signal, options.readDetailFile)
  async function dressing(id: string, budget: number, crop?: PreviewGltfOptions['crop']) {
    try { return await loadModel(id, budget, options, detail, signal, crop) }
    catch (error) {
      if (signal?.aborted) throw error
      warnings.push(id + ' could not load: ' + (error instanceof Error ? error.message : String(error)))
    }
  }
  async function cargo(id: string, kind: Parameters<typeof makeLocalCargo>[0], budget: number, crop?: PreviewGltfOptions['crop']) {
    try {
      return await loadModel(id, budget, options, detail, signal, crop)
    } catch (error) {
      if (signal?.aborted) throw error
    }
    warnings.push(`Using the project-modeled ${kind} in place of ${id}.`)
    return makeLocalCargo(kind)
  }
  // Limit simultaneous high-resolution sources; each downloaded model can be
  // placed more than once without fetching or decoding another copy.
  const [shore, rocks, dock] = await Promise.all([
    loadShore(options, signal), loadModel('coast_land_rocks_02', coveScanBudgets.coast_land_rocks_02, options, detail, signal).then(supplyCoveRockColor),
    dressing('modular_wooden_pier', 42_000, dockBounds),
  ])
  const [chest, crate, barrels, lantern] = await Promise.all([
    loadModel('treasure_chest', coveScanBudgets.treasure_chest, options, detail, signal), cargo('wooden_crate_01', 'crate', 14_000),
    cargo('wooden_barrels_01', 'barrel', 16_000, barrelBounds), cargo('wooden_lantern_01', 'lantern', 8_000),
  ])
  const parts: Part[] = [shore,
    outcrop(rocks, -3.82, 0.22, 4.20, -0.32), outcrop(rocks, -6.46, -0.39, 2.86, 1.56),
    outcrop(rocks, 4.92, 1.30, 1.96, -1.22)]

  const approach = pierApproach, deckY = shoreHeight(approach.x, approach.z) + 0.12
  const quay = makeQuayside(deckY)
  parts.push(quay, makeCoveCargoGear(deckY), makeDryingRack(), makeCovePotMarkers(),
    makeCoveFishMarket(), makeCoveLiftNet(), makeCoveBreakwater(),
    place(makeCoveLugger(), { x: -2.35, y: 0, z: 5.17, cx: 0, cy: 0, cz: 0, scale: 1, angle: -0.16 }),
    makeCoveTidePools(quay.materials.get('weathered limestone')!, shore.materials.get('water')!),
    place(makeCoveNetLoft(), { x: 3.71, y: quayLevel, z: -2.93,
    cx: 0, cy: 0, cz: 0, scale: 1, angle: 0 }))
  const deck = dock && inspectDeck(dock)
  const skiff = makeCoveSkiff(deck?.planks)
  const boat = place(skiff, { x: 0.90, y: 0, z: 3.20,
    cx: 0, cy: 0, cz: 0, scale: 1, angle: -0.06 })
  if (dock && deck) {
    const heading = 0.06
    const segment = 2.04, scale = segment / Math.max(deck.width, deck.depth)
    const angle = heading + (deck.width > deck.depth ? Math.PI / 2 : 0)
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
    const [stern,bow] = skiff.mooringVertices.map((vertex): [number,number,number] => {
      const i = vertex*sceneVertexStride, v = boat.mesh.vertices
      return [v[i],v[i+1],v[i+2]]
    })
    parts.push(makeMoorings([{ post: cleat(1.65), boat: stern }, { post: cleat(3.80), boat: bow }]))
  }
  parts.push(boat)
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
  await addHarborDetail(result.materials, detail)
  for (const warning of warnings) result.warnings.add(warning)
  return result
}

