import type { MaterialUVTransform, SceneMapSampling, SceneMaterial } from '../rendering/scene-mesh'

export type CoveSourceSlot = 'map' | 'normalMap' | 'metallicRoughnessMap' | 'occlusionMap' | 'roughnessMap'

// The URL, byte count and MD5 come from the publisher's asset metadata. All
// other fields describe the material once this particular image is available.
export interface CoveSourceTextureRef {
  url: string
  bytes: number
  md5: string
  upgrade?: true // A verified lower-resolution version of this source is already in the scene.
  uvTransform?: MaterialUVTransform
  clamp?: boolean
  sampling?: SceneMapSampling
  normalScale?: number
  normalYSign?: 1 | -1
  roughness?: number
  metallic?: number
  color?: [number, number, number]
  occlusionStrength?: number
}
export type CoveSourceSlots = Partial<Record<CoveSourceSlot, CoveSourceTextureRef>>
export type CoveSourceManifest = { version: 1; materials: Record<string, CoveSourceSlots> }
declare module '../rendering/scene-mesh' {
  interface SceneMaterial { recoverableMaps?: CoveSourceSlots }
}

export interface CoveSourceState {
  readonly missing: number
  readonly upgrades: number
  restore(materials: ReadonlyMap<string, SceneMaterial>, signal?: AbortSignal): Promise<{
    materials: Map<string, SceneMaterial>; missing: number; upgrades: number; restored: number; warnings: string[]
  }>
}

// Poly Haven publishes an MD5 for each file. Web Crypto does not expose this
// legacy digest; use it for byte identity with the publisher, over HTTPS.
const shifts = [[7, 12, 17, 22], [5, 9, 14, 20], [4, 11, 16, 23], [6, 10, 15, 21]]
const constants = Int32Array.from({ length: 64 }, (_, i) => Math.floor(2 ** 32 * Math.abs(Math.sin(i + 1))))
async function digest(bytes: Uint8Array, signal?: AbortSignal) {
  const tail = new Uint8Array(bytes.length % 64 < 56 ? 64 : 128)
  const whole = bytes.length - bytes.length % 64
  tail.set(bytes.subarray(whole))
  tail[bytes.length % 64] = 128
  const end = new DataView(tail.buffer)
  end.setUint32(tail.length - 8, bytes.length * 8, true)
  end.setUint32(tail.length - 4, Math.floor(bytes.length / 2 ** 29), true)
  const words = new Int32Array([0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476])
  function compress(block: DataView, offset: number) {
    let a = words[0], b = words[1], c = words[2], d = words[3]
    for (let i = 0; i < 64; ++i) {
      let f: number, g: number
      if (i < 16) { f = (b & c) | (~b & d); g = i }
      else if (i < 32) { f = (d & b) | (~d & c); g = (5 * i + 1) % 16 }
      else if (i < 48) { f = b ^ c ^ d; g = (3 * i + 5) % 16 }
      else { f = c ^ (b | ~d); g = (7 * i) % 16 }
      const step = a + f + constants[i] + block.getInt32(offset + 4 * g, true)
      const rotate = shifts[i >> 4][i % 4]
      a = d
      d = c
      c = b
      b = (b + ((step << rotate) | (step >>> (32 - rotate)))) | 0
    }
    words[0] += a
    words[1] += b
    words[2] += c
    words[3] += d
  }
  const input = new DataView(bytes.buffer, bytes.byteOffset, bytes.length)
  // Periodically let the browser render while validating multi-megabyte scans.
  const batchSize = 128 * 1024
  for (let start = 0; start < whole; start += batchSize) {
    if (start) await new Promise(resolve => setTimeout(resolve, 0))
    signal?.throwIfAborted()
    const limit = Math.min(start + batchSize, whole)
    for (let offset = start; offset < limit; offset += 64) compress(input, offset)
  }
  for (let offset = 0; offset < tail.length; offset += 64) compress(end, offset)
  return Array.from(words, word => [0, 8, 16, 24].map(bit =>
    ((word >>> bit) & 255).toString(16).padStart(2, '0')).join('')).join('')
}

function valid(ref: CoveSourceTextureRef) {
  // Prepared manifests come from the deployed scene, never an imported model.
  return /^https:\/\/dl\.polyhaven\.(org|com)\//.test(ref.url)
    && /^([a-f\d]{32})$/i.test(ref.md5) && Number.isSafeInteger(ref.bytes) && ref.bytes > 0
}
function cacheKey(ref: CoveSourceTextureRef) {
  const url = new URL(ref.url)
  url.searchParams.set('caustic-checksum', ref.md5)
  return url.href
}
async function verify(response: Response, ref: CoveSourceTextureRef, signal?: AbortSignal) {
  const fileName = new URL(ref.url).pathname.split('/').pop()!
  if (!response.ok) throw new Error('HTTP ' + response.status)
  const bytes = new Uint8Array(await response.arrayBuffer())
  if (bytes.length !== ref.bytes || await digest(bytes, signal) !== ref.md5.toLowerCase()) throw new Error('File checksum mismatch')
  const type = bytes[0] === 0xff && bytes[1] === 0xd8 ? 'image/jpeg'
    : bytes[0] === 137 && bytes[1] === 80 && bytes[2] === 78 && bytes[3] === 71 ? 'image/png'
    : String.fromCharCode(...bytes.subarray(8,12)) === 'WEBP' ? 'image/webp' : undefined
  if (!type) throw new Error('Unsupported source image')
  return new File([bytes], fileName, { type })
}

const properties = {
  map: ['uvTransform','clamp'], normalMap: ['normalUVTransform','normalClamp'],
  metallicRoughnessMap: ['metallicRoughnessUVTransform','metallicRoughnessClamp'],
  occlusionMap: ['occlusionUVTransform','occlusionClamp'], roughnessMap: ['roughnessUVTransform','roughnessClamp'],
} as const

function applyTexture(material: SceneMaterial, slot: CoveSourceSlot, ref: CoveSourceTextureRef, image: File) {
  material[slot] = image
  const [transform, clamp] = properties[slot]
  material[transform] = ref.uvTransform
  material[clamp] = ref.clamp
  material.sampling = { ...material.sampling, [slot]: ref.sampling }
  if (slot === 'map' && ref.color) material.color = [...ref.color]
  if (slot === 'metallicRoughnessMap') {
    material.metallic = ref.metallic ?? 1
    material.roughness = ref.roughness ?? 1
    delete material.metallicMap
    delete material.roughnessMap
  }
  if (slot === 'roughnessMap') material.roughness = ref.roughness ?? 1
  if (slot === 'normalMap') {
    material.normalScale = ref.normalScale ?? 1
    material.normalYSign = ref.normalYSign ?? 1
  }
  if (slot === 'occlusionMap') material.occlusionStrength = ref.occlusionStrength ?? 1
}

export async function loadCoveSourceTextures(materials: ReadonlyMap<string, SceneMaterial>,
  manifest: CoveSourceManifest, signal?: AbortSignal, localSources?: ReadonlyMap<string, string>) {
  if (manifest.version !== 1) throw new Error('Unsupported source texture manifest.')
  const images = new Map<string, File>(), sources = new Map<string, CoveSourceTextureRef>()
  const required = new Set<string>()
  const localWarnings: string[] = []
  const bindings: { material: string; slot: CoveSourceSlot; ref: CoveSourceTextureRef }[] = []
  for (const [material, slots] of Object.entries(manifest.materials)) {
    if (!materials.has(material)) continue
    for (const slot of Object.keys(properties) as CoveSourceSlot[]) {
      let ref = slots[slot]
      if (!ref || !valid(ref)) continue
      if (ref.md5 !== ref.md5.toLowerCase()) ref = { ...ref, md5: ref.md5.toLowerCase() }
      bindings.push({ material, slot, ref })
      sources.set(ref.md5, ref)
      if (!ref.upgrade) required.add(ref.md5)
    }
  }
  function apply(base: ReadonlyMap<string,SceneMaterial>) {
    const updated = new Map(base)
    for (const { material, slot, ref } of bindings) {
      const image = images.get(ref.md5), original = base.get(material)
      if (!image || !original) continue
      if (updated.get(material) === original) updated.set(material, { ...original })
      applyTexture(updated.get(material)!, slot, ref, image)
    }
    return updated
  }
  let cache: Cache | undefined
  try { cache = await globalThis.caches?.open('caustic-canvas-source-textures-v1') } catch { /* Private browsing */ }
  for (const ref of sources.values()) {
    signal?.throwIfAborted()
    const localUrl = localSources?.get(ref.md5)
    if (localUrl) {
      try {
        const response = await fetch(localUrl, { signal })
        images.set(ref.md5, await verify(response, ref, signal))
        continue
      } catch (error) {
        signal?.throwIfAborted()
        localWarnings.push('Could not read local ' + new URL(ref.url).pathname.split('/').pop() + ': '
          + (error instanceof Error ? error.message : String(error)))
      }
    }
    try {
      const cached = await cache?.match(cacheKey(ref))
      if (cached) images.set(ref.md5, await verify(cached, ref, signal))
    } catch { signal?.throwIfAborted() /* Damaged cache entries are retried on explicit restore. */ }
  }
  signal?.throwIfAborted()
  let active: Promise<string[]> | undefined
  async function download(signal?: AbortSignal) {
    const warnings: string[] = []
    const remaining = Array.from(sources.values()).filter(ref => !images.has(ref.md5))
    const timeout = AbortSignal.timeout(60_000)
    const requestSignal = signal ? AbortSignal.any([signal, timeout]) : timeout
    let persistFailed = false
    async function worker() {
      while (remaining.length) {
        signal?.throwIfAborted()
        if (timeout.aborted) { warnings.push('Source download timed out.'); break }
        const ref = remaining.shift()!
        try {
          const fileSignal = AbortSignal.any([requestSignal, AbortSignal.timeout(20_000)])
          const response = await fetch(ref.url, { signal: fileSignal, credentials: 'omit' })
          const file = await verify(response, ref, fileSignal)
          signal?.throwIfAborted()
          images.set(ref.md5, file)
          if (cache) {
            try { await cache.put(cacheKey(ref), new Response(file)) } catch { persistFailed = true }
          } else { persistFailed = true }
        } catch (error) {
          signal?.throwIfAborted()
          const file = new URL(ref.url).pathname.split('/').pop()
          warnings.push(file + ': ' + (error instanceof Error ? error.message : String(error)))
        }
      }
    }
    await Promise.all(Array.from({ length: Math.min(3,remaining.length) }, worker))
    if (persistFailed) warnings.push('Restored sources are available for this session. Browser storage is unavailable.')
    return [...new Set(warnings)]
  }
  const state: CoveSourceState = {
    get missing() {
      let count = 0
      for (const id of required) if (!images.has(id)) count++
      return count
    },
    get upgrades() { return sources.size - images.size - state.missing },
    async restore(base, signal) {
      signal?.throwIfAborted()
      const before = images.size
      active ??= download(signal).finally(() => { active = undefined })
      const warnings = await active
      signal?.throwIfAborted()
      return { materials: apply(base), missing: state.missing, upgrades: state.upgrades,
        restored: images.size - before, warnings }
    },
  }
  return { materials: apply(materials), sourceTextures: state, localWarnings }
}
