import { createHash, randomUUID } from 'node:crypto'
import { createReadStream, openAsBlob } from 'node:fs'
import { mkdir, open, readFile, rename, rm, stat, writeFile } from 'node:fs/promises'
import { extname, join } from 'node:path'
import { pipeline } from 'node:stream/promises'
import { localSceneSources } from './local-scene-sources.mjs'
import coveCatalog from '../scene/smugglers-cove/source-catalog.json' with { type: 'json' }

// Cache the public input files used by the preparation command. The browser
// never sees this cache. A changed layout can be baked from the same sources.
function publicSource(url) {
  return url.protocol === 'https:' && !url.username && !url.password &&
    (url.hostname.endsWith('.polyhaven.org') || url.hostname.endsWith('.polyhaven.com'))
}

export function originalImageName(ref) {
  if (!/^[\da-f]{32}$/i.test(ref.md5)) return undefined
  const ext = extname(new URL(ref.url).pathname).toLowerCase()
  return ['.jpg', '.jpeg', '.png', '.webp'].includes(ext) ? ref.md5.toLowerCase() + ext : undefined
}

export function sceneSourceCache(directory, network = fetch, {
  sourcePath = process.env.CAUSTIC_SCENE_SOURCE_DIR, originalsDirectory, catalog = coveCatalog,
} = {}) {
  const counts = { reused: 0, downloaded: 0 }
  const originals = sourcePath && localSceneSources(sourcePath)
  const expected = new Map()
  function register(value) {
    if (value && typeof value === 'object') {
      if (typeof value.url === 'string' && /^[\da-f]{32}$/i.test(value.md5) && value.size > 0) expected.set(value.url, value)
      for (const child of Object.values(value)) register(child)
    }
  }
  register(catalog)
  function cacheBase(url) { return join(directory, createHash('sha256').update(url.href).digest('hex')) }
  function installedPath(signature) {
    const name = signature && originalImageName(signature)
    return name && originalsDirectory && join(originalsDirectory, name)
  }
  async function sameSize(path, size) {
    if (!path) return false
    try { return (await stat(path)).size === size }
    catch (error) { if (error.code === 'ENOENT') return false; throw error }
  }
  // This inexpensive lookup picks preferred resolutions. fetchSource verifies
  // the actual MD5 of the selected candidate before any bytes can be used.
  async function hasLocal(input) {
    const url = new URL(input?.url ?? input)
    const signature = expected.get(url.href)
    if (!signature) return false
    return await sameSize(cacheBase(url) + '.body', signature.size)
      || await sameSize(installedPath(signature), signature.size)
      || !!(originals && await originals.has(url, signature))
  }
  async function matchesFile(path, signature) {
    if (!signature) return true
    if ((await stat(path)).size !== signature.size) return false
    const md5 = createHash('md5')
    for await (const bytes of createReadStream(path)) md5.update(bytes)
    return md5.digest('hex') === signature.md5.toLowerCase()
  }
  async function diskResponse(path, meta) {
    const blob = await openAsBlob(path, { type: meta.type || 'application/octet-stream' })
    if (blob.size !== meta.size || !blob.size) return undefined
    const response = new Response(blob, { headers: { 'Content-Type': blob.type, 'Content-Length': String(blob.size) } })
    Object.defineProperty(response, 'url', { value: meta.finalUrl || meta.url })
    return response
  }
  async function fetchSource(input, init) {
    const req = input instanceof Request ? input : undefined
    const url = new URL(req ? req.url : input)
    const headers = new Headers(init?.headers ?? req?.headers)
    if (!publicSource(url) || (init?.method ?? req?.method ?? 'GET') !== 'GET' ||
        headers.has('Range') || headers.has('Authorization') || headers.has('Cookie')) return network(input, init)
    const signal = init?.signal ?? req?.signal
    signal?.throwIfAborted()
    const base = cacheBase(url)
    const apiId = url.hostname === 'api.polyhaven.com' && /^\/files\/([\w-]+)$/.exec(url.pathname)?.[1]
    const signature = expected.get(url.href)
    const pinned = apiId && Object.hasOwn(catalog, apiId) && catalog[apiId]
    function pinnedManifest() {
      const response = Response.json(pinned)
      Object.defineProperty(response, 'url', { value: url.href })
      counts.reused++
      return response
    }
    try {
      const meta = JSON.parse(await readFile(base + '.json', 'utf8'))
      if (meta.url === url.href) {
        const response = await diskResponse(base + '.body', meta)
        if (response && await matchesFile(base + '.body', signature)) {
          if (apiId) register(JSON.parse(await readFile(base + '.body', 'utf8')))
          counts.reused++
          return response
        }
      }
    } catch (error) {
      if (error.code !== 'ENOENT' && !(error instanceof SyntaxError)) throw error
    }
    // Pinned official file lists allow verification on a fresh offline checkout.
    if (originals && pinned) return pinnedManifest()
    const installed = installedPath(signature)
    const local = installed && await sameSize(installed, signature.size) && await matchesFile(installed, signature)
      ? createReadStream(installed) : originals && signature && await originals.read(url, signature)
    const mime = { '.jpg': 'image/jpeg', '.png': 'image/png', '.gltf': 'model/gltf+json' }[extname(url.pathname)] ?? 'application/octet-stream'
    let response
    try { response = local ? new Response(local, { headers: { 'Content-Type': mime } }) : await network(input, init) }
    catch (error) {
      signal?.throwIfAborted()
      if (pinned) return pinnedManifest()
      throw error
    }
    if (!response.ok && pinned) return pinnedManifest()
    if (response.status !== 200 || !response.body) return response
    await mkdir(directory, { recursive: true })
    const temp = base + '.' + randomUUID()
    try {
      const file = await open(temp + '.body', 'wx')
      try { await pipeline(response.body, file.createWriteStream()) }
      finally { await file.close() }
      const { size } = await stat(temp + '.body')
      if (!size) throw new Error('Empty scene download: ' + url.href)
      if (!await matchesFile(temp + '.body', signature)) throw new Error('Original asset checksum mismatch: ' + url.href)
      if (apiId) register(JSON.parse(await readFile(temp + '.body', 'utf8')))
      const meta = { url: url.href, finalUrl: response.url || url.href, type: response.headers.get('Content-Type'), size }
      await writeFile(temp + '.json', JSON.stringify(meta))
      await rename(temp + '.body', base + '.body')
      await rename(temp + '.json', base + '.json')
      if (local) counts.reused++
      else counts.downloaded++
      return await diskResponse(base + '.body', meta)
    } finally {
      await Promise.all([rm(temp + '.body', { force: true }), rm(temp + '.json', { force: true })])
    }
  }
  return { fetch: fetchSource, hasLocal, counts }
}
