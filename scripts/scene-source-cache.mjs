import { createHash, randomUUID } from 'node:crypto'
import { createWriteStream, openAsBlob } from 'node:fs'
import { mkdir, readFile, rename, rm, stat, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { pipeline } from 'node:stream/promises'

// Cache the public input files used by the preparation command. The browser
// never sees this cache. A changed layout can be baked from the same sources.
function publicSource(url) {
  return url.protocol === 'https:' && !url.username && !url.password &&
    (url.hostname.endsWith('.polyhaven.org') || url.hostname.endsWith('.polyhaven.com'))
}

export function sceneSourceCache(directory, network = fetch) {
  const counts = { reused: 0, downloaded: 0 }
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
    init?.signal?.throwIfAborted()
    const key = createHash('sha256').update(url.href).digest('hex')
    const base = join(directory, key)
    try {
      const meta = JSON.parse(await readFile(base + '.json', 'utf8'))
      if (meta.url === url.href) {
        const response = await diskResponse(base + '.body', meta)
        if (response) { counts.reused++; return response }
      }
    } catch (error) {
      if (error.code !== 'ENOENT' && !(error instanceof SyntaxError)) throw error
    }
    const response = await network(input, init)
    if (response.status !== 200 || !response.body) return response
    await mkdir(directory, { recursive: true })
    const temp = base + '.' + randomUUID()
    try {
      await pipeline(response.body, createWriteStream(temp + '.body'))
      const { size } = await stat(temp + '.body')
      if (!size) throw new Error('Empty scene download: ' + url.href)
      const meta = { url: url.href, finalUrl: response.url || url.href, type: response.headers.get('Content-Type'), size }
      await writeFile(temp + '.json', JSON.stringify(meta))
      await rename(temp + '.body', base + '.body')
      await rename(temp + '.json', base + '.json')
      counts.downloaded++
      return await diskResponse(base + '.body', meta)
    } finally {
      await Promise.all([rm(temp + '.body', { force: true }), rm(temp + '.json', { force: true })])
    }
  }
  return { fetch: fetchSource, counts }
}
