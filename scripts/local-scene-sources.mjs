import { createReadStream } from 'node:fs'
import { open, readdir, stat } from 'node:fs/promises'
import { basename, extname, join } from 'node:path'
import { createInflateRaw } from 'node:zlib'
import { createHash } from 'node:crypto'

async function readAt(file, size, offset) {
  const bytes = Buffer.alloc(size)
  if ((await file.read(bytes, 0, size, offset)).bytesRead !== size) throw new Error('Truncated source archive.')
  return bytes
}

// Read individual members of ordinary ZIP downloads without extracting paths
// from the archive. Extracted directories also work for ZIP64 or other formats.
async function indexZip(path, size, add) {
  const file = await open(path, 'r')
  try {
    const tailSize = Math.min(size, 22 + 65535), tail = await readAt(file, tailSize, size - tailSize)
    const end = tail.lastIndexOf(Buffer.from([0x50, 0x4b, 5, 6]))
    if (end < 0 || end + 22 > tail.length || tail.readUInt16LE(end + 4) || tail.readUInt16LE(end + 6)) {
      throw new Error('Unsupported or truncated ZIP.')
    }
    const entries = tail.readUInt16LE(end + 10)
    let cursor = tail.readUInt32LE(end + 16)
    if (entries === 65535 || cursor === 0xffffffff) throw new Error('ZIP64 archive.')
    for (let i = 0; i < entries; i++) {
      const entry = await readAt(file, 46, cursor)
      if (entry.readUInt32LE(0) !== 0x02014b50) throw new Error('Invalid ZIP central directory.')
      const nameLength = entry.readUInt16LE(28), flags = entry.readUInt16LE(8), method = entry.readUInt16LE(10)
      const name = (await readAt(file, nameLength, cursor + 46)).toString('utf8')
      const packed = entry.readUInt32LE(20), original = entry.readUInt32LE(24), header = entry.readUInt32LE(42)
      cursor += 46 + nameLength + entry.readUInt16LE(30) + entry.readUInt16LE(32)
      if (!packed || (flags & 1) || (method !== 0 && method !== 8)) continue
      const local = await readAt(file, 30, header)
      if (local.readUInt32LE(0) !== 0x04034b50) throw new Error('Invalid ZIP member.')
      const start = header + 30 + local.readUInt16LE(26) + local.readUInt16LE(28)
      if (start + packed > size) throw new Error('Invalid ZIP member size.')
      add(name, original, () => {
        const stream = createReadStream(path, { start, end: start + packed - 1 })
        if (!method) return stream
        const inflate = createInflateRaw()
        stream.on('error', error => inflate.destroy(error))
        inflate.on('close', () => stream.destroy())
        return stream.pipe(inflate)
      })
    }
  } catch (error) {
    throw new Error(`Cannot read ${basename(path)}. Extract it to a folder and use that folder as the source. ${error.message}`)
  } finally { await file.close() }
}

export function localSceneSources(path) {
  let index
  async function scan() {
    const byName = new Map()
    function add(name, size, stream) {
      const key = basename(name.replaceAll('\\', '/')).toLowerCase()
      if (!byName.has(key)) byName.set(key, [])
      byName.get(key).push({ size, stream })
    }
    async function visit(location) {
      const info = await stat(location)
      if (info.isDirectory()) {
        for (const entry of await readdir(location, { withFileTypes: true })) {
          if (entry.isFile() || entry.isDirectory()) await visit(join(location, entry.name))
        }
      } else if (extname(location).toLowerCase() === '.zip') {
        await indexZip(location, info.size, add)
      } else if (info.isFile()) add(location, info.size, () => createReadStream(location))
    }
    await visit(path)
    return byName
  }
  async function candidates(url, expected) {
    index ??= scan()
    const name = basename(decodeURIComponent(url.pathname)).toLowerCase()
    return ((await index).get(name) ?? []).filter(source => source.size === expected.size)
  }
  async function read(url, expected) {
    for (const source of await candidates(url, expected)) {
      const hash = createHash('md5')
      let size = 0
      for await (const chunk of source.stream()) {
        size += chunk.length
        if (size > expected.size) break
        hash.update(chunk)
      }
      if (size === expected.size && hash.digest('hex') === expected.md5.toLowerCase()) return source.stream()
    }
  }
  return { read, has: async (url, expected) => (await candidates(url, expected)).length > 0 }
}
