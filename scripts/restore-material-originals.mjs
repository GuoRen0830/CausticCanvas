import { createHash } from 'node:crypto'
import { createReadStream, createWriteStream } from 'node:fs'
import { mkdir, mkdtemp, open, readFile, rename, rm, stat } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'
import { pipeline } from 'node:stream/promises'
import { fileURLToPath } from 'node:url'
import { originalImageName, sceneSourceCache } from './scene-source-cache.mjs'

function manifestImages(manifest) {
  if (manifest.version !== 1 || !manifest.materials || typeof manifest.materials !== 'object') {
    throw new Error('Unsupported harbor source texture manifest.')
  }
  const images = new Map()
  for (const slots of Object.values(manifest.materials)) {
    for (const ref of Object.values(slots)) {
      const url = new URL(ref.url)
      const filename = originalImageName(ref)
      if (!filename || !Number.isSafeInteger(ref.bytes) || ref.bytes <= 0 || url.protocol !== 'https:' ||
          !['dl.polyhaven.org', 'dl.polyhaven.com'].includes(url.hostname) || url.username || url.password) {
        throw new Error('Invalid original source reference: ' + ref.url)
      }
      const md5 = ref.md5.toLowerCase()
      const previous = images.get(md5)
      if (previous && (previous.bytes !== ref.bytes || previous.filename !== filename)) {
        throw new Error('Conflicting original source references: ' + ref.url)
      }
      images.set(md5, { url: ref.url, bytes: ref.bytes, md5, filename, upgrade: ref.upgrade === true && (!previous || previous.upgrade) })
    }
  }
  return [...images.values()]
}

async function verified(path, ref) {
  try {
    if ((await stat(path)).size !== ref.bytes) return false
    const digest = createHash('md5')
    for await (const bytes of createReadStream(path)) digest.update(bytes)
    return digest.digest('hex') === ref.md5
  } catch (error) {
    if (error.code === 'ENOENT') return false
    throw error
  }
}

export async function restoreMaterialOriginals(preview, sources, log = console.log) {
  const scene = dirname(preview)
  const manifestPath = join(preview, 'source-textures.json')
  const manifestBytes = await readFile(manifestPath)
  const images = manifestImages(JSON.parse(manifestBytes))
  const originals = join(preview, 'originals')
  const staging = await mkdtemp(join(scene, '.restore-originals-'))
  const staged = []
  const missing = []
  try {
    const remaining = images.values()
    async function fetchNext() {
      for (const ref of remaining) {
        try {
          if (await verified(join(originals, ref.filename), ref)) continue
          const temp = join(staging, ref.filename)
          const response = await sources.fetch(ref.url, { signal: AbortSignal.timeout(40_000) })
          if (!response.ok || !response.body) throw new Error('HTTP ' + response.status)
          await pipeline(response.body, createWriteStream(temp, { flags: 'wx' }))
          if (!await verified(temp, ref)) throw new Error('Publisher size or MD5 mismatch.')
          staged.push(ref)
        } catch (error) {
          missing.push(ref)
          log(new URL(ref.url).pathname.split('/').pop() + ': ' + error.message)
        }
      }
    }
    await Promise.all(Array.from({ length: Math.min(3, images.length) }, fetchNext))
    const lockPath = join(scene, '.install-preview.lock')
    let lock
    try { lock = await open(lockPath, 'wx') }
    catch (error) {
      if (error.code === 'EEXIST') throw new Error('Another process is installing this harbor preview. Retry after it finishes.')
      throw error
    }
    try {
      if (!manifestBytes.equals(await readFile(manifestPath))) {
        throw new Error('The harbor source manifest changed during restoration. Retry against the current preview.')
      }
      if (staged.length) await mkdir(originals, { recursive: true })
      for (const ref of staged) await rename(join(staging, ref.filename), join(originals, ref.filename))
    } finally {
      await lock.close()
      await rm(lockPath)
    }
    return { installed: staged.length, available: images.length - missing.length,
      total: images.length, missing: missing.filter(ref => !ref.upgrade).length,
      missingUpgrades: missing.filter(ref => ref.upgrade).length }
  } finally {
    await rm(staging, { recursive: true, force: true })
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2)
  const usage = 'Usage: npm run restore:materials -- [original-folder-or-zip]'
  if (args.includes('--help')) { console.log(usage); process.exit(0) }
  if (args.length > 1 || args.some(arg => arg.startsWith('-'))) { console.error(usage); process.exit(1) }
  const root = fileURLToPath(new URL('../', import.meta.url))
  const preview = join(root, 'scene', 'smugglers-cove', 'preview')
  const sources = sceneSourceCache(join(root, 'node_modules', '.cache', 'caustic-source-cache'), fetch,
    { originalsDirectory: join(preview, 'originals'), sourcePath: args[0] })
  try {
    const result = await restoreMaterialOriginals(preview, sources)
    console.log('Verified original images: ' + result.available + '/' + result.total +
      '; newly installed: ' + result.installed + '; missing: ' + result.missing +
      '; optional resolution upgrades unavailable: ' + result.missingUpgrades + '.')
    if (result.available !== result.total) process.exitCode = 1
  } catch (error) { console.error(error.message); process.exitCode = 1 }
}
