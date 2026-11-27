import { dirname, join, relative, resolve } from 'node:path'
import { randomUUID } from 'node:crypto'
import { fileURLToPath } from 'node:url'
import { createWriteStream } from 'node:fs'
import { access, copyFile, cp, mkdir, mkdtemp, open, readdir, readFile, rename, rm, writeFile } from 'node:fs/promises'
import { pipeline } from 'node:stream/promises'
import { setTimeout as delay } from 'node:timers/promises'
import { createServer } from 'vite'
import { sceneSourceCache } from './scene-source-cache.mjs'

const root = fileURLToPath(new URL('../', import.meta.url))
const readers = {
  'smugglers-cove': ['/src/geometry/smugglers-cove-preview.ts', 'readCovePreview'],
}

function imageFormat(bytes) {
  if (bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))) return 'png'
  if (bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255) return 'jpg'
  if (bytes.toString('ascii', 0, 4) === 'RIFF' && bytes.toString('ascii', 8, 12) === 'WEBP') return 'webp'
  if (bytes.toString('ascii', 4, 8) === 'ftyp') {
    const end = Math.min(bytes.length, bytes.readUInt32BE(0))
    for (let offset = 8; offset + 4 <= end; offset += 4) {
      if (offset === 12) continue // Minor version, between the major and compatible brands.
      if (['avif', 'avis'].includes(bytes.toString('ascii', offset, offset + 4))) return 'avif'
    }
  }
  throw new Error('A base color image could not be identified as a PNG, JPEG, WebP or AVIF.')
}

function* objLines(mesh, drawNames, id, stride) {
  yield `mtllib ${id}.mtl`
  const count = mesh.vertices.length / stride
  for (let i = 0; i < count; i++) {
    const v = i * stride
    yield `v ${mesh.vertices[v]} ${mesh.vertices[v + 1]} ${mesh.vertices[v + 2]}`
    yield `vn ${mesh.vertices[v + 3]} ${mesh.vertices[v + 4]} ${mesh.vertices[v + 5]}`
    // SceneMesh UVs are already in OBJ space. The renderer applies the V flip.
    if (mesh.uvs) yield `vt ${mesh.uvs[2 * i]} ${mesh.uvs[2 * i + 1]}`
  }
  let lastObject = -1
  const draws = mesh.draws ?? [{ firstIndex: 0, indexCount: mesh.indices.length }]
  for (const draw of draws) {
    yield 'usemtl ' + (drawNames.get(draw) ?? '')
    for (let i = draw.firstIndex; i < draw.firstIndex + draw.indexCount; i += 3) {
      const triangle = mesh.indices.subarray(i, i + 3)
      const object = mesh.vertices[triangle[0] * stride + 10]
      if (object !== lastObject) {
        yield 'o object-' + object
        lastObject = object
      }
      const corners = Array.from(triangle, vertex => {
        const index = vertex + 1
        return draw.hasUV && mesh.uvs ? `${index}/${index}/${index}` : `${index}//${index}`
      })
      yield 'f ' + corners.join(' ')
    }
  }
}

function* textChunks(lines) {
  let chunk = ''
  for (const line of lines) {
    chunk += line + '\n'
    if (chunk.length >= 128 * 1024) { yield chunk; chunk = '' }
  }
  if (chunk) yield chunk
}

// Emit only the textures used in the base-color preview. Source material names
// can contain slashes or punctuation, so use our own portable disk filenames.
export async function writeScenePreview(asset, destination, id, stride) {
  const plain = new Map(), textured = new Map(), drawNames = new Map(), images = new Map(), mtl = []
  // First draw order stays stable when source texture requests finish out of order.
  for (const draw of asset.mesh.draws ?? []) {
    const key = draw.material
    if (!key) continue
    const material = asset.materials.get(key)
    if (!material) throw new Error('Missing scene material ' + key)
    // MTL applies map_Kd to every face using the material. The interactive
    // renderer uses only Kd for a draw without a complete UV stream.
    const useMap = material.map && draw.hasUV && asset.mesh.uvs
    const variants = useMap ? textured : plain
    const existing = variants.get(key)
    if (existing !== undefined) {
      drawNames.set(draw, existing)
      continue
    }
    const name = 'material-' + (plain.size + textured.size)
    drawNames.set(draw, name)
    variants.set(key, name)
    mtl.push('newmtl ' + name, 'Kd ' + material.color.join(' '))
    if (useMap) {
      let path = images.get(material.map)
      if (!path) {
        const bytes = Buffer.from(await material.map.arrayBuffer())
        path = 'textures/base-color-' + images.size + '.' + imageFormat(bytes)
        await mkdir(join(destination, 'textures'), { recursive: true })
        await writeFile(join(destination, path), bytes)
        images.set(material.map, path)
      }
      const flags = []
      if (material.clamp) flags.push('-clamp on')
      if (material.uvTransform) {
        const [sx, sy, ox, oy] = material.uvTransform
        flags.push(`-s ${sx} ${sy}`, `-o ${ox} ${oy}`)
      }
      mtl.push(['map_Kd', ...flags, path].join(' '))
    }
    mtl.push('')
  }
  await writeFile(join(destination, id + '.mtl'), mtl.join('\n'))
  await pipeline(textChunks(objLines(asset.mesh, drawNames, id, stride)), createWriteStream(join(destination, id + '.obj')))
}

async function readFiles(directory, prefix = '') {
  const files = []
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name), relativePath = prefix + entry.name
    if (entry.isDirectory()) files.push(...await readFiles(path, relativePath + '/'))
    else if (entry.isFile()) {
      const file = new File([await readFile(path)], entry.name)
      Object.defineProperty(file, 'webkitRelativePath', { value: relativePath })
      files.push(file)
    }
  }
  return files
}

async function checkPreview(directory, readObjAsset) {
  const files = await readFiles(directory), models = files.filter(file => /\.obj$/i.test(file.name))
  if (models.length !== 1) throw new Error('Expected a single OBJ in ' + directory)
  const decoded = await readObjAsset(models[0], files)
  if (decoded.warnings.size) throw new Error([...decoded.warnings].join(' '))
  return decoded.mesh.indices.length / 3
}

async function exists(path) {
  try { await access(path); return true }
  catch (error) { if (error.code === 'ENOENT') return false; throw error }
}

async function movePreview(from, to) {
  // Windows can briefly hold a newly verified OBJ open after closing the file.
  for (let attempt = 0; ; attempt++) {
    try { return await rename(from, to) }
    catch (error) {
      if (process.platform !== 'win32' || !['EPERM', 'EBUSY'].includes(error.code) || attempt >= 6) throw error
      await delay(100 * (attempt + 1))
    }
  }
}

async function installVerifiedPreview(next, destination, id) {
  try {
    return await movePreview(next, destination)
  } catch (error) {
    if (process.platform !== 'win32' || !['EPERM', 'EBUSY'].includes(error.code)) throw error
  }
  // A Windows file watcher can hold a directory handle that prevents renaming
  // its parent. Publish the verified files into a newly claimed directory,
  // with the OBJ appearing only when every referenced texture and MTL is ready.
  await mkdir(destination)
  const obj = id + '.obj'
  try {
    for (const file of await readdir(next)) {
      if (file !== obj) await cp(join(next, file), join(destination, file), { recursive: true, errorOnExist: true, force: false })
    }
    const pending = join(destination, obj + '.installing')
    await copyFile(join(next, obj), pending)
    await rename(pending, join(destination, obj))
  } catch (error) {
    if (destination === join(root, 'scene', id, 'preview') && Object.hasOwn(readers, id)) {
      await rm(destination, { recursive: true, force: true })
    }
    throw error
  }
}

// Directory handles held by Vite or Explorer may prevent replacement even
// after a retry. Keep the previous verified package until all new files are
// present. Withhold the entry OBJ while its companion MTL and images change.
async function replaceWatchedPreview(next, destination, previous, id) {
  const lockPath = join(dirname(destination), '.install-preview.lock')
  const lock = await open(lockPath, 'wx')
  const temporary = join(destination, '.install-' + randomUUID())
  const entry = id + '.obj'
  async function publish(source) {
    const files = await readdir(source, { recursive: true, withFileTypes: true })
    const textures = join(destination, 'textures'), currentImages = new Set()
    for (const file of files) {
      if (!file.isFile() || (file.parentPath === source && file.name === entry)) continue
      const target = join(destination, relative(source, file.parentPath), file.name)
      if (dirname(target) === textures) currentImages.add(file.name)
      await mkdir(dirname(target), { recursive: true })
      await copyFile(join(file.parentPath, file.name), temporary)
      await rename(temporary, target)
    }
    // Vite's scene glob loads every image in the directory. A rebake can move
    // a JPEG to a new slot and put a PNG at its former index.
    if (await exists(textures)) {
      for (const name of await readdir(textures)) {
        if (/^base-color-\d+\.(jpg|png|webp|avif)$/.test(name) && !currentImages.has(name)) {
          await rm(join(textures, name))
        }
      }
    }
    await copyFile(join(source, entry), temporary)
    await rename(temporary, join(destination, entry))
  }
  try {
    await cp(destination, previous, { recursive: true, errorOnExist: true, force: false })
    await rm(join(destination, entry))
    try {
      await publish(next)
    } catch (error) {
      await publish(previous)
      throw error
    }
  } finally {
    await rm(temporary, { force: true })
    await lock.close()
    await rm(lockPath, { force: true })
  }
}

async function prepare(id, force, sourcesOnly, vite, readObjAsset, vertexStride) {
  const parent = join(root, 'scene', id), destination = join(parent, 'preview')
  await mkdir(parent, { recursive: true })
  if (!sourcesOnly && !force && await exists(destination)) {
    console.log(`${id}: using existing preview (${await checkPreview(destination, readObjAsset)} triangles).`)
    return
  }
  const [module, entry] = readers[id]
  const reader = (await vite.ssrLoadModule(module))[entry]
  console.log(`${id}: fetching the scene geometry and base colors.`)
  const asset = await reader()
  for (const warning of asset.warnings) console.warn(id + ': ' + warning)
  const notes = [...asset.warnings].filter(message => message.startsWith('Using the project-'))
  // A transient download failure must not become the preferred offline scene.
  if ([...asset.warnings].some(message => /could not load/i.test(message))) {
    throw new Error('Some models or textures could not be downloaded. Retry to save a complete local preview.')
  }
  if (sourcesOnly) {
    console.log(id + ': source files are ready on disk.')
    return
  }
  if (asset.warnings.size > notes.length && force && await exists(destination)) {
    throw new Error('The source loaded with missing assets. Keeping the existing local preview.')
  }
  // Both staging and the target are on the same filesystem. A failed fetch,
  // texture write or OBJ re-import never modifies an existing preview.
  const stage = await mkdtemp(join(parent, '.prepare-'))
  const next = join(stage, 'next'), previous = join(stage, 'previous')
  let previousMoved = false
  try {
    await mkdir(next)
    await writeScenePreview(asset, next, id, vertexStride)
    if (notes.length) await writeFile(join(next, 'ASSET-NOTES.txt'), notes.join('\n') + '\n')
    const triangles = await checkPreview(next, readObjAsset)
    if (triangles * 3 !== asset.mesh.indices.length) throw new Error('Triangle count changed during OBJ conversion.')
    if (force && await exists(destination)) {
      try {
        await movePreview(destination, previous)
      } catch (error) {
        if (process.platform !== 'win32' || !['EPERM', 'EBUSY'].includes(error.code)) throw error
        // Retain the backup if rollback itself fails.
        previousMoved = true
        await replaceWatchedPreview(next, destination, previous, id)
        previousMoved = false
        console.log(`${id}: saved ${triangles} triangles to ${destination}`)
        return
      }
      previousMoved = true
    }
    try {
      await installVerifiedPreview(next, destination, id)
    } catch (error) {
      if (previousMoved) {
        await movePreview(previous, destination)
        previousMoved = false
      } else if (!force && await exists(destination)) {
        await checkPreview(destination, readObjAsset)
        console.log(id + ': another preparation finished first.')
        return
      }
      throw error
    }
    previousMoved = false
    console.log(`${id}: saved ${triangles} triangles to ${destination}`)
  } finally {
    if (!previousMoved) await rm(stage, { recursive: true, force: true })
    else console.error('Previous preview retained at ' + previous)
  }
}

async function main(args) {
  if (args.includes('--help') || args.includes('-h')) {
    console.log('npm run prepare:scenes -- [all | smugglers-cove] [--force | --sources-only]')
    return
  }
  const force = args.includes('--force'), sourcesOnly = args.includes('--sources-only')
  const requested = args.filter(a => a !== '--force' && a !== '--sources-only')
  const ids = requested.length && !requested.includes('all') ? [...new Set(requested)] : Object.keys(readers)
  if (requested.some(a => a !== 'all' && !Object.hasOwn(readers, a))) throw new Error('Unknown scene; run with --help.')
  const vite = await createServer({ root, configFile: false, server: { middlewareMode: true, hmr: false, watch: null }, appType: 'custom' })
  const originalFetch = globalThis.fetch
  const sources = sceneSourceCache(join(root, 'node_modules', '.cache', 'caustic-source-cache'), originalFetch)
  globalThis.fetch = sources.fetch
  let failed = false
  try {
    const { readObjAsset } = await vite.ssrLoadModule('/src/geometry/obj-materials.ts')
    const { sceneVertexStride } = await vite.ssrLoadModule('/src/rendering/scene-mesh.ts')
    for (const id of ids) {
      try { await prepare(id, force, sourcesOnly, vite, readObjAsset, sceneVertexStride) }
      catch (error) {
        failed = true
        console.error(id + ': ' + error.message)
        if (error.cause) console.error(error.cause.message ?? error.cause)
      }
    }
  } finally {
    globalThis.fetch = originalFetch
    console.log(`Source files: ${sources.counts.reused} read from disk, ${sources.counts.downloaded} downloaded.`)
    await vite.close()
  }
  if (failed) process.exitCode = 1
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main(process.argv.slice(2)).catch(error => { console.error(error.message); process.exitCode = 1 })
}
