import { dirname, join, relative, resolve, sep } from 'node:path'
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
  throw new Error('A material image could not be identified as a PNG, JPEG, WebP or AVIF.')
}

function* objLines(mesh, materials, drawNames, id, stride) {
  yield `mtllib ${id}.mtl`
  const count = mesh.vertices.length / stride
  const colored = new Set()
  for (const draw of mesh.draws ?? []) {
    if (!materials.get(draw.material)?.vertexColor) continue
    for (let j = draw.firstIndex; j < draw.firstIndex + draw.indexCount; j++) colored.add(mesh.indices[j])
  }
  for (let i = 0; i < count; i++) {
    const v = i * stride
    const color = colored.has(i) ? ' ' + mesh.vertices.subarray(v + 6, v + 9).join(' ') : ''
    yield `v ${mesh.vertices[v]} ${mesh.vertices[v + 1]} ${mesh.vertices[v + 2]}${color}`
    // OBJ tooling can ignore optional streams while retaining the UV0 model.
    if (mesh.uvs1) yield '# caustic-uv1 ' + mesh.uvs1.subarray(i * 2, i * 2 + 2).join(' ')
    if (mesh.tangents) yield '# caustic-tangent ' + mesh.tangents.subarray(i * 4, i * 4 + 4).join(' ')
    yield `vn ${mesh.vertices[v + 3]} ${mesh.vertices[v + 4]} ${mesh.vertices[v + 5]}`
    // SceneMesh UVs are already in OBJ space. The renderer applies the V flip.
    if (mesh.uvs) yield `vt ${mesh.uvs[2 * i]} ${mesh.uvs[2 * i + 1]}`
  }
  let lastObject = -1
  const draws = mesh.draws ?? [{ firstIndex: 0, indexCount: mesh.indices.length }]
  for (const draw of draws) {
    yield 'usemtl ' + (drawNames.get(draw) ?? '')
    if (draw.uvMask !== undefined) yield '# caustic-uv-mask ' + draw.uvMask
    for (let i = draw.firstIndex; i < draw.firstIndex + draw.indexCount; i += 3) {
      const triangle = mesh.indices.subarray(i, i + 3)
      const object = mesh.vertices[triangle[0] * stride + 10]
      if (object !== lastObject) {
        yield 'o object-' + object
        lastObject = object
      }
      const corners = Array.from(triangle, vertex => {
        const index = vertex + 1
        const hasPrimaryUV = draw.uvMask === undefined ? draw.hasUV : (draw.uvMask & 1) !== 0
        return hasPrimaryUV && mesh.uvs ? `${index}/${index}/${index}` : `${index}//${index}`
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

// Emit only the textures used by the preview. Source material names
// can contain slashes or punctuation, so use our own portable disk filenames.
export async function writeScenePreview(asset, destination, id, stride) {
  const plain = new Map(), textured = new Map(), drawNames = new Map(), images = new Map(), mtl = []
  const recoverable = {}
  const counts = new Map()
  const vertexColors = Array.from(asset.materials.values()).some(material => material.vertexColor)
  const uvStreams = (asset.mesh.uvs ? 1 : 0) | (asset.mesh.uvs1 ? 2 : 0)
  async function mapLine(op, image, role, transform, clamp, options = []) {
    if (!image) return
    let path = images.get(image)
    if (!path) {
      const bytes = Buffer.from(await image.arrayBuffer()), count = counts.get(role) ?? 0
      path = 'textures/' + role + '-' + count + '.' + imageFormat(bytes)
      counts.set(role, count + 1)
      await mkdir(join(destination, 'textures'), { recursive: true })
      await writeFile(join(destination, path), bytes)
      images.set(image, path)
      mtl.push('# ' + path + ' <- ' + image.name.replace(/[\r\n]/g, ' '))
    }
    const flags = [...options]
    if (clamp) flags.push('-clamp on')
    if (transform) {
      const [sx, sy, ox, oy] = transform
      flags.push(`-s ${sx} ${sy}`, `-o ${ox} ${oy}`)
    }
    mtl.push([op, ...flags, path].join(' '))
  }
  // First draw order stays stable when source texture requests finish out of order.
  for (const draw of asset.mesh.draws ?? []) {
    const key = draw.material
    if (!key) continue
    const material = asset.materials.get(key)
    if (!material) throw new Error('Missing scene material ' + key)
    // The interactive renderer uses factors alone when a draw is missing UVs.
    const mask = (draw.uvMask ?? Number(!!draw.hasUV)) & uvStreams
    const uses = slot => (mask & (1 << (material.sampling?.[slot]?.uvSet ?? 0))) !== 0
    const refs = Object.fromEntries(Object.entries(material.recoverableMaps ?? {}).filter(([, ref]) =>
      (mask & (1 << (ref.sampling?.uvSet ?? 0))) !== 0))
    const useMap = ['map','normalMap','metallicRoughnessMap','roughnessMap','metallicMap','occlusionMap','emissiveMap']
      .some(slot => material[slot] && uses(slot)) || Object.keys(refs).length > 0
    const variants = useMap ? textured : plain
    const variant = useMap ? mask + '/' + key : key
    const existing = variants.get(variant)
    if (existing !== undefined) {
      drawNames.set(draw, existing)
      continue
    }
    const name = 'material-' + (plain.size + textured.size)
    drawNames.set(draw, name)
    variants.set(variant, name)
    if (Object.keys(refs).length) recoverable[name] = refs
    mtl.push('newmtl ' + name, 'Kd ' + material.color.join(' '))
    // A single vertex can be shared by a colored and an uncolored draw.
    if (vertexColors) mtl.push('# caustic-vertex-color ' + Number(!!material.vertexColor))
    if (material.vertexColor && material.vertexWetness) {
      const { saturatedRed, saturatedRoughness } = material.vertexWetness
      mtl.push(`# caustic-vertex-wetness ${saturatedRed} ${saturatedRoughness}`)
    }
    if (material.doubleSided) mtl.push('# caustic-double-sided 1')
    if (material.roughness !== undefined) mtl.push('Pr ' + material.roughness)
    if (material.metallic !== undefined) mtl.push('Pm ' + material.metallic)
    if (material.emissive) mtl.push('Ke ' + material.emissive.join(' '))
    if (useMap) {
      const image = slot => uses(slot) ? material[slot] : undefined
      for (const [slot, sampling] of Object.entries(material.sampling ?? {})) {
        if (image(slot)) mtl.push('# caustic-texture-sampling ' + slot + ' ' + JSON.stringify(sampling))
      }
      await mapLine('map_Kd', image('map'), 'base-color', material.uvTransform, material.clamp)
      await mapLine('norm', image('normalMap'), 'normal', material.normalUVTransform, material.normalClamp,
        material.normalScale !== undefined ? ['-bm ' + material.normalScale] : [])
      if (image('normalMap') && material.normalYSign === -1) mtl.push('# caustic-normal-y-sign -1')
      if (image('metallicRoughnessMap')) {
        const { metallicRoughnessMap: map, metallicRoughnessUVTransform: uv, metallicRoughnessClamp: clamp } = material
        await mapLine('map_Pr', map, 'metallic-roughness', uv, clamp, ['-imfchan g'])
        await mapLine('map_Pm', map, 'metallic-roughness', uv, clamp, ['-imfchan b'])
      } else {
        await mapLine('map_Pr', image('roughnessMap'), 'roughness', material.roughnessUVTransform, material.roughnessClamp)
        await mapLine('map_Pm', image('metallicMap'), 'metallic', material.metallicUVTransform, material.metallicClamp)
      }
      await mapLine('map_AO', image('occlusionMap'), 'occlusion', material.occlusionUVTransform, material.occlusionClamp, ['-imfchan r'])
      if (image('occlusionMap') && material.occlusionStrength !== undefined) mtl.push('# caustic-occlusion-strength ' + material.occlusionStrength)
      await mapLine('map_Ke', image('emissiveMap'), 'emissive', material.emissiveUVTransform, material.emissiveClamp)
    }
    mtl.push('')
  }
  await writeFile(join(destination, id + '.mtl'), mtl.join('\n'))
  if (Object.keys(recoverable).length) {
    await writeFile(join(destination, 'source-textures.json'), JSON.stringify({ version: 1, materials: recoverable }, null, 2) + '\n')
  }
  await pipeline(textChunks(objLines(asset.mesh, asset.materials, drawNames, id, stride)), createWriteStream(join(destination, id + '.obj')))
}

async function readFiles(directory, prefix = '') {
  const files = []
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    // Restored source images are referenced by the manifest, separately from the MTL.
    if (!prefix && entry.name === 'originals') continue
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
  const temporary = join(destination, '.install-' + randomUUID())
  const entry = id + '.obj'
  async function publish(source) {
    const files = await readdir(source, { recursive: true, withFileTypes: true })
    const textures = join(destination, 'textures'), currentImages = new Set()
    for (const file of files) {
      if (!file.isFile() || (file.parentPath === source && file.name === entry)) continue
      const folder = relative(source, file.parentPath)
      if (folder.split(sep)[0] === 'originals') continue // Existing restored sources stay in place.
      const target = join(destination, folder, file.name)
      if (dirname(target) === textures) currentImages.add(file.name)
      await mkdir(dirname(target), { recursive: true })
      await copyFile(join(file.parentPath, file.name), temporary)
      await rename(temporary, target)
    }
    // Vite's scene glob loads every image in the directory. A rebake can move
    // a JPEG to a new slot and put a PNG at its former index.
    if (await exists(textures)) {
      for (const name of await readdir(textures)) {
        if (/^(base-color|normal|roughness|metallic|metallic-roughness|occlusion|emissive)-\d+\.(jpg|png|webp|avif)$/.test(name) && !currentImages.has(name)) {
          await rm(join(textures, name))
        }
      }
    }
    for (const optional of ['ASSET-NOTES.txt', 'source-textures.json']) {
      if (!await exists(join(source, optional))) await rm(join(destination, optional), { force: true })
    }
    if (await exists(join(source, entry))) {
      await copyFile(join(source, entry), temporary)
      await rename(temporary, join(destination, entry))
    }
  }
  try {
    await cp(destination, previous, { recursive: true, errorOnExist: true, force: false })
    await rm(join(destination, entry), { force: true })
    try {
      await publish(next)
    } catch (error) {
      await publish(previous)
      throw error
    }
  } finally {
    await rm(temporary, { force: true })
  }
}

async function prepare(id, force, sourcesOnly, vite, readObjAsset, vertexStride, hasLocal) {
  const parent = join(root, 'scene', id), destination = join(parent, 'preview')
  await mkdir(parent, { recursive: true })
  if (!sourcesOnly && !force && await exists(destination)) {
    console.log(`${id}: using existing preview (${await checkPreview(destination, readObjAsset)} triangles).`)
    return
  }
  const [module, entry] = readers[id]
  const reader = (await vite.ssrLoadModule(module))[entry]
  console.log(`${id}: fetching scene geometry and material channels.`)
  const asset = await reader(undefined, { hasLocal,
    readDetailFile: async filename => new Blob([await readFile(join(parent, filename))], { type: 'image/png' }) })
  for (const warning of asset.warnings) console.warn(id + ': ' + warning)
  const notes = [...asset.warnings].filter(message =>
    message.startsWith('Using the project-') || message.startsWith('Source material ') || message.startsWith('Source geometry ')).sort()
  // A transient download failure must not become the preferred offline scene.
  if ([...asset.warnings].some(message => !notes.includes(message) && /could not load/i.test(message))) {
    throw new Error('Some models or textures could not be downloaded. Retry to save a complete local preview.')
  }
  if (sourcesOnly) {
    console.log(id + ': available sources saved to disk.')
    return
  }
  if (asset.warnings.size > notes.length && force && await exists(destination)) {
    throw new Error('The source loaded with missing assets. Keeping the existing local preview.')
  }
  // Both staging and the target are on the same filesystem. A failed fetch,
  // texture write or OBJ re-import never modifies an existing preview.
  const stage = await mkdtemp(join(parent, '.prepare-'))
  const next = join(stage, 'next'), previous = join(stage, 'previous')
  const lockPath = join(parent, '.install-preview.lock')
  let lock
  let previousMoved = false
  try {
    await mkdir(next)
    await writeScenePreview(asset, next, id, vertexStride)
    if (notes.length) await writeFile(join(next, 'ASSET-NOTES.txt'), notes.join('\n') + '\n')
    const triangles = await checkPreview(next, readObjAsset)
    if (triangles * 3 !== asset.mesh.indices.length) throw new Error('Triangle count changed during OBJ conversion.')
    // The texture restorer uses the same short-lived lock when publishing originals.
    // Take the snapshot under the lock so a concurrent restore cannot be lost.
    lock = await open(lockPath, 'wx')
    if (force && await exists(destination)) {
      const originals = join(destination, 'originals')
      if (await exists(originals)) await cp(originals, join(next, 'originals'), { recursive: true })
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
    if (lock) {
      await lock.close()
      await rm(lockPath, { force: true })
    }
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
  const sources = sceneSourceCache(join(root, 'node_modules', '.cache', 'caustic-source-cache'), originalFetch,
    { originalsDirectory: join(root, 'scene', 'smugglers-cove', 'preview', 'originals') })
  globalThis.fetch = sources.fetch
  let failed = false
  try {
    const { readObjAsset } = await vite.ssrLoadModule('/src/geometry/obj-materials.ts')
    const { sceneVertexStride } = await vite.ssrLoadModule('/src/rendering/scene-mesh.ts')
    for (const id of ids) {
      try { await prepare(id, force, sourcesOnly, vite, readObjAsset, sceneVertexStride, sources.hasLocal) }
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
