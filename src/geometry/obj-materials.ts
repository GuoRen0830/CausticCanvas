import type { BaseColorMaterial } from '../rendering/scene-mesh'
import { readObjMesh } from './obj-mesh'

const unquote = (s: string) => s.replace(/^"(.*)"$/, '$1')
const tokens = (s: string) => (s.match(/"[^"]*"|[^\s]+/g) ?? []).map(unquote)
const filePath = (f: File) => (f.webkitRelativePath || f.name).replaceAll('\\', '/')

function normalize(path: string) {
  const parts: string[] = []
  for (const part of path.replaceAll('\\', '/').split('/')) {
    if (part === '..') parts.pop()
    else if (part && part !== '.') parts.push(part)
  }
  return parts.join('/')
}

// Folder imports retain relative paths. Loose selections and absolute paths from
// exporters can use a basename only when it identifies exactly one supplied file.
function localFiles(files: File[]) {
  const paths = new Map<string, File | null>(), folded = new Map<string, File | null>(), names = new Map<string, File | null>()
  function add(index: Map<string, File | null>, key: string, file: File) {
    index.set(key, index.has(key) ? null : file)
  }
  for (const file of files) {
    const path = filePath(file)
    add(paths, path, file)
    add(folded, path.toLowerCase(), file)
    add(names, file.name.toLowerCase(), file)
  }
  return (owner: File, ref: string) => {
    const path = unquote(ref).replaceAll('\\', '/')
    const dir = filePath(owner).split('/').slice(0, -1)
    const key = normalize([...dir, path].join('/'))
    const absolute = /^(?:[a-z]:\/|\/)/i.test(path)
    if (!absolute) {
      const exact = paths.get(key) ?? folded.get(key.toLowerCase())
      if (exact) return exact
      if (owner.webkitRelativePath) return undefined
    }
    return names.get(path.split('/').at(-1)!.toLowerCase()) ?? undefined
  }
}

type ImportedMaterial = BaseColorMaterial & { mapPath?: string }

function colorMap(record: string, material: ImportedMaterial) {
  const args = tokens(record)
  const transform: [number, number, number, number] = [1, 1, 0, 0]
  let i = 0
  while (args[i]?.startsWith('-')) {
    const option = args[i++]
    if (option === '-clamp') {
      const value = args[i++]
      if (value !== 'on' && value !== 'off') throw new Error('Invalid map_Kd -clamp option')
      material.clamp = value === 'on'
    } else if (option === '-s' || option === '-o') {
      const values: number[] = []
      while (i < args.length - 1 && values.length < 3 && Number.isFinite(Number(args[i]))) values.push(Number(args[i++]))
      if (!values.length) throw new Error('Invalid map_Kd ' + option)
      const start = option === '-s' ? 0 : 2
      transform[start] = values[0]
      if (values.length > 1) transform[start + 1] = values[1]
    } else {
      throw new Error('Unsupported map_Kd option ' + option)
    }
  }
  if (i === args.length) throw new Error('Missing map_Kd filename')
  material.mapPath = args.slice(i).join(' ')
  material.uvTransform = transform
}

function parseMtl(text: string, file: File, warnings: Set<string>) {
  const materials = new Map<string, ImportedMaterial>()
  let current: ImportedMaterial | undefined
  for (const [index, raw] of text.split(/\r\n?|\n/).entries()) {
    const line = raw.split('#', 1)[0].trim()
    const [, op, value = ''] = /^(\S+)(?:\s+(.*))?$/.exec(line) ?? []
    if (op === 'newmtl') {
      current = { color: [1, 1, 1] }
      materials.set(unquote(value), current)
    } else if (current && op === 'Kd') {
      const rgb = tokens(value).map(Number)
      if (rgb.length === 3 && rgb.every(v => Number.isFinite(v) && v >= 0 && v <= 1)) {
        current.color = [rgb[0], rgb[1], rgb[2]]
      } else warnings.add(file.name + ' line ' + (index + 1) + ': invalid RGB Kd.')
    } else if (current && op === 'map_Kd') {
      try { colorMap(value, current) }
      catch (error) { warnings.add(file.name + ' line ' + (index + 1) + ': ' + (error as Error).message + '.') }
    }
  }
  return materials
}

export async function readObjAsset(obj: File, files: File[]) {
  const mesh = await readObjMesh(obj)
  const resolve = localFiles(files), warnings = new Set<string>()
  const used = new Set(mesh.draws?.map(draw => draw.material))
  const materials = new Map<string, BaseColorMaterial>()
  const libraries = new Set<File>()
  for (const record of mesh.materialLibraries) {
    // A single unquoted filename can contain spaces. Also accept a list of libraries.
    const parts = tokens(record)
    const multiple = parts.length > 1 && parts.every(part => /\.mtl$/i.test(part))
    const refs = multiple ? parts : [unquote(record)]
    for (const ref of refs) {
      const file = resolve(obj, ref)
      if (file) libraries.add(file)
      else warnings.add('Missing material library: ' + ref + '.')
    }
  }
  for (const file of libraries) {
    const parsed = parseMtl(await file.text(), file, warnings)
    for (const [name, material] of parsed) {
      if (!used.has(name)) continue
      if (material.mapPath) {
        material.map = resolve(file, material.mapPath)
        if (!material.map) warnings.add('Missing Base Color image: ' + material.mapPath + '. Using Kd.')
      }
      materials.set(name, material)
    }
  }
  for (const draw of mesh.draws ?? []) {
    if (!draw.material) continue
    if (!materials.has(draw.material)) warnings.add('Missing material: ' + draw.material + '. Using gray.')
    else if (!draw.hasUV && materials.get(draw.material)?.map) {
      warnings.add(draw.material + ': faces without complete UVs use Kd.')
    }
  }
  return { mesh, materials, warnings }
}
