import type { SceneMapSampling, SceneMapSlot, SceneMaterial } from '../rendering/scene-mesh'
import { localFiles } from './local-asset-files'
import { readObjMesh } from './obj-mesh'

const unquote = (s: string) => s.replace(/^"(.*)"$/, '$1')
const tokens = (s: string) => (s.match(/"[^"]*"|[^\s]+/g) ?? []).map(unquote)

type Slot = 'map_Kd' | 'map_Pr' | 'map_Pm' | 'norm' | 'map_AO' | 'map_Ke'
type TextureMap = {
  path: string
  transform: [number, number, number, number]
  clamp: boolean
  channel?: string
  strength?: number
}
type ImportedMaterial = { material: SceneMaterial; maps: Partial<Record<Slot, TextureMap>>; shininess?: number }

function parseMap(record: string, slot: Slot): TextureMap {
  const args = tokens(record)
  const transform: TextureMap['transform'] = [1, 1, 0, 0]
  let clamp = false, channel: string | undefined, strength: number | undefined, i = 0
  while (args[i]?.startsWith('-')) {
    const option = args[i++]
    if (option === '-clamp') {
      const value = args[i++]
      if (value !== 'on' && value !== 'off') throw new Error('Invalid ' + slot + ' -clamp option')
      clamp = value === 'on'
    } else if (option === '-s' || option === '-o') {
      const values: number[] = []
      while (i < args.length - 1 && values.length < 3 && Number.isFinite(Number(args[i]))) values.push(Number(args[i++]))
      if (!values.length) throw new Error('Invalid ' + slot + ' ' + option)
      const start = option === '-s' ? 0 : 2
      transform[start] = values[0]
      if (values.length > 1) transform[start + 1] = values[1]
    } else if (option === '-imfchan' && (slot === 'map_Pr' || slot === 'map_Pm' || slot === 'map_AO')) {
      channel = args[i++]
      if (!['r', 'g', 'b'].includes(channel)) throw new Error('Unsupported ' + slot + ' channel ' + channel)
    } else if (option === '-bm' && slot === 'norm') {
      strength = Number(args[i++])
      if (!Number.isFinite(strength)) throw new Error('Invalid normal scale')
    } else {
      throw new Error('Unsupported ' + slot + ' option ' + option)
    }
  }
  if (i === args.length) throw new Error('Missing ' + slot + ' filename')
  return { path: args.slice(i).join(' '), transform, clamp, channel, strength }
}

const samplingSlots = new Set<SceneMapSlot>(['map', 'metallicRoughnessMap', 'normalMap', 'roughnessMap',
  'metallicMap', 'occlusionMap', 'emissiveMap'])

// Wavefront map options cover scale and offset. Preserve glTF sampling options
// that have no MTL equivalent in optional comments readable by this importer.
function parseSampling(text: string): SceneMapSampling {
  const source = JSON.parse(text)
  if (!source || typeof source !== 'object') throw new Error('Expected texture sampling settings')
  const value: SceneMapSampling = {}
  if (source.uvSet !== undefined) {
    if (source.uvSet !== 0 && source.uvSet !== 1) throw new Error('Invalid texture coordinate set')
    value.uvSet = source.uvSet
  }
  if (source.rotation !== undefined) {
    if (!Number.isFinite(source.rotation)) throw new Error('Invalid texture rotation')
    value.rotation = source.rotation
  }
  if (source.sampler !== undefined) {
    if (!source.sampler || typeof source.sampler !== 'object') throw new Error('Invalid texture sampler')
    const sampler: GPUSamplerDescriptor = {}
    for (const key of ['addressModeU', 'addressModeV'] as const) {
      const mode = source.sampler[key]
      if (mode === undefined) continue
      if (!['repeat','clamp-to-edge','mirror-repeat'].includes(mode)) throw new Error('Invalid texture address mode')
      sampler[key] = mode
    }
    for (const key of ['minFilter','magFilter','mipmapFilter'] as const) {
      const mode = source.sampler[key]
      if (mode === undefined) continue
      if (mode !== 'linear' && mode !== 'nearest') throw new Error('Invalid texture filter')
      sampler[key] = mode
    }
    if (source.sampler.lodMaxClamp !== undefined) {
      if (!Number.isFinite(source.sampler.lodMaxClamp) || source.sampler.lodMaxClamp < 0) throw new Error('Invalid texture LOD')
      sampler.lodMaxClamp = source.sampler.lodMaxClamp
    }
    value.sampler = sampler
  }
  return value
}

function parseMtl(text: string, file: File, warnings: Set<string>) {
  const materials = new Map<string, ImportedMaterial>()
  let current: ImportedMaterial | undefined
  for (const [index, raw] of text.split(/\r\n?|\n/).entries()) {
    // A baked glTF normal texture retains its top-left tangent-space Y convention.
    const comment = raw.trim()
    if (current && comment === '# caustic-normal-y-sign -1') current.material.normalYSign = -1
    if (current && comment === '# caustic-double-sided 1') current.material.doubleSided = true
    if (current && /^# caustic-vertex-color [01]$/.test(comment)) current.material.vertexColor = comment.endsWith('1')
    if (current && comment.startsWith('# caustic-vertex-wetness ')) {
      const fields = comment.slice('# caustic-vertex-wetness '.length).trim().split(/\s+/).map(Number)
      const [saturatedRed, saturatedRoughness] = fields
      if (fields.length === 2 && saturatedRed >= 0 && saturatedRed < 1 && saturatedRoughness >= 0 && saturatedRoughness <= 1) {
        current.material.vertexWetness = { saturatedRed, saturatedRoughness }
      } else warnings.add(file.name + ' line ' + (index + 1) + ': invalid vertex wetness settings.')
    }
    if (current && comment.startsWith('# caustic-occlusion-strength ')) {
      const strength = Number(comment.slice('# caustic-occlusion-strength '.length))
      if (Number.isFinite(strength) && strength >= 0 && strength <= 1) current.material.occlusionStrength = strength
    }
    if (current && comment.startsWith('# caustic-texture-sampling ')) {
      const match = /^# caustic-texture-sampling (\S+) (.+)$/.exec(comment)
      const slot = match?.[1] as SceneMapSlot
      try {
        if (!samplingSlots.has(slot)) throw new Error('Invalid texture sampling slot')
        const sampling = current.material.sampling ??= {}
        sampling[slot] = parseSampling(match![2])
      } catch (error) { warnings.add(file.name + ' line ' + (index + 1) + ': ' + (error as Error).message + '.') }
    }
    const line = raw.split('#', 1)[0].trim()
    const [, op, value = ''] = /^(\S+)(?:\s+(.*))?$/.exec(line) ?? []
    const location = file.name + ' line ' + (index + 1) + ': '
    if (op === 'newmtl') {
      current = { material: { color: [1, 1, 1] }, maps: {} }
      materials.set(unquote(value), current)
    } else if (current && op === 'Kd') {
      const rgb = tokens(value).map(Number)
      if (rgb.length === 3 && rgb.every(v => Number.isFinite(v) && v >= 0 && v <= 1)) {
        current.material.color = [rgb[0], rgb[1], rgb[2]]
      } else warnings.add(location + 'invalid RGB Kd.')
    } else if (current && op === 'Ke') {
      const rgb = tokens(value).map(Number)
      if (rgb.length === 3 && rgb.every(v => Number.isFinite(v) && v >= 0)) {
        current.material.emissive = [rgb[0], rgb[1], rgb[2]]
      } else warnings.add(location + 'invalid emissive color Ke.')
    } else if (current && op === 'Ns') {
      const exponent = Number(value)
      if (value && Number.isFinite(exponent) && exponent >= 0) current.shininess = exponent
      else warnings.add(location + 'invalid specular exponent Ns.')
    } else if (current && (op === 'Pr' || op === 'Pm')) {
      const factor = Number(value)
      if (value && Number.isFinite(factor) && factor >= 0 && factor <= 1) {
        current.material[op === 'Pr' ? 'roughness' : 'metallic'] = factor
      } else warnings.add(location + 'invalid ' + op + ' factor.')
    } else if (current && (op === 'map_Kd' || op === 'map_Pr' || op === 'map_Pm' || op === 'norm' || op === 'map_AO' || op === 'map_Ke')) {
      try { current.maps[op] = parseMap(value, op) }
      catch (error) { warnings.add(location + (error as Error).message + '.') }
    }
  }
  return materials
}

function bindMaps({ material, maps, shininess }: ImportedMaterial, library: File,
  resolve: ReturnType<typeof localFiles>, warnings: Set<string>) {
  function image(ref: TextureMap, label: string) {
    const file = resolve(library, ref.path)
    if (!file) warnings.add('Missing ' + label + ' image: ' + ref.path + '.')
    return file
  }
  const base = maps.map_Kd, rough = maps.map_Pr, metal = maps.map_Pm, normal = maps.norm
  // Approximate a legacy Blinn-Phong lobe width. GGX uses alpha = roughness².
  if (material.roughness === undefined && !rough && shininess !== undefined) {
    material.roughness = (2 / (shininess + 2)) ** 0.25
  }
  if (base) {
    material.map = resolve(library, base.path)
    if (!material.map) warnings.add('Missing Base Color image: ' + base.path + '. Using Kd.')
    material.uvTransform = base.transform
    material.clamp = base.clamp
  }
  if (normal) {
    material.normalMap = image(normal, 'normal')
    material.normalUVTransform = normal.transform
    material.normalClamp = normal.clamp
    material.normalScale = normal.strength ?? 1
  }
  const ao = maps.map_AO, glow = maps.map_Ke
  if (ao && (!ao.channel || ao.channel === 'r')) {
    material.occlusionMap = image(ao, 'ambient occlusion')
    material.occlusionUVTransform = ao.transform
    material.occlusionClamp = ao.clamp
  } else if (ao) warnings.add('Unsupported occlusion channel for ' + ao.path + '.')
  if (glow) {
    material.emissiveMap = image(glow, 'emissive')
    material.emissiveUVTransform = glow.transform
    material.emissiveClamp = glow.clamp
    material.emissive ??= [1,1,1]
  }
  if (rough?.channel === 'g' && metal?.channel === 'b' && rough.path === metal.path &&
      rough.clamp === metal.clamp && rough.transform.every((n, i) => n === metal.transform[i])) {
    material.metallicRoughnessMap = image(rough, 'metallic-roughness')
    material.metallicRoughnessUVTransform = rough.transform
    material.metallicRoughnessClamp = rough.clamp
    if (material.metallicRoughnessMap) {
      material.roughness ??= 1
      material.metallic ??= 1
    }
  } else {
    if (rough && (!rough.channel || rough.channel === 'r')) {
      material.roughnessMap = image(rough, 'roughness')
      material.roughnessUVTransform = rough.transform
      material.roughnessClamp = rough.clamp
      if (material.roughnessMap) material.roughness ??= 1
    } else if (rough) warnings.add('Unsupported roughness channel for ' + rough.path + '.')
    if (metal && (!metal.channel || metal.channel === 'r')) {
      material.metallicMap = image(metal, 'metallic')
      material.metallicUVTransform = metal.transform
      material.metallicClamp = metal.clamp
      if (material.metallicMap) material.metallic ??= 1
    } else if (metal) warnings.add('Unsupported metallic channel for ' + metal.path + '.')
  }
}

export async function readObjAsset(obj: File, files: File[]) {
  const mesh = await readObjMesh(obj)
  const resolve = localFiles(files), warnings = new Set<string>()
  const used = new Set(mesh.draws?.map(draw => draw.material))
  const materials = new Map<string, SceneMaterial>()
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
    for (const [name, record] of parsed) {
      if (!used.has(name)) continue
      bindMaps(record, file, resolve, warnings)
      if (mesh.vertexColorMaterials.has(name)) record.material.vertexColor ??= true
      materials.set(name, record.material)
    }
  }
  for (const draw of mesh.draws ?? []) {
    if (!draw.material) continue
    const m = materials.get(draw.material)
    if (!m) {
      warnings.add('Missing material: ' + draw.material + '. Using gray.')
      continue
    }
    if (m.doubleSided) draw.doubleSided = true
    if (!(draw.uvMask ?? Number(!!draw.hasUV)) &&
        (m.map || m.normalMap || m.roughnessMap || m.metallicMap || m.metallicRoughnessMap || m.occlusionMap || m.emissiveMap)) {
      warnings.add(draw.material + ': faces without complete UVs use material factors.')
    }
  }
  return { mesh, materials, warnings }
}
