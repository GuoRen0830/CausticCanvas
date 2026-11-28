import type { SceneMaterial } from '../rendering/scene-mesh'

const root = '../../scene/smugglers-cove/'
const urls = import.meta.glob<string>([
  '../../scene/smugglers-cove/detail/*.png',
  '../../scene/smugglers-cove/{chest-scan-normal,reef-scan-normal,quay-stone-roughness,tide-normal}.png',
], { eager: true, query: '?url&no-inline', import: 'default' })
export type ReadCoveDetail = (filename: string) => Promise<Blob>

// A build reads the same PNGs from disk. In the live source assembler Vite
// resolves these URLs for the deployment base; baked OBJ users read the MTL.
export function coveDetailFiles(signal?: AbortSignal, readFile?: ReadCoveDetail) {
  const files = new Map<string, Promise<File>>()
  async function read(filename: string) {
    signal?.throwIfAborted()
    let blob: Blob
    if (readFile) blob = await readFile(filename)
    else {
      const url = urls[root + filename]
      if (!url) throw new Error('Missing local harbor material image: ' + filename)
      const response = await fetch(url, { signal })
      if (!response.ok) throw new Error('Cannot read ' + filename + ' (HTTP ' + response.status + ').')
      blob = await response.blob()
    }
    return new File([blob], filename.split('/').at(-1)!, { type: 'image/png' })
  }
  return {
    get(filename: string) {
      let file = files.get(filename)
      if (!file) { file = read(filename); files.set(filename, file) }
      return file
    },
    reuse(file: File) { files.set(file.name, Promise.resolve(file)) },
  }
}
type Files = ReturnType<typeof coveDetailFiles>

// These names belong to the local mesh builders, before OBJ material renaming.
// Scanned meshes use a separate, source-material-specific match below.
const groups = {
  sand: ['dry', 'sediment'],
  water: ['water', 'rock pool'],
  grass: ['grass-olive', 'grass-sage', 'grass-straw', 'olive kelp', 'amber kelp'],
  stone: ['lime mortar', 'sandstone', 'weathered limestone', 'warm freestone', 'capstone',
    'tidal stone', 'armor stone', 'bedrock', 'wet strata', 'algal rock'],
  timber: ['dock oak', 'sawn cedar', 'ochre boards', 'harbor blue', 'salt-bleached slats',
    'lugger deck', 'lugger petrol', 'lugger lapstrake', 'lugger ivory stripe', 'varnished trim', 'ash oars',
    'deep teal paint', 'faded teal paint', 'plank edges', 'cream band', 'interior oak', 'interior cedar', 'steamed willow'],
  fittings: ['wrought iron', 'aged zinc', 'weathered zinc', 'zinc flashing', 'verdigris', 'lens brass'],
  glass: ['lamp glass', 'lantern amber'],
  rope: ['hemp rope', 'sail bolt rope', 'tanbark seam', 'tarred shroud', 'tarred seine'],
  sailcloth: ['tanbark linen', 'sun-faded tanbark', 'reefed oxblood linen', 'lug sail corner doubler', 'flax jib', 'waxed flax',
    'pilot signal indigo', 'pilot signal cream'],
  linen: ['fish market indigo', 'fish market sailcloth', 'fish market madder'],
  'fired-clay': ['fish market terracotta', 'fish market fired clay'],
  'sea-star': ['ochre sea star', 'sea star ossicles'],
  enamel: ['beacon ivory', 'beacon oxblood', 'fish market enamel', 'fish market marigold', 'vermilion buoy paint', 'ivory buoy paint'],
  cork: ['oiled cork'],
  fish: ['fish market silver', 'fish market mackerel'],
  shell: ['limpet lip', 'limpet shell', 'limpet rib', 'tidepool mussel', 'tidepool nacre',
    'acorn barnacle', 'barnacle aperture'],
}
type Family = keyof typeof groups | 'pier-planks' | 'pier-poles' | 'reef' | 'chest'
const finishes = new Map<string, Family>()
for (const [family, names] of Object.entries(groups)) for (const name of names) finishes.set(name, family as Family)

async function addDetail(material: SceneMaterial, family: Family, files: Files) {
  if (!material.normalMap) {
    const scanBake = family === 'reef' || family === 'chest'
    const path = scanBake ? family + '-scan-normal.png' : family === 'water' ? 'tide-normal.png' : 'detail/' + family + '-normal.png'
    material.normalMap = await files.get(path)
    material.normalYSign = scanBake ? -1 : 1
    material.normalScale = family === 'timber' && (material.roughness ?? 1) < 0.6 ? 0.5 : 1
    material.normalClamp = material.clamp
  }
  if (!material.metallicRoughnessMap && !material.roughnessMap) {
    if (family === 'chest' || family === 'fittings') {
      material.metallicRoughnessMap = await files.get('detail/' + family + '-mr.png')
      // The chest atlas distinguishes its wood from the iron straps in one draw.
      // Harbor hardware uses its existing finish factors times the patina map.
      if (family === 'chest') { material.roughness = 1; material.metallic = 1 }
    } else {
      const path = family === 'stone' ? 'quay-stone-roughness.png' : 'detail/' + family + '-roughness.png'
      material.roughnessMap = await files.get(path)
      material.roughnessClamp = material.clamp
    }
  }
  if (family === 'pier-planks' && !material.metallicRoughnessMap && !material.metallicMap) {
    material.metallicMap = await files.get('detail/pier-planks-metallic.png')
    material.metallic = 1
  }
  if (family === 'sand' && !material.occlusionMap) material.occlusionMap = await files.get('detail/sand-ao.png')
}

export async function addScanDetail(material: SceneMaterial, sourceName: string | undefined, files: Files) {
  const scans: Record<string, Family> = {
    coast_land_rocks_02: 'reef', treasure_chest: 'chest',
    modular_wooden_pier_planks: 'pier-planks', modular_wooden_pier_poles: 'pier-poles',
  }
  const family = sourceName && scans[sourceName]
  if (family) await addDetail(material, family, files)
}

export async function addHarborDetail(materials: Map<string, SceneMaterial>, files: Files) {
  for (const material of materials.values()) {
    if (material.normalMap?.name === 'tide-normal.png') files.reuse(material.normalMap)
    if (material.roughnessMap?.name === 'quay-stone-roughness.png') files.reuse(material.roughnessMap)
  }
  await Promise.all(Array.from(materials, async ([id, source]) => {
    const name = id.slice(id.indexOf('/') + 1)
    const family = finishes.get(name)
    if (!family) return
    // HarborMesh factories share finish objects. A later source load can have
    // different local originals; keep each prepared scene's bindings separate.
    const material = { ...source }
    await addDetail(material, family, files)
    materials.set(id, material)
  }))
}
