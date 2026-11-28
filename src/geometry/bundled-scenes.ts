import skateCredit from '../../scene/greer-skatepark/CREDITS.md?url&no-inline'
import fairCredit from '../../scene/fairground/CREDITS.md?url&no-inline'
import coveCredit from '../../scene/smugglers-cove/CREDITS.md?url&no-inline'
import { readObjAsset } from './obj-materials'
import type { CoveSourceManifest, CoveSourceState } from './cove-source-textures'
import type { SceneMaterial } from '../rendering/scene-mesh'

export const bundledScenes = {
  'greer-skatepark': { name: 'Greer Park Skating Bowl', attribution: 'Model by scott.anderson · CC BY 4.0', credit: skateCredit },
  fairground: { name: 'Fairground', attribution: 'Model by Brian Trepanier · CC BY 4.0', credit: fairCredit },
  'smugglers-cove': {
    name: 'Smuggler’s Cove · Tidal Landing', attribution: 'Poly Haven assets · CC0', credit: coveCredit,
    roomCamera: { azimuth: -0.43, elevation: 0.65 },
  },
} as const
export type BundledScene = keyof typeof bundledScenes
export const defaultScene: BundledScene = 'smugglers-cove'

// Vite emits local asset URLs with the deployment base and cache hashes.
const assets = import.meta.glob<string>(
  [
    '../../scene/{greer-skatepark,fairground}/**/*.{obj,mtl,png,jpg,jpeg,webp}',
    '../../scene/smugglers-cove/preview/**/*.{obj,mtl,png,jpg,jpeg,webp,avif,bmp}',
    '!../../scene/smugglers-cove/preview/originals/**',
  ],
  { eager: true, query: '?url&no-inline', import: 'default' },
)

const coveManifests = import.meta.glob<CoveSourceManifest>(
  '../../scene/smugglers-cove/preview/source-textures*.json',
  { eager: true, import: 'default' },
)
const originalAssets = import.meta.glob<string>(
  '../../scene/smugglers-cove/preview/originals/*.{jpg,jpeg,png,webp}',
  { eager: true, query: '?url&no-inline', import: 'default' },
)
const coveOriginals = new Map(Object.entries(originalAssets).map(([path, url]) =>
  [path.split('/').at(-1)!.split('.')[0].toLowerCase(), url]))

async function withSourceTextures<T extends { materials: Map<string, SceneMaterial>; warnings: Set<string> }>(
  asset: T, manifest?: CoveSourceManifest, signal?: AbortSignal,
): Promise<T & { sourceTextures: CoveSourceState | undefined }> {
  if (manifest) {
    try {
      const { loadCoveSourceTextures } = await import('./cove-source-textures')
      const recovered = await loadCoveSourceTextures(asset.materials, manifest, signal, coveOriginals)
      for (const warning of recovered.localWarnings) asset.warnings.add(warning)
      return { ...asset, materials: recovered.materials, sourceTextures: recovered.sourceTextures }
    } catch (error) {
      signal?.throwIfAborted()
      asset.warnings.add('Could not read the source texture manifest: ' + (error instanceof Error ? error.message : String(error)))
    }
  }
  return { ...asset, sourceTextures: undefined }
}

export async function readBundledScene(id: BundledScene, signal?: AbortSignal) {
  const root = '../../scene/' + id + '/'
  const entries = Object.entries(assets).filter(([path]) => path.startsWith(root))
  if (!entries.some(([path]) => path.endsWith('.obj'))) {
    if (id !== 'smugglers-cove') throw new Error('The model files are missing from scene/' + id + '/.')
    try {
      const { readCovePreview } = await import('./smugglers-cove-preview')
      const asset = await readCovePreview(signal)
      const materials: CoveSourceManifest['materials'] = {}
      for (const [name, material] of asset.materials) {
        if (material.recoverableMaps) materials[name] = material.recoverableMaps
      }
      return await withSourceTextures(asset, { version: 1, materials }, signal)
    } catch (error) {
      if (!signal?.aborted && error instanceof TypeError && /fetch|network|load failed/i.test(error.message)) {
        throw new Error('No local copy is installed. The first download needs a connection to Poly Haven.', { cause: error })
      }
      throw error
    }
  }
  const files = await Promise.all(entries.map(async ([path, url]) => {
    const response = await fetch(url, { signal })
    if (!response.ok) throw new Error('Could not load ' + path.slice(root.length))
    const blob = await response.blob()
    const file = new File([blob], path.split('/').at(-1)!, { type: blob.type })
    Object.defineProperty(file, 'webkitRelativePath', { value: path.slice(root.length) })
    return file
  }))
  const asset = await readObjAsset(files.find(file => file.name.endsWith('.obj'))!, files)
  return withSourceTextures(asset, id === 'smugglers-cove' ? Object.values(coveManifests)[0] : undefined, signal)
}
