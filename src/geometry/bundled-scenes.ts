import skateCredit from '../../scene/greer-skatepark/CREDITS.md?url&no-inline'
import fairCredit from '../../scene/fairground/CREDITS.md?url&no-inline'
import { readObjAsset } from './obj-materials'

export const bundledScenes = {
  'greer-skatepark': { name: 'Greer Park Skating Bowl', author: 'scott.anderson', credit: skateCredit },
  fairground: { name: 'Fairground', author: 'Brian Trepanier', credit: fairCredit },
} as const
export type BundledScene = keyof typeof bundledScenes
export const defaultScene: BundledScene = 'greer-skatepark'

// Vite emits local asset URLs with the deployment base and cache hashes.
const assets = import.meta.glob<string>(
  '../../scene/{greer-skatepark,fairground}/**/*.{obj,mtl,png,jpg,jpeg,webp}',
  { eager: true, query: '?url&no-inline', import: 'default' },
)

export async function readBundledScene(id: BundledScene, signal?: AbortSignal) {
  const root = '../../scene/' + id + '/'
  const entries = Object.entries(assets).filter(([path]) => path.startsWith(root))
  if (!entries.some(([path]) => path.endsWith('.obj'))) {
    throw new Error('The model files are missing from scene/' + id + '/.')
  }
  const files = await Promise.all(entries.map(async ([path, url]) => {
    const response = await fetch(url, { signal })
    if (!response.ok) throw new Error('Could not load ' + path.slice(root.length))
    const blob = await response.blob()
    const file = new File([blob], path.split('/').at(-1)!, { type: blob.type })
    Object.defineProperty(file, 'webkitRelativePath', { value: path.slice(root.length) })
    return file
  }))
  return readObjAsset(files.find(file => file.name.endsWith('.obj'))!, files)
}
