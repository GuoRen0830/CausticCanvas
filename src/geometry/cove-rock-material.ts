import type { readPreviewGltf } from './preview-gltf'

type RockAsset = Awaited<ReturnType<typeof readPreviewGltf>>

// The secondary atlas follows this scan's own UV charts. Keep the original
// geometry and the source photograph whenever its download succeeds.
export async function supplyCoveRockColor(asset: RockAsset): Promise<RockAsset> {
  const original = asset.materials.get('0')
  if (!original || original.map) return asset
  const failure = [...asset.warnings].find(message => /^Could not load coast_land_rocks_02 base color coast_land_rocks_02_diff(?: \([\s\S]*\))?\. Using the material color\.$/.test(message))
  if (!failure) return asset

  const { default: dataUrl } = await import('../../scene/smugglers-cove/rock-shore-color.jpg?inline')
  const bytes = Uint8Array.from(atob(dataUrl.slice(dataUrl.indexOf(',') + 1)), c => c.charCodeAt(0))
  const map = new File([bytes], 'rock-shore-color.jpg', { type: 'image/jpeg' })
  const materials = new Map(asset.materials)
  materials.set('0', { ...original, map })
  const warnings = new Set(asset.warnings)
  warnings.delete(failure)
  warnings.add('Using the project-authored shoreline color atlas for coastal rocks because the original texture is unavailable.')
  return { mesh: asset.mesh, materials, warnings }
}
