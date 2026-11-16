import type { DesignView } from '../src/optics/design-view'

// The fixed horizontal stroke that crosses the box top and the floor in the
// 256 x 256 light view. Shared by the ?stroke acceptance gate and the study page.
export const strokeReceiver = { width: 256, height: 256, sizeX: 0.4, sizeY: 0.4, z: 0.5 }
export const strokeBox = { center: [0.05, 0.03, 0.3], size: [0.08, 0.08, 0.1] } as const

export function strokeWeights(): Float32Array {
  return Float32Array.from({ length: 256 * 256 }, (_, i) => {
    const x = i % 256 + 0.5 - 70, y = Math.floor(i / 256) + 0.5 - 140
    const t = Math.max(0, Math.min(1, (94 * x + 2.8 * y) / (94 ** 2 + 2.8 ** 2)))
    return Math.max(0, Math.min(1, 6.9 - Math.hypot(x - 94 * t, y - 2.8 * t)))
  })
}

// Power (W) received on the box top (object 2) and over the whole light view,
// and the fraction of each inside the brush (weight >= 0.5).
export function strokeFractions(image: Float32Array, weights: Float32Array, view: DesignView) {
  let total = 0, painted = 0, top = 0, paintedTop = 0
  for (let i = 0; i < weights.length; i++) {
    const triangle = view.samples[2 * i]
    const object = triangle ? view.bvh.triangles[(triangle - 1) * 12 + 3] : 0
    const power = image[i] * view.surfaceAreas[i]
    total += power
    if (weights[i] >= 0.5) painted += power
    if (object === 2) {
      top += power
      if (weights[i] >= 0.5) paintedTop += power
    }
  }
  return { top, total, topFraction: paintedTop / top, sceneFraction: painted / total }
}
