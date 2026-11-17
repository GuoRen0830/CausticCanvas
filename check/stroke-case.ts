import type { DesignView } from '../src/optics/design-view'
import type { ReceiverPlane } from '../src/contracts'

export function strokeWeights({ width, height }: ReceiverPlane): Float32Array {
  return Float32Array.from({ length: width * height }, (_, i) => {
    const x = (i % width + 0.5) * 256 / width - 70, y = (Math.floor(i / width) + 0.5) * 256 / height - 140
    const t = Math.max(0, Math.min(1, (94 * x + 2.8 * y) / (94 ** 2 + 2.8 ** 2)))
    return Math.max(0, Math.min(1, 6.9 - Math.hypot(x - 94 * t, y - 2.8 * t)))
  })
}

// Power (W) received on visible model geometry and over the whole light view,
// and the fraction of each inside the brush (weight >= 0.5).
export function strokeFractions(image: Float32Array, weights: Float32Array, view: DesignView) {
  let total = 0, painted = 0, model = 0, paintedModel = 0
  for (let i = 0; i < weights.length; i++) {
    const triangle = view.samples[2 * i]
    const object = triangle ? view.bvh.triangles[(triangle - 1) * 12 + 3] : 0
    const power = image[i] * view.surfaceAreas[i]
    total += power
    if (weights[i] >= 0.5) painted += power
    if (object >= 2) {
      model += power
      if (weights[i] >= 0.5) paintedModel += power
    }
  }
  return { model, total, modelFraction: model > 0 ? paintedModel / model : 0, sceneFraction: total > 0 ? painted / total : 0 }
}
