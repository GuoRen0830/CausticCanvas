import type { DesignView } from '../src/optics/design-view'
import type { ReceiverPlane } from '../src/contracts'

export function strokeWeights({ width, height }: ReceiverPlane): Float32Array {
  return Float32Array.from({ length: width * height }, (_, i) => {
    const x = (i % width + 0.5) * 256 / width - 70, y = (Math.floor(i / width) + 0.5) * 256 / height - 140
    const t = Math.max(0, Math.min(1, (94 * x + 2.8 * y) / (94 ** 2 + 2.8 ** 2)))
    return Math.max(0, Math.min(1, 6.9 - Math.hypot(x - 94 * t, y - 2.8 * t)))
  })
}

// A face like the ones users draw: ring, two eye dots and a smile arc, laid
// out on a 256 x 256 grid and stretched to the receiver. The eyes are small
// features that blurry solutions lose first.
export const faceEyes = [[105, 105], [160, 105]] as const
function faceCoordinates(i: number, { width, height }: ReceiverPlane) {
  return [(i % width + 0.5) * 256 / width, (Math.floor(i / width) + 0.5) * 256 / height]
}
export function faceWeights(receiver: ReceiverPlane): Float32Array {
  return Float32Array.from({ length: receiver.width * receiver.height }, (_, i) => {
    const [x, y] = faceCoordinates(i, receiver)
    const ring = Math.abs(Math.hypot(x - 130, y - 130) - 80) - 7
    const eye = Math.min(...faceEyes.map(([ex, ey]) => Math.hypot(x - ex, y - ey) - 8))
    const r = Math.hypot(x - 130, y - 125), angle = Math.atan2(y - 125, x - 130)
    const mouth = angle > 0.5 && angle < Math.PI - 0.5 ? Math.abs(r - 45) - 6 : Infinity
    return Math.max(0, Math.min(1, 0.5 - Math.min(ring, eye, mouth)))
  })
}

// Fraction of the target power the eyes were asked for that actually lands
// within 3 px of them, and the mean background-to-target irradiance ratio.
export function faceMetrics(image: Float32Array, weights: Float32Array, view: DesignView, receiver: ReceiverPlane) {
  let eyeTarget = 0, eyeReceived = 0, inside = 0, insideArea = 0, outside = 0, outsideArea = 0
  for (let i = 0; i < weights.length; i++) {
    const [x, y] = faceCoordinates(i, receiver)
    const area = view.surfaceAreas[i]
    const power = image[i] * area
    const d = Math.min(...faceEyes.map(([ex, ey]) => Math.hypot(x - ex, y - ey)))
    if (d < 11) eyeTarget += weights[i] * area
    if (d < 11) eyeReceived += power
    if (weights[i] >= 0.5) { inside += power; insideArea += area }
    else if (area > 0) { outside += power; outsideArea += area }
  }
  let targetPower = 0, weightedArea = 0
  for (let i = 0; i < weights.length; i++) weightedArea += weights[i] * view.surfaceAreas[i]
  for (let i = 0; i < weights.length; i++) targetPower += image[i] * view.surfaceAreas[i]
  // Eye share of the total target versus eye share of the received power.
  const eyeRecall = (eyeReceived / Math.max(targetPower, 1e-12)) / (eyeTarget / weightedArea)
  const haze = (outside / Math.max(outsideArea, 1e-12)) / (inside / Math.max(insideArea, 1e-12))
  return { eyeRecall, haze }
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
