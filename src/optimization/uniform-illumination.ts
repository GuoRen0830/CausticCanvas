import type { LensState, OpticalConfig, ReceiverPlane } from '../contracts'
import type { DesignView } from '../optics/design-view'

type LampGrid = Pick<LensState, 'columns' | 'rows' | 'sizeX' | 'sizeY'> & { thickness: number }

// A diverging exit surface magnifies the uniformly lit aperture onto the fitted
// canvas. At a representative scene depth, the XY map is a constant scale and
// therefore conserves uniform irradiance across a flat receiver. The ordinary
// forward pass still traces this surface against the actual model and its shadows.
export function uniformIlluminationHeights(lens: LampGrid,
  optics: Pick<OpticalConfig, 'lensIOR' | 'outsideIOR'>, receiver: ReceiverPlane,
  view: Pick<DesignView, 'samples' | 'surfaceAreas'>): Float32Array<ArrayBuffer> {
  const { columns, rows, sizeX, sizeY, thickness } = lens
  const heights = new Float32Array(columns * rows)
  const eta = optics.lensIOR / optics.outsideIOR
  const magnification = Math.max(receiver.sizeX / sizeX, receiver.sizeY / sizeY)
  if (eta === 1 || magnification <= 1) return heights.fill(thickness)

  // Use the depth of surfaces visible from above. The floor between objects
  // participates just like roofs and bowls; empty pixels do not pull the plane away.
  const samples = new Float32Array(view.samples.buffer, view.samples.byteOffset, view.samples.length)
  let weightedDepth = 0, area = 0
  for (let i = 0; i < view.surfaceAreas.length; i++) {
    const depth = samples[2 * i + 1], weight = view.surfaceAreas[i]
    if (weight > 0 && depth > thickness) {
      weightedDepth += depth * weight
      area += weight
    }
  }
  const planeZ = area > 0 ? weightedDepth / area : receiver.z
  const radius = 0.5 * Math.hypot(sizeX, sizeY)
  const steps = Math.max(128, columns, rows)
  const dr = radius / steps
  const profile = new Float64Array(steps + 1)
  // Even for an unreachable magnification the exit surface must stay in
  // front of its receiver. For glass-to-air leave a margin below the TIR slope.
  const clearanceSlope = Math.max(0, (planeZ - thickness) / radius) * 0.99
  const slopeLimit = eta > 1 ? Math.min(clearanceSlope, 0.995 / Math.sqrt(eta * eta - 1)) : clearanceSlope
  // To send r to magnification*r, Snell gives dh/dr = tx / (eta - tz).
  // h changes the remaining travel distance, so integrate it along the radius.
  const shift = magnification - 1
  const slopeAt = (r: number, h: number) => {
    const x = shift * r, z = Math.max(planeZ - h, 0)
    const length = Math.hypot(x, z)
    if (!x || !length) return 0
    if (eta > 1) {
      if (eta * z <= length) return slopeLimit
      return Math.min(slopeLimit, x / (eta * length - z))
    }
    // A lower-index diverger is thick at its centre. At grazing transmission
    // a finite grid cannot represent a vertical tangent.
    if (z <= eta * length) return -slopeLimit
    return -Math.min(slopeLimit, x / (z - eta * length))
  }
  // With eta < 1 the minimum thickness occurs at the rim. Integrate towards
  // the centre starting from the prescribed rim thickness.
  const start = eta > 1 ? 0 : steps, direction = eta > 1 ? 1 : -1
  profile[start] = thickness
  for (let i = start; i + direction >= 0 && i + direction <= steps; i += direction) {
    const step = direction * dr, h = profile[i], r = i * dr
    const midpoint = h + 0.5 * step * slopeAt(r, h)
    profile[i + direction] = h + step * slopeAt(r + 0.5 * step, midpoint)
  }
  const dx = sizeX / (columns - 1), dy = sizeY / (rows - 1)
  for (let row = 0; row < rows; row++) {
    const y = row * dy - 0.5 * sizeY
    for (let col = 0; col < columns; col++) {
      const x = col * dx - 0.5 * sizeX
      const t = Math.min(steps, Math.sqrt(x * x + y * y) / dr)
      const i = Math.min(steps - 1, Math.floor(t))
      heights[row * columns + col] = profile[i] + (t - i) * (profile[i + 1] - profile[i])
    }
  }
  return heights
}
