import type { ForwardInput } from '../src/contracts'

type Point = [number, number]

function clip(points: Point[], axis: number, boundary: number, greater: boolean) {
  const output: Point[] = []
  if (!points.length) return output
  let a = points[points.length - 1]
  let aInside = greater ? a[axis] >= boundary : a[axis] <= boundary
  for (const b of points) {
    const bInside = greater ? b[axis] >= boundary : b[axis] <= boundary
    if (aInside !== bInside) {
      const t = (boundary - a[axis]) / (b[axis] - a[axis])
      output.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])])
    }
    if (bInside) output.push(b)
    a = b
    aInside = bInside
  }
  return output
}

function area(points: Point[]) {
  let sum = 0
  for (let i = 0; i < points.length; i++) {
    const a = points[i], b = points[(i + 1) % points.length]
    sum += a[0] * b[1] - b[0] * a[1]
  }
  return Math.abs(sum) / 2
}

export function reference(input: ForwardInput, heights: ArrayLike<number>) {
  const { lens, optics, receiver: r } = input
  const image = new Float64Array(r.width * r.height)
  const beams: number[] = []
  const dx = lens.sizeX / (lens.columns - 1)
  const dy = lens.sizeY / (lens.rows - 1)
  const flux = optics.incidentIrradiance * dx * dy / 2
  const eta = optics.lensIOR / optics.outsideIOR
  const pixelArea = r.sizeX * r.sizeY / (r.width * r.height)
  for (let y = 0; y < lens.rows - 1; y++) {
    for (let x = 0; x < lens.columns - 1; x++) {
      const a = y * lens.columns + x
      for (const ids of [[a, a + 1, a + lens.columns], [a + 1, a + lens.columns + 1, a + lens.columns]]) {
        const vertices = ids.map(i => [(i % lens.columns) * dx - lens.sizeX / 2,
          Math.floor(i / lens.columns) * dy - lens.sizeY / 2, heights[i]])
        const u = vertices[1].map((v, j) => v - vertices[0][j])
        const v = vertices[2].map((v, j) => v - vertices[0][j])
        const n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
        const length = Math.hypot(...n)
        n.forEach((_, j) => n[j] /= length)
        const k = 1 - eta * eta * (1 - n[2] * n[2])
        if (k <= 1e-8) { beams.push(0, 0, 1, 0); continue }
        const factor = Math.sqrt(k) - eta * n[2]
        const direction = [factor * n[0], factor * n[1], eta + factor * n[2]]
        if (direction[2] <= 1e-6 || vertices.some(v => v[2] >= r.z)) { beams.push(0, 0, 1, 0); continue }
        beams.push(...direction, flux)
        const q: Point[] = vertices.map(v => [
          (v[0] + (r.z - v[2]) * direction[0] / direction[2] + r.sizeX / 2) * r.width / r.sizeX,
          (v[1] + (r.z - v[2]) * direction[1] / direction[2] + r.sizeY / 2) * r.height / r.sizeY,
        ])
        const footprintArea = area(q)
        if (footprintArea <= 1e-8) continue
        for (let py = 0; py < r.height; py++) {
          for (let px = 0; px < r.width; px++) {
            const covered = clip(clip(clip(clip(q, 0, px, true), 0, px + 1, false), 1, py, true), 1, py + 1, false)
            image[py * r.width + px] += flux * area(covered) / (footprintArea * pixelArea)
          }
        }
      }
    }
  }
  return { image, beams }
}
