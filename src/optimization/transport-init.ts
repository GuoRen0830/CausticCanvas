// Global initial guess for the lens: send every part of the lens to the target
// at once, then turn that assignment into exit-surface heights.
//
// Gradient descent on heights only moves light that already lands near the
// target; light from distant lens regions never feels a pull and stays spread
// out as faint filaments. Here each lens grid vertex is a particle carrying
// equal power. Sliced transport (iterative 1D optimal transport along many
// directions) moves the particles until they are distributed like the target.
// Each vertex then knows the receiver point it should light, which fixes the
// surface slope by Snell's law, and the slopes are integrated into heights by
// least squares; the gradient optimizer refines the rest.

export interface TransportInitInput {
  // Target power per receiver pixel, row 0 at -Y; any positive scale.
  target: Float32Array
  width: number
  height: number
  receiverSizeX: number
  receiverSizeY: number
  receiverZ: number
  // Optional z of the surface seen in each pixel (0 = use receiverZ), so
  // light aimed at a raised object stops there instead of at the floor.
  depths?: Float32Array
  // Output lens grid.
  columns: number
  rows: number
  lensSizeX: number
  lensSizeY: number
  // Lowest exit-surface z of the result, in meters.
  baseHeight: number
  // lensIOR / outsideIOR.
  eta: number
  // Slopes are clamped to this tangent to stay clear of total internal reflection.
  maxSlope: number
  // Sliced-transport passes (one direction each).
  iterations?: number
  // Seed for the target samples.
  seed?: number
}

// Small deterministic generator so a target always gives the same lens.
function random(seed: number) {
  let state = seed >>> 0 || 1
  return () => {
    state ^= state << 13
    state ^= state >>> 17
    state ^= state << 5
    return (state >>> 0) / 4294967296
  }
}

// Conjugate gradient for a symmetric positive semidefinite operator whose
// null space is the constant vector; the right-hand side is kept at zero mean.
function conjugateGradient(size: number, apply: (v: Float64Array, out: Float64Array) => void,
  rhs: Float64Array, x: Float64Array, maxIter: number, tol: number) {
  let mean = 0
  for (let k = 0; k < size; k++) mean += rhs[k]
  mean /= size
  const r = new Float64Array(size), p = new Float64Array(size), q = new Float64Array(size)
  apply(x, q)
  let rr = 0, bb = 0
  for (let k = 0; k < size; k++) {
    const b = rhs[k] - mean
    r[k] = b - q[k]; p[k] = r[k]; rr += r[k] * r[k]; bb += b * b
  }
  const stop = tol * tol * Math.max(bb, 1e-300)
  for (let it = 0; it < maxIter && rr > stop; it++) {
    apply(p, q)
    let pq = 0
    for (let k = 0; k < size; k++) pq += p[k] * q[k]
    if (!(pq > 0)) break
    const alpha = rr / pq
    let next = 0
    for (let k = 0; k < size; k++) { x[k] += alpha * p[k]; r[k] -= alpha * q[k]; next += r[k] * r[k] }
    const beta = next / rr
    rr = next
    for (let k = 0; k < size; k++) p[k] = r[k] + beta * p[k]
  }
}

// Receiver points (meters) that each lens vertex should light.
export function transportTargets(input: TransportInitInput): { x: Float64Array, y: Float64Array } {
  const { columns, rows, width, height } = input
  const count = columns * rows
  const next = random(input.seed ?? 1)

  // Target samples: systematic sampling of the cumulative pixel power,
  // jittered inside each pixel.
  const cumulative = new Float64Array(width * height)
  let total = 0
  for (let i = 0; i < cumulative.length; i++) {
    const value = input.target[i]
    if (value > 0 && Number.isFinite(value)) total += value
    cumulative[i] = total
  }
  if (!(total > 0)) throw new Error('Transport init needs a target with positive power.')
  const sampleX = new Float64Array(count), sampleY = new Float64Array(count)
  let pixel = 0
  for (let s = 0; s < count; s++) {
    const level = (s + next()) / count * total
    while (pixel < cumulative.length - 1 && cumulative[pixel] < level) pixel++
    const px = pixel % width, py = (pixel - px) / width
    sampleX[s] = ((px + next()) / width - 0.5) * input.receiverSizeX
    sampleY[s] = ((py + next()) / height - 0.5) * input.receiverSizeY
  }

  // Particles start where a flat lens sends them: straight down.
  const x = new Float64Array(count), y = new Float64Array(count)
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < columns; c++) {
      x[r * columns + c] = (c / (columns - 1) - 0.5) * input.lensSizeX
      y[r * columns + c] = (r / (rows - 1) - 0.5) * input.lensSizeY
    }
  }

  // Each pass: project both clouds on one direction, sort, and move every
  // particle along that direction to the sample of the same rank.
  const order = new Uint32Array(count), sampleOrder = new Uint32Array(count)
  const projection = new Float64Array(count), sampleProjection = new Float64Array(count)
  const iterations = input.iterations ?? 300
  for (let it = 0; it < iterations; it++) {
    // Golden-angle directions cover the circle evenly.
    const angle = it * 2.399963229728653
    const dx = Math.cos(angle), dy = Math.sin(angle)
    for (let k = 0; k < count; k++) {
      projection[k] = x[k] * dx + y[k] * dy
      sampleProjection[k] = sampleX[k] * dx + sampleY[k] * dy
      order[k] = k
      sampleOrder[k] = k
    }
    order.sort((a, b) => projection[a] - projection[b])
    sampleOrder.sort((a, b) => sampleProjection[a] - sampleProjection[b])
    for (let k = 0; k < count; k++) {
      const particle = order[k]
      const shift = sampleProjection[sampleOrder[k]] - projection[particle]
      x[particle] += shift * dx
      y[particle] += shift * dy
    }
  }
  return { x, y }
}

export function transportHeights(input: TransportInitInput): Float32Array {
  const { columns, rows } = input
  const { x: targetX, y: targetY } = transportTargets(input)

  // Exit direction toward the receiver point, then the slope that refracts
  // the vertical ray onto it: grad h = t_xy / (eta - t_z).
  const slopeX = new Float64Array(columns * rows), slopeY = new Float64Array(columns * rows)
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < columns; c++) {
      const k = r * columns + c
      const px = (c / (columns - 1) - 0.5) * input.lensSizeX
      const py = (r / (rows - 1) - 0.5) * input.lensSizeY
      const pixelX = Math.min(input.width - 1, Math.max(0, Math.floor((targetX[k] / input.receiverSizeX + 0.5) * input.width)))
      const pixelY = Math.min(input.height - 1, Math.max(0, Math.floor((targetY[k] / input.receiverSizeY + 0.5) * input.height)))
      const depth = input.depths?.[pixelY * input.width + pixelX] || input.receiverZ
      const distance = depth - input.baseHeight
      const dx = targetX[k] - px, dy = targetY[k] - py, length = Math.hypot(dx, dy, distance)
      const tx = dx / length, ty = dy / length, tz = distance / length
      let gx = tx / (input.eta - tz), gy = ty / (input.eta - tz)
      const slope = Math.hypot(gx, gy)
      if (slope > input.maxSlope) { gx *= input.maxSlope / slope; gy *= input.maxSlope / slope }
      slopeX[k] = gx
      slopeY[k] = gy
    }
  }

  // Least-squares integration: each edge difference matches the averaged
  // slopes; the normal equations are -L(h) = e(k-1, k) - e(k, k+1) per axis.
  const dx = input.lensSizeX / (columns - 1), dy = input.lensSizeY / (rows - 1)
  const rhs = new Float64Array(columns * rows)
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < columns; c++) {
      const k = r * columns + c
      if (c + 1 < columns) {
        const e = 0.5 * (slopeX[k] + slopeX[k + 1]) * dx
        rhs[k] -= e
        rhs[k + 1] += e
      }
      if (r + 1 < rows) {
        const e = 0.5 * (slopeY[k] + slopeY[k + columns]) * dy
        rhs[k] -= e
        rhs[k + columns] += e
      }
    }
  }
  // -L with unit spacing (positive semidefinite, Neumann boundaries).
  const negativeLaplacian = (v: Float64Array, out: Float64Array) => {
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < columns; c++) {
        const k = r * columns + c, center = v[k]
        let sum = 0
        if (c > 0) sum += center - v[k - 1]
        if (c + 1 < columns) sum += center - v[k + 1]
        if (r > 0) sum += center - v[k - columns]
        if (r + 1 < rows) sum += center - v[k + columns]
        out[k] = sum
      }
    }
  }
  const surface = new Float64Array(columns * rows)
  conjugateGradient(columns * rows, negativeLaplacian, rhs, surface, 40 * (columns + rows), 1e-8)
  let lowest = Infinity
  for (const value of surface) lowest = Math.min(lowest, value)
  const heights = new Float32Array(columns * rows)
  for (let k = 0; k < heights.length; k++) heights[k] = surface[k] - lowest + input.baseHeight
  return heights
}
