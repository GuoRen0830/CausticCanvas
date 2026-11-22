// CPU diagnostics for a lens height field read back from the GPU.
// Definitions match the regularizers in shaders/optimizer.wgsl.

export interface LensShapeMetrics {
  // RMS of the discrete Laplacian over interior vertices, in 1/m.
  // Independent of grid spacing for a smooth surface.
  laplacianRms: number
  // RMS of laplacian * mean spacing: the dimensionless curvature penalized by
  // smoothnessWeight. Scales with spacing, so compare it only on one grid.
  curvatureRms: number
  // RMS of the Laplacian minus the mean Laplacian of its four neighbors, in 1/m.
  // A focusing lens needs large but slowly varying curvature; this high-pass
  // part isolates vertex-scale ripples that laplacianRms mixes with focusing.
  rippleRms: number
  // 99th percentile of the dimensionless curvature magnitude.
  curvatureP99: number
  // Fraction of triangles whose slope exceeds maxSlope, and the largest slope.
  slopeExcessFraction: number
  maxSlope: number
  // Height range in meters.
  minHeight: number
  maxHeight: number
}

// With vertex motion (offsets: row-major x/y displacement in meters), face
// slopes come from the displaced triangles, and the curvature terms from the
// surface resampled onto the uniform grid, so they describe the lens shape
// rather than how its vertices are spread.
export function lensShapeMetrics(
  vertexHeights: Float32Array,
  grid: { columns: number, rows: number, sizeX: number, sizeY: number },
  maxSurfaceSlope: number,
  offsets?: Float32Array,
): LensShapeMetrics {
  const { columns, rows } = grid
  const dx = grid.sizeX / (columns - 1), dy = grid.sizeY / (rows - 1)
  const heights = offsets ? resampleToGrid(vertexHeights, offsets, grid) : vertexHeights
  const meanSpacing = 0.5 * (dx + dy)
  const curvatures: number[] = []
  const laplacians = new Float64Array(columns * rows)
  let laplacianSq = 0
  for (let row = 1; row + 1 < rows; row++) {
    for (let column = 1; column + 1 < columns; column++) {
      const i = row * columns + column, h = heights[i]
      const laplacian = (heights[i - 1] + heights[i + 1] - 2 * h) / (dx * dx) +
        (heights[i - columns] + heights[i + columns] - 2 * h) / (dy * dy)
      laplacians[i] = laplacian
      laplacianSq += laplacian * laplacian
      curvatures.push(Math.abs(laplacian * meanSpacing))
    }
  }
  const interior = Math.max(1, curvatures.length)
  let rippleSq = 0, rippleCount = 0
  for (let row = 2; row + 2 < rows; row++) {
    for (let column = 2; column + 2 < columns; column++) {
      const i = row * columns + column
      const ripple = laplacians[i] - 0.25 * (laplacians[i - 1] + laplacians[i + 1] + laplacians[i - columns] + laplacians[i + columns])
      rippleSq += ripple * ripple
      rippleCount++
    }
  }
  curvatures.sort((a, b) => a - b)
  let excess = 0, maxSlope = 0
  const slope = (gx: number, gy: number) => {
    const value = Math.hypot(gx, gy)
    if (value > maxSurfaceSlope) excess++
    maxSlope = Math.max(maxSlope, value)
  }
  for (let row = 0; row + 1 < rows; row++) {
    for (let column = 0; column + 1 < columns; column++) {
      const a = row * columns + column
      if (offsets) {
        for (const face of cellFaces(a, columns)) slope(...faceGradient(face, vertexHeights, offsets, columns, dx, dy))
        continue
      }
      const ha = heights[a], hb = heights[a + 1], hc = heights[a + columns], hd = heights[a + columns + 1]
      // Faces follow buildLensIndices: (a, b, c) and (b, d, c).
      slope((hb - ha) / dx, (hc - ha) / dy)
      slope((hd - hc) / dx, (hd - hb) / dy)
    }
  }
  let minHeight = Infinity, maxHeight = -Infinity
  for (const h of vertexHeights) { minHeight = Math.min(minHeight, h); maxHeight = Math.max(maxHeight, h) }
  return {
    laplacianRms: Math.sqrt(laplacianSq / interior),
    rippleRms: Math.sqrt(rippleSq / Math.max(1, rippleCount)),
    curvatureRms: Math.sqrt(curvatures.reduce((s, q) => s + q * q, 0) / interior),
    curvatureP99: curvatures[Math.floor(0.99 * (curvatures.length - 1))] ?? 0,
    slopeExcessFraction: excess / (2 * (columns - 1) * (rows - 1)),
    maxSlope,
    minHeight,
    maxHeight,
  }
}

// The two faces of the cell whose lower-left vertex is a (buildLensIndices order).
function cellFaces(a: number, columns: number): [number, number, number][] {
  const b = a + 1, c = a + columns, d = c + 1
  return [[a, b, c], [b, d, c]]
}

function displaced(vertex: number, offsets: Float32Array, columns: number, dx: number, dy: number) {
  return [(vertex % columns) * dx + offsets[2 * vertex], Math.floor(vertex / columns) * dy + offsets[2 * vertex + 1]]
}

// Height gradient of the plane through a displaced triangle.
function faceGradient(face: number[], heights: Float32Array, offsets: Float32Array,
  columns: number, dx: number, dy: number): [number, number] {
  const [p, q, r] = face.map(v => displaced(v, offsets, columns, dx, dy))
  const e1x = q[0] - p[0], e1y = q[1] - p[1], e2x = r[0] - p[0], e2y = r[1] - p[1]
  const h1 = heights[face[1]] - heights[face[0]], h2 = heights[face[2]] - heights[face[0]]
  const determinant = e1x * e2y - e1y * e2x
  if (!(Math.abs(determinant) > 0)) return [0, 0]
  return [(h1 * e2y - h2 * e1y) / determinant, (e1x * h2 - e2x * h1) / determinant]
}

// Piecewise-linear lens surface of the displaced mesh, sampled at the uniform
// grid points. Border vertices only slide along the edge, so the displaced
// mesh covers the aperture; a point no face claims keeps its vertex height.
function resampleToGrid(heights: Float32Array, offsets: Float32Array,
  grid: { columns: number, rows: number, sizeX: number, sizeY: number }): Float32Array {
  const { columns, rows } = grid
  const dx = grid.sizeX / (columns - 1), dy = grid.sizeY / (rows - 1)
  const out = Float32Array.from(heights)
  const tolerance = 1e-6
  for (let row = 0; row + 1 < rows; row++) {
    for (let column = 0; column + 1 < columns; column++) {
      for (const face of cellFaces(row * columns + column, columns)) {
        const [p, q, r] = face.map(v => displaced(v, offsets, columns, dx, dy))
        const determinant = (q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1])
        if (!(Math.abs(determinant) > 0)) continue
        const i0 = Math.max(0, Math.ceil(Math.min(p[0], q[0], r[0]) / dx - tolerance))
        const i1 = Math.min(columns - 1, Math.floor(Math.max(p[0], q[0], r[0]) / dx + tolerance))
        const j0 = Math.max(0, Math.ceil(Math.min(p[1], q[1], r[1]) / dy - tolerance))
        const j1 = Math.min(rows - 1, Math.floor(Math.max(p[1], q[1], r[1]) / dy + tolerance))
        for (let j = j0; j <= j1; j++) {
          for (let i = i0; i <= i1; i++) {
            const x = i * dx, y = j * dy
            const u = ((q[0] - x) * (r[1] - y) - (r[0] - x) * (q[1] - y)) / determinant
            const v = ((r[0] - x) * (p[1] - y) - (p[0] - x) * (r[1] - y)) / determinant
            const w = 1 - u - v
            if (u < -tolerance || v < -tolerance || w < -tolerance) continue
            out[j * columns + i] = u * heights[face[0]] + v * heights[face[1]] + w * heights[face[2]]
          }
        }
      }
    }
  }
  return out
}
