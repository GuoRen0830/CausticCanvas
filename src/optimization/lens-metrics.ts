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

export function lensShapeMetrics(
  heights: Float32Array,
  grid: { columns: number, rows: number, sizeX: number, sizeY: number },
  maxSurfaceSlope: number,
): LensShapeMetrics {
  const { columns, rows } = grid
  const dx = grid.sizeX / (columns - 1), dy = grid.sizeY / (rows - 1)
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
      const ha = heights[a], hb = heights[a + 1], hc = heights[a + columns], hd = heights[a + columns + 1]
      // Faces follow buildLensIndices: (a, b, c) and (b, d, c).
      slope((hb - ha) / dx, (hc - ha) / dy)
      slope((hd - hc) / dx, (hd - hb) / dy)
    }
  }
  let minHeight = Infinity, maxHeight = -Infinity
  for (const h of heights) { minHeight = Math.min(minHeight, h); maxHeight = Math.max(maxHeight, h) }
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
