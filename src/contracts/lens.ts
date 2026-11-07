export interface LensState {
  // Vertex counts along each axis, including boundaries.
  columns: number
  rows: number

  // Aperture dimensions in meters.
  sizeX: number
  sizeY: number

  // Exit-surface z in meters; the entry plane is z = 0.
  // Row-major: x increases first, then y.
  heights: Float32Array

  // Increment whenever the geometry changes.
  version: number
}