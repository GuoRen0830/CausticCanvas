import type { LensState } from '../contracts/lens'

interface FlatLensOptions {
  columns: number
  rows: number
  sizeX: number
  sizeY: number
  thickness: number
}

export function createFlatLens(options: FlatLensOptions): LensState {
  const { columns, rows, sizeX, sizeY, thickness } = options

  // A surface grid needs at least one cell along each axis.
  if (
    !Number.isInteger(columns) ||
    !Number.isInteger(rows) ||
    columns < 2 ||
    rows < 2
  ) {
    throw new Error('Lens grid requires at least 2 vertices per axis.')
  }

  // All physical dimensions use meters and must be positive.
  const dimensions = [sizeX, sizeY, thickness]
  if (dimensions.some((value) => !Number.isFinite(value) || value <= 0)) {
    throw new Error('Lens dimensions must be finite and positive.')
  }

  // A flat exit surface has the same z coordinate at every vertex.
  const heights = new Float32Array(columns * rows)
  heights.fill(thickness)

  return {
    columns,
    rows,
    sizeX,
    sizeY,
    heights,
    version: 0,
  }
}