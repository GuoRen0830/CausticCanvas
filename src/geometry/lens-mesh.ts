import type { LensState } from '../contracts/lens'

export function buildLensPositions(lens: LensState): Float32Array {
  const { columns, rows, sizeX, sizeY, heights } = lens

  if (heights.length !== columns * rows) {
    throw new Error('Lens height count does not match the grid.')
  }

  // Adjacent vertices are evenly spaced in the lens-local XY plane.
  const spacingX = sizeX / (columns - 1)
  const spacingY = sizeY / (rows - 1)

  const positions = new Float32Array(columns * rows * 3)

  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const vertex = row * columns + column
      const offset = vertex * 3

      // Preserve physical scale and use the stored exit-surface height.
      positions[offset] = -sizeX / 2 + column * spacingX
      positions[offset + 1] = -sizeY / 2 + row * spacingY
      positions[offset + 2] = heights[vertex]
    }
  }

  return positions
}

export function buildLensIndices(lens: LensState): Uint32Array {
  const { columns, rows } = lens
  const cellCount = (columns - 1) * (rows - 1)

  // Each cell contains two triangles, with three indices each.
  const indices = new Uint32Array(cellCount * 6)
  let offset = 0

  for (let row = 0; row < rows - 1; row++) {
    for (let column = 0; column < columns - 1; column++) {
      const a = row * columns + column
      const b = a + 1
      const c = a + columns
      const d = c + 1

      // Use the b-c diagonal and counterclockwise winding toward +Z.
      indices[offset++] = a
      indices[offset++] = b
      indices[offset++] = c

      indices[offset++] = b
      indices[offset++] = d
      indices[offset++] = c
    }
  }

  return indices
}