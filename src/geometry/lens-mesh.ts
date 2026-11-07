import type { LensState } from '../contracts'

interface FlatLensOptions {
  columns: number
  rows: number
  sizeX: number
  sizeY: number
  thickness: number
}

export function createFlatLens(
  device: GPUDevice,
  options: FlatLensOptions,
): LensState {
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

  // Keep CPU heights only for the initial upload.
  const initialHeights = new Float32Array(columns * rows)
  initialHeights.fill(thickness)

  const heightBuffer = device.createBuffer({
    label: 'Lens heights',
    size: initialHeights.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  })

  // Upload the initial heights once.
  device.queue.writeBuffer(heightBuffer, 0, initialHeights)

  return {
    columns,
    rows,
    sizeX,
    sizeY,
    heightBuffer,
  }
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