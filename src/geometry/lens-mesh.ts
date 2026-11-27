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
  return createLens(device, options)
}

export function createLens(
  device: GPUDevice,
  options: FlatLensOptions,
  exitHeights?: Float32Array<ArrayBuffer>,
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

  const vertexCount = columns * rows
  if (exitHeights && exitHeights.length !== vertexCount) {
    throw new Error('Exit heights must match the lens grid.')
  }
  const initialHeights = exitHeights ?? new Float32Array(vertexCount).fill(thickness)

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
    flatThickness: thickness,
  }
}
