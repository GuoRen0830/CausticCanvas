export interface LensState {
  // Vertex counts along each axis, including boundaries.
  columns: number
  rows: number

  // Aperture dimensions in meters.
  sizeX: number
  sizeY: number

  // Exit-surface z in meters; the entry plane is z = 0.
  // Row-major f32 values, starting at byte offset 0.
  heightBuffer: GPUBuffer
}

export interface TargetImage {
  width: number
  height: number
  version: number

  // Row-major weights in [0, 1], with row 0 at the bottom.
  values: Float32Array
}

export interface ReceiverPlane {
  // Centered on the optical Z axis and parallel to the XY plane.
  // Absolute z in meters, measured from the lens entry plane.
  z: number

  // Physical dimensions in meters.
  sizeX: number
  sizeY: number

  // Image resolution; row 0 corresponds to the bottom edge.
  width: number
  height: number
}

export type CausticImage = {
  // Linear irradiance; no exposure or tone mapping.
  readonly width: number
  readonly height: number

  // Row 0 maps to the receiver's bottom edge.
} & (
  | {
      readonly kind: 'buffer'

      // Row-major f32 values, starting at byte offset 0.
      readonly buffer: GPUBuffer
    }
  | {
      readonly kind: 'texture'

      // A single-mip, single-layer r32float 2D view.
      readonly view: GPUTextureView
    }
)

export interface AppState {
  // Latest drawing snapshot, including an empty target.
  target: TargetImage | null

  // Current lens geometry used by the preview.
  lens: LensState

  // Reference plane used for the caustic image.
  receiver: ReceiverPlane
}