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

export interface OpticalConfig {
  // Refractive indices of the lens and surrounding medium.
  lensIOR: number
  outsideIOR: number

  // Uniform parallel light travels along +Z.
  // Irradiance just before the entry plane, in W/m^2.
  incidentIrradiance: number
}

export interface TriangleBeamField {
  // Snapshot of the lens metadata and buffer handle used to generate the beams.
  // GPU readers must use the matching height data, before either buffer is reused.
  readonly lens: Readonly<LensState>

  // One uniform parallel beam per triangle, in buildLensIndices order.
  // Count: 2 * (lens.columns - 1) * (lens.rows - 1); no compaction.
  // STORAGE array<vec4<f32>>, byte offset 0, 16-byte stride.
  // xyz: unit propagation direction in lens-local coordinates.
  // w: total transmitted radiant flux in W, finite and nonnegative.
  // Inactive beams use (0, 0, 1, 0); consumers skip zero flux.
  // Consumers borrow both buffers; resource owners manage their lifetime.
  readonly buffer: GPUBuffer
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

export interface ForwardInput {
  // Borrowed inputs; the forward model must not modify them.
  readonly lens: Readonly<LensState>
  readonly optics: Readonly<OpticalConfig>
  readonly receiver: Readonly<ReceiverPlane>
}

export interface ForwardResult {
  // Beams and image belong to the same evaluation.
  readonly beams: TriangleBeamField

  // Snapshot of the receiver used for this evaluation.
  readonly receiver: Readonly<ReceiverPlane>

  // Plane irradiance generated from these beams, at the receiver resolution.
  // Each call replaces prior values; no accumulation across evaluations.
  // Export buffers with STORAGE usage, or textures with TEXTURE_BINDING usage.
  readonly caustic: CausticImage

  // GPU outputs are borrowed, reusable storage, not persistent copies.
  // Record dependent reads before the next encode or input-height overwrite.
  // Submit command buffers in that same order and keep resources alive.
}

export interface ForwardModel {
  // Pipelines must be ready; all resources must use the encoder's device.
  // Call between passes; record a full evaluation and close all opened passes.
  // Never finish or submit the encoder; the caller controls submission.
  // Return GPU handles without waiting for execution or reading data back.
  // Multiple calls may share an encoder; preserve each call's parameter values.
  encode(encoder: GPUCommandEncoder, input: ForwardInput): ForwardResult

  // Release owned resources only; never destroy the borrowed lens heights.
  destroy(): void
}

export interface AppState {
  // Latest drawing snapshot, including an empty target.
  target: TargetImage | null

  // Current lens geometry used by the preview.
  lens: LensState

  // Optical inputs shared by the forward model and optimizer.
  optics: OpticalConfig

  // Reference plane used for the caustic image.
  receiver: ReceiverPlane
}