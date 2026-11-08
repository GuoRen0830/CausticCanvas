export interface LensState {
  // Vertex counts along each axis, including boundaries.
  columns: number
  rows: number

  // Centered XY aperture in meters; uniformly spaced vertices.
  // Columns run from -sizeX/2 to +sizeX/2; rows from -sizeY/2 to +sizeY/2.
  sizeX: number
  sizeY: number

  // Exit-surface z in meters; the entry plane is z = 0.
  // Row-major f32 values, starting at byte offset 0.
  // All shared height buffers require STORAGE | COPY_SRC | COPY_DST.
  heightBuffer: GPUBuffer
}

export interface OpticalConfig {
  // Ideal lossless refraction; omit absorption and Fresnel reflection.
  // The optimizer must keep the surface within the valid refraction domain.
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
  // For valid lossless beams: incidentIrradiance * triangle XY area.
  // Inactive beams use (0, 0, 1, 0); consumers skip zero flux.
  // Consumers borrow both buffers; resource owners manage their lifetime.
  readonly buffer: GPUBuffer
}

export interface TargetImage {
  width: number
  height: number
  version: number

  // Linear shape weights in [0, 1]; no gamma decoding.
  // Row-major, with row 0 at the bottom; not absolute irradiance.
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
  // Linear irradiance in W/m^2; no exposure or tone mapping.
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
  // Keep the caustic image kind stable for this model instance.
  // Pipelines must be ready; all resources must use the encoder's device.
  // Call between passes; record a full evaluation and close all opened passes.
  // Never finish or submit the encoder; the caller controls submission.
  // Return GPU handles without waiting for execution or reading data back.
  // Multiple calls may share an encoder; preserve each call's parameter values.
  encode(encoder: GPUCommandEncoder, input: ForwardInput): ForwardResult

  // Overwrite one f32 gradient per height using the latest matching forward input.
  encodeBackward?(
    encoder: GPUCommandEncoder,
    input: ForwardInput,
    result: ForwardResult,
    pixelGradient: GPUBuffer,
    heightGradient: GPUBuffer,
  ): void

  // Release owned resources only; never destroy the borrowed lens heights.
  destroy(): void
}

export interface OptimizationStepResult {
  readonly lens: LensState

  // Stop scheduling this target after submitting the current batch.
  // A CPU-known decision, not a GPU-completion or readback signal.
  readonly done: boolean
}

export interface LensOptimizer {
  // Call before encoding a batch, after submitting any previous batch.
  // Upload new target versions once; never modify the caller's values.
  // A new version starts a new run; unchanged versions preserve progress.
  // Normalize weights to P = incidentIrradiance * lens.sizeX * lens.sizeY.
  // For pixel area dA, target E[i] = P * weight[i] / (dA * sum(weights)).
  // Do not normalize by the current received power or display exposure.
  // Null or all-zero targets disable updates without resetting the lens.
  setTarget(target: Readonly<TargetImage> | null): void

  // Encode one update using the forward model captured during creation.
  // Nonempty target dimensions must match input.receiver.
  // Preserve grid and aperture metadata; height storage may be reused.
  // Consume old forward and gradient data before overwriting source heights.
  // Missing or empty targets return { lens: input.lens, done: true }.
  // A completed run stays done until setTarget receives a new version.
  // Report done using a step limit or already available convergence information;
  // never wait for GPU results here. The last call may still update the lens.
  // Pipelines must be ready and all resources must use the same device.
  // Call between passes; close all opened passes, never finish or submit.
  // Multiple steps may share an encoder; preserve each step's parameters.
  // Return without waiting for GPU execution or reading data back.
  encodeStep(
    encoder: GPUCommandEncoder,
    input: ForwardInput,
  ): OptimizationStepResult

  // Release owned resources only; keep the shared forward model and borrowed heights.
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