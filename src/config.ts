import type { OpticalConfig, ReceiverPlane } from './contracts'

// Cells per axis. Each cell emits two triangles and shares its boundary vertices.
const lensCellsPerSide = 1024

// Startup settings. Reload to apply grid, image, or optical changes.
export const config = {
  lens: {
    columns: lensCellsPerSide + 1,
    rows: lensCellsPerSide + 1,
    sizeX: 0.2,
    sizeY: 0.2,
    thickness: 0.01,
  },
  optics: {
    lensIOR: 1.5,
    outsideIOR: 1,
    incidentIrradiance: 1,
  } satisfies OpticalConfig,
  receiver: {
    z: 0.5,
    sizeX: 0.4,
    sizeY: 0.4,
    // Shared by the drawing pad, solver, and caustic preview.
    width: 256,
    height: 256,
  } satisfies ReceiverPlane,
  optimizer: {
    // Initial Adam step size in meters; decreases during each run.
    learningRate: 1e-4,
    // Initial loss-filter width in receiver pixels; anneals linearly.
    lossBlurSigma: 16,
    // Final loss-filter width in receiver pixels.
    lossBlurFinalSigma: 1,
    // Fraction of maxIterations over which the loss filter anneals.
    lossBlurAnnealFraction: 0.8,
    // Final step size as a fraction of learningRate (quadratic decay).
    learningRateFloor: 0.05,
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8,
    // Iterations per target before reporting done.
    maxIterations: 1500,
    // Weight of the mean squared discrete Laplacian (dimensionless curvature).
    smoothnessWeight: 10,
    // Length in meters that makes the Laplacian dimensionless. 0 uses the mean
    // vertex spacing, which weakens the penalty as the grid gets finer; a fixed
    // length keeps the same physical smoothness on every grid.
    curvatureLengthScale: 0,
    // Penalize face slopes above this tangent; total internal reflection
    // starts near tan(asin(outsideIOR / lensIOR)) = 0.894 for 1.5 / 1.
    maxSurfaceSlope: 0.7,
    slopeWeight: 10,
    // Gaussian width in lens meters for blurring the optical height gradient
    // before regularization and Adam. Damps vertex-scale ripples on fine grids,
    // where each beam is smaller than a receiver pixel; 0 disables.
    gradientSmoothingSigma: 0.003,
    // Gaussian width in lens meters for blurring each Adam step; 0 disables.
    updateSmoothingSigma: 0,
    // Heights are projected to at least this exit-surface z in meters.
    minHeight: 0.002,
    // Read back the loss for logging every N iterations; 0 disables it.
    lossLogInterval: 100,
  },
  scene: {
    box: {
      // Lens-local meters. The lower face is 0.15 m above the receiver.
      center: [0.05, 0.03, 0.3],
      size: [0.08, 0.08, 0.1],
    },
  },
  // Scheduling limit, independent of the optimizer's stopping policy.
  maxStepsPerFrame: 1,
  rendering: {
    maxPixelRatio: 2,
    // Display only; never changes optical power or the optimization target.
    causticExposure: 1,
  },
} as const
