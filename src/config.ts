import type { OpticalConfig, ReceiverPlane } from './contracts'

// Startup settings. Reload to apply grid, image, or optical changes.
export const config = {
  lens: {
    columns: 33,
    rows: 33,
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
    // Adam step size in meters of height per iteration.
    learningRate: 5e-5,
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8,
    // Iterations per target before reporting done.
    maxIterations: 3000,
    // Weight of the mean squared discrete Laplacian (dimensionless curvature).
    smoothnessWeight: 0.1,
    // Penalize face slopes above this tangent; total internal reflection
    // starts near tan(asin(outsideIOR / lensIOR)) = 0.894 for 1.5 / 1.
    maxSurfaceSlope: 0.7,
    slopeWeight: 10,
    // Heights are projected to at least this exit-surface z in meters.
    minHeight: 0.002,
    // Read back the loss for logging every N iterations; 0 disables it.
    lossLogInterval: 100,
  },
  // Scheduling limit, independent of the optimizer's stopping policy.
  maxStepsPerFrame: 1,
  rendering: {
    maxPixelRatio: 2,
    // Display only; never changes optical power or the optimization target.
    causticExposure: 1,
  },
} as const
