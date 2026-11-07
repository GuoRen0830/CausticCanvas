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
  // Scheduling limit, independent of the optimizer's stopping policy.
  maxStepsPerFrame: 1,
  rendering: {
    maxPixelRatio: 2,
    // Display only; never changes optical power or the optimization target.
    causticExposure: 1,
  },
} as const
