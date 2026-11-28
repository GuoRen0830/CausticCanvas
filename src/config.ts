import type { OpticalConfig, ReceiverPlane } from './contracts'

// Cells per axis. Each cell emits two triangles and shares its boundary vertices.
const lensCellsPerSide = 4096

// Startup settings. Reload to apply grid, image, or optical changes.
export const config = {
  // Linear aperture coverage of the fitted scene, independent of the drawing area.
  sceneApertureFraction: 0.10,
  // This slope lets the compact lamp sit closer while its rays still reach
  // the scene corners. Total internal reflection starts at about 0.727 in air.
  sceneLensIOR: 1.7,
  sceneSurfaceSlope: 0.714,
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
    incidentIrradiance: 3,
  } satisfies OpticalConfig,
  receiver: {
    // Maximum floor distance. Imported scenes fit the lamp height to the model.
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
    gradientSmoothingSigma: 0.0015,
    // Make that blur stop at lens creases: a neighbor whose surface slope differs
    // by more than about this much (dimensionless) barely contributes, so the
    // push on one side of a crease no longer smears it across. 0 keeps the
    // plain Gaussian.
    creaseSlope: 0.3,
    // Weight of a spill penalty: light on the receiver costs its distance in
    // pixels to the nearest painted pixel. Unlike the squared error, whose
    // gradient vanishes inside an evenly lit dark area, this pulls stray light
    // toward the drawing from any distance; 0 disables. Above about 0.03 it
    // starts to pull light out of small features such as the face's eyes.
    spillWeight: 0.02,
    // Gaussian width in lens meters for blurring each Adam step; 0 disables.
    updateSmoothingSigma: 0,
    // Vertex motion during the fine refinement: vertices also slide in x and y,
    // so they can gather at the seams between regions of the lens that light
    // different strokes. A face straddling a seam lands between the strokes;
    // gathering vertices there shrinks such faces and the light they spill.
    // Adam step in vertex spacings per iteration; 0 keeps the uniform grid.
    xyLearningRate: 0.02,
    // Weight of the mean squared Laplacian of the x/y displacement in vertex
    // spacings, which keeps the mesh from wrinkling.
    xySmoothnessWeight: 0.1,
    // No face may shrink below this fraction of its grid area, so faces never fold.
    minAreaRatio: 0.02,
    // Gaussian width in lens meters for blurring the x/y gradient (stops at
    // creases like the height gradient blur); 0 disables.
    xySmoothingSigma: 0.0015,
    // 1: a moving vertex keeps to the current surface (its height follows the
    // local slope), so x/y motion redistributes faces without reshaping the
    // lens; 0: heights stay put and every x/y step also tilts the faces.
    xyTangential: 1,
    // Coarse-to-fine: first optimize a grid with this many cells per side (it
    // must divide the fine cell count; 0 disables), then interpolate it onto
    // the fine grid and refine it for refineIterations steps at
    // refineLearningRate. A gentle refinement keeps the coarse grid's sharp,
    // low-haze focus; the full learning rate brings the haze back.
    coarseGridCells: 128,
    coarseIterations: 1500,
    coarseLearningRate: 1e-4,
    // Start the coarse grid from a global lens-to-target transport map
    // (transport-init.ts) instead of the previous lens; 0 disables. Without
    // it, light far from every stroke never gets pulled in and stays as
    // faint filaments.
    transportInit: 1,
    // Sliced-transport passes for that map (about 7 ms each on 129 x 129).
    transportIterations: 100,
    // Coarse steps per encodeStep call; coarse steps are much cheaper.
    coarseStepsPerCall: 10,
    // Initial loss-filter width in pixels for the fine refinement after a coarse run.
    refineLossBlurSigma: 2,
    refineIterations: 2000,
    refineLearningRate: 3e-6,
    // Heights are projected to at least this exit-surface z in meters.
    minHeight: 0.002,
    // Read back the loss for logging every N iterations; 0 disables it.
    lossLogInterval: 100,
  },
  // Scheduling limit, independent of the optimizer's stopping policy.
  maxStepsPerFrame: 3,
  rendering: {
    maxPixelRatio: 2,
    // Presentation settings are independent of the forward model.
    environmentLight: 0.05,
    // Display scattering in inverse metres for the thin medium in Room.
    roomScattering: 1.05,
    // Display only; never changes optical power or the optimization target.
    causticExposure: 1,
    // Light-view exposure once something is drawn, as the panel value of a
    // fully painted pixel that receives its share of the light (x in x/(1+x)).
    // Zero keeps the same exposure used for the initial floodlit scene.
    drawingExposure: 3,
    // Lamp exposure in the lit 3D view once something is drawn, in the same
    // units as drawingExposure. Lower keeps more texture inside the light.
    roomExposure: 6,
  },
} as const

// The artist paints over the receiver. A smaller, centered collimator supplies
// the same lamp power through a smaller area. Optical lengths are in metres.
export function sceneLamp(receiver: Pick<ReceiverPlane, 'sizeX' | 'sizeY'>) {
  const scale = config.sceneApertureFraction
  const settings = config.optimizer
  return {
    lens: {
      ...config.lens, sizeX: receiver.sizeX * scale, sizeY: receiver.sizeY * scale,
      thickness: config.lens.thickness * scale,
    },
    optics: { ...config.optics, lensIOR: config.sceneLensIOR,
      incidentIrradiance: config.optics.incidentIrradiance / (scale * scale) },
    // Keep smoothing radii in lens grid cells, and each height update in the
    // same proportion to the vertex spacing. Receiver-pixel filters keep their units.
    optimizer: {
      ...settings,
      maxSurfaceSlope: config.sceneSurfaceSlope,
      learningRate: settings.learningRate * scale,
      coarseLearningRate: settings.coarseLearningRate * scale,
      refineLearningRate: settings.refineLearningRate * scale,
      curvatureLengthScale: settings.curvatureLengthScale * scale,
      gradientSmoothingSigma: settings.gradientSmoothingSigma * scale,
      updateSmoothingSigma: settings.updateSmoothingSigma * scale,
      xySmoothingSigma: settings.xySmoothingSigma * scale,
      minHeight: settings.minHeight * scale,
    },
  }
}

// Axial throw needed per metre of receiver radius. A homothetic light field
// moves a ray at aperture radius f*r outwards by (1-f)*r. The extra distance
// budgets the rising rim of the diverging exit surface above the peripheral rays.
export function sceneThrowPerRadius() {
  const f = config.sceneApertureFraction
  const tilt = Math.atan(config.sceneSurfaceSlope)
  const eta = config.sceneLensIOR / config.optics.outsideIOR
  const bend = Math.abs(Math.asin(Math.min(1, eta * Math.sin(tilt))) - tilt)
  return (1 - f) / Math.max(1e-6, Math.tan(bend)) + 0.6 * f
}

// Visible air above the flat entry face of the lens, in optical meters.
export function incidentColumnHeight(receiverZ: number, apertureSpan: number) {
  return Math.min(0.25 * receiverZ, 0.14 * apertureSpan)
}
