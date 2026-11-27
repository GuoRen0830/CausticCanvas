import { workgroupGrid } from '../gpu/dispatch'
import { config } from '../config'
import type {
  CausticImage,
  ForwardInput,
  ForwardModel,
  LensOptimizer,
  LensState,
  OptimizationStepResult,
  ReceiverPlane,
  TargetImage,
} from '../contracts'
import shaderSource from './shaders/optimizer.wgsl?raw'
import { transportHeights } from './transport-init'
import multigridSource from './shaders/multigrid.wgsl?raw'

export type OptimizerSettings = { [K in keyof typeof config.optimizer]: number }

const LOSS_WORKGROUP = 256
const VERTEX_WORKGROUP = 64
const PARAMS_SIZE = 128

// Total transmitted power P = incidentIrradiance * aperture area.
export function incidentPower(input: ForwardInput): number {
  return input.optics.incidentIrradiance * input.lens.sizeX * input.lens.sizeY
}

// Normalize incident power over the physical surface area represented by each pixel.
export function normalizeTarget(
  target: Readonly<TargetImage>,
  receiver: Readonly<ReceiverPlane>,
  power: number,
  surfaceAreas?: Float32Array,
): Float32Array | null {
  if (target.width !== receiver.width || target.height !== receiver.height) {
    throw new Error('Target size must match the receiver resolution.')
  }
  if (target.values.length !== target.width * target.height) {
    throw new Error('Target values do not match the target size.')
  }
  if (surfaceAreas && surfaceAreas.length !== target.values.length) throw new Error('Invalid surface area map.')
  const pixelArea = (receiver.sizeX / receiver.width) * (receiver.sizeY / receiver.height)
  let weightedArea = 0
  for (let i = 0; i < target.values.length; i++) {
    const value = target.values[i]
    if (!Number.isFinite(value) || value < 0) {
      throw new Error('Target weights must be finite and nonnegative.')
    }
    weightedArea += value * (surfaceAreas?.[i] ?? pixelArea)
  }
  if (weightedArea <= 0) return null
  const scale = power / weightedArea
  const normalized = new Float32Array(target.values.length)
  for (let i = 0; i < normalized.length; i++) {
    normalized[i] = surfaceAreas?.[i] === 0 ? 0 : target.values[i] * scale
  }
  return normalized
}

function validateSettings(settings: OptimizerSettings) {
  const positive = [
    settings.learningRate,
    settings.epsilon,
    settings.maxSurfaceSlope,
    settings.minHeight,
  ]
  const nonnegative = [settings.smoothnessWeight, settings.slopeWeight, settings.lossBlurSigma,
    settings.curvatureLengthScale, settings.gradientSmoothingSigma, settings.lossBlurFinalSigma,
    settings.updateSmoothingSigma, settings.spillWeight, settings.creaseSlope,
    settings.xyLearningRate, settings.xySmoothnessWeight, settings.xySmoothingSigma]
  const fractions = [settings.lossBlurAnnealFraction, settings.learningRateFloor]
  if (
    positive.some((value) => !Number.isFinite(value) || value <= 0) ||
    nonnegative.some((value) => !Number.isFinite(value) || value < 0) ||
    fractions.some((value) => !(value > 0 && value <= 1)) ||
    !(settings.beta1 >= 0 && settings.beta1 < 1) ||
    !(settings.beta2 >= 0 && settings.beta2 < 1) ||
    !Number.isSafeInteger(settings.maxIterations) || settings.maxIterations < 1 ||
    !Number.isSafeInteger(settings.lossLogInterval) || settings.lossLogInterval < 0 ||
    !Number.isSafeInteger(settings.coarseGridCells) || settings.coarseGridCells < 0 ||
    !Number.isSafeInteger(settings.coarseIterations) || settings.coarseIterations < 1 ||
    !Number.isSafeInteger(settings.coarseStepsPerCall) || settings.coarseStepsPerCall < 1 ||
    !Number.isSafeInteger(settings.refineIterations) || settings.refineIterations < 1 ||
    !Number.isSafeInteger(settings.transportIterations) || settings.transportIterations < 1 ||
    !(settings.coarseLearningRate > 0) || !(settings.refineLearningRate > 0) ||
    !(settings.refineLossBlurSigma >= 0) ||
    !(settings.minAreaRatio >= 0 && settings.minAreaRatio < 1)
  ) {
    throw new Error('Invalid optimizer settings.')
  }
}

function sameGrid(a: Readonly<LensState>, b: Readonly<LensState>) {
  return a.columns === b.columns && a.rows === b.rows &&
    a.sizeX === b.sizeX && a.sizeY === b.sizeY
}

function sameReceiver(a: Readonly<ReceiverPlane>, b: Readonly<ReceiverPlane>) {
  return a.width === b.width && a.height === b.height &&
    a.sizeX === b.sizeX && a.sizeY === b.sizeY && a.z === b.z
}

// Exact squared Euclidean distance transform along one line (Felzenszwalb and
// Huttenlocher): out[q] = min_p (q - p)^2 + f[p].
function distanceLine(f: Float64Array, out: Float64Array, n: number) {
  const v = new Int32Array(n), z = new Float64Array(n + 1)
  let k = 0
  v[0] = 0; z[0] = -Infinity; z[1] = Infinity
  for (let q = 1; q < n; q++) {
    if (f[q] === Infinity) continue
    if (f[v[k]] === Infinity) { v[k] = q; continue }
    let s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
    while (k > 0 && s <= z[k]) {
      k--
      s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
    }
    k++
    v[k] = q; z[k] = s; z[k + 1] = Infinity
  }
  k = 0
  for (let q = 0; q < n; q++) {
    while (z[k + 1] < q) k++
    const d = q - v[k]
    out[q] = f[v[k]] === Infinity ? Infinity : d * d + f[v[k]]
  }
}

// Gradient of the spill penalty spillWeight * mean(E * D) / meanIrradiance with
// respect to each pixel's irradiance E, where D is the distance in pixels to the
// nearest painted pixel (a target above a quarter of the painted level).
export function spillGradient(target: Float32Array, { width, height }: { width: number, height: number },
  spillWeight: number, meanIrradiance: number) {
  const gradient = new Float32Array(target.length)
  if (!(spillWeight > 0) || !(meanIrradiance > 0)) return gradient
  let sum = 0, sumSquares = 0
  for (const value of target) { sum += value; sumSquares += value * value }
  if (!(sum > 0)) return gradient
  const threshold = 0.25 * sumSquares / sum
  const squared = new Float64Array(target.length)
  for (let i = 0; i < target.length; i++) squared[i] = target[i] > threshold ? 0 : Infinity
  const line = new Float64Array(Math.max(width, height)), result = new Float64Array(Math.max(width, height))
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) line[x] = squared[y * width + x]
    distanceLine(line, result, width)
    for (let x = 0; x < width; x++) squared[y * width + x] = result[x]
  }
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) line[y] = squared[y * width + x]
    distanceLine(line, result, height)
    for (let y = 0; y < height; y++) squared[y * width + x] = result[y]
  }
  const scale = spillWeight / (meanIrradiance * target.length)
  for (let i = 0; i < target.length; i++) {
    const distance = Math.sqrt(squared[i])
    gradient[i] = Number.isFinite(distance) ? scale * distance : 0
  }
  return gradient
}

function writeParams(
  input: ForwardInput,
  settings: OptimizerSettings,
): ArrayBuffer {
  const { lens, receiver } = input
  const vertexCount = lens.columns * lens.rows
  const faceCount = 2 * (lens.columns - 1) * (lens.rows - 1)
  const pixelCount = receiver.width * receiver.height
  const meanIrradiance = incidentPower(input) / (receiver.sizeX * receiver.sizeY)
  const data = new ArrayBuffer(PARAMS_SIZE)
  const view = new DataView(data)
  // Match the WGSL Params layout.
  view.setUint32(0, lens.columns, true)
  view.setUint32(4, lens.rows, true)
  view.setUint32(8, receiver.width, true)
  view.setUint32(12, receiver.height, true)
  view.setFloat32(16, lens.sizeX / (lens.columns - 1), true)
  view.setFloat32(20, lens.sizeY / (lens.rows - 1), true)
  const floats = [
    1 / (meanIrradiance * meanIrradiance),
    1 / pixelCount,
    settings.learningRate,
    settings.beta1,
    settings.beta2,
    settings.epsilon,
    settings.smoothnessWeight,
    settings.slopeWeight,
    settings.maxSurfaceSlope,
    settings.minHeight,
    1 / vertexCount,
    1 / faceCount,
  ]
  floats.forEach((value, i) => view.setFloat32(24 + 4 * i, value, true))
  view.setUint32(72, settings.maxIterations, true)
  view.setFloat32(76, settings.lossBlurSigma, true)
  view.setFloat32(80, settings.lossBlurAnnealFraction, true)
  view.setFloat32(84, settings.learningRateFloor, true)
  const meanSpacing = 0.5 * (lens.sizeX / (lens.columns - 1) + lens.sizeY / (lens.rows - 1))
  view.setFloat32(88, settings.curvatureLengthScale || meanSpacing, true)
  view.setFloat32(92, settings.gradientSmoothingSigma, true)
  view.setFloat32(96, settings.lossBlurFinalSigma, true)
  view.setFloat32(100, settings.updateSmoothingSigma, true)
  view.setFloat32(104, settings.creaseSlope, true)
  view.setFloat32(108, settings.xyLearningRate, true)
  view.setFloat32(112, settings.xySmoothnessWeight, true)
  view.setFloat32(116, settings.minAreaRatio, true)
  view.setFloat32(120, settings.xySmoothingSigma, true)
  view.setFloat32(124, settings.xyTangential, true)
  return data
}

// Prepare optimizer resources on the shared device without changing input heights.
// Capture the shared forward model for encodeStep; the caller retains ownership.
// Clean up partial allocations on failure.
export async function createLensOptimizer(
  device: GPUDevice,
  initial: ForwardInput,
  forward: ForwardModel,
  settings: OptimizerSettings = config.optimizer,
  options: {
    timestamps?: (stage: string) => GPUComputePassTimestampWrites | undefined
    // Forward model on a coarse grid with coarseGridCells cells per side. When
    // given, each run first optimizes that grid and then refines the fine one.
    coarseForward?: ForwardModel
  } = {},
): Promise<LensOptimizer | null> {
  if (!forward.encodeBackward) {
    // Without optical gradients there is no honest update; stay in preview mode.
    console.warn('Lens optimizer disabled: forward model has no encodeBackward.')
    return null
  }
  validateSettings(settings)

  const lensMeta = { ...initial.lens }
  const receiver = { ...initial.receiver }
  const power = incidentPower(initial)
  if (!Number.isFinite(power) || power < 0) throw new Error('Invalid incident power.')
  if (power === 0) return null

  const vertexCount = lensMeta.columns * lensMeta.rows
  const pixelCount = receiver.width * receiver.height
  const lossGroups = Math.ceil(pixelCount / LOSS_WORKGROUP)
  const limit = device.limits.maxComputeWorkgroupsPerDimension
  const lossGrid = workgroupGrid(lossGroups, limit)
  const vertexGrid = workgroupGrid(Math.ceil(vertexCount / VERTEX_WORKGROUP), limit)

  const owned: GPUBuffer[] = []
  const createBuffer = (label: string, size: number, usage: GPUBufferUsageFlags) => {
    const buffer = device.createBuffer({ label, size, usage })
    owned.push(buffer)
    return buffer
  }

  try {
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    const params = createBuffer(
      'Optimizer params',
      PARAMS_SIZE,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    )
    // Optimized heights; the borrowed initial buffer is only read once.
    const heights = createBuffer('Optimized lens heights', vertexCount * 4, storage)
    const heightGradient = createBuffer('Lens height gradient', vertexCount * 4, storage)
    const curvature = createBuffer('Lens curvature', vertexCount * 4, storage)
    const moments = createBuffer('Adam moments', vertexCount * 8, storage)
    const stepState = createBuffer('Optimizer step state', 16, storage)
    const targetBuffer = createBuffer('Target irradiance', pixelCount * 4, storage)
    const pixelGradient = createBuffer('Pixel loss gradient', pixelCount * 4, storage)
    const filterA = createBuffer('Loss filter A', pixelCount * 4, storage)
    const filterB = createBuffer('Loss filter B', pixelCount * 4, storage)
    // Spill-penalty gradient per pixel, added to the filtered data gradient.
    const spillGradients = createBuffer('Spill gradient', pixelCount * 4, storage)
    const lossPartials = createBuffer('Loss partial sums', lossGroups * 4, storage)
    const lossStaging = createBuffer(
      'Loss readback',
      lossGroups * 4,
      GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    )

    // After a coarse run the fine refinement starts close to the answer, so it
    // begins from a narrower loss filter.
    const refining = settings.coarseGridCells > 0 && !!options.coarseForward
    const fineSettings = refining
      ? { ...settings, lossBlurSigma: settings.refineLossBlurSigma,
          learningRate: settings.refineLearningRate, maxIterations: settings.refineIterations }
      : settings
    device.queue.writeBuffer(params, 0, writeParams(initial, fineSettings))

    // Vertex motion (x/y displacement of each vertex). Without it the offsets
    // stay zero and only place the slope penalty on the actual faces.
    const motion = fineSettings.xyLearningRate > 0
    const offsets = createBuffer('Lens vertex offsets', vertexCount * 8, storage)
    const motionBuffer = (label: string, bytes: number) => createBuffer(label, motion ? vertexCount * bytes : 16, storage)
    const offsetGradient = motionBuffer('Lens offset gradient', 8)
    const offsetMoments = motionBuffer('Offset Adam moments', 16)
    const proposal = motionBuffer('Proposed lens offsets', 16)
    const scratch = motionBuffer('Lens offset scratch', 8)

    const visibility = GPUShaderStage.COMPUTE
    const storageEntry = (binding: number, type: GPUBufferBindingType) => ({
      binding,
      visibility,
      buffer: { type },
    })
    const lossLayout = device.createBindGroupLayout({
      label: 'Optimizer loss layout',
      entries: [
        { binding: 0, visibility, buffer: { type: 'uniform' } },
        storageEntry(1, 'read-only-storage'),
        storageEntry(2, 'storage'),
        storageEntry(3, 'storage'),
        storageEntry(8, 'storage'),
        storageEntry(9, 'storage'),
        storageEntry(10, 'storage'),
        storageEntry(11, 'read-only-storage'),
      ],
    })
    const causticLayouts = {
      buffer: device.createBindGroupLayout({
        label: 'Optimizer caustic buffer layout',
        entries: [storageEntry(0, 'read-only-storage')],
      }),
      texture: device.createBindGroupLayout({
        label: 'Optimizer caustic texture layout',
        entries: [{
          binding: 1,
          visibility,
          texture: { sampleType: 'unfilterable-float' },
        }],
      }),
    }
    const updateLayout = device.createBindGroupLayout({
      label: 'Optimizer update layout',
      entries: [
        { binding: 0, visibility, buffer: { type: 'uniform' } },
        storageEntry(4, 'storage'),
        storageEntry(5, 'storage'),
        storageEntry(6, 'storage'),
        storageEntry(7, 'storage'),
        storageEntry(8, 'storage'),
        storageEntry(12, 'storage'),
      ],
    })
    const motionLayout = device.createBindGroupLayout({
      label: 'Optimizer vertex motion layout',
      entries: [
        { binding: 0, visibility, buffer: { type: 'uniform' } },
        ...[4, 5, 8, 12, 13, 14, 15, 16].map(binding => storageEntry(binding, 'storage')),
      ],
    })

    const module = device.createShaderModule({ label: 'Optimizer shader', code: shaderSource })
    const pipeline = (entryPoint: string, layouts: GPUBindGroupLayout[]) =>
      device.createComputePipelineAsync({
        label: `Optimizer ${entryPoint}`,
        layout: device.createPipelineLayout({ bindGroupLayouts: layouts }),
        compute: { module, entryPoint },
      })
    const [
      lossFromBuffer,
      lossFromTexture,
      filterResidualX,
      filterResidualY,
      filterGradientY,
      filterGradientX,
      smoothGradientX,
      smoothGradientY,
      computeCurvature,
      addRegularization,
      adamUpdate,
      adamStep,
      smoothUpdateX,
      applySmoothUpdateY,
      advanceStep,
      tangentialGradient,
      blurOffsetGradientX,
      blurOffsetGradientY,
      offsetCurvature,
      proposeOffsets,
      acceptOffsets,
      commitOffsets,
    ] = await Promise.all([
      pipeline('lossFromBuffer', [lossLayout, causticLayouts.buffer]),
      pipeline('lossFromTexture', [lossLayout, causticLayouts.texture]),
      pipeline('filterResidualX', [lossLayout]),
      pipeline('filterResidualY', [lossLayout]),
      pipeline('filterGradientY', [lossLayout]),
      pipeline('filterGradientX', [lossLayout]),
      pipeline('smoothGradientX', [updateLayout]),
      pipeline('smoothGradientY', [updateLayout]),
      pipeline('computeCurvature', [updateLayout]),
      pipeline('addRegularization', [updateLayout]),
      pipeline('adamUpdate', [updateLayout]),
      pipeline('adamStep', [updateLayout]),
      pipeline('smoothUpdateX', [updateLayout]),
      pipeline('applySmoothUpdateY', [updateLayout]),
      pipeline('advanceStep', [updateLayout]),
      ...['tangentialGradient', 'blurOffsetGradientX', 'blurOffsetGradientY', 'offsetCurvature', 'proposeOffsets',
        'acceptOffsets', 'commitOffsets'].map(entryPoint => pipeline(entryPoint, [motionLayout])),
    ])

    const lossBindings = device.createBindGroup({
      label: 'Optimizer loss bindings',
      layout: lossLayout,
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: targetBuffer } },
        { binding: 2, resource: { buffer: pixelGradient } },
        { binding: 3, resource: { buffer: lossPartials } },
        { binding: 8, resource: { buffer: stepState } },
        { binding: 9, resource: { buffer: filterA } },
        { binding: 10, resource: { buffer: filterB } },
        { binding: 11, resource: { buffer: spillGradients } },
      ],
    })
    const updateBindings = device.createBindGroup({
      label: 'Optimizer update bindings',
      layout: updateLayout,
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 4, resource: { buffer: heights } },
        { binding: 5, resource: { buffer: heightGradient } },
        { binding: 6, resource: { buffer: curvature } },
        { binding: 7, resource: { buffer: moments } },
        { binding: 8, resource: { buffer: stepState } },
        { binding: 12, resource: { buffer: offsets } },
      ],
    })
    const motionBindings = device.createBindGroup({
      label: 'Optimizer vertex motion bindings',
      layout: motionLayout,
      entries: [
        { binding: 0, resource: { buffer: params } },
        ...([[4, heights], [5, heightGradient], [8, stepState], [12, offsets], [13, offsetGradient], [14, offsetMoments],
          [15, proposal], [16, scratch]] as const).map(([binding, buffer]) => ({ binding, resource: { buffer } })),
      ],
    })

    // Forward outputs are reusable storage, so cache bindings per GPU resource.
    const causticBindings = new WeakMap<GPUBuffer | GPUTextureView, GPUBindGroup>()
    const causticBindingsFor = (image: CausticImage) => {
      const resource = image.kind === 'buffer' ? image.buffer : image.view
      let group = causticBindings.get(resource)
      if (!group) {
        group = device.createBindGroup({
          label: `Optimizer caustic ${image.kind} bindings`,
          layout: causticLayouts[image.kind],
          entries: image.kind === 'buffer'
            ? [{ binding: 0, resource: { buffer: image.buffer, size: pixelCount * 4 } }]
            : [{ binding: 1, resource: image.view }],
        })
        causticBindings.set(resource, group)
      }
      return group
    }

    // Coarse-to-fine: a nested optimizer trains a coarse copy of the lens, whose
    // heights are interpolated onto the fine grid before the fine refinement.
    const coarseCells = settings.coarseGridCells
    let coarse: {
      optimizer: LensOptimizer
      heights: GPUBuffer
      lens: LensState
      transfer: GPUBindGroup
      restrict: GPUComputePipeline
      prolong: GPUComputePipeline
      coarseGrid: [number, number]
      // Transport-initialized coarse heights for the next reset, if any.
      initial: GPUBuffer
      initialPending: boolean
      meta: { columns: number, rows: number, sizeX: number, sizeY: number }
    } | null = null
    if (coarseCells > 0 && options.coarseForward) {
      const fineCells = [lensMeta.columns - 1, lensMeta.rows - 1]
      if (fineCells.some(cells => cells % coarseCells !== 0 || cells === coarseCells)) {
        throw new Error('coarseGridCells must be a proper divisor of the fine cell count.')
      }
      const coarseMeta = { ...lensMeta, columns: coarseCells + 1, rows: coarseCells + 1 }
      const coarseCount = coarseMeta.columns * coarseMeta.rows
      const coarseHeights = createBuffer('Coarse lens heights', coarseCount * 4, storage)
      const coarseLens: LensState = { ...coarseMeta, heightBuffer: coarseHeights }
      const inner = await createLensOptimizer(device,
        { lens: coarseLens, optics: initial.optics, receiver }, options.coarseForward,
        { ...settings, coarseGridCells: 0, maxIterations: settings.coarseIterations,
          learningRate: settings.coarseLearningRate, lossLogInterval: 0, xyLearningRate: 0 })
      if (!inner) throw new Error('Coarse optimizer could not be created.')
      const transferParams = createBuffer('Multigrid transfer', 32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)
      device.queue.writeBuffer(transferParams, 0, new Uint32Array([
        lensMeta.columns, lensMeta.rows, coarseMeta.columns, coarseMeta.rows,
        fineCells[0] / coarseCells, fineCells[1] / coarseCells, 0, 0]))
      const transferModule = device.createShaderModule({ label: 'Multigrid transfer', code: multigridSource })
      // Both entry points use the same three bindings, so one group serves both.
      const transferLayout = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ] })
      const layout = device.createPipelineLayout({ bindGroupLayouts: [transferLayout] })
      const [restrictPipeline, prolongPipeline] = await Promise.all(
        ['restrictHeights', 'prolongHeights'].map(entryPoint =>
          device.createComputePipelineAsync({ label: entryPoint, layout,
            compute: { module: transferModule, entryPoint } })))
      coarse = {
        optimizer: inner,
        heights: coarseHeights,
        lens: coarseLens,
        transfer: device.createBindGroup({ layout: transferLayout, entries: [
          { binding: 0, resource: { buffer: transferParams } },
          { binding: 1, resource: { buffer: heights } },
          { binding: 2, resource: { buffer: coarseHeights } },
        ] }),
        restrict: restrictPipeline,
        prolong: prolongPipeline,
        coarseGrid: workgroupGrid(Math.ceil(coarseCount / 64), limit),
        initial: createBuffer('Coarse transport heights', coarseCount * 4,
          GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST),
        initialPending: false,
        meta: coarseMeta,
      }
    }
    let coarsePhase = false
    // The offsets buffer starts zeroed; track it to avoid clearing it every step.
    let offsetsZero = true

    let targetVersion: number | null = null
    let active = false
    let done = true
    let resetPending = false
    let iteration = 0
    let destroyed = false

    // Loss logging: copy during one batch, then map in the next setTarget call,
    // which the caller makes only after submitting the previous batch.
    // Mapping inside encodeStep would invalidate a multi-step batch.
    let lossState: 'idle' | 'copied' | 'mapping' = 'idle'
    let targetRun = 0
    let lossRun = 0
    let lossIteration = 0
    const lossHistory: { iteration: number, loss: number }[] = []
    const flushLossReadback = () => {
      if (lossState !== 'copied' || destroyed) return
      lossState = 'mapping'
      const loggedIteration = lossIteration
      const loggedRun = lossRun
      lossStaging.mapAsync(GPUMapMode.READ).then(() => {
        if (destroyed) return
        const values = new Float32Array(lossStaging.getMappedRange())
        let loss = 0
        for (const value of values) loss += value
        lossStaging.unmap()
        lossState = 'idle'
        if (loggedRun !== targetRun) return
        lossHistory.push({ iteration: loggedIteration, loss })
        console.info(`[optimizer] iteration ${loggedIteration} loss ${loss.toExponential(4)}`)
      }, () => {
        lossState = 'idle'
      })
    }

    const optimizer: LensOptimizer & {
      readonly lossHistory: typeof lossHistory
      // Read-only diagnostics for check pages: the last combined height
      // gradient and the Adam moments (m, v) per vertex.
      readonly diagnostics: { heightGradient: GPUBuffer, moments: GPUBuffer, offsets: GPUBuffer }
    } = {
      lossHistory,
      diagnostics: { heightGradient, moments, offsets },

      setTarget(target) {
        if (destroyed) return
        flushLossReadback()
        if (!target) {
          if (targetVersion !== null) {
            targetRun++
            coarse?.optimizer.setTarget(null)
          }
          targetVersion = null
          active = false
          done = true
          return
        }
        if (target.version === targetVersion) return
        const normalized = normalizeTarget(target, receiver, power, forward.targetSurfaceAreas)
        targetVersion = target.version
        targetRun++
        active = normalized !== null
        done = !active
        if (!normalized) {
          coarse?.optimizer.setTarget(null)
          return
        }
        device.queue.writeBuffer(targetBuffer, 0, normalized)
        device.queue.writeBuffer(spillGradients, 0, spillGradient(normalized, receiver, settings.spillWeight,
          power / (receiver.sizeX * receiver.sizeY)))
        if (coarse && settings.transportInit > 0) {
          // Power per light-view pixel, then a global lens-to-target assignment.
          const pixelArea = (receiver.sizeX / receiver.width) * (receiver.sizeY / receiver.height)
          const areas = forward.targetSurfaceAreas
          const power = normalized.map((value, i) => value * (areas?.[i] ?? pixelArea))
          const start = transportHeights({
            target: power, width: receiver.width, height: receiver.height,
            receiverSizeX: receiver.sizeX, receiverSizeY: receiver.sizeY, receiverZ: receiver.z,
            depths: forward.targetDepths,
            columns: coarse.meta.columns, rows: coarse.meta.rows,
            lensSizeX: coarse.meta.sizeX, lensSizeY: coarse.meta.sizeY,
            baseHeight: Math.max(settings.minHeight, lensMeta.flatThickness ?? config.lens.thickness),
            eta: initial.optics.lensIOR / initial.optics.outsideIOR,
            maxSlope: settings.maxSurfaceSlope,
            iterations: settings.transportIterations,
          })
          device.queue.writeBuffer(coarse.initial, 0, start)
          coarse.initialPending = true
        }
        resetPending = true
        iteration = 0
        lossHistory.length = 0
        coarse?.optimizer.setTarget(target)
      },

      encodeStep(encoder, input): OptimizationStepResult {
        if (destroyed) throw new Error('Optimizer was destroyed.')
        if (!active || done) return { lens: input.lens, done: true }
        if (!sameGrid(input.lens, lensMeta) || !sameReceiver(input.receiver, receiver)) {
          throw new Error('Lens grid and receiver must match the optimizer setup.')
        }

        if (input.lens.heightBuffer !== heights) {
          encoder.copyBufferToBuffer(input.lens.heightBuffer, 0, heights, 0, vertexCount * 4)
        }
        if (input.lens.offsetBuffer && input.lens.offsetBuffer !== offsets) {
          encoder.copyBufferToBuffer(input.lens.offsetBuffer, 0, offsets, 0, vertexCount * 8)
          offsetsZero = false
        } else if (!input.lens.offsetBuffer && !offsetsZero) {
          encoder.clearBuffer(offsets)
          offsetsZero = true
        }
        // Without vertex motion the caller's offsets (if any) pass through unchanged.
        const lens: LensState = motion ? { ...input.lens, heightBuffer: heights, offsetBuffer: offsets }
          : { ...input.lens, heightBuffer: heights }
        if (resetPending) {
          // New target: restart Adam from the current lens shape.
          encoder.clearBuffer(moments)
          encoder.clearBuffer(stepState)
          if (motion) {
            encoder.clearBuffer(offsetMoments)
            // The coarse grid has no offsets and its prolongation rewrites the
            // fine heights on the uniform grid. Without it, the warm start keeps
            // the displaced mesh, since the heights belong to those positions.
            if (coarse) {
              encoder.clearBuffer(offsets)
              offsetsZero = true
            }
          }
          resetPending = false
          if (coarse && coarse.initialPending) {
            // Start the coarse grid from the global transport solution.
            encoder.copyBufferToBuffer(coarse.initial, 0, coarse.heights, 0, coarse.heights.size)
            coarse.initialPending = false
            coarse.lens = { ...coarse.lens, heightBuffer: coarse.heights }
            coarsePhase = true
          } else if (coarse) {
            // Warm-start the coarse grid from the current fine heights.
            const pass = encoder.beginComputePass({ label: 'Restrict lens heights' })
            pass.setPipeline(coarse.restrict)
            pass.setBindGroup(0, coarse.transfer)
            pass.dispatchWorkgroups(...coarse.coarseGrid)
            pass.end()
            coarse.lens = { ...coarse.lens, heightBuffer: coarse.heights }
            coarsePhase = true
          }
        }
        if (coarse && coarsePhase) {
          // Several cheap coarse steps per call, then one interpolation onto the
          // fine grid. The caller previews the fine heights with its own forward.
          let finished = false
          for (let k = 0; k < settings.coarseStepsPerCall && !finished; k++) {
            const update = coarse.optimizer.encodeStep(encoder,
              { lens: coarse.lens, optics: input.optics, receiver: input.receiver })
            coarse.lens = update.lens
            finished = update.done
          }
          if (coarse.lens.heightBuffer !== coarse.heights) {
            encoder.copyBufferToBuffer(coarse.lens.heightBuffer, 0, coarse.heights, 0, coarse.heights.size)
          }
          const pass = encoder.beginComputePass({ label: 'Prolong lens heights' })
          pass.setPipeline(coarse.prolong)
          pass.setBindGroup(0, coarse.transfer)
          pass.dispatchWorkgroups(...vertexGrid)
          pass.end()
          coarsePhase = !finished
          return { lens, done: false }
        }
        const stepInput = { lens, optics: input.optics, receiver: input.receiver }

        const result = forward.encode(encoder, stepInput)
        if (result.caustic.width !== receiver.width || result.caustic.height !== receiver.height) {
          throw new Error('Forward image size does not match the receiver.')
        }

        const lossPass = encoder.beginComputePass({ label: 'Optimizer loss', timestampWrites: options.timestamps?.('loss') })
        lossPass.setPipeline(result.caustic.kind === 'buffer' ? lossFromBuffer : lossFromTexture)
        lossPass.setBindGroup(0, lossBindings)
        lossPass.setBindGroup(1, causticBindingsFor(result.caustic))
        lossPass.dispatchWorkgroups(...lossGrid)
        lossPass.end()

        const filterPass = encoder.beginComputePass({ label: 'Optimizer loss filter', timestampWrites: options.timestamps?.('filter') })
        filterPass.setBindGroup(0, lossBindings)
        for (const filter of [filterResidualX, filterResidualY, filterGradientY, filterGradientX]) {
          filterPass.setPipeline(filter)
          filterPass.dispatchWorkgroups(...lossGrid)
        }
        filterPass.end()

        // Optical vector-Jacobian product overwrites heightGradient (and offsetGradient).
        forward.encodeBackward!(encoder, stepInput, result, pixelGradient, heightGradient,
          motion ? offsetGradient : undefined)

        const updatePass = encoder.beginComputePass({ label: 'Optimizer update', timestampWrites: options.timestamps?.('update') })
        updatePass.setBindGroup(0, updateBindings)
        if (settings.gradientSmoothingSigma > 0) {
          updatePass.setPipeline(smoothGradientX)
          updatePass.dispatchWorkgroups(...vertexGrid)
          updatePass.setPipeline(smoothGradientY)
          updatePass.dispatchWorkgroups(...vertexGrid)
        }
        updatePass.setPipeline(computeCurvature)
        updatePass.dispatchWorkgroups(...vertexGrid)
        updatePass.setPipeline(addRegularization)
        updatePass.dispatchWorkgroups(...vertexGrid)
        if (settings.updateSmoothingSigma > 0) {
          for (const stage of [adamStep, smoothUpdateX, applySmoothUpdateY]) {
            updatePass.setPipeline(stage)
            updatePass.dispatchWorkgroups(...vertexGrid)
          }
        } else {
          updatePass.setPipeline(adamUpdate)
          updatePass.dispatchWorkgroups(...vertexGrid)
        }
        if (motion) {
          updatePass.setBindGroup(0, motionBindings)
          const stages = [offsetCurvature, proposeOffsets, acceptOffsets, commitOffsets]
          if (settings.xySmoothingSigma > 0) stages.unshift(blurOffsetGradientX, blurOffsetGradientY)
          stages.unshift(tangentialGradient)
          for (const stage of stages) {
            updatePass.setPipeline(stage)
            updatePass.dispatchWorkgroups(...vertexGrid)
          }
          updatePass.setBindGroup(0, updateBindings)
          offsetsZero = false
        }
        updatePass.setPipeline(advanceStep)
        updatePass.dispatchWorkgroups(1)
        updatePass.end()

        iteration++
        // The loss belongs to the heights before this update.
        if (
          settings.lossLogInterval > 0 && lossState === 'idle' &&
          (iteration === 1 || iteration % settings.lossLogInterval === 0)
        ) {
          encoder.copyBufferToBuffer(lossPartials, 0, lossStaging, 0, lossGroups * 4)
          lossState = 'copied'
          lossRun = targetRun
          lossIteration = iteration - 1
        }

        done = iteration >= fineSettings.maxIterations
        return { lens, done, preview: result }
      },

      destroy() {
        if (destroyed) return
        destroyed = true
        coarse?.optimizer.destroy()
        for (const buffer of owned) buffer.destroy()
      },
    }
    return optimizer
  } catch (error) {
    for (const buffer of owned) buffer.destroy()
    throw error
  }
}
