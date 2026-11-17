import { config } from '../src/config'
import { createFlatLens } from '../src/geometry/lens-mesh'
import { buildDesignView } from '../src/optics/design-view'
import { createForwardModel } from '../src/optics/forward-model'
import { createLensOptimizer, type OptimizerSettings } from '../src/optimization/lens-optimizer'
import { lensShapeMetrics } from '../src/optimization/lens-metrics'
import { buildSceneGeometry } from '../src/rendering/scene-mesh'
import { createSimulation } from '../src/simulation'
import type { ForwardResult } from '../src/contracts'
import { bundledScenes, defaultScene, readBundledScene, type BundledScene } from '../src/geometry/bundled-scenes'
import { placeObjOnReceiver } from '../src/geometry/place-obj'
import { faceMetrics, faceWeights, strokeFractions, strokeWeights } from './stroke-case'

// /check/training-study.html?smoothnessWeight=30&maxIterations=1500&every=250
// Any config.optimizer key can be overridden from the query string. Extra keys:
//   grid    lens vertices per side (default 33)
//   every   checkpoint interval in iterations (default 250)
//   timing  GPU timestamp steps to average per pass (default 20; 0 disables)
//   scene   bundled scene id (default: the app's default scene)
//   target  face for the face case (default: the horizontal stroke)
// Reports image quality (stroke fractions), lens shape metrics and timing as JSON.
const lines: string[] = []
function log(value: unknown) {
  const text = typeof value === 'string' ? value : JSON.stringify(value)
  lines.push(text)
  document.querySelector('#output')!.textContent = lines.join('\n')
  console.log(text)
}

async function run() {
  const query = new URLSearchParams(location.search)
  const number = (key: string, fallback: number) => query.has(key) ? Number(query.get(key)) : fallback
  const settings: { -readonly [K in keyof OptimizerSettings]: number } = { ...config.optimizer, lossLogInterval: 0 }
  for (const key of Object.keys(settings) as (keyof OptimizerSettings)[]) settings[key] = number(key, settings[key])
  const grid = number('grid', 33), every = number('every', 250), timingSteps = number('timing', 20)

  const adapter = (await navigator.gpu.requestAdapter())!
  const canTime = timingSteps > 0 && adapter.features.has('timestamp-query')
  const device = await adapter.requestDevice({ requiredFeatures: canTime ? ['timestamp-query'] : [] })
  const errors: string[] = []
  device.addEventListener('uncapturederror', e => errors.push(e.error.message))

  // Timestamp plumbing: each labelled pass writes a begin/end pair while recording.
  const capacity = 32
  const querySet = canTime ? device.createQuerySet({ type: 'timestamp', count: capacity }) : null
  const resolve = device.createBuffer({ size: capacity * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC })
  const readTimes = device.createBuffer({ size: capacity * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
  let recording = false
  let stages: string[] = []
  const timestamps = (stage: string): GPUComputePassTimestampWrites | undefined => {
    if (!recording || !querySet || 2 * stages.length + 2 > capacity) return undefined
    stages.push(stage)
    const index = 2 * (stages.length - 1)
    return { querySet, beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1 }
  }

  const sceneId = (query.get('scene') ?? defaultScene) as BundledScene
  if (!(sceneId in bundledScenes)) throw new Error('Unknown scene ' + sceneId)
  const { mesh } = await readBundledScene(sceneId)
  const { receiver } = placeObjOnReceiver(mesh, config.lens, config.receiver)
  const lensSpec = { ...config.lens, columns: grid, rows: grid, sizeX: receiver.sizeX, sizeY: receiver.sizeY }
  const lens = createFlatLens(device, lensSpec)
  const state = { lens, receiver, optics: config.optics, target: null }
  const view = buildDesignView(buildSceneGeometry(receiver, { mesh }).mesh, receiver)
  const solver = (await createForwardModel(device, state, { designView: view, timestamps }))!
  // Coarse-to-fine runs need a forward model on the coarse grid as well.
  // The default coarse grid only applies where it properly divides this grid.
  if ((grid - 1) % settings.coarseGridCells !== 0 || grid - 1 <= settings.coarseGridCells) settings.coarseGridCells = 0
  const coarseCells = settings.coarseGridCells
  const coarseLens = coarseCells > 0
    ? createFlatLens(device, { ...lensSpec, columns: coarseCells + 1, rows: coarseCells + 1 }) : null
  const coarseForward = coarseLens
    ? (await createForwardModel(device, { ...state, lens: coarseLens }, { designView: view }))! : undefined
  const optimizer = (await createLensOptimizer(device, state, solver, settings, { timestamps, coarseForward }))!
  const sim = createSimulation(state, solver, optimizer)
  const face = query.get('target') === 'face'
  const weights = face ? faceWeights(receiver) : strokeWeights(receiver)
  const pixels = receiver.width * receiver.height, vertices = grid * grid
  const staging = device.createBuffer({ size: Math.max(pixels, 2 * vertices) * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
  async function read(source: GPUBuffer, count: number) {
    const enc = device.createCommandEncoder()
    enc.copyBufferToBuffer(source, 0, staging, 0, count * 4)
    device.queue.submit([enc.finish()])
    await staging.mapAsync(GPUMapMode.READ, 0, count * 4)
    const values = new Float32Array(staging.getMappedRange(0, count * 4).slice(0))
    staging.unmap()
    return values
  }

  // encodeStep calls until done: coarse calls (several coarse steps each) plus fine steps.
  const totalCalls = coarseCells > 0
    ? settings.refineIterations + Math.ceil(settings.coarseIterations / settings.coarseStepsPerCall)
    : settings.maxIterations
  log({ settings, grid, totalCalls, vertices, faces: 2 * (grid - 1) ** 2, timestampQuery: canTime, adapter: adapter.info?.description || adapter.info?.vendor })
  optimizer.setTarget({ version: 1, width: receiver.width, height: receiver.height, values: weights })

  // Per-pass GPU time, averaged over the first steps (widest loss filter).
  const passTotals = new Map<string, number>()
  let timed = 0
  let result: ForwardResult | null = null
  let iteration = 0
  if (canTime) {
    for (; timed < timingSteps && iteration < totalCalls; timed++, iteration++) {
      stages = []
      recording = true
      const enc = device.createCommandEncoder()
      result = sim.encode(enc, 1).result
      recording = false
      enc.resolveQuerySet(querySet!, 0, 2 * stages.length, resolve, 0)
      enc.copyBufferToBuffer(resolve, 0, readTimes, 0, 2 * stages.length * 8)
      device.queue.submit([enc.finish()])
      await readTimes.mapAsync(GPUMapMode.READ)
      const ticks = new BigUint64Array(readTimes.getMappedRange().slice(0))
      readTimes.unmap()
      // The trailing forward evaluation of sim.encode repeats sceneTrace/sceneForward: label it separately.
      const seen = new Set<string>()
      stages.forEach((stage, i) => {
        const label = seen.has(stage) ? `${stage} (preview)` : stage
        seen.add(stage)
        passTotals.set(label, (passTotals.get(label) ?? 0) + Number(ticks[2 * i + 1] - ticks[2 * i]) / 1e6)
      })
    }
    const perPass = Object.fromEntries([...passTotals].map(([k, v]) => [k, +(v / timed).toFixed(4)]))
    const step = [...passTotals].filter(([k]) => !k.endsWith('(preview)')).reduce((s, [, v]) => s + v, 0) / timed
    log({ gpuMsPerPass: perPass, gpuMsPerStep: +step.toFixed(4), timedSteps: timed })
  }

  const checkpoints = []
  let lastHeights = new Float32Array(0)
  let lastImage = new Float32Array(0)
  let trainingMs = 0
  const batch = 10
  while (iteration < totalCalls) {
    const target = Math.min(totalCalls, (Math.floor(iteration / every) + 1) * every)
    const start = performance.now()
    while (iteration < target) {
      const steps = Math.min(batch, target - iteration)
      const enc = device.createCommandEncoder()
      result = sim.encode(enc, steps).result
      device.queue.submit([enc.finish()])
      await device.queue.onSubmittedWorkDone()
      iteration += steps
    }
    trainingMs += performance.now() - start
    if (result!.caustic.kind !== 'buffer') throw new Error('Expected scene image buffer')
    const image = await read(result!.caustic.buffer, pixels)
    const heights = await read(state.lens.heightBuffer, vertices)
    const fractions = strokeFractions(image, weights, view)
    const faceStats = face ? faceMetrics(image, weights, view, receiver) : null
    const shape = lensShapeMetrics(heights, lensSpec, settings.maxSurfaceSlope)
    // Adam scale check: when sqrt(v) approaches epsilon, epsilon shrinks the step.
    const { heightGradient, moments } = (optimizer as any).diagnostics
    const gradient = await read(heightGradient, vertices)
    const mv = await read(moments, 2 * vertices)
    const rootV = Array.from({ length: vertices }, (_, i) => Math.sqrt(mv[2 * i + 1])).sort((a, b) => a - b)
    const point = {
      iteration,
      modelPct: +(100 * fractions.modelFraction).toFixed(2),
      scenePct: +(100 * fractions.sceneFraction).toFixed(2),
      ...(faceStats ? { eyeRecall: +faceStats.eyeRecall.toFixed(3), haze: +faceStats.haze.toFixed(4) } : {}),
      modelW: +fractions.model.toFixed(5),
      totalW: +fractions.total.toFixed(5),
      lapRms: +shape.laplacianRms.toFixed(3),
      rippleRms: +shape.rippleRms.toFixed(3),
      curvRms: +shape.curvatureRms.toExponential(3),
      curvP99: +shape.curvatureP99.toExponential(3),
      slopeOverPct: +(100 * shape.slopeExcessFraction).toFixed(3),
      maxSlope: +shape.maxSlope.toFixed(3),
      gradRms: +Math.sqrt(gradient.reduce((sum, g) => sum + g * g, 0) / vertices).toExponential(2),
      medianRootV: +rootV[vertices >> 1].toExponential(2),
      wallMsPerStep: +(trainingMs / (iteration - timed)).toFixed(2),
    }
    checkpoints.push(point)
    lastHeights = heights
    lastImage = image
    log(point)
  }
  const final = checkpoints[checkpoints.length - 1]
  const summary = { final, spill: spillBands(lastImage, weights, view.surfaceAreas, receiver), errors }
  ;(window as any).studyResult = { settings, grid, checkpoints, ...summary, heights: Array.from(lastHeights),
    image: Array.from(lastImage), dump: relativeDump(lastImage, weights, view.surfaceAreas, receiver) }
  log(summary)
  staging.destroy(); resolve.destroy(); readTimes.destroy(); querySet?.destroy()
  optimizer.destroy(); solver.destroy(); lens.heightBuffer.destroy()
  coarseForward?.destroy(); coarseLens?.heightBuffer.destroy(); device.destroy()
  log('STUDY DONE')
}
run().catch(error => { log('Error: ' + String(error)); console.error(error) })

// Share of the received power (%) by distance in pixels from the brush
// (weight >= 0.5): where the light that misses the drawing goes.
function spillBands(image: Float32Array, weights: Float32Array, areas: Float32Array, { width, height }: { width: number, height: number }) {
  const distance = new Float32Array(width * height).fill(Infinity)
  const queue: number[] = []
  for (let i = 0; i < distance.length; i++) if (weights[i] >= 0.5) { distance[i] = 0; queue.push(i) }
  // Breadth-first chamfer distance (4-neighbour steps of 1 px).
  for (let head = 0; head < queue.length; head++) {
    const i = queue[head], x = i % width, y = (i - x) / width
    for (const [nx, ny] of [[x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]]) {
      if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue
      const j = ny * width + nx
      if (distance[j] > distance[i] + 1) { distance[j] = distance[i] + 1; queue.push(j) }
    }
  }
  const edges = [0, 2, 4, 8, Infinity], bands = [0, 0, 0, 0, 0]
  let total = 0
  for (let i = 0; i < image.length; i++) {
    const power = image[i] * areas[i]
    total += power
    bands[edges.findIndex(edge => distance[i] <= edge)] += power
  }
  const pct = (value: number) => +(100 * value / Math.max(total, 1e-30)).toFixed(2)
  return { inside: pct(bands[0]), to2px: pct(bands[1]), to4px: pct(bands[2]), to8px: pct(bands[3]), beyond8px: pct(bands[4]) }
}

// Half-resolution map of irradiance relative to the mean inside the brush,
// one hex digit per pixel on a log scale: 0 below 1e-4, f at 1 or above.
function relativeDump(image: Float32Array, weights: Float32Array, areas: Float32Array, { width, height }: { width: number, height: number }) {
  let inside = 0, insideArea = 0
  for (let i = 0; i < image.length; i++) if (weights[i] >= 0.5) { inside += image[i] * areas[i]; insideArea += areas[i] }
  const reference = inside / Math.max(insideArea, 1e-30)
  const rows: string[] = []
  for (let y = 0; y + 1 < height; y += 2) {
    let row = ''
    for (let x = 0; x + 1 < width; x += 2) {
      const i = y * width + x
      const mean = 0.25 * (image[i] + image[i + 1] + image[i + width] + image[i + width + 1])
      const level = Math.round(15 * (1 + Math.log10(Math.max(mean / reference, 1e-5)) / 4))
      row += Math.max(0, Math.min(15, level)).toString(16)
    }
    rows.push(row)
  }
  return rows
}
