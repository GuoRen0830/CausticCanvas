import { config } from '../src/config'
import { createFlatLens } from '../src/geometry/lens-mesh'
import { buildDesignView } from '../src/optics/design-view'
import { createForwardModel } from '../src/optics/forward-model'
import { createLensOptimizer, type OptimizerSettings } from '../src/optimization/lens-optimizer'
import { lensShapeMetrics } from '../src/optimization/lens-metrics'
import { buildSceneMesh } from '../src/rendering/scene-mesh'
import { createSimulation } from '../src/simulation'
import type { ForwardResult } from '../src/contracts'
import { strokeBox, strokeFractions, strokeOptics, strokeReceiver, strokeWeights } from './stroke-case'

// /check/training-study.html?smoothnessWeight=30&maxIterations=1500&every=250
// Any config.optimizer key can be overridden from the query string. Extra keys:
//   grid    lens vertices per side (default 33, as in the ?stroke gate)
//   every   checkpoint interval in iterations (default 250)
//   timing  GPU timestamp steps to average per pass (default 20; 0 disables)
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

  const receiver = { ...strokeReceiver }
  const lensSpec = { columns: grid, rows: grid, sizeX: 0.2, sizeY: 0.2, thickness: 0.01 }
  const lens = createFlatLens(device, lensSpec)
  const state = { lens, receiver, optics: strokeOptics, target: null }
  const view = buildDesignView(buildSceneMesh(receiver, strokeBox), receiver)
  const solver = (await createForwardModel(device, state, { designView: view, timestamps }))!
  const optimizer = (await createLensOptimizer(device, state, solver, settings, { timestamps }))!
  const sim = createSimulation(state, solver, optimizer)
  const weights = strokeWeights()
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

  log({ settings, grid, vertices, faces: 2 * (grid - 1) ** 2, timestampQuery: canTime, adapter: adapter.info?.description || adapter.info?.vendor })
  optimizer.setTarget({ version: 1, width: 256, height: 256, values: weights })

  // Per-pass GPU time, averaged over the first steps (widest loss filter).
  const passTotals = new Map<string, number>()
  let timed = 0
  let result: ForwardResult | null = null
  let iteration = 0
  if (canTime) {
    for (; timed < timingSteps && iteration < settings.maxIterations; timed++, iteration++) {
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
  let trainingMs = 0
  const batch = 10
  while (iteration < settings.maxIterations) {
    const target = Math.min(settings.maxIterations, (Math.floor(iteration / every) + 1) * every)
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
    const shape = lensShapeMetrics(heights, lensSpec, settings.maxSurfaceSlope)
    // Adam scale check: when sqrt(v) approaches epsilon, epsilon shrinks the step.
    const { heightGradient, moments } = (optimizer as any).diagnostics
    const gradient = await read(heightGradient, vertices)
    const mv = await read(moments, 2 * vertices)
    const rootV = Array.from({ length: vertices }, (_, i) => Math.sqrt(mv[2 * i + 1])).sort((a, b) => a - b)
    const point = {
      iteration,
      topPct: +(100 * fractions.topFraction).toFixed(2),
      scenePct: +(100 * fractions.sceneFraction).toFixed(2),
      topW: +fractions.top.toFixed(5),
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
    log(point)
  }
  const gate = checkpoints.find(p => p.topPct > 82 && p.scenePct > 72 && p.topW > 0.005 && p.totalW > 0.025)
  const final = checkpoints[checkpoints.length - 1]
  const summary = { final, firstGateIteration: gate?.iteration ?? null, errors }
  ;(window as any).studyResult = { settings, grid, checkpoints, ...summary, heights: Array.from(lastHeights) }
  log(summary)
  staging.destroy(); resolve.destroy(); readTimes.destroy(); querySet?.destroy()
  optimizer.destroy(); solver.destroy(); lens.heightBuffer.destroy(); device.destroy()
  log('STUDY DONE')
}
run().catch(error => { log('Error: ' + String(error)); console.error(error) })
