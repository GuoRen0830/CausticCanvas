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
//   diagnose 1 to report where on the lens the far spill comes from (slow readback)
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
  let lastOffsets: Float32Array | undefined
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
    // Vertex motion: how far vertices moved (in cells) and the smallest face.
    const offsets = state.lens.offsetBuffer ? await read(state.lens.offsetBuffer, 2 * vertices) : undefined
    const shape = lensShapeMetrics(heights, lensSpec, settings.maxSurfaceSlope, offsets)
    // Adam scale check: when sqrt(v) approaches epsilon, epsilon shrinks the step.
    const { heightGradient, moments } = (optimizer as any).diagnostics
    const gradient = await read(heightGradient, vertices)
    const mv = await read(moments, 2 * vertices)
    const rootV = Array.from({ length: vertices }, (_, i) => Math.sqrt(mv[2 * i + 1])).sort((a, b) => a - b)
    const motion = offsets ? meshMotion(offsets, grid, lensSpec) : undefined
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
      ...motion,
    }
    checkpoints.push(point)
    lastHeights = heights
    lastImage = image
    lastOffsets = offsets
    log(point)
  }
  const final = checkpoints[checkpoints.length - 1]
  const diagnosis = query.get('diagnose') === '1' && result?.firstHits
    ? await diagnoseSpill(device, result, grid, lensSpec, receiver, weights, lastOffsets) : undefined
  const summary = { final, spill: spillBands(lastImage, weights, view.surfaceAreas, receiver), diagnosis, errors }
  ;(window as any).studyResult = { settings, grid, checkpoints, ...summary, heights: Array.from(lastHeights),
    image: Array.from(lastImage), dump: relativeDump(lastImage, weights, view.surfaceAreas, receiver) }
  log(summary)
  staging.destroy(); resolve.destroy(); readTimes.destroy(); querySet?.destroy()
  optimizer.destroy(); solver.destroy(); lens.heightBuffer.destroy()
  coarseForward?.destroy(); coarseLens?.heightBuffer.destroy(); device.destroy()
  log('STUDY DONE')
}
run().catch(error => { log('Error: ' + String(error)); console.error(error) })

// Largest and mean vertex displacement in cells, and the smallest face area
// as a fraction of its grid area.
function meshMotion(offsets: Float32Array, grid: number, lens: { sizeX: number, sizeY: number }) {
  const cells = grid - 1, dx = lens.sizeX / cells, dy = lens.sizeY / cells
  let largest = 0, total = 0, smallest = Infinity
  for (let v = 0; v < grid * grid; v++) {
    const shift = Math.hypot(offsets[2 * v] / dx, offsets[2 * v + 1] / dy)
    largest = Math.max(largest, shift)
    total += shift
  }
  const at = (v: number, k: number) => (k ? Math.floor(v / grid) * dy : (v % grid) * dx) + offsets[2 * v + k]
  for (let r = 0; r < cells; r++) {
    for (let c = 0; c < cells; c++) {
      const a = r * grid + c
      for (const [i, j, k] of [[a, a + 1, a + grid], [a + 1, a + grid + 1, a + grid]]) {
        const area = (at(j, 0) - at(i, 0)) * (at(k, 1) - at(i, 1)) - (at(k, 0) - at(i, 0)) * (at(j, 1) - at(i, 1))
        smallest = Math.min(smallest, area / (dx * dy))
      }
    }
  }
  return { maxShiftCells: +largest.toFixed(3), meanShiftCells: +(total / (grid * grid)).toFixed(4),
    minAreaRatio: +smallest.toFixed(3) }
}

// Distance in pixels from the brush (weight >= 0.5), in 4-neighbour steps.
function brushDistance(weights: Float32Array, width: number, height: number) {
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
  return distance
}

// Share of the received power (%) by distance in pixels from the brush
// (weight >= 0.5): where the light that misses the drawing goes.
function spillBands(image: Float32Array, weights: Float32Array, areas: Float32Array, { width, height }: { width: number, height: number }) {
  const distance = brushDistance(weights, width, height)
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

// Where on the lens the light that lands more than 3 px from the brush comes
// from. Each lens triangle's beam lands at its centroid plus t times its
// direction. Its stretch is how far (in light-view pixels) it lands from its
// neighbouring triangles: large across a crease, small in a smooth region.
// lensMap: share of each 8 x 8-cell block's power that spills (0 below 1e-4,
// f = all of it, log scale); creaseMap: largest stretch in the block
// (4 = 1 px, 8 = 10 px, 12 = 100 px).
async function diagnoseSpill(device: GPUDevice, result: ForwardResult, grid: number,
  lens: { sizeX: number, sizeY: number }, receiver: { width: number, height: number },
  weights: Float32Array, offsets?: Float32Array) {
  const faces = 2 * (grid - 1) * (grid - 1), cells = grid - 1
  async function readAll(source: GPUBuffer, bytes: number) {
    const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    const enc = device.createCommandEncoder()
    enc.copyBufferToBuffer(source, 0, staging, 0, bytes)
    device.queue.submit([enc.finish()])
    await staging.mapAsync(GPUMapMode.READ)
    const data = staging.getMappedRange().slice(0)
    staging.unmap()
    staging.destroy()
    return data
  }
  const beams = new Float32Array(await readAll(result.beams.buffer, faces * 16))
  const hitData = await readAll(result.firstHits!.buffer, faces * 8)
  const hitIds = new Uint32Array(hitData), hitT = new Float32Array(hitData)
  const { width, height } = receiver
  const distance = brushDistance(weights, width, height)
  const landX = new Float32Array(faces).fill(NaN), landY = new Float32Array(faces).fill(NaN)
  for (let f = 0; f < faces; f++) {
    if (!hitIds[2 * f] || !(beams[4 * f + 3] > 0)) continue
    const cell = f >> 1, row = Math.floor(cell / cells), col = cell % cells
    const a = row * grid + col
    const ids = f % 2 === 0 ? [a, a + 1, a + grid] : [a + 1, a + grid + 1, a + grid]
    let cx = 0, cy = 0
    for (const v of ids) {
      cx += (((v % grid) / cells - 0.5) * lens.sizeX + (offsets?.[2 * v] ?? 0)) / 3
      cy += ((Math.floor(v / grid) / cells - 0.5) * lens.sizeY + (offsets?.[2 * v + 1] ?? 0)) / 3
    }
    const t = hitT[2 * f + 1]
    landX[f] = ((cx + t * beams[4 * f]) / lens.sizeX + 0.5) * width
    landY[f] = ((cy + t * beams[4 * f + 1]) / lens.sizeY + 0.5) * height
  }
  const apart = (f: number, g: number) => g < 0 || Number.isNaN(landX[g]) ? 0
    : Math.hypot(landX[f] - landX[g], landY[f] - landY[g])
  const edges = [0.5, 2, 8, 32, Infinity]
  const spillByStretch = edges.map(() => 0), allByStretch = edges.map(() => 0)
  const blocks = 128, per = cells / blocks
  const blockPower = new Float64Array(blocks * blocks), blockSpill = new Float64Array(blocks * blocks)
  const blockStretch = new Float64Array(blocks * blocks)
  let total = 0, spill = 0, offImage = 0
  for (let f = 0; f < faces; f++) {
    if (Number.isNaN(landX[f])) continue
    const power = beams[4 * f + 3]
    const cell = f >> 1, row = Math.floor(cell / cells), col = cell % cells
    const stretch = Math.max(apart(f, f ^ 1), apart(f, col + 1 < cells ? 2 * (cell + 1) : -1),
      apart(f, row + 1 < cells ? 2 * (cell + cells) + 1 : -1))
    const bin = edges.findIndex(edge => stretch <= edge)
    const block = Math.floor(row / per) * blocks + Math.floor(col / per)
    total += power
    allByStretch[bin] += power
    blockPower[block] += power
    blockStretch[block] = Math.max(blockStretch[block], stretch)
    const px = Math.floor(landX[f]), py = Math.floor(landY[f])
    if (!(px >= 0 && py >= 0 && px < width && py < height)) { offImage += power; continue }
    if (distance[py * width + px] > 3) { spill += power; spillByStretch[bin] += power; blockSpill[block] += power }
  }
  const pct = (value: number, of: number) => +(100 * value / Math.max(of, 1e-30)).toFixed(2)
  const lensMap: string[] = [], creaseMap: string[] = []
  for (let r = 0; r < blocks; r++) {
    let spillRow = '', creaseRow = ''
    for (let c = 0; c < blocks; c++) {
      const b = r * blocks + c
      const share = blockSpill[b] / Math.max(blockPower[b], 1e-30)
      spillRow += Math.max(0, Math.min(15, Math.round(15 * (1 + Math.log10(Math.max(share, 1e-5)) / 4)))).toString(16)
      creaseRow += Math.max(0, Math.min(15, Math.round(4 + 4 * Math.log10(Math.max(blockStretch[b], 1e-3))))).toString(16)
    }
    lensMap.push(spillRow)
    creaseMap.push(creaseRow)
  }
  return {
    spillPct: pct(spill, total), offImagePct: pct(offImage, total),
    stretchBinsPx: '<=0.5, <=2, <=8, <=32, >32',
    spillByStretchPct: spillByStretch.map(v => pct(v, spill)),
    allByStretchPct: allByStretch.map(v => pct(v, total)),
    lensMap, creaseMap,
  }
}
