import { config } from '../src/config'
import { createFlatLens } from '../src/geometry/lens-mesh'
import { createForwardModel } from '../src/optics/forward-model'
import { createLensOptimizer, normalizeTarget, incidentPower } from '../src/optimization/lens-optimizer'
import { createSimulation } from '../src/simulation'
import type { AppState, ForwardResult, TargetImage } from '../src/contracts'

const log: string[] = []
function say(text: string) {
  log.push(text)
  document.querySelector('#output')!.textContent = log.join('\n')
  console.info(text)
}
function check(ok: boolean, text: string) {
  if (!ok) throw new Error(text)
  say(`PASS ${text}`)
}
async function run() {
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) throw new Error('WebGPU unavailable.')
  const device = await adapter.requestDevice()
  const errors: string[] = []
  device.addEventListener('uncapturederror', e => errors.push(e.error.message))
  const initialLens = createFlatLens(device, { ...config.lens, columns: 33, rows: 33 })
  const state: AppState = { lens: initialLens, optics: { ...config.optics }, receiver: { ...config.receiver }, target: null }
  const strategy = new URLSearchParams(location.search).get('strategy')
  const forward = (await createForwardModel(device, state, {
    strategy: strategy === 'striped' ? 'striped' : strategy === 'cooperative' ? 'cooperative' : strategy === 'cached' ? 'cached' : 'baseline',
  }))!
  const optimizer = (await createLensOptimizer(device, state, forward))!
  check(!!optimizer, 'B accepts A encodeBackward')
  const simulation = createSimulation(state, forward, optimizer)
  const r = state.receiver
  const shape = new URLSearchParams(location.search).get('shape')
  const values = Float32Array.from({ length:r.width*r.height }, (_,i) => {
    const x = ((i%r.width+0.5)/r.width-0.5)*r.sizeX
    const y = ((Math.floor(i/r.width)+0.5)/r.height-0.5)*r.sizeY
    if (shape === 'ring') return x*x+y*y < 0.08**2 && x*x+y*y > 0.045**2 ? 1 : 0
    if (shape === 'two-circles') return Math.min((x-0.065)**2+y*y,(x+0.065)**2+y*y) < 0.035**2 ? 1 : 0
    if (shape === 'circle') return x*x+y*y < 0.06**2 ? 1 : 0
    if (shape === 'offset') return (x-0.052)**2+(y+0.032)**2 < 0.06**2 ? 1 : 0
    if (shape === 'bar') return Math.abs(x) < 0.092 && Math.abs(y) < 0.024 ? 1 : 0
    return Math.exp(-(x*x+y*y)/(2*0.045**2))
  })
  const target: TargetImage = { width:r.width,height:r.height,version:1,values }
  const normalized = normalizeTarget(target,r,incidentPower(state))!
  const mean = incidentPower(state)/(r.sizeX*r.sizeY)
  async function read(source: GPUBuffer, count: number) {
    const b = device.createBuffer({ size:count*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST })
    const e = device.createCommandEncoder()
    e.copyBufferToBuffer(source,0,b,0,count*4)
    device.queue.submit([e.finish()])
    await b.mapAsync(GPUMapMode.READ)
    const data = new Float32Array(b.getMappedRange().slice(0))
    b.unmap();b.destroy()
    return data
  }
  async function measure(result: ForwardResult) {
    if (result.caustic.kind !== 'buffer') throw new Error('Expected buffer.')
    const image = await read(result.caustic.buffer,values.length)
    return image.reduce((s,v,i)=>s+((v-normalized[i])/mean)**2,0)/values.length
  }
  let e = device.createCommandEncoder()
  let batch = simulation.encode(e,0)
  device.queue.submit([e.finish()])
  const initialLoss = await measure(batch.result)
  const losses = [{step:0,loss:initialLoss}]
  say(`Initial loss ${initialLoss}`)
  let steps = 0
  const start = performance.now()
  while (steps < config.optimizer.maxIterations) {
    optimizer.setTarget(target)
    e = device.createCommandEncoder()
    batch = simulation.encode(e,Math.min(5,config.optimizer.maxIterations-steps))
    device.queue.submit([e.finish()])
    await device.queue.onSubmittedWorkDone()
    steps += Math.min(5,config.optimizer.maxIterations-steps)
    if (steps%100===0) {
      const loss = await measure(batch.result)
      losses.push({step:steps,loss})
      say(`Step ${steps}: ${loss}`)
    }
    if (batch.done) break
  }
  const finalLoss = await measure(batch.result)
  check(batch.done && steps === config.optimizer.maxIterations,'stops after configured iteration limit')
  check(finalLoss < initialLoss*0.9,'real optical loss decreases by at least 10%')
  if (shape === 'circle' && batch.result.caustic.kind === 'buffer') {
    const image = await read(batch.result.caustic.buffer, values.length)
    const pixelArea = r.sizeX * r.sizeY / values.length
    const power = incidentPower(state)
    const received = image.reduce((sum,v) => sum+v*pixelArea,0) / power
    const outside = image.reduce((sum,v,i) => sum+(values[i] === 0 ? v*pixelArea : 0),0) / power
    check(outside < 0.1, 'circle spill is below 10% of incident power')
    check(received > 0.95 && received < 1.01, 'circle retains at least 95% of incident power')
    say(`Circle power: received ${received}, outside ${outside}`)
  }
  const optimized = await read(state.lens.heightBuffer,state.lens.columns*state.lens.rows)
  check(optimized.every(v=>Number.isFinite(v)&&v>=config.optimizer.minHeight-1e-9),'optimized heights remain finite and thick enough')
  check(optimized.some(v=>Math.abs(v-config.lens.thickness)>1e-5),'lens actually changes')
  check((await read(initialLens.heightBuffer,optimized.length)).every(v=>Math.abs(v-config.lens.thickness)<1e-8),'borrowed initial heights unchanged')
  // With vertex motion the heights belong to displaced vertex positions.
  const vertices = optimized.length
  const optimizedOffsets = state.lens.offsetBuffer ? await read(state.lens.offsetBuffer,2*vertices) : null
  optimizer.setTarget(target)
  e = device.createCommandEncoder()
  const finished = optimizer.encodeStep(e,state)
  check(finished.done && finished.lens.heightBuffer===state.lens.heightBuffer,'same target version remains done')
  optimizer.setTarget(null)
  e = device.createCommandEncoder()
  const empty = simulation.encode(e,1)
  device.queue.submit([e.finish()])
  check(empty.done,'empty target stops optimization')
  check((await read(state.lens.heightBuffer,optimized.length)).every((v,i)=>v===optimized[i]),'empty target preserves lens')
  if (optimizedOffsets) {
    check((await read(state.lens.offsetBuffer!,2*vertices)).every((v,i)=>v===optimizedOffsets[i]),'empty target preserves vertex positions')
  }
  optimizer.setTarget({...target,version:2})
  e = device.createCommandEncoder()
  const restarted = simulation.encode(e,1)
  device.queue.submit([e.finish()])
  check(!restarted.done,'new version restarts optimization')
  const warm = await read(state.lens.heightBuffer,optimized.length)
  // Each height moves by at most one Adam step. A vertex that slides along the
  // surface also follows its slope in height; the optimizer takes that slope
  // after the height step, so neighbor steps of up to 1.1 learningRate may
  // change the slide by the slope's sensitivity to them.
  const learningRate = config.optimizer.learningRate
  let expected = optimized
  let tolerance = (_: number) => learningRate*1.1
  if (optimizedOffsets) {
    const moved = await read(state.lens.offsetBuffer!,2*vertices)
    const { columns, rows } = state.lens
    const dx = state.lens.sizeX/(columns-1), dy = state.lens.sizeY/(rows-1)
    const xyStep = config.optimizer.xyLearningRate*1.1
    check(moved.every((v,i)=>Math.abs(v-optimizedOffsets[i])<=xyStep*(i%2===0 ? dx : dy)),'new target keeps the displaced vertices')
    if (config.optimizer.xyTangential > 0) {
      const slides = Array.from(optimized,(_,v)=>tangentialSlide(optimized,optimizedOffsets,moved,columns,rows,dx,dy,v))
      expected = optimized.map((h,v)=>Math.max(h+slides[v].height,config.optimizer.minHeight))
      tolerance = v => learningRate*(1.1+2.2*slides[v].sensitivity)
    }
  }
  let worst = 0, worstVertex = 0
  warm.forEach((v,i)=>{
    const ratio = Math.abs(v-expected[i])/tolerance(i)
    if (ratio > worst) { worst = ratio; worstVertex = i }
  })
  check(worst < 1,`new target warm-starts from current lens (worst vertex ${worstVertex}: off by ${Math.abs(warm[worstVertex]-expected[worstVertex]).toExponential(2)} m, tolerance ${tolerance(worstVertex).toExponential(2)} m)`)
  // A running batch displays the forward pass its last step already evaluated,
  // i.e. the heights before that update, instead of tracing the scene again.
  const previous = await measure(restarted.result)
  const before = device.createBuffer({ size: optimized.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
  device.queue.writeBuffer(before, 0, optimized)
  e = device.createCommandEncoder()
  const beforeOffsets = optimizedOffsets ? device.createBuffer({ size: optimizedOffsets.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }) : undefined
  if (beforeOffsets) device.queue.writeBuffer(beforeOffsets, 0, optimizedOffsets!)
  const repeated = forward.encode(e,{ ...state, lens: { ...state.lens, heightBuffer: before, offsetBuffer: beforeOffsets } })
  device.queue.submit([e.finish()])
  check(Math.abs((await measure(repeated))-previous)<1e-6,'running preview reuses the step forward of the pre-update heights')
  before.destroy()
  beforeOffsets?.destroy()
  await device.queue.onSubmittedWorkDone()
  check(errors.length===0,`no GPU validation errors: ${errors.join('; ')}`)
  say('ALL PASSED')
  ;(window as any).integrationResult = {passed:true,initialLoss,finalLoss,losses,steps,wallMs:performance.now()-start,adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture},log}
  optimizer.destroy();forward.destroy();initialLens.heightBuffer.destroy();device.destroy()
}
// The optimizer's tangential slide of vertex v by its x/y step: the slope from
// central differences between displaced neighbors (displacedSlope in
// optimizer.wgsl) dotted with the step, written as c1 * (h[xp] - h[xm]) +
// c2 * (h[yp] - h[ym]). sensitivity = |c1| + |c2| bounds how much changes of
// those height differences change the slide.
function tangentialSlide(heights: Float32Array, offsets: Float32Array, moved: Float32Array,
  columns: number, rows: number, dx: number, dy: number, v: number) {
  const column = v%columns, row = Math.floor(v/columns)
  const xm = column===0 ? v : v-1, xp = column+1===columns ? v : v+1
  const ym = row===0 ? v : v-columns, yp = row+1===rows ? v : v+columns
  const position = (i: number) => [(i%columns)*dx+offsets[2*i], Math.floor(i/columns)*dy+offsets[2*i+1]]
  const [ax,ay] = position(xm), [bx,by] = position(xp), [cx,cy] = position(ym), [ex,ey] = position(yp)
  const e1x = bx-ax, e1y = by-ay, e2x = ex-cx, e2y = ey-cy
  const determinant = e1x*e2y-e1y*e2x
  const sx = moved[2*v]-offsets[2*v], sy = moved[2*v+1]-offsets[2*v+1]
  if (!(Math.abs(determinant) > 0)) return { height: 0, sensitivity: 0 }
  const c1 = (e2y*sx-e2x*sy)/determinant, c2 = (e1x*sy-e1y*sx)/determinant
  return { height: c1*(heights[xp]-heights[xm])+c2*(heights[yp]-heights[ym]), sensitivity: Math.abs(c1)+Math.abs(c2) }
}
run().catch(error=>{
  say(`FAIL ${error.stack}`)
  ;(window as any).integrationResult={passed:false,error:String(error),log}
})
