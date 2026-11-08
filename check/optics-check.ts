import { createForwardModel } from '../src/optics/forward-model'
import type { ForwardInput } from '../src/contracts'
import { reference } from './optics-reference'

const output = document.querySelector('#output')!
const log: string[] = []
function assert(ok: boolean, message: string) {
  if (!ok) throw new Error(message)
  log.push(`PASS ${message}`)
  output.textContent = log.join('\n')
}
function relative(a: ArrayLike<number>, b: ArrayLike<number>) {
  let error = 0, scale = 0
  for (let i = 0; i < a.length; i++) { error += (a[i] - b[i]) ** 2; scale += b[i] ** 2 }
  return Math.sqrt(error / Math.max(scale, 1e-20))
}
async function run() {
  const adapter = await navigator.gpu?.requestAdapter()
  if (!adapter) throw new Error('WebGPU adapter unavailable.')
  const device = await adapter.requestDevice()
  const errors: string[] = []
  device.addEventListener('uncapturederror', e => errors.push(e.error.message))
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  function buffer(data: Float32Array) {
    const b = device.createBuffer({ size: data.byteLength, usage: storage })
    device.queue.writeBuffer(b, 0, data)
    return b
  }
  function copy(encoder: GPUCommandEncoder, source: GPUBuffer, length: number) {
    const b = device.createBuffer({ size: length * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    encoder.copyBufferToBuffer(source, 0, b, 0, length * 4)
    return b
  }
  async function read(b: GPUBuffer) {
    await b.mapAsync(GPUMapMode.READ)
    const data = new Float32Array(b.getMappedRange().slice(0))
    b.unmap(); b.destroy()
    return data
  }
  const h = new Float32Array([0.011,0.0123,0.0108,0.0131,0.0097,0.0114,0.012,0.0102,0.0133])
  const hb = buffer(h)
  const input: ForwardInput = {
    lens: { columns: 3, rows: 3, sizeX: 0.2, sizeY: 0.18, heightBuffer: hb },
    optics: { lensIOR: 1.5, outsideIOR: 1, incidentIrradiance: 1 },
    receiver: { width: 17, height: 19, sizeX: 0.43, sizeY: 0.39, z: 0.5 },
  }
  const model = (await createForwardModel(device, input))!
  const n = input.receiver.width * input.receiver.height
  async function evaluate(values: Float32Array, data = input) {
    device.queue.writeBuffer(hb, 0, values)
    const encoder = device.createCommandEncoder()
    const result = model.encode(encoder, data)
    if (result.caustic.kind !== 'buffer') throw new Error('Expected buffer image.')
    const imageCopy = copy(encoder, result.caustic.buffer, n)
    const beamCopy = copy(encoder, result.beams.buffer, 8 * 4)
    device.queue.submit([encoder.finish()])
    return { image: await read(imageCopy), beams: await read(beamCopy) }
  }
  for (const [name, values, data] of [
    ['flat', new Float32Array(9).fill(0.01), input],
    ['perturbed', h, input],
    ['equal indices', h, { ...input, optics: { ...input.optics, lensIOR: 1 } }],
    ['clipped', Float32Array.from(h, (_, i) => 0.01 + (i % 3) * 0.065), input],
    ['total internal reflection', Float32Array.from(h, (_, i) => 0.01 + (i % 3) * 0.1), input],
  ] as [string, Float32Array, ForwardInput][]) {
    const gpu = await evaluate(values, data)
    const cpu = reference(data, values)
    assert(relative(gpu.image, cpu.image) < 3e-4, `${name}: receiver agrees with CPU`)
    assert(relative(gpu.beams, cpu.beams) < 3e-5, `${name}: beam directions and flux agree with CPU`)
    const power = gpu.image.reduce((a,b) => a+b, 0) * data.receiver.sizeX * data.receiver.sizeY / n
    if (name === 'flat') assert(Math.abs(power-0.036) < 1e-6, 'flat received power = 0.036 W')
    if (name === 'clipped') assert(power < 0.035 && power > 0, 'escaped power is not renormalized')
  }
  const first = await evaluate(h), second = await evaluate(h)
  assert(relative(first.image, second.image) < 1e-6, 'repeated forward does not accumulate')
  const pixel = Float32Array.from({ length: n }, (_, i) => Math.sin(i*0.31)*0.03 + Math.cos(i*0.17)*0.02)
  const pg = buffer(pixel), hg = buffer(new Float32Array(9).fill(999))
  device.queue.writeBuffer(hb, 0, h)
  const encoder = device.createCommandEncoder()
  const result = model.encode(encoder, input)
  model.encodeBackward!(encoder, input, result, pg, hg)
  const gradientCopy = copy(encoder, hg, 9)
  device.queue.submit([encoder.finish()])
  const gradient = await read(gradientCopy)
  function objective(values: ArrayLike<number>) {
    return reference(input, values).image.reduce((s, v, i) => s + v * pixel[i], 0)
  }
  const finite = Array.from(h, (_, i) => {
    const plus = Array.from(h), minus = Array.from(h)
    plus[i] += 1e-7; minus[i] -= 1e-7
    return (objective(plus) - objective(minus)) / 2e-7
  })
  const error = relative(gradient, finite)
  log.push(`Gradient relative L2 error: ${error}`)
  assert(error < 0.003, 'all height gradients agree with independent CPU central differences')
  const gpuFinite = []
  for (let i = 0; i < h.length; i++) {
    const plus = h.slice(), minus = h.slice()
    plus[i] += 1e-5; minus[i] -= 1e-5
    const a = await evaluate(plus), b = await evaluate(minus)
    const value = a.image.reduce((s, v, j) => s + (v - b.image[j]) * pixel[j], 0) / (plus[i] - minus[i])
    gpuFinite.push(value)
  }
  assert(relative(gradient, gpuFinite) < 0.02, 'height gradients agree with GPU central differences')
  // Record two forward evaluations with different parameters before one submission.
  device.queue.writeBuffer(hb, 0, h)
  const batch = device.createCommandEncoder()
  const r1 = model.encode(batch, input)
  const c1 = copy(batch, (r1.caustic as { buffer: GPUBuffer }).buffer, n)
  const changed = { ...input, receiver: { ...input.receiver, z: 0.7 } }
  const r2 = model.encode(batch, changed)
  const c2 = copy(batch, (r2.caustic as { buffer: GPUBuffer }).buffer, n)
  device.queue.submit([batch.finish()])
  assert(relative(await read(c1), reference(input,h).image) < 3e-4, 'first same-batch parameters preserved')
  assert(relative(await read(c2), reference(changed,h).image) < 3e-4, 'second same-batch parameters preserved')
  const zero = buffer(new Float32Array(n))
  const batch2 = device.createCommandEncoder()
  const r3 = model.encode(batch2, input)
  model.encodeBackward!(batch2, input, r3, zero, hg)
  const c3 = copy(batch2, hg, 9)
  device.queue.submit([batch2.finish()])
  assert((await read(c3)).every(v => v === 0), 'backward overwrites every height gradient')
  const stale = device.createCommandEncoder()
  let rejected = false
  try { model.encodeBackward!(stale, input, r1, pg, hg) } catch { rejected = true }
  assert(rejected, 'stale forward result rejected')
  // One real optical gradient step must decrease the same objective.
  const target = new Float32Array(n)
  const targetImage = reference(input, Float32Array.from(h, (v,i) => v+(i===4 ? 0.001 : 0))).image
  const start = await evaluate(h)
  for (let i=0;i<n;i++) target[i] = 2*(start.image[i]-targetImage[i])/n
  device.queue.writeBuffer(pg, 0, target)
  const step = device.createCommandEncoder()
  const sr = model.encode(step, input)
  model.encodeBackward!(step,input,sr,pg,hg)
  const sg = copy(step,hg,9)
  device.queue.submit([step.finish()])
  const g = await read(sg)
  const adjusted = Float32Array.from(h,(v,i)=>v-1e-5*g[i])
  const after = await evaluate(adjusted)
  const loss = (a: ArrayLike<number>) => Array.from(a).reduce((s,v,i)=>s+(v-targetImage[i])**2,0)/n
  assert(loss(after.image)<loss(start.image), 'one true optical gradient step reduces image loss')
  const clippedHeights = Float32Array.from(h, (_, i) => 0.01 + (i % 3) * 0.065 + (i === 4 ? 0.00013 : 0))
  device.queue.writeBuffer(hb, 0, clippedHeights)
  device.queue.writeBuffer(pg, 0, pixel)
  const clippedEncoder = device.createCommandEncoder()
  const clippedResult = model.encode(clippedEncoder, input)
  model.encodeBackward!(clippedEncoder, input, clippedResult, pg, hg)
  const clippedCopy = copy(clippedEncoder, hg, 9)
  device.queue.submit([clippedEncoder.finish()])
  const clippedGradient = await read(clippedCopy)
  const clippedFinite = Array.from(clippedHeights, (_, i) => {
    const a = Array.from(clippedHeights), b = Array.from(clippedHeights)
    a[i] += 1e-7; b[i] -= 1e-7
    return (objective(a) - objective(b)) / 2e-7
  })
  assert(relative(clippedGradient, clippedFinite) < 0.003, 'clipped footprint gradients agree with central differences')
  const noRefraction = { ...input, optics: { ...input.optics, lensIOR: 1 } }
  const noEncoder = device.createCommandEncoder()
  const noResult = model.encode(noEncoder, noRefraction)
  model.encodeBackward!(noEncoder, noRefraction, noResult, pg, hg)
  const noCopy = copy(noEncoder, hg, 9)
  device.queue.submit([noEncoder.finish()])
  assert((await read(noCopy)).every(v => Math.abs(v) < 1e-6), 'equal-index optical height gradient is zero')
  model.destroy(); model.destroy()
  assert(hb.size === 36, 'borrowed height buffer retained')
  // Full default-size smoke check.
  const fullHeights = buffer(new Float32Array(33*33).fill(0.01))
  const full: ForwardInput = { ...input,
    lens: { ...input.lens, rows:33, columns:33, sizeY:0.2, heightBuffer:fullHeights },
    receiver: { width:256,height:256,sizeX:0.4,sizeY:0.4,z:0.5 },
  }
  const fullModel = (await createForwardModel(device, full))!
  const fullEncoder = device.createCommandEncoder()
  const fullResult = fullModel.encode(fullEncoder,full)
  const fullCopy = copy(fullEncoder,(fullResult.caustic as {buffer:GPUBuffer}).buffer,256*256)
  device.queue.submit([fullEncoder.finish()])
  const fullImage = await read(fullCopy)
  const power = fullImage.reduce((s,v)=>s+v,0)*0.16/(256*256)
  assert(Math.abs(power-0.04)<1e-6, 'default 33x33 / 256x256 flat power = 0.04 W')
  assert(fullImage.every(Number.isFinite), 'default image is finite')
  const desiredHeights = Float32Array.from({ length:33*33 }, (_,i) => {
    const x = (i%33)/32-0.5, y = Math.floor(i/33)/32-0.5
    return 0.012-0.004*(x*x+y*y)
  })
  const desiredBuffer = buffer(desiredHeights)
  const desiredInput = { ...full, lens: { ...full.lens, heightBuffer:desiredBuffer } }
  const desiredEncoder = device.createCommandEncoder()
  const desiredResult = fullModel.encode(desiredEncoder,desiredInput)
  const desiredCopy = copy(desiredEncoder,(desiredResult.caustic as {buffer:GPUBuffer}).buffer,256*256)
  device.queue.submit([desiredEncoder.finish()])
  const desiredImage = await read(desiredCopy)
  const flatPixel = buffer(Float32Array.from(fullImage,(v,i)=>2*(v-desiredImage[i])/fullImage.length))
  const flatGradient = buffer(new Float32Array(33*33))
  const flatEncoder = device.createCommandEncoder()
  const flatResult = fullModel.encode(flatEncoder,full)
  fullModel.encodeBackward!(flatEncoder,full,flatResult,flatPixel,flatGradient)
  const flatCopy = copy(flatEncoder,flatGradient,33*33)
  device.queue.submit([flatEncoder.finish()])
  const flatG = await read(flatCopy)
  assert(flatG.every(Number.isFinite) && flatG.some(v=>Math.abs(v)>1e-6), 'flat initialization has a finite nonzero gradient')
  const fullLoss = (a: Float32Array) => a.reduce((s,v,i)=>s+(v-desiredImage[i])**2,0)/a.length
  let improved = false
  for (const rate of [1e-4,1e-5,1e-6,1e-7]) {
    device.queue.writeBuffer(fullHeights,0,Float32Array.from(flatG,g=>0.01-rate*g))
    const e = device.createCommandEncoder()
    const r = fullModel.encode(e,full)
    const c = copy(e,(r.caustic as {buffer:GPUBuffer}).buffer,fullImage.length)
    device.queue.submit([e.finish()])
    if (fullLoss(await read(c)) < fullLoss(fullImage)) { improved=true; break }
  }
  assert(improved,'default flat lens can take a loss-reducing optical gradient step')
  desiredBuffer.destroy(); flatPixel.destroy(); flatGradient.destroy()
  fullModel.destroy()
  await device.queue.onSubmittedWorkDone()
  assert(errors.length === 0, `no WebGPU validation errors: ${errors.join('; ')}`)
  for (const b of [hb,pg,hg,zero,fullHeights]) b.destroy()
  device.destroy()
  output.textContent = log.join('\n') + '\nALL PASSED'
  ;(window as any).testResult = { passed:true, log }
}
run().catch(error => {
  output.textContent = log.join('\n') + '\nFAIL ' + error.stack
  ;(window as any).testResult = { passed:false, error:String(error), log }
})
