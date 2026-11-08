import { createLensOptimizer, normalizeTarget, type OptimizerSettings } from '../src/optimization/lens-optimizer'
import { config } from '../src/config'
import type { ForwardInput, ForwardModel, ForwardResult, LensState, TargetImage } from '../src/contracts'

const log: string[] = []
const say = (s: string) => { log.push(s); console.log(s) }
let failures = 0
const check = (ok: boolean, name: string) => { if (!ok) failures++; say(`${ok ? 'PASS' : 'FAIL'} ${name}`) }

async function readBuffer(device: GPUDevice, src: GPUBuffer, size: number) {
  const staging = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
  const enc = device.createCommandEncoder()
  enc.copyBufferToBuffer(src, 0, staging, 0, size)
  device.queue.submit([enc.finish()])
  await staging.mapAsync(GPUMapMode.READ)
  const out = new Float32Array(staging.getMappedRange().slice(0))
  staging.unmap(); staging.destroy()
  return out
}

// Toy differentiable "forward": E_j = mean + k * (h[v(j)] - h0). Not optics; test only.
const toyShader = /* wgsl */`
struct P { grid: vec2<u32>, image: vec2<u32>, k: f32, mean: f32, h0: f32, pad: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> h: array<f32>;
@group(0) @binding(2) var<storage, read_write> e: array<f32>;
@group(0) @binding(3) var<storage, read> g: array<f32>;
@group(0) @binding(4) var<storage, read_write> hg: array<f32>;
fn vertexOf(j: u32) -> u32 {
  let x = j % p.image.x; let y = j / p.image.x;
  let c = (x * p.grid.x) / p.image.x; let r = (y * p.grid.y) / p.image.y;
  return r * p.grid.x + c;
}
@compute @workgroup_size(64) fn fwd(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= p.image.x * p.image.y) { return; }
  e[id.x] = p.mean + p.k * (h[vertexOf(id.x)] - p.h0);
}
@compute @workgroup_size(64) fn bwd(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= p.grid.x * p.grid.y) { return; }
  var s = 0.0;
  for (var j = 0u; j < p.image.x * p.image.y; j++) { if (vertexOf(j) == id.x) { s += g[j]; } }
  hg[id.x] = p.k * s;
}`

function makeToyForward(device: GPUDevice, input: ForwardInput, kind: 'buffer' | 'texture', opts: { zeroOptical?: boolean }) {
  const { lens, receiver } = input
  const n = receiver.width * receiver.height
  const params = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  const mean = input.optics.incidentIrradiance * lens.sizeX * lens.sizeY / (receiver.sizeX * receiver.sizeY)
  const pd = new DataView(new ArrayBuffer(32))
  pd.setUint32(0, lens.columns, true); pd.setUint32(4, lens.rows, true)
  pd.setUint32(8, receiver.width, true); pd.setUint32(12, receiver.height, true)
  pd.setFloat32(16, 200, true); pd.setFloat32(20, mean, true); pd.setFloat32(24, 0.01, true)
  device.queue.writeBuffer(params, 0, pd.buffer)
  const image = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
  const texture = device.createTexture({ size: [receiver.width, receiver.height], format: 'r32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST })
  const view = texture.createView()
  const module = device.createShaderModule({ code: toyShader })
  const fwd = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'fwd' } })
  const bwd = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'bwd' } })
  const beams = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE })
  const captured: { pixelGradient?: GPUBuffer, heightGradient?: GPUBuffer } = {}
  const model: ForwardModel = {
    encode(encoder, inp): ForwardResult {
      const pass = encoder.beginComputePass()
      pass.setPipeline(fwd)
      pass.setBindGroup(0, device.createBindGroup({ layout: fwd.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: params } }, { binding: 1, resource: { buffer: inp.lens.heightBuffer } }, { binding: 2, resource: { buffer: image } }] }))
      pass.dispatchWorkgroups(Math.ceil(n / 64)); pass.end()
      if (kind === 'texture') encoder.copyBufferToTexture({ buffer: image, bytesPerRow: receiver.width * 4 }, { texture }, [receiver.width, receiver.height])
      return {
        beams: { lens: { ...inp.lens }, buffer: beams },
        receiver: inp.receiver,
        caustic: kind === 'buffer' ? { kind, width: receiver.width, height: receiver.height, buffer: image }
          : { kind, width: receiver.width, height: receiver.height, view },
      }
    },
    encodeBackward(encoder, _inp, _res, pixelGradient, heightGradient) {
      captured.pixelGradient = pixelGradient; captured.heightGradient = heightGradient
      if (opts.zeroOptical) { encoder.clearBuffer(heightGradient); return }
      const pass = encoder.beginComputePass()
      pass.setPipeline(bwd)
      pass.setBindGroup(0, device.createBindGroup({ layout: bwd.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: params } }, { binding: 3, resource: { buffer: pixelGradient } }, { binding: 4, resource: { buffer: heightGradient } }] }))
      pass.dispatchWorkgroups(Math.ceil(lens.columns * lens.rows / 64)); pass.end()
    },
    destroy() {},
  }
  return { model, captured, image }
}

function heightsBuffer(device: GPUDevice, values: Float32Array) {
  const b = device.createBuffer({ size: values.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST })
  device.queue.writeBuffer(b, 0, values)
  return b
}

function regLoss(h: Float64Array, cols: number, rows: number, dx: number, dy: number, s: OptimizerSettings) {
  const N = cols * rows, F = 2 * (cols - 1) * (rows - 1), ms = 0.5 * (dx + dy)
  let ls = 0
  for (let r = 1; r < rows - 1; r++) for (let c = 1; c < cols - 1; c++) {
    const i = r * cols + c
    const lap = (h[i - 1] + h[i + 1] - 2 * h[i]) / (dx * dx) + (h[i - cols] + h[i + cols] - 2 * h[i]) / (dy * dy)
    ls += (lap * ms) ** 2
  }
  let lt = 0
  const pen = (gx: number, gy: number) => Math.max(0, Math.hypot(gx, gy) - s.maxSurfaceSlope) ** 2
  for (let r = 0; r < rows - 1; r++) for (let c = 0; c < cols - 1; c++) {
    const a = r * cols + c, b = a + 1, cc = a + cols, d = cc + 1
    lt += pen((h[b] - h[a]) / dx, (h[cc] - h[a]) / dy) + pen((h[d] - h[cc]) / dx, (h[d] - h[b]) / dy)
  }
  return s.smoothnessWeight * ls / N + s.slopeWeight * lt / F
}

async function main() {
  const adapter = await navigator.gpu.requestAdapter()
  const device = await adapter!.requestDevice()
  device.addEventListener('uncapturederror', (e) => { failures++; say('GPU ERROR ' + (e as GPUUncapturedErrorEvent).error.message) })

  const cols = 9, rows = 9
  const receiver = { z: 0.5, sizeX: 0.4, sizeY: 0.4, width: 64, height: 64 }
  const optics = { ...config.optics }
  // Disable loss filtering so the pixel gradient has a closed form; loss-filter-check covers the filter.
  const settings: OptimizerSettings = { ...config.optimizer, maxIterations: 400, lossLogInterval: 50, lossBlurSigma: 0 }
  const dx = 0.2 / (cols - 1), dy = 0.2 / (rows - 1)
  const lensOf = (buf: GPUBuffer): LensState => ({ columns: cols, rows, sizeX: 0.2, sizeY: 0.2, heightBuffer: buf })

  // 1. Target normalization conserves power.
  const w = new Float32Array(64 * 64); for (let i = 0; i < 300; i++) w[(i * 37) % w.length] = Math.random()
  const T = normalizeTarget({ width: 64, height: 64, version: 1, values: w }, receiver, 0.04)!
  const dA = (0.4 / 64) ** 2
  const total = T.reduce((a, b) => a + b, 0) * dA
  check(Math.abs(total - 0.04) < 1e-6, `normalized target power ${total.toFixed(8)} W == 0.04 W`)
  check(normalizeTarget({ width: 64, height: 64, version: 2, values: new Float32Array(64 * 64) }, receiver, 0.04) === null, 'empty target normalizes to null')

  // 2. Regularization gradient vs finite differences (optical gradient zeroed).
  {
    const h = new Float32Array(cols * rows)
    for (let i = 0; i < h.length; i++) h[i] = 0.01 + 0.03 * Math.sin(i * 1.7) * Math.cos(i * 0.37)
    const input: ForwardInput = { lens: lensOf(heightsBuffer(device, h)), optics, receiver }
    const toy = makeToyForward(device, input, 'buffer', { zeroOptical: true })
    const opt = (await createLensOptimizer(device, input, toy.model, settings))!
    opt.setTarget({ width: 64, height: 64, version: 1, values: w })
    const enc = device.createCommandEncoder()
    opt.encodeStep(enc, input)
    device.queue.submit([enc.finish()])
    const gpu = await readBuffer(device, toy.captured.heightGradient!, cols * rows * 4)
    const h64 = Float64Array.from(h)
    let maxErr = 0, maxMag = 0
    for (let k = 0; k < h.length; k++) {
      const eps = 1e-6, save = h64[k]
      h64[k] = save + eps; const lp = regLoss(h64, cols, rows, dx, dy, settings)
      h64[k] = save - eps; const lm = regLoss(h64, cols, rows, dx, dy, settings)
      h64[k] = save
      const fd = (lp - lm) / (2 * eps)
      maxErr = Math.max(maxErr, Math.abs(fd - gpu[k])); maxMag = Math.max(maxMag, Math.abs(fd))
    }
    check(maxMag > 0 && maxErr / maxMag < 1e-3, `regularization gradient rel err ${(maxErr / maxMag).toExponential(2)} (max |g| ${maxMag.toExponential(2)})`)
    opt.destroy()
  }

  // 3. Pixel gradient, convergence, lifecycle, for both caustic kinds.
  for (const kind of ['buffer', 'texture'] as const) {
    const h = new Float32Array(cols * rows).fill(0.01)
    const input: ForwardInput = { lens: lensOf(heightsBuffer(device, h)), optics, receiver }
    const toy = makeToyForward(device, input, kind, {})
    const opt = (await createLensOptimizer(device, input, toy.model, settings))! as any
    // Smooth reachable target so the toy problem has a low-loss solution.
    const tw = new Float32Array(64 * 64)
    for (let y = 0; y < 64; y++) for (let x = 0; x < 64; x++) tw[y * 64 + x] = 0.5 + 0.4 * Math.sin(x / 10) * Math.cos(y / 13)
    const target: TargetImage = { width: 64, height: 64, version: 7, values: tw }
    const before = Float32Array.from(tw)
    opt.setTarget(target)
    check(tw.every((v, i) => v === before[i]), `${kind}: setTarget leaves caller data unchanged`)

    // One step, then compare captured pixel gradient with CPU.
    let lens = input.lens
    let enc = device.createCommandEncoder()
    let r = opt.encodeStep(enc, { lens, optics, receiver }); lens = r.lens
    device.queue.submit([enc.finish()])
    check(lens.heightBuffer !== input.lens.heightBuffer, `${kind}: step returns optimizer-owned heights`)
    const Tn = normalizeTarget(target, receiver, 0.04)!
    const E = await readBuffer(device, toy.image, 64 * 64 * 4)
    const g = await readBuffer(device, toy.captured.pixelGradient!, 64 * 64 * 4)
    const mean = 0.04 / 0.16
    let gErr = 0, gMax = 0
    for (let j = 0; j < g.length; j++) {
      // E was overwritten by the step's forward on old heights? Recompute from flat heights: E = mean.
      const ref = 2 * (mean - Tn[j]) / (mean * mean * g.length)
      gErr = Math.max(gErr, Math.abs(ref - g[j])); gMax = Math.max(gMax, Math.abs(ref))
    }
    void E
    check(gErr / gMax < 1e-5, `${kind}: pixel gradient matches 2(E-T)/(mean^2 N), rel err ${(gErr / gMax).toExponential(2)}`)

    // Multi-step batches until done.
    let batches = 0, done = false
    while (!done && batches < 1000) {
      opt.setTarget(target) // same version: must not reset
      enc = device.createCommandEncoder()
      for (let s = 0; s < 4 && !done; s++) { r = opt.encodeStep(enc, { lens, optics, receiver }); lens = r.lens; done = r.done }
      device.queue.submit([enc.finish()])
      await device.queue.onSubmittedWorkDone()
      batches++
    }
    await new Promise((res) => setTimeout(res, 200))
    // Let a pending loss readback flush.
    opt.setTarget(target); await device.queue.onSubmittedWorkDone(); await new Promise((res) => setTimeout(res, 200))
    const hist: { iteration: number, loss: number }[] = opt.lossHistory
    say(`${kind}: loss history ${hist.map((p) => `${p.iteration}:${p.loss.toExponential(2)}`).join(' ')}`)
    check(done && batches === 100, `${kind}: done after maxIterations (${batches} batches of 4)`)
    // Best achievable toy loss: each vertex block can only match its block mean.
    const sums = new Float64Array(cols * rows), counts = new Float64Array(cols * rows)
    const vOf = (j: number) => Math.floor(((j % 64) * cols) / 64) + cols * Math.floor((Math.floor(j / 64) * rows) / 64)
    Tn.forEach((t, j) => { sums[vOf(j)] += t; counts[vOf(j)]++ })
    let floor = 0
    Tn.forEach((t, j) => { floor += ((sums[vOf(j)] / counts[vOf(j)] - t) / mean) ** 2 / Tn.length })
    const last = hist[hist.length - 1].loss
    say(`${kind}: final loss ${last.toExponential(3)}, achievable floor ${floor.toExponential(3)}`)
    check(hist.length >= 3 && last < 1.1 * floor + 1e-4 && last < 0.1 * hist[0].loss, `${kind}: loss reaches the toy optimum`)
    const r2 = opt.encodeStep(device.createCommandEncoder(), { lens, optics, receiver })
    check(r2.done && r2.lens === lens, `${kind}: stays done without new version`)
    opt.setTarget(null)
    const r3 = opt.encodeStep(device.createCommandEncoder(), { lens, optics, receiver })
    check(r3.done && r3.lens.heightBuffer === lens.heightBuffer, `${kind}: null target keeps lens, done`)
    opt.setTarget({ ...target, version: 8 })
    enc = device.createCommandEncoder()
    const r4 = opt.encodeStep(enc, { lens, optics, receiver })
    device.queue.submit([enc.finish()])
    check(!r4.done, `${kind}: new version restarts`)
    const hs = await readBuffer(device, r4.lens.heightBuffer, cols * rows * 4)
    check(hs.every((v) => Number.isFinite(v) && v >= settings.minHeight - 1e-9), `${kind}: heights finite and >= minHeight`)
    opt.destroy()
  }

  // 4. No backward -> null.
  const noBackward: ForwardModel = { encode: () => { throw new Error('x') }, destroy() {} }
  const none = await createLensOptimizer(device, { lens: lensOf(heightsBuffer(device, new Float32Array(81))), optics, receiver }, noBackward)
  check(none === null, 'returns null when forward has no encodeBackward')

  await device.queue.onSubmittedWorkDone()
  say(failures === 0 ? 'ALL PASSED' : `${failures} FAILURES`)
  ;(window as any).__done = log
}
main().catch((e) => { say('EXCEPTION ' + e.stack); (window as any).__done = log })
