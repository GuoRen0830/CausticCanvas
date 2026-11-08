// Compare optimizer settings on the real optics; ?variant=name&shape=circle&size=128&iterations=1500
import { config } from '../src/config'
import { createFlatLens } from '../src/geometry/lens-mesh'
import { createForwardModel } from '../src/optics/forward-model'
import { createLensOptimizer, incidentPower, normalizeTarget, type OptimizerSettings } from '../src/optimization/lens-optimizer'
import type { ForwardInput, TargetImage } from '../src/contracts'

const variants: Record<string, Partial<OptimizerSettings>> = {
  adjusted: {},
  original: { lossBlurSigma: 0, smoothnessWeight: 0.1 },
  blurOnly: { smoothnessWeight: 0.1 },
  smoothOnly: { lossBlurSigma: 0 },
}

async function run() {
  const query = new URLSearchParams(location.search)
  const name = query.get('variant') ?? 'adjusted'
  const shape = query.get('shape') ?? 'circle'
  const size = Number(query.get('size') ?? 128)
  const iterations = Number(query.get('iterations') ?? 1500)
  const adapter = (await navigator.gpu.requestAdapter())!
  const device = await adapter.requestDevice()
  const errors: string[] = []
  device.addEventListener('uncapturederror', (e) => errors.push(e.error.message))
  const receiver = { ...config.receiver, width: size, height: size }
  const initialLens = createFlatLens(device, config.lens)
  const input: ForwardInput = { lens: initialLens, optics: { ...config.optics }, receiver }
  const forward = (await createForwardModel(device, input))!
  const settings = { ...config.optimizer, ...variants[name], maxIterations: iterations, lossLogInterval: 0 }
  const optimizer = (await createLensOptimizer(device, input, forward, settings))!
  const values = Float32Array.from({ length: size * size }, (_, i) => {
    const x = ((i % size + 0.5) / size - 0.5) * receiver.sizeX
    const y = ((Math.floor(i / size) + 0.5) / size - 0.5) * receiver.sizeY
    if (shape === 'two-circles') return Math.min((x - 0.065) ** 2 + y * y, (x + 0.065) ** 2 + y * y) < 0.035 ** 2 ? 1 : 0
    if (shape === 'ring') return x * x + y * y < 0.08 ** 2 && x * x + y * y > 0.045 ** 2 ? 1 : 0
    return x * x + y * y < 0.06 ** 2 ? 1 : 0
  })
  const target: TargetImage = { width: size, height: size, version: 1, values }
  const power = incidentPower(input)
  const normalized = normalizeTarget(target, receiver, power)!
  const mean = power / (receiver.sizeX * receiver.sizeY)
  const pixelArea = receiver.sizeX * receiver.sizeY / (size * size)
  async function evaluate(lens: ForwardInput['lens']) {
    const e = device.createCommandEncoder()
    const result = forward.encode(e, { ...input, lens })
    if (result.caustic.kind !== 'buffer') throw new Error('Expected buffer')
    const staging = device.createBuffer({ size: size * size * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    e.copyBufferToBuffer(result.caustic.buffer, 0, staging, 0, size * size * 4)
    device.queue.submit([e.finish()])
    await staging.mapAsync(GPUMapMode.READ)
    const image = new Float32Array(staging.getMappedRange().slice(0))
    staging.unmap(); staging.destroy()
    let loss = 0, received = 0, outside = 0
    image.forEach((v, i) => {
      loss += ((v - normalized[i]) / mean) ** 2 / image.length
      received += v * pixelArea / power
      if (values[i] === 0) outside += v * pixelArea / power
    })
    return { loss, received, outside }
  }
  const curve = [{ step: 0, ...(await evaluate(initialLens)) }]
  let lens = initialLens
  let step = 0, done = false
  const start = performance.now()
  while (!done) {
    optimizer.setTarget(target)
    const e = device.createCommandEncoder()
    for (let s = 0; s < 5 && !done; s++) {
      const r = optimizer.encodeStep(e, { ...input, lens })
      lens = r.lens; done = r.done; step++
    }
    device.queue.submit([e.finish()])
    await device.queue.onSubmittedWorkDone()
    if (step % 250 === 0 || done) {
      curve.push({ step, ...(await evaluate(lens)) })
      console.info(`${name} ${shape} step ${step} ${JSON.stringify(curve[curve.length - 1])}`)
    }
  }
  const result = { name, shape, size, iterations, curve, errors, seconds: (performance.now() - start) / 1000 }
  document.querySelector('#output')!.textContent = JSON.stringify(result, null, 1)
  ;(window as any).ablationResult = result
}
run().catch((e) => { (window as any).ablationResult = { error: String(e.stack) } })
