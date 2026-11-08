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

type OptimizerSettings = { [K in keyof typeof config.optimizer]: number }

const LOSS_WORKGROUP = 256
const VERTEX_WORKGROUP = 64
const PARAMS_SIZE = 80

// Total transmitted power P = incidentIrradiance * aperture area.
export function incidentPower(input: ForwardInput): number {
  return input.optics.incidentIrradiance * input.lens.sizeX * input.lens.sizeY
}

// Target irradiance E[i] = P * w[i] / (dA * sum(w)); null for empty targets.
export function normalizeTarget(
  target: Readonly<TargetImage>,
  receiver: Readonly<ReceiverPlane>,
  power: number,
): Float32Array | null {
  if (target.width !== receiver.width || target.height !== receiver.height) {
    throw new Error('Target size must match the receiver resolution.')
  }
  if (target.values.length !== target.width * target.height) {
    throw new Error('Target values do not match the target size.')
  }
  let sum = 0
  for (const value of target.values) {
    if (!Number.isFinite(value) || value < 0) {
      throw new Error('Target weights must be finite and nonnegative.')
    }
    sum += value
  }
  if (sum <= 0) return null
  const pixelArea = (receiver.sizeX / receiver.width) * (receiver.sizeY / receiver.height)
  const scale = power / (pixelArea * sum)
  const normalized = new Float32Array(target.values.length)
  for (let i = 0; i < normalized.length; i++) {
    normalized[i] = target.values[i] * scale
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
  const nonnegative = [settings.smoothnessWeight, settings.slopeWeight, settings.lossBlurSigma]
  if (
    positive.some((value) => !Number.isFinite(value) || value <= 0) ||
    nonnegative.some((value) => !Number.isFinite(value) || value < 0) ||
    !(settings.beta1 >= 0 && settings.beta1 < 1) ||
    !(settings.beta2 >= 0 && settings.beta2 < 1) ||
    !Number.isSafeInteger(settings.maxIterations) || settings.maxIterations < 1 ||
    !Number.isSafeInteger(settings.lossLogInterval) || settings.lossLogInterval < 0
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
  if (!(power > 0)) throw new Error('Incident power must be positive.')

  const vertexCount = lensMeta.columns * lensMeta.rows
  const pixelCount = receiver.width * receiver.height
  const lossGroups = Math.ceil(pixelCount / LOSS_WORKGROUP)
  const vertexGroups = Math.ceil(vertexCount / VERTEX_WORKGROUP)

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
    const lossPartials = createBuffer('Loss partial sums', lossGroups * 4, storage)
    const lossStaging = createBuffer(
      'Loss readback',
      lossGroups * 4,
      GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    )

    device.queue.writeBuffer(params, 0, writeParams(initial, settings))

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
      computeCurvature,
      addRegularization,
      adamUpdate,
      advanceStep,
    ] = await Promise.all([
      pipeline('lossFromBuffer', [lossLayout, causticLayouts.buffer]),
      pipeline('lossFromTexture', [lossLayout, causticLayouts.texture]),
      pipeline('filterResidualX', [lossLayout]),
      pipeline('filterResidualY', [lossLayout]),
      pipeline('filterGradientY', [lossLayout]),
      pipeline('filterGradientX', [lossLayout]),
      pipeline('computeCurvature', [updateLayout]),
      pipeline('addRegularization', [updateLayout]),
      pipeline('adamUpdate', [updateLayout]),
      pipeline('advanceStep', [updateLayout]),
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
    let lossIteration = 0
    const lossHistory: { iteration: number, loss: number }[] = []
    const flushLossReadback = () => {
      if (lossState !== 'copied' || destroyed) return
      lossState = 'mapping'
      const loggedIteration = lossIteration
      lossStaging.mapAsync(GPUMapMode.READ).then(() => {
        if (destroyed) return
        const values = new Float32Array(lossStaging.getMappedRange())
        let loss = 0
        for (const value of values) loss += value
        lossStaging.unmap()
        lossState = 'idle'
        lossHistory.push({ iteration: loggedIteration, loss })
        console.info(`[optimizer] iteration ${loggedIteration} loss ${loss.toExponential(4)}`)
      }, () => {
        // Device loss or destruction; logging is diagnostic only.
      })
    }

    const optimizer: LensOptimizer & { readonly lossHistory: typeof lossHistory } = {
      lossHistory,

      setTarget(target) {
        if (destroyed) return
        flushLossReadback()
        if (!target) {
          targetVersion = null
          active = false
          done = true
          return
        }
        if (target.version === targetVersion) return
        targetVersion = target.version
        const normalized = normalizeTarget(target, receiver, power)
        active = normalized !== null
        done = !active
        if (!normalized) return
        device.queue.writeBuffer(targetBuffer, 0, normalized)
        resetPending = true
        iteration = 0
        lossHistory.length = 0
      },

      encodeStep(encoder, input): OptimizationStepResult {
        if (destroyed) throw new Error('Optimizer was destroyed.')
        if (!active || done) return { lens: input.lens, done: true }
        if (!sameGrid(input.lens, lensMeta) || !sameReceiver(input.receiver, receiver)) {
          throw new Error('Lens grid and receiver must match the optimizer setup.')
        }

        if (resetPending) {
          // New target: restart Adam from the current lens shape.
          encoder.clearBuffer(moments)
          encoder.clearBuffer(stepState)
          resetPending = false
        }
        if (input.lens.heightBuffer !== heights) {
          encoder.copyBufferToBuffer(input.lens.heightBuffer, 0, heights, 0, vertexCount * 4)
        }
        const lens: LensState = { ...input.lens, heightBuffer: heights }
        const stepInput = { lens, optics: input.optics, receiver: input.receiver }

        const result = forward.encode(encoder, stepInput)
        if (result.caustic.width !== receiver.width || result.caustic.height !== receiver.height) {
          throw new Error('Forward image size does not match the receiver.')
        }

        const lossPass = encoder.beginComputePass({ label: 'Optimizer loss' })
        lossPass.setPipeline(result.caustic.kind === 'buffer' ? lossFromBuffer : lossFromTexture)
        lossPass.setBindGroup(0, lossBindings)
        lossPass.setBindGroup(1, causticBindingsFor(result.caustic))
        lossPass.dispatchWorkgroups(lossGroups)
        lossPass.end()

        const filterPass = encoder.beginComputePass({ label: 'Optimizer loss filter' })
        filterPass.setBindGroup(0, lossBindings)
        for (const filter of [filterResidualX, filterResidualY, filterGradientY, filterGradientX]) {
          filterPass.setPipeline(filter)
          filterPass.dispatchWorkgroups(lossGroups)
        }
        filterPass.end()

        // Optical vector-Jacobian product overwrites heightGradient.
        forward.encodeBackward!(encoder, stepInput, result, pixelGradient, heightGradient)

        const updatePass = encoder.beginComputePass({ label: 'Optimizer update' })
        updatePass.setBindGroup(0, updateBindings)
        updatePass.setPipeline(computeCurvature)
        updatePass.dispatchWorkgroups(vertexGroups)
        updatePass.setPipeline(addRegularization)
        updatePass.dispatchWorkgroups(vertexGroups)
        updatePass.setPipeline(adamUpdate)
        updatePass.dispatchWorkgroups(vertexGroups)
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
          lossIteration = iteration - 1
        }

        done = iteration >= settings.maxIterations
        return { lens, done }
      },

      destroy() {
        if (destroyed) return
        destroyed = true
        for (const buffer of owned) buffer.destroy()
      },
    }
    return optimizer
  } catch (error) {
    for (const buffer of owned) buffer.destroy()
    throw error
  }
}
