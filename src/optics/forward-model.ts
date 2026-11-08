import type { ForwardInput, ForwardModel, ForwardResult } from '../contracts'
import shaderSource from './optics.wgsl?raw'
import cooperativeSource from './cooperative.wgsl?raw'

export interface OpticsOptions {
  strategy?: 'baseline' | 'cached' | 'cooperative' | 'striped'
  timestamps?: (stage: string) => GPUComputePassTimestampWrites | undefined
}

export async function createForwardModel(
  device: GPUDevice,
  initial: ForwardInput,
  options: OpticsOptions = {},
): Promise<ForwardModel | null> {
  const strategy = options.strategy ?? 'baseline'
  const cooperative = strategy === 'cooperative' || strategy === 'striped'
  const lanes = strategy === 'striped' ? 8 : 64
  const buffers: GPUBuffer[] = []
  const parameters = new Map<string, GPUBuffer>()
  const columns = initial.lens.columns
  const rows = initial.lens.rows
  const width = initial.receiver.width
  const height = initial.receiver.height
  const vertexCount = columns * rows
  const faceCount = 2 * (columns - 1) * (rows - 1)
  let disposed = false
  let lastResult: ForwardResult | null = null
  let lastParameters = ''

  function validate(input: ForwardInput) {
    const { lens, receiver, optics } = input
    const dimensions = [lens.columns, lens.rows, receiver.width, receiver.height]
    if (dimensions.some(v => !Number.isSafeInteger(v) || v < 1) || lens.columns < 2 || lens.rows < 2) {
      throw new Error('Invalid optical grid dimensions.')
    }
    if (lens.columns !== columns || lens.rows !== rows || receiver.width !== width || receiver.height !== height) {
      throw new Error('Recreate the forward model when grid or image resolution changes.')
    }
    const positive = [lens.sizeX, lens.sizeY, receiver.sizeX, receiver.sizeY, receiver.z, optics.lensIOR, optics.outsideIOR]
    if (positive.some(v => !Number.isFinite(v) || v <= 0) || !Number.isFinite(optics.incidentIrradiance) || optics.incidentIrradiance < 0) {
      throw new Error('Invalid optical dimensions or illumination.')
    }
    if (lens.heightBuffer.size < vertexCount * 4 || !(lens.heightBuffer.usage & GPUBufferUsage.STORAGE)) {
      throw new Error('The height buffer must contain one storage f32 per vertex.')
    }
  }

  function allocate(label: string, size: number, usage: GPUBufferUsageFlags) {
    if (!Number.isSafeInteger(size) || size > device.limits.maxBufferSize ||
        ((usage & GPUBufferUsage.STORAGE) && size > device.limits.maxStorageBufferBindingSize)) {
      throw new Error(`Optical buffer exceeds device limits: ${label}`)
    }
    const buffer = device.createBuffer({ label, size, usage })
    buffers.push(buffer)
    return buffer
  }

  function parameterKey(input: ForwardInput) {
    const { lens, receiver, optics } = input
    return JSON.stringify([lens.sizeX, lens.sizeY, receiver.sizeX, receiver.sizeY,
      receiver.z, optics.lensIOR / optics.outsideIOR, optics.incidentIrradiance])
  }

  function uniforms(input: ForwardInput) {
    const key = parameterKey(input)
    let buffer = parameters.get(key)
    if (!buffer) {
      const data = new ArrayBuffer(48)
      new Uint32Array(data, 0, 4).set([columns, rows, width, height])
      const { lens, receiver, optics } = input
      new Float32Array(data, 16).set([lens.sizeX, lens.sizeY, receiver.sizeX, receiver.sizeY,
        receiver.z, optics.lensIOR / optics.outsideIOR, optics.incidentIrradiance, 0])
      buffer = allocate('Optical parameters', 48, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)
      // Immutable parameter buffers preserve multiple evaluations in one encoder.
      device.queue.writeBuffer(buffer, 0, data)
      parameters.set(key, buffer)
    }
    return buffer
  }

  try {
    validate(initial)
    const forwardGroups = cooperative ? Math.ceil(faceCount / (64 / lanes)) : Math.ceil(faceCount / 64)
    if (forwardGroups > device.limits.maxComputeWorkgroupsPerDimension) {
      throw new Error('Optical dispatch exceeds the device workgroup limit.')
    }
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    const beams = allocate('Triangle beams', faceCount * 16, storage)
    const image = allocate('Receiver irradiance', width * height * 4, storage)
    const triangleGradients = allocate('Triangle height derivatives', faceCount * 16, storage)
    const footprints = allocate('Projected footprints', strategy === 'baseline' ? 128 : faceCount * 128, storage)
    const unusedPixelGradient = allocate('Unused pixel derivative', 4, storage)
    const unusedHeightGradient = allocate('Unused height derivative', 4, storage)
    const layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    })
    const module = device.createShaderModule({ label: 'Triangle optics', code: shaderSource + '\n' + cooperativeSource })
    const info = await module.getCompilationInfo()
    const errors = info.messages.filter(message => message.type === 'error')
    if (errors.length) {
      throw new Error(errors.map(message => `optics.wgsl:${message.lineNum}: ${message.message}`).join('\n'))
    }
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] })
    const [forward, backward, gather, project] = await Promise.all(
      [cooperative ? 'forwardCooperative' : 'forward',
       cooperative ? 'backwardCooperative' : 'backward', 'gather', 'prepareFootprints'].map(entryPoint => device.createComputePipelineAsync({
        label: `Optics ${entryPoint}`,
        layout: pipelineLayout,
        compute: { module, entryPoint, constants: { cacheFootprints: strategy === 'baseline' ? 0 : 1, lanesPerTriangle: lanes } },
      })),
    )

    function bind(input: ForwardInput, pixel = unusedPixelGradient, gradient = unusedHeightGradient) {
      return device.createBindGroup({
        layout,
        entries: [uniforms(input), input.lens.heightBuffer, beams, image, pixel, triangleGradients, gradient, footprints]
          .map((buffer, binding) => ({ binding, resource: { buffer } })),
      })
    }

    return {
      encode(encoder, input) {
        if (disposed) throw new Error('Forward model has been destroyed.')
        validate(input)
        encoder.clearBuffer(image)
        if (cooperative) {
          const prepare = encoder.beginComputePass({ label: 'Project footprints', timestampWrites: options.timestamps?.('project') })
          prepare.setPipeline(project)
          prepare.setBindGroup(0, bind(input))
          prepare.dispatchWorkgroups(Math.ceil(faceCount / 64))
          prepare.end()
        }
        const pass = encoder.beginComputePass({ label: 'Forward optics', timestampWrites: options.timestamps?.('forward') })
        pass.setPipeline(forward)
        pass.setBindGroup(0, bind(input))
        pass.dispatchWorkgroups(forwardGroups)
        pass.end()
        lastParameters = parameterKey(input)
        lastResult = {
          beams: { lens: { ...input.lens }, buffer: beams },
          receiver: { ...input.receiver },
          caustic: { kind: 'buffer', width, height, buffer: image },
        }
        return lastResult
      },

      encodeBackward(encoder, input, result, pixelGradient, heightGradient) {
        if (disposed) throw new Error('Forward model has been destroyed.')
        validate(input)
        if (result !== lastResult || result.beams.lens.heightBuffer !== input.lens.heightBuffer || parameterKey(input) !== lastParameters) {
          throw new Error('Backward requires the latest matching forward evaluation.')
        }
        if (pixelGradient.size < width * height * 4 || heightGradient.size < vertexCount * 4 ||
            !(pixelGradient.usage & GPUBufferUsage.STORAGE) || !(heightGradient.usage & GPUBufferUsage.STORAGE)) {
          throw new Error('Invalid optical gradient buffers.')
        }
        if (heightGradient === input.lens.heightBuffer || heightGradient === pixelGradient ||
            [beams, image, triangleGradients].includes(heightGradient) ||
            [beams, image, triangleGradients, input.lens.heightBuffer].includes(pixelGradient)) {
          throw new Error('Optical input and output buffers must not alias.')
        }
        const group = bind(input, pixelGradient, heightGradient)
        const pass = encoder.beginComputePass({ label: 'Backward optics', timestampWrites: options.timestamps?.('backward') })
        pass.setBindGroup(0, group)
        pass.setPipeline(backward)
        pass.dispatchWorkgroups(forwardGroups)
        pass.end()
        const gatherPass = encoder.beginComputePass({ label: 'Gradient gather', timestampWrites: options.timestamps?.('gather') })
        gatherPass.setBindGroup(0, group)
        gatherPass.setPipeline(gather)
        gatherPass.dispatchWorkgroups(Math.ceil(vertexCount / 64))
        gatherPass.end()
      },

      destroy() {
        if (disposed) return
        disposed = true
        lastResult = null
        parameters.clear()
        for (const buffer of buffers) buffer.destroy()
      },
    }
  } catch (error) {
    for (const buffer of buffers) buffer.destroy()
    throw error
  }
}
