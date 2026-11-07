import type { CausticImage } from '../contracts'
import shaderSource from './shaders/caustic-display.wgsl?raw'

async function createCausticPipeline(
  device: GPUDevice,
  format: GPUTextureFormat,
  kind: CausticImage['kind'],
) {
  // Each variant exposes only the image resource it reads.
  const imageLayout: GPUBindGroupLayoutEntry =
    kind === 'buffer'
      ? {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: 'read-only-storage' },
        }
      : {
          binding: 2,
          visibility: GPUShaderStage.FRAGMENT,
          texture: {
            sampleType: 'unfilterable-float',
            viewDimension: '2d',
          },
        }

  const bindGroupLayout = device.createBindGroupLayout({
    label: `Caustic ${kind} bindings`,
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: {
          type: 'uniform',
          minBindingSize: 16,
        },
      },
      imageLayout,
    ],
  })

  const module = device.createShaderModule({
    label: 'Caustic display shader',
    code: shaderSource,
  })

  const pipeline = await device.createRenderPipelineAsync({
    label: `Caustic ${kind} pipeline`,
    layout: device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    }),
    vertex: {
      module,
      entryPoint: 'vertexMain',
    },
    fragment: {
      module,
      entryPoint:
        kind === 'buffer' ? 'fragmentBuffer' : 'fragmentTexture',
      targets: [{ format }],
    },
    primitive: {
      topology: 'triangle-list',
    },
  })

  return { pipeline, bindGroupLayout }
}

function createDisplayUniforms(device: GPUDevice) {
  const buffer = device.createBuffer({
    label: 'Caustic display uniforms',
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })

  const data = new ArrayBuffer(16)
  const view = new DataView(data)

  function update(width: number, height: number, exposure: number) {
    if (
      !Number.isInteger(width) ||
      !Number.isInteger(height) ||
      width <= 0 ||
      height <= 0
    ) {
      throw new Error('Image dimensions must be positive integers.')
    }

    if (!Number.isFinite(exposure) || exposure < 0) {
      throw new Error('Exposure must be finite and non-negative.')
    }

    // Match the WGSL field types and byte offsets.
    view.setUint32(0, width, true)
    view.setUint32(4, height, true)
    view.setFloat32(8, exposure, true)

    device.queue.writeBuffer(buffer, 0, data)
  }

  return { buffer, update }
}

function createCausticBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  uniformBuffer: GPUBuffer,
  image: CausticImage,
) {
  // Bind the producer's GPU resource directly.
  const imageEntry: GPUBindGroupEntry =
    image.kind === 'buffer'
      ? {
          binding: 1,
          resource: {
            buffer: image.buffer,
            size: image.width * image.height * 4,
          },
        }
      : {
          binding: 2,
          resource: image.view,
        }

  return device.createBindGroup({
    label: 'Caustic display bindings',
    layout,
    entries: [
      {
        binding: 0,
        resource: {
          buffer: uniformBuffer,
          size: 16,
        },
      },
      imageEntry,
    ],
  })
}

export interface CausticDisplay {
  setImage(image: CausticImage): Promise<void>
  setExposure(exposure: number): void
  draw(pass: GPURenderPassEncoder): void
  destroy(): void
}

export function createCausticDisplay(
  device: GPUDevice,
  format: GPUTextureFormat,
): CausticDisplay {
  const uniforms = createDisplayUniforms(device)

  // Create each pipeline variant only when first needed.
  const pipelines = new Map<
    CausticImage['kind'],
    ReturnType<typeof createCausticPipeline>
  >()

  let active: {
    image: CausticImage
    pipeline: GPURenderPipeline
    bindGroup: GPUBindGroup
  } | null = null

  let exposure = 1
  let requestId = 0
  let disposed = false

  return {
    async setImage(image) {
      if (disposed) return

      const currentRequest = ++requestId
      let pending = pipelines.get(image.kind)

      if (!pending) {
        pending = createCausticPipeline(device, format, image.kind)
        pipelines.set(image.kind, pending)
      }

      const { pipeline, bindGroupLayout } = await pending

      // Ignore results superseded while pipeline creation was pending.
      if (disposed || currentRequest !== requestId) return

      uniforms.update(image.width, image.height, exposure)

      const bindGroup = createCausticBindGroup(
        device,
        bindGroupLayout,
        uniforms.buffer,
        image,
      )

      active = { image, pipeline, bindGroup }
    },

    setExposure(value) {
      if (disposed) return

      if (!Number.isFinite(value) || value < 0) {
        throw new Error('Exposure must be finite and non-negative.')
      }

      if (active) {
        uniforms.update(active.image.width, active.image.height, value)
      }

      exposure = value
    },

    draw(pass) {
      if (!active) return

      pass.setPipeline(active.pipeline)
      pass.setBindGroup(0, active.bindGroup)
      pass.draw(3)
    },

    destroy() {
      disposed = true
      active = null
      pipelines.clear()

      // The producer owns the image resource.
      uniforms.buffer.destroy()
    },
  }
}