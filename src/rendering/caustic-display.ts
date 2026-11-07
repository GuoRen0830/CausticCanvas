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
  prepare(kind: CausticImage['kind']): Promise<void>
  setImage(image: CausticImage): void
  setExposure(exposure: number): void
  draw(pass: GPURenderPassEncoder): void
  destroy(): void
}

export function createCausticDisplay(
  device: GPUDevice,
  format: GPUTextureFormat,
): CausticDisplay {
  const uniforms = createDisplayUniforms(device)

  type Pipeline = Awaited<ReturnType<typeof createCausticPipeline>>
  const pipelines = new Map<CausticImage['kind'], Pipeline>()
  const preparing = new Map<CausticImage['kind'], Promise<void>>()

  // Reuse bindings when the producer alternates GPU resources.
  let bindings = new WeakMap<
    GPUBuffer | GPUTextureView,
    { width: number; height: number; bindGroup: GPUBindGroup }
  >()

  let active: {
    image: CausticImage
    pipeline: GPURenderPipeline
    bindGroup: GPUBindGroup
  } | null = null

  let exposure = 1
  let disposed = false

  return {
    async prepare(kind) {
      if (disposed || pipelines.has(kind)) return

      // Compile outside the frame loop and share concurrent preparation requests.
      let pending = preparing.get(kind)
      if (!pending) {
        pending = createCausticPipeline(device, format, kind).then((pipeline) => {
          if (!disposed) pipelines.set(kind, pipeline)
        })
        preparing.set(kind, pending)
      }

      try {
        await pending
      } finally {
        if (preparing.get(kind) === pending) preparing.delete(kind)
      }
    },

    setImage(image) {
      if (disposed) return

      const ready = pipelines.get(image.kind)
      if (!ready) {
        throw new Error('Prepare the caustic pipeline before binding an image.')
      }

      if (
        !active ||
        active.image.width !== image.width ||
        active.image.height !== image.height
      ) {
        uniforms.update(image.width, image.height, exposure)
      }

      const resource = image.kind === 'buffer' ? image.buffer : image.view
      let binding = bindings.get(resource)

      if (
        !binding ||
        binding.width !== image.width ||
        binding.height !== image.height
      ) {
        binding = {
          width: image.width,
          height: image.height,
          bindGroup: createCausticBindGroup(
            device,
            ready.bindGroupLayout,
            uniforms.buffer,
            image,
          ),
        }
        bindings.set(resource, binding)
      }

      // Snapshot image metadata without copying the shared GPU contents.
      active = {
        image: { ...image },
        pipeline: ready.pipeline,
        bindGroup: binding.bindGroup,
      }
    },

    setExposure(value) {
      if (disposed) return

      if (!Number.isFinite(value) || value < 0) {
        throw new Error('Exposure must be finite and non-negative.')
      }
      if (value === exposure) return

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
      preparing.clear()
      bindings = new WeakMap()

      // The producer owns the image resource.
      uniforms.buffer.destroy()
    },
  }
}
