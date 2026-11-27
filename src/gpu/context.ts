import type { LensState } from '../contracts'

export async function initializeGPU(canvas: HTMLCanvasElement, lens: Pick<LensState, 'columns' | 'rows'>) {
  if (!navigator.gpu) {
    throw new Error('WebGPU is unavailable.')
  }

  const adapter = await navigator.gpu.requestAdapter()

  if (!adapter) {
    throw new Error('No GPU adapter available.')
  }

  // Scene optics stores one vec4 per face for beams and another for gradients.
  const faces = 2 * (lens.columns - 1) * (lens.rows - 1)
  const storageBytes = Math.max(faces * 16, lens.columns * lens.rows * 8)
  if (storageBytes > adapter.limits.maxStorageBufferBindingSize || storageBytes > adapter.limits.maxBufferSize) {
    throw new Error(`Lens ${lens.columns} x ${lens.rows} needs a ${storageBytes}-byte buffer. This GPU supports ${Math.min(adapter.limits.maxStorageBufferBindingSize, adapter.limits.maxBufferSize)} bytes per storage binding.`)
  }
  const device = await adapter.requestDevice({
    // Imported scene vertices and BVH triangles can be larger than the lens buffers.
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
    requiredFeatures: adapter.features.has('float32-blendable') ? ['float32-blendable'] : [],
  })
  const context = canvas.getContext('webgpu')

  if (!context) {
    throw new Error('Failed to create WebGPU canvas context.')
  }

  const format = navigator.gpu.getPreferredCanvasFormat()

  context.configure({
    device,
    format,
    alphaMode: 'opaque',
  })

  return { device, context, format }
}
