export async function initializeGPU(canvas: HTMLCanvasElement) {
  if (!navigator.gpu) {
    throw new Error('WebGPU is unavailable.')
  }

  const adapter = await navigator.gpu.requestAdapter()

  if (!adapter) {
    throw new Error('No GPU adapter available.')
  }

  const device = await adapter.requestDevice({
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