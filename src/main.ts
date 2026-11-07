import './style.css'
import { initializeGPU } from './gpu/context'
import { createDrawingPad } from './ui/drawing-pad'
import { createFlatLens } from './geometry/create-flat-lens'
import {
  createLensBuffers,
  createLensTransforms,
  createLensPipeline,
} from './rendering/lens-renderer'

async function main() {
  const canvas = document.querySelector('#gpu-canvas')
  const status = document.querySelector('#status')

  if (
    !(canvas instanceof HTMLCanvasElement) ||
    !(status instanceof HTMLElement)
  ) {
    throw new Error('Missing canvas or status element.')
  }

  const targetCanvas = document.querySelector('#target-canvas')
  const clearButton = document.querySelector('#clear-target')
  const targetStatus = document.querySelector('#target-status')

  if (
    !(targetCanvas instanceof HTMLCanvasElement) ||
    !(clearButton instanceof HTMLButtonElement) ||
    !(targetStatus instanceof HTMLElement)
  ) {
    throw new Error('Missing drawing controls.')
  }

  // This callback is the handoff point for the optimizer.
  const drawingPad = createDrawingPad(targetCanvas, (target) => {
    const hasInput = target.values.some((value) => value > 0)
    targetStatus.textContent = hasInput ? 'Target updated' : 'Draw a pattern to begin.'

    console.info('Target updated', {
      version: target.version,
      width: target.width,
      height: target.height,
      hasInput,
    })
  })

  clearButton.addEventListener('click', drawingPad.clear)

  const { device, context, format } = await initializeGPU(canvas)

  device.addEventListener('uncapturederror', (event) => {
    console.error(event.error)
    status.textContent = `GPU error: ${event.error.message}`
  })

  // Create a flat exit surface in physical units.
  const lens = createFlatLens({
    columns: 33,
    rows: 33,
    sizeX: 0.2,
    sizeY: 0.2,
    thickness: 0.01,
  })

  const lensBuffers = createLensBuffers(device, lens)
  const transforms = createLensTransforms(device, lens)

  const { pipeline, bindGroup } = createLensPipeline(
    device,
    format,
    transforms.buffer,
  )

  // Allocate the depth texture after computing the canvas size.
  let depthTexture: GPUTexture | null = null

  const render = () => {
    const pixelRatio = Math.min(window.devicePixelRatio, 2)
    const limit = device.limits.maxTextureDimension2D

    const width = Math.min(
      limit,
      Math.max(1, Math.round(canvas.clientWidth * pixelRatio)),
    )

    const height = Math.min(
      limit,
      Math.max(1, Math.round(canvas.clientHeight * pixelRatio)),
    )

    if (
      !depthTexture ||
      canvas.width !== width ||
      canvas.height !== height
    ) {
      canvas.width = width
      canvas.height = height
    
      // Recreate depth storage when the render size changes.
      depthTexture?.destroy()
      depthTexture = device.createTexture({
        label: 'Viewport depth',
        size: [width, height],
        format: 'depth24plus',
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      })
    }

    // Match the projection to the actual render dimensions.
    transforms.update(width / height)

    const encoder = device.createCommandEncoder()

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          clearValue: { r: 0.03, g: 0.18, b: 0.22, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'discard',
      },
    })
    
    // Bind the rendering rules and their input resources.
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bindGroup)
    pass.setVertexBuffer(0, lensBuffers.vertexBuffer)
    pass.setIndexBuffer(lensBuffers.indexBuffer, 'uint32')
    
    // Draw the complete indexed surface.
    pass.drawIndexed(lensBuffers.indexCount)
    
    pass.end()

    device.queue.submit([encoder.finish()])
  }

  render()
  window.addEventListener('resize', render)
  status.textContent = 'WebGPU ready'

  void device.lost.then((info) => {
    window.removeEventListener('resize', render)
    status.textContent = `GPU device lost. Reload the page. ${info.message}`
  })
}

main().catch((error: unknown) => {
  console.error(error)

  const status = document.querySelector('#status')

  if (status) {
    status.textContent =
      error instanceof Error ? error.message : String(error)
  }
})