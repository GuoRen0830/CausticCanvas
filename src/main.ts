import './style.css'
import { initializeGPU } from './gpu/context'
import { createDrawingPad } from './ui/drawing-pad'
import { createFlatLens } from './geometry/lens-mesh'
import { createLensRenderer } from './rendering/lens-renderer'
import type { AppState } from './contracts'
import { createCausticDisplay } from './rendering/caustic-display'

async function main() {
  const canvas = document.querySelector('#gpu-canvas')
  const causticCanvas = document.querySelector('#caustic-canvas')
  const status = document.querySelector('#status')

  if (
    !(canvas instanceof HTMLCanvasElement) ||
    !(causticCanvas instanceof HTMLCanvasElement) ||
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

  const { device, context, format } = await initializeGPU(canvas)

  device.addEventListener('uncapturederror', (event) => {
    console.error(event.error)
    status.textContent = `GPU error: ${event.error.message}`
  })

  // Create design state before the drawing pad publishes its first snapshot.
  const state: AppState = {
    target: null,
    lens: createFlatLens(device, {
      columns: 33,
      rows: 33,
      sizeX: 0.2,
      sizeY: 0.2,
      thickness: 0.01,
    }),
    optics: {
      lensIOR: 1.5,
      outsideIOR: 1,
      incidentIrradiance: 1,
    },
    receiver: {
      z: 0.5,
      sizeX: 0.4,
      sizeY: 0.4,
      width: 256,
      height: 256,
    },
  }

  // Retain ownership even if the optimizer later replaces the active buffer.
  const initialHeightBuffer = state.lens.heightBuffer

  // This callback is the handoff point for the optimizer.
  const drawingPad = createDrawingPad(targetCanvas, (target) => {
    // Store every snapshot, including clear operations.
    state.target = target

    const hasInput = target.values.some((value) => value > 0)
    targetStatus.textContent = hasInput
      ? 'Target updated'
      : 'Draw a pattern to begin.'
  })

  clearButton.addEventListener('click', drawingPad.clear)

  const causticContext = causticCanvas.getContext('webgpu')

  if (!causticContext) {
    throw new Error('Failed to create caustic canvas context.')
  }

  // Start with one display pixel per receiver sample.
  causticCanvas.width = state.receiver.width
  causticCanvas.height = state.receiver.height

  // Share the device with the solver and lens preview.
  causticContext.configure({
    device,
    format,
    alphaMode: 'opaque',
  })

  const lensRenderer = createLensRenderer(device, format, state.lens)
  const causticDisplay = createCausticDisplay(device, format)

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

    // Bind the current GPU heights and update camera parameters.
    lensRenderer.update(state.lens, width / height)

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
    
    lensRenderer.draw(pass)
    
    pass.end()

    const causticPass = encoder.beginRenderPass({
      label: 'Caustic preview',
      colorAttachments: [
        {
          view: causticContext.getCurrentTexture().createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    })

    causticDisplay.draw(causticPass)
    causticPass.end()

    device.queue.submit([encoder.finish()])
  }

  render()
  window.addEventListener('resize', render)
  status.textContent = 'WebGPU ready'

  void device.lost.then((info) => {
    window.removeEventListener('resize', render)

    lensRenderer.destroy()
    depthTexture?.destroy()
    causticDisplay.destroy()
    initialHeightBuffer.destroy()

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