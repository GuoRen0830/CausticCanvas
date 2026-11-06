import './style.css'
import { initializeGPU } from './gpu/context'

async function main() {
  const canvas = document.querySelector('#gpu-canvas')
  const status = document.querySelector('#status')

  if (
    !(canvas instanceof HTMLCanvasElement) ||
    !(status instanceof HTMLElement)
  ) {
    throw new Error('Missing canvas or status element.')
  }

  const { device, context } = await initializeGPU(canvas)

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

    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width
      canvas.height = height
    }

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
    })

    pass.end()
    device.queue.submit([encoder.finish()])
  }

  device.addEventListener('uncapturederror', (event) => {
    console.error(event.error)
    status.textContent = `GPU error: ${event.error.message}`
  })

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