import './style.css'
import { mat4 } from 'wgpu-matrix'
import { config } from './config'
import { initializeGPU } from './gpu/context'
import { createDrawingPad } from './ui/drawing-pad'
import { createFlatLens } from './geometry/lens-mesh'
import { createLensRenderer } from './rendering/lens-renderer'
import { createOrbitCamera } from './rendering/orbit-camera'
import { createSceneRenderer } from './rendering/scene-renderer'
import { createCausticDisplay } from './rendering/caustic-display'
import { createFrameLoop } from './frame-loop'
import { createSimulation } from './simulation'
import { createForwardModel } from './optics/forward-model'
import { createLensOptimizer } from './optimization/lens-optimizer'
import type { AppState, ForwardModel, ForwardResult, LensOptimizer } from './contracts'

async function main() {
  const canvas = document.querySelector('#gpu-canvas')
  const causticCanvas = document.querySelector('#caustic-canvas')
  const statusElement = document.querySelector('#status')
  const causticStatusElement = document.querySelector('#caustic-status')
  const targetCanvas = document.querySelector('#target-canvas')
  const clearButton = document.querySelector('#clear-target')
  const targetStatus = document.querySelector('#target-status')

  if (
    !(canvas instanceof HTMLCanvasElement) ||
    !(causticCanvas instanceof HTMLCanvasElement) ||
    !(statusElement instanceof HTMLElement) ||
    !(causticStatusElement instanceof HTMLElement) ||
    !(targetCanvas instanceof HTMLCanvasElement) ||
    !(clearButton instanceof HTMLButtonElement) ||
    !(targetStatus instanceof HTMLElement)
  ) {
    throw new Error('Missing application controls.')
  }

  const status = statusElement
  const causticStatus = causticStatusElement

  const { device, context, format } = await initializeGPU(canvas)
  const events = new AbortController()
  let disposed = false
  let hasTarget = false
  let optimizationDone = false
  let initialHeightBuffer: GPUBuffer | null = null
  let depthTexture: GPUTexture | null = null
  let forward: ForwardModel | null = null
  let optimizer: LensOptimizer | null = null
  let lastResult: ForwardResult | null = null
  let startupEncoder: GPUCommandEncoder | null = null
  let drawingPad: ReturnType<typeof createDrawingPad> | null = null
  let lensRenderer: ReturnType<typeof createLensRenderer> | null = null
  let causticDisplay: ReturnType<typeof createCausticDisplay> | null = null
  let frameLoop: ReturnType<typeof createFrameLoop> | null = null
  let camera: ReturnType<typeof createOrbitCamera> | null = null
  let sceneRenderer: ReturnType<typeof createSceneRenderer> | null = null

  function dispose() {
    if (disposed) return
    disposed = true
    events.abort()
    frameLoop?.destroy()
    camera?.destroy()
    drawingPad?.destroy()
    startupEncoder = null
    lastResult = null

    try {
      lensRenderer?.destroy()
      sceneRenderer?.destroy()
      causticDisplay?.destroy()
      optimizer?.destroy()
      forward?.destroy()
      depthTexture?.destroy()
      initialHeightBuffer?.destroy()
    } finally {
      device.destroy()
    }
  }

  function fail(error: unknown) {
    if (disposed) return
    console.error(error)
    status.textContent = error instanceof Error ? error.message : String(error)
    causticStatus.textContent = 'Computation stopped.'
    dispose()
  }

  device.addEventListener('uncapturederror', (event) => {
    fail(new Error(`GPU error: ${event.error.message}`))
  }, { signal: events.signal })

  // Register before awaiting module creation so late results cannot restart the app.
  void device.lost.then((info) => {
    if (disposed) return
    status.textContent = `GPU device lost. Reload the page. ${info.message}`
    causticStatus.textContent = 'Computation stopped.'
    dispose()
  })

  if (import.meta.hot) {
    import.meta.hot.dispose(dispose)
  }

  try {
    // Create design state before the drawing pad publishes its first snapshot.
    const state: AppState = {
      target: null,
      lens: createFlatLens(device, config.lens),
      optics: { ...config.optics },
      receiver: { ...config.receiver },
    }

    // Keep ownership of the initial buffer if the optimizer later swaps it out.
    initialHeightBuffer = state.lens.heightBuffer

    function updateRunState() {
      if (!frameLoop || disposed) return
      const running = optimizer !== null && hasTarget && !optimizationDone
      frameLoop.setRunning(running)

      if (!forward) {
        causticStatus.textContent = 'Forward model not implemented.'
      } else if (!optimizer) {
        causticStatus.textContent = 'Forward preview. Optimizer not implemented.'
      } else if (!hasTarget) {
        causticStatus.textContent = 'Draw a pattern to optimize.'
      } else {
        causticStatus.textContent = optimizationDone
          ? 'Optimization complete.'
          : 'Optimizing.'
      }
    }

    drawingPad = createDrawingPad(targetCanvas, state.receiver, (target) => {
      // Keep only the latest target; upload it at the next batch boundary.
      state.target = target
      optimizationDone = false
      hasTarget = target.values.some((value) => value > 0)
      targetStatus.textContent = hasTarget
        ? 'Target updated'
        : 'Draw a pattern to begin.'
      updateRunState()
      frameLoop?.requestFrame()
    })
    clearButton.addEventListener('click', drawingPad.clear, { signal: events.signal })

    const causticContext = causticCanvas.getContext('webgpu')
    if (!causticContext) {
      throw new Error('Failed to create caustic canvas context.')
    }
    causticCanvas.width = state.receiver.width
    causticCanvas.height = state.receiver.height
    causticContext.configure({ device, format, alphaMode: 'opaque' })

    // Rotate local +Z downward and place the receiver on world Y = 0.
    const opticalToWorld = mat4.multiply(
      mat4.translation([0, state.receiver.z, 0]),
      mat4.rotationX(Math.PI / 2),
    )
    const orbitCamera = createOrbitCamera(canvas, {
      target: [0, (state.receiver.z - config.lens.thickness) / 2, 0],
      span: Math.max(
        state.lens.sizeX, state.lens.sizeY,
        state.receiver.sizeX, state.receiver.sizeY, state.receiver.z,
      ),
    }, () => frameLoop?.requestFrame())
    camera = orbitCamera

    const lensView = createLensRenderer(device, format, state.lens, opticalToWorld)
    lensRenderer = lensView
    const sceneView = createSceneRenderer(
      device, format, state.receiver, config.scene.box, opticalToWorld,
    )
    sceneRenderer = sceneView
    const causticView = createCausticDisplay(device, format)
    causticDisplay = causticView
    causticView.setExposure(config.rendering.causticExposure)
    status.textContent = 'Preparing optical modules.'

    // Both modules share this device and the same initial optical configuration.
    const initial = {
      lens: { ...state.lens },
      optics: { ...state.optics },
      receiver: { ...state.receiver },
    }
    forward = await createForwardModel(device, initial)
    if (disposed) {
      forward?.destroy()
      return
    }

    let simulation: ReturnType<typeof createSimulation> | null = null
    if (forward) {
      optimizer = await createLensOptimizer(device, initial, forward)
      if (disposed) {
        optimizer?.destroy()
        return
      }
      simulation = createSimulation(state, forward, optimizer ?? undefined)

      // Record the first evaluation now; submit it with the first rendered frame.
      startupEncoder = device.createCommandEncoder({ label: 'Initial optical frame' })
      lastResult = simulation.encode(startupEncoder, 0).result
      await causticView.prepare(lastResult.caustic.kind)
      if (disposed) return
      causticView.setImage(lastResult.caustic)
    }

    const encodeFrame = (steps: number) => {
      const initialFrame = startupEncoder !== null
      const encoder = startupEncoder ?? device.createCommandEncoder()
      startupEncoder = null

      // Deliver the latest target once per batch, including clear and redraw.
      optimizer?.setTarget(state.target)

      // A pure redraw reuses the last result; running frames evaluate updated heights.
      if (simulation && !initialFrame && steps > 0) {
        const batch = simulation.encode(encoder, steps)
        lastResult = batch.result
        causticView.setImage(lastResult.caustic)
        if (batch.done) {
          optimizationDone = true
          updateRunState()
        }
      }

      const pixelRatio = Math.min(window.devicePixelRatio, config.rendering.maxPixelRatio)
      const limit = device.limits.maxTextureDimension2D
      const width = Math.min(
        limit,
        Math.max(1, Math.round(canvas.clientWidth * pixelRatio)),
      )
      const height = Math.min(
        limit,
        Math.max(1, Math.round(canvas.clientHeight * pixelRatio)),
      )

      if (!depthTexture || canvas.width !== width || canvas.height !== height) {
        canvas.width = width
        canvas.height = height
        depthTexture?.destroy()
        depthTexture = device.createTexture({
          label: 'Viewport depth',
          size: [width, height],
          format: 'depth24plus',
          usage: GPUTextureUsage.RENDER_ATTACHMENT,
        })
      }

      // Bind the geometry that produced the displayed caustic image.
      const viewProjection = orbitCamera.update(width / height)
      lensView.update(lastResult?.beams.lens ?? state.lens, viewProjection)
      sceneView.update(viewProjection)
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: context.getCurrentTexture().createView(),
          clearValue: { r: 0.03, g: 0.18, b: 0.22, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        }],
        depthStencilAttachment: {
          view: depthTexture.createView(),
          depthClearValue: 1,
          depthLoadOp: 'clear',
          depthStoreOp: 'discard',
        },
      })
      // One shared depth attachment resolves visibility across both pipelines.
      sceneView.draw(pass)
      lensView.draw(pass)
      pass.end()

      const causticPass = encoder.beginRenderPass({
        label: 'Caustic preview',
        colorAttachments: [{
          view: causticContext.getCurrentTexture().createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        }],
      })
      causticView.draw(causticPass)
      causticPass.end()
      return encoder.finish()
    }

    frameLoop = createFrameLoop(device, encodeFrame, {
      maxStepsPerFrame: config.maxStepsPerFrame,
      onError: fail,
    })
    window.addEventListener('resize', frameLoop.requestFrame, { signal: events.signal })
    status.textContent = 'WebGPU ready'
    updateRunState()
    frameLoop.requestFrame()
  } catch (error) {
    fail(error)
  }
}

main().catch((error: unknown) => {
  console.error(error)
  const status = document.querySelector('#status')
  if (status) {
    status.textContent = error instanceof Error ? error.message : String(error)
  }
})
