import './style.css'
import { mat4 } from 'wgpu-matrix'
import { config } from './config'
import { initializeGPU } from './gpu/context'
import { createDrawingPad } from './ui/drawing-pad'
import { readObjMesh } from './geometry/obj-mesh'
import { placeObjOnReceiver } from './geometry/place-obj'
import { createOrbitCamera, type OrbitCameraOptions } from './rendering/orbit-camera'
import { buildSceneGeometry, type SceneGeometry } from './rendering/scene-mesh'
import type { BeamSource, BeamView, SceneMethod } from './rendering/beam-renderer'
import { createFrameLoop } from './frame-loop'
import { buildDesignView } from './optics/design-view'
import { createSceneSession, type SceneSession } from './scene-session'

function element<K extends keyof HTMLElementTagNameMap>(id: string, tag: K) {
  const node = document.querySelector<HTMLElementTagNameMap[K]>(`${tag}#${id}`)
  if (!node) throw new Error(`Missing application control: ${id}.`)
  return node
}

const lifetime = new AbortController()
if (import.meta.hot) import.meta.hot.dispose(() => lifetime.abort())

async function main() {
  const canvas = element('gpu-canvas', 'canvas'), causticCanvas = element('caustic-canvas', 'canvas')
  const targetCanvas = element('target-canvas', 'canvas'), clearButton = element('clear-target', 'button')
  const status = element('status', 'p'), causticStatus = element('caustic-status', 'p')
  const targetStatus = element('target-status', 'p'), sceneStatus = element('scene-status', 'p')
  const importButton = element('import-obj', 'button'), objFile = element('obj-file', 'input')
  const defaultButton = element('default-scene', 'button')
  const methodSelect = element('scene-method', 'select'), beamSelect = element('beam-view', 'select')
  const sourceSelect = element('beam-source', 'select'), cameraHelp = element('camera-help', 'p')
  const events = { signal: lifetime.signal }

  importButton.disabled = true
  const { device, context, format } = await initializeGPU(canvas, config.lens)
  if (events.signal.aborted) { device.destroy(); return }
  let active: SceneSession | undefined
  let frameLoop: ReturnType<typeof createFrameLoop> | undefined
  let pad: ReturnType<typeof createDrawingPad> | undefined
  let camera: ReturnType<typeof createOrbitCamera> | undefined
  let hasTarget = false, optimizationDone = false, switching = false
  let activeLabel = 'Default box', hasReferenceBox = true

  lifetime.signal.addEventListener('abort', () => {
    importButton.disabled = true
    defaultButton.disabled = true
    frameLoop?.destroy()
    camera?.destroy()
    pad?.destroy()
    active?.destroy()
    device.destroy()
  }, { once: true })

  function fail(error: unknown) {
    if (events.signal.aborted) return
    console.error(error)
    status.textContent = error instanceof Error ? error.message : String(error)
    causticStatus.textContent = 'Computation stopped.'
    lifetime.abort()
  }
  device.addEventListener('uncapturederror', event => fail(new Error(`GPU error: ${event.error.message}`)), events)
  void device.lost.then(info => fail(new Error(`GPU device lost. Reload the page. ${info.message}`)))

  const previewContext = causticCanvas.getContext('webgpu')
  if (!previewContext) throw new Error('Failed to create caustic canvas context.')
  const { receiver, lens } = config
  causticCanvas.width = receiver.width
  causticCanvas.height = receiver.height
  previewContext.configure({ device, format, alphaMode: 'opaque' })
  const opticalToWorld = mat4.multiply(mat4.translation([0, receiver.z, 0]), mat4.rotationX(Math.PI / 2))
  const defaultFraming: OrbitCameraOptions = {
    target: [0, (receiver.z - lens.thickness) / 2, 0],
    span: Math.max(lens.sizeX, lens.sizeY, receiver.sizeX, receiver.sizeY, receiver.z),
  }
  let sceneFraming = defaultFraming, currentFraming = defaultFraming
  camera = createOrbitCamera(canvas, defaultFraming, () => frameLoop?.requestFrame())

  function updateRunState() {
    frameLoop?.setRunning(!!active?.canOptimize && hasTarget && !optimizationDone)
    if (!active?.canOptimize) {
      causticStatus.textContent = active?.state.optics.incidentIrradiance === 0 ? 'Source is off.' : 'Forward preview only.'
    } else {
      causticStatus.textContent = !hasTarget ? 'Draw a pattern to optimize.'
        : optimizationDone ? 'Optimization complete.' : 'Optimizing.'
    }
  }

  const viewHelp: Record<BeamView, string> = {
    lit: 'Drag to orbit · Scroll to zoom · Double-click to reset',
    lens: 'Inspect the lens surface · Drag to orbit · Scroll to zoom',
    candidates: 'Candidate count · Blue 1 · Cyan 2 · Green 4 · Yellow 16 · Red 64+',
    footprint: 'Visible floor irradiance after scene occlusion',
    blocked: 'Floor irradiance intercepted by the box · Brighter means more light blocked',
    plane: 'Irradiance these beams would produce on the unobstructed reference plane',
    difference: 'Absolute irradiance difference between the scene and the unobstructed plane',
  }
  function changeView() {
    const solver = sourceSelect.value === 'solver'
    for (const value of ['plane', 'difference']) {
      beamSelect.querySelector<HTMLOptionElement>(`option[value="${value}"]`)!.disabled = !solver
    }
    if (!solver && ['plane', 'difference'].includes(beamSelect.value)) beamSelect.value = 'footprint'
    const mode = beamSelect.value as BeamView
    const framing = mode === 'lens' ? defaultFraming : sceneFraming
    if (currentFraming !== framing) {
      camera!.frame(framing)
      currentFraming = framing
    }
    sourceSelect.disabled = mode === 'lens'
    methodSelect.disabled = !hasReferenceBox || ['plane', 'difference', 'blocked', 'lens'].includes(mode)
    cameraHelp.textContent = viewHelp[mode]
    frameLoop?.requestFrame()
  }
  for (const select of [methodSelect, sourceSelect, beamSelect]) select.addEventListener('change', changeView, events)
  clearButton.addEventListener('click', () => pad?.clear(), events)
  window.addEventListener('resize', () => frameLoop?.requestFrame(), events)

  function drawSceneFrame(scene: SceneSession, steps: number, source: BeamSource, mode: BeamView, method: SceneMethod) {
    const ratio = Math.min(window.devicePixelRatio, config.rendering.maxPixelRatio)
    const limit = device.limits.maxTextureDimension2D
    const width = Math.min(limit, Math.max(1, Math.round(canvas.clientWidth * ratio)))
    const height = Math.min(limit, Math.max(1, Math.round(canvas.clientHeight * ratio)))
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width
      canvas.height = height
    }
    const encoder = device.createCommandEncoder()
    const done = scene.render(encoder, context.getCurrentTexture(), previewContext!.getCurrentTexture(),
      camera!.update(width / height), source, mode, method, steps)
    return { done, commands: encoder.finish() }
  }

  function encodeFrame(steps: number) {
    const { done, commands } = drawSceneFrame(active!, steps, sourceSelect.value as BeamSource,
      beamSelect.value as BeamView, methodSelect.value === 'analytic' ? 'analytic' : 'first-hit')
    if (done) { optimizationDone = true; updateRunState() }
    return commands
  }

  async function loadScene(file?: File) {
    if (switching || events.signal.aborted) return
    switching = true
    importButton.disabled = defaultButton.disabled = true
    const name = file?.name ?? 'Default box'
    sceneStatus.textContent = `Loading ${name}…`
    let candidate: SceneSession | undefined
    try {
      let geometry: SceneGeometry
      let framing = defaultFraming
      if (file) {
        const model = await readObjMesh(file)
        if (events.signal.aborted) return
        const bounds = placeObjOnReceiver(model, lens, receiver)
        geometry = buildSceneGeometry(receiver, { mesh: model })
        framing = { target: [0, bounds.height / 2, 0], span: Math.max(bounds.width, bounds.height, bounds.depth) }
      } else {
        geometry = buildSceneGeometry(receiver, { box: config.scene.box })
      }
      const { mesh } = geometry
      const storageLimit = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize)
      if (mesh.vertices.byteLength > device.limits.maxBufferSize || mesh.indices.byteLength > device.limits.maxBufferSize ||
          mesh.indices.length / 3 * 48 > storageLimit) {
        throw new Error('Scene geometry exceeds the GPU buffer limit.')
      }
      const designView = buildDesignView(mesh, receiver)
      if (designView.bvh.nodes.byteLength > storageLimit) throw new Error('Scene BVH exceeds the GPU buffer limit.')

      // Hold the old lens and target until the new GPU resources and first evaluation succeed.
      frameLoop?.destroy()
      frameLoop = undefined
      await device.queue.onSubmittedWorkDone()
      if (events.signal.aborted) return
      device.pushErrorScope('out-of-memory')
      device.pushErrorScope('validation')
      try {
        candidate = await createSceneSession(device, format, opticalToWorld, geometry, designView)
        if (events.signal.aborted) return
        // Include G-buffer allocation and first-hit splatting in the same validation scope.
        device.queue.submit([drawSceneFrame(candidate, 0, 'solver', 'lit', 'first-hit').commands])
        await device.queue.onSubmittedWorkDone()
      } finally {
        const [validation, memory] = await Promise.all([device.popErrorScope(), device.popErrorScope()])
        const error = validation ?? memory
        if (error) throw new Error(error.message || 'GPU could not prepare this scene.')
      }
      if (events.signal.aborted) return

      const next = candidate
      const newPad = createDrawingPad(targetCanvas, receiver, target => {
        next.state.target = target
        optimizationDone = false
        hasTarget = target.values.some(value => value > 0)
        targetStatus.textContent = hasTarget ? 'Target updated' : 'Draw a pattern to begin.'
        updateRunState()
        frameLoop?.requestFrame()
      }, designView)
      pad?.destroy()
      active?.destroy()
      pad = newPad
      active = next
      candidate = undefined
      sceneFraming = framing
      hasReferenceBox = !!geometry.referenceBox
      defaultButton.hidden = hasReferenceBox
      methodSelect.querySelector<HTMLOptionElement>('option[value="analytic"]')!.disabled = !hasReferenceBox
      beamSelect.querySelector<HTMLOptionElement>('option[value="blocked"]')!.disabled = !hasReferenceBox
      methodSelect.value = 'first-hit'
      sourceSelect.value = 'solver'
      beamSelect.value = 'lit'
      changeView()
      activeLabel = name
      sceneStatus.textContent = name
      status.textContent = 'WebGPU ready'
    } catch (error) {
      if (events.signal.aborted) return
      if (!active) { fail(error); return }
      const reason = error instanceof Error ? error.message : String(error)
      sceneStatus.textContent = `Could not load ${name}. ${reason} Current scene: ${activeLabel}.`
    } finally {
      candidate?.destroy()
      switching = false
      if (!events.signal.aborted && active) {
        importButton.disabled = defaultButton.disabled = false
        frameLoop ??= createFrameLoop(device, encodeFrame, { maxStepsPerFrame: config.maxStepsPerFrame, onError: fail })
        updateRunState()
        frameLoop.requestFrame()
      }
    }
  }

  importButton.addEventListener('click', () => objFile.click(), events)
  objFile.addEventListener('change', () => {
    const file = objFile.files?.[0]
    objFile.value = '' // Allow retrying the same file after editing it on disk.
    if (file) void loadScene(file)
  }, events)
  defaultButton.addEventListener('click', () => { void loadScene() }, events)
  await loadScene()
}

main().catch((error: unknown) => {
  console.error(error)
  if (lifetime.signal.aborted) return
  lifetime.abort()
  const status = document.querySelector('#status')
  if (status) status.textContent = error instanceof Error ? error.message : String(error)
})
