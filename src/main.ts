import './style.css'
import { mat4 } from 'wgpu-matrix'
import { config } from './config'
import { initializeGPU } from './gpu/context'
import { createDrawingPad } from './ui/drawing-pad'
import { bundledScenes, defaultScene, readBundledScene, type BundledScene } from './geometry/bundled-scenes'
import { readObjAsset } from './geometry/obj-materials'
import { placeObjOnReceiver } from './geometry/place-obj'
import { createOrbitCamera, type OrbitCameraOptions } from './rendering/orbit-camera'
import { buildSceneGeometry } from './rendering/scene-mesh'
import type { BeamView } from './rendering/beam-renderer'
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
  const folderButton = element('import-folder', 'button'), folderInput = element('asset-folder', 'input')
  const sceneSelect = element('bundled-scene', 'select'), beamSelect = element('beam-view', 'select')
  const credit = element('scene-credit', 'a')
  const cameraHelp = element('camera-help', 'p')
  const panel = element('design-panel', 'aside'), undoButton = element('undo-target', 'button')
  const brushSize = element('brush-size', 'input')
  const viewButtons = Array.from(document.querySelectorAll<HTMLButtonElement>('#view-switch button[data-view]'))
  // Diagnostic views stay available behind ?debug.
  beamSelect.hidden = !new URLSearchParams(location.search).has('debug')
  for (const [id, preset] of Object.entries(bundledScenes)) sceneSelect.add(new Option(preset.name, id))
  sceneSelect.value = defaultScene
  const events = { signal: lifetime.signal }

  importButton.disabled = folderButton.disabled = true
  const { device, context, format } = await initializeGPU(canvas, config.lens)
  if (events.signal.aborted) { device.destroy(); return }
  let active: SceneSession | undefined
  let frameLoop: ReturnType<typeof createFrameLoop> | undefined
  let pad: ReturnType<typeof createDrawingPad> | undefined
  let camera: ReturnType<typeof createOrbitCamera> | undefined
  let hasTarget = false, optimizationDone = false, switching = false
  let runStarted = 0, runTimer: number | undefined
  let activeLabel = '', activePreset: BundledScene | undefined

  lifetime.signal.addEventListener('abort', () => {
    importButton.disabled = folderButton.disabled = true
    sceneSelect.disabled = true
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
  previewContext.configure({ device, format, alphaMode: 'opaque' })
  const opticalToWorld = mat4.multiply(mat4.translation([0, receiver.z, 0]), mat4.rotationX(Math.PI / 2))
  const defaultFraming: OrbitCameraOptions = {
    target: [0, (receiver.z - lens.thickness) / 2, 0],
    size: [lens.sizeX, receiver.z, lens.sizeY],
  }
  let sceneFraming = defaultFraming, lensFraming = defaultFraming, currentFraming = defaultFraming
  camera = createOrbitCamera(canvas, defaultFraming, () => frameLoop?.requestFrame())

  const seconds = () => Math.round((performance.now() - runStarted) / 1000)
  function updateRunState() {
    const running = !!active?.canOptimize && hasTarget && !optimizationDone
    frameLoop?.setRunning(running)
    panel.classList.toggle('is-working', running)
    if (running && runTimer === undefined) {
      runStarted = performance.now()
      runTimer = window.setInterval(() => { causticStatus.textContent = `Shaping the glass · ${seconds()} s` }, 1000)
    } else if (!running && runTimer !== undefined) {
      window.clearInterval(runTimer)
      runTimer = undefined
    }
    if (!active?.canOptimize) {
      causticStatus.textContent = active?.state.optics.incidentIrradiance === 0 ? 'The lamp is off' : 'Preview only'
    } else if (!hasTarget) {
      causticStatus.textContent = 'Plain glass'
    } else if (optimizationDone) {
      causticStatus.textContent = runStarted ? `Done in ${seconds()} s` : 'Done'
    } else {
      causticStatus.textContent = `Shaping the glass · ${seconds()} s`
    }
  }

  const viewHelp: Record<BeamView, string> = {
    lit: 'Drag to look around · Scroll to zoom · Double-click to reset',
    lens: 'The glass surface that makes the pattern · Drag to look around',
    candidates: 'Candidate count · Blue 1 · Cyan 2 · Green 4 · Yellow 16 · Red 64+',
    footprint: 'Visible floor irradiance after scene occlusion',
    plane: 'Irradiance these beams would produce on the unobstructed reference plane',
    difference: 'Absolute irradiance difference between the scene and the unobstructed plane',
  }
  function changeView() {
    const mode = beamSelect.value as BeamView
    const framing = mode === 'lens' ? lensFraming : sceneFraming
    if (currentFraming !== framing) {
      camera!.frame(framing)
      currentFraming = framing
    }
    cameraHelp.textContent = viewHelp[mode]
    for (const button of viewButtons) button.setAttribute('aria-pressed', String(button.dataset.view === mode))
    frameLoop?.requestFrame()
  }
  beamSelect.addEventListener('change', changeView, events)
  for (const button of viewButtons) {
    button.addEventListener('click', () => {
      beamSelect.value = button.dataset.view!
      changeView()
    }, events)
  }
  clearButton.addEventListener('click', () => pad?.clear(), events)
  undoButton.addEventListener('click', () => pad?.undo(), events)
  brushSize.addEventListener('input', () => pad?.setBrush(Number(brushSize.value)), events)
  window.addEventListener('resize', () => frameLoop?.requestFrame(), events)

  function drawSceneFrame(scene: SceneSession, steps: number, mode: BeamView) {
    const ratio = Math.min(window.devicePixelRatio, config.rendering.maxPixelRatio)
    const limit = device.limits.maxTextureDimension2D
    const width = Math.min(limit, Math.max(1, Math.round(canvas.clientWidth * ratio)))
    const height = Math.min(limit, Math.max(1, Math.round(canvas.clientHeight * ratio)))
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width
      canvas.height = height
    }
    const encoder = device.createCommandEncoder()
    const viewProjection = camera!.update(width / height)
    const covered = canvas.clientWidth > 760 ? panel.getBoundingClientRect().right : 0
    if (covered > 0) mat4.multiply(mat4.translation([covered / canvas.clientWidth, 0, 0]), viewProjection, viewProjection)
    const done = scene.render(encoder, context.getCurrentTexture(), previewContext!.getCurrentTexture(),
      viewProjection, mode, steps)
    return { done, commands: encoder.finish() }
  }

  function encodeFrame(steps: number) {
    const { done, commands } = drawSceneFrame(active!, steps, beamSelect.value as BeamView)
    if (done) { optimizationDone = true; updateRunState() }
    return commands
  }

  async function loadScene(source: BundledScene | File, files: File[] = []) {
    if (switching || events.signal.aborted) return
    switching = true
    importButton.disabled = folderButton.disabled = sceneSelect.disabled = true
    const name = typeof source === 'string' ? bundledScenes[source].name : source.name
    sceneStatus.textContent = `Loading ${name}…`
    let candidate: SceneSession | undefined
    let warnings = new Set<string>()
    try {
      const asset = typeof source === 'string'
        ? await readBundledScene(source, events.signal) : await readObjAsset(source, files)
      if (events.signal.aborted) return
      warnings = asset.warnings
      const bounds = placeObjOnReceiver(asset.mesh, lens, receiver)
      const fitted = bounds.receiver
      const geometry = buildSceneGeometry(fitted, { mesh: asset.mesh })
      geometry.materials = asset.materials
      const framing: OrbitCameraOptions = {
        target: [0, bounds.height / 2, 0], size: [fitted.sizeX, bounds.height, fitted.sizeY],
      }
      const { mesh } = geometry
      const storageLimit = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize)
      if (mesh.vertices.byteLength > device.limits.maxBufferSize || mesh.indices.byteLength > device.limits.maxBufferSize ||
          mesh.indices.length / 3 * 48 > storageLimit) {
        throw new Error('Scene geometry exceeds the GPU buffer limit.')
      }
      const designView = buildDesignView(mesh, fitted)
      if (designView.bvh.nodes.byteLength > storageLimit) throw new Error('Scene BVH exceeds the GPU buffer limit.')

      // Hold the old lens and target until the new GPU resources and first evaluation succeed.
      frameLoop?.destroy()
      frameLoop = undefined
      await device.queue.onSubmittedWorkDone()
      if (events.signal.aborted) return
      device.pushErrorScope('out-of-memory')
      device.pushErrorScope('validation')
      try {
        candidate = await createSceneSession(device, format, opticalToWorld, geometry, fitted, designView, message => warnings.add(message))
        if (events.signal.aborted) return
        // Include G-buffer allocation and first-hit splatting in the same validation scope.
        device.queue.submit([drawSceneFrame(candidate, 0, 'lit').commands])
        await device.queue.onSubmittedWorkDone()
      } finally {
        const [validation, memory] = await Promise.all([device.popErrorScope(), device.popErrorScope()])
        const error = validation ?? memory
        if (error) throw new Error(error.message || 'GPU could not prepare this scene.')
      }
      if (events.signal.aborted) return

      const next = candidate
      const newPad = createDrawingPad(targetCanvas, fitted, target => {
        next.state.target = target
        optimizationDone = false
        hasTarget = target.values.some(value => value > 0)
        targetStatus.textContent = hasTarget ? '' : 'Try a word, a shape, or a single line.'
        if (runTimer !== undefined) runStarted = performance.now()
        updateRunState()
        frameLoop?.requestFrame()
      }, designView, canUndo => { undoButton.disabled = !canUndo })
      newPad.setBrush(Number(brushSize.value))
      pad?.destroy()
      active?.destroy()
      pad = newPad
      active = next
      candidate = undefined
      sceneFraming = framing
      lensFraming = { target: [0, fitted.z - lens.thickness, 0], size: [fitted.sizeX, lens.thickness, fitted.sizeY] }
      causticCanvas.width = fitted.width
      causticCanvas.height = fitted.height
      for (const c of [targetCanvas, causticCanvas]) {
        c.style.setProperty('--scene-aspect', String(fitted.sizeX / fitted.sizeY))
        c.style.aspectRatio = String(fitted.sizeX / fitted.sizeY)
      }
      activePreset = typeof source === 'string' ? source : undefined
      sceneSelect.value = activePreset ?? ''
      credit.hidden = !activePreset
      if (activePreset) {
        const { author, credit: link } = bundledScenes[activePreset]
        credit.href = link
        credit.textContent = `Model by ${author} · CC BY 4.0`
      }
      // The model credit lives in the scene picker's tooltip.
      sceneSelect.title = activePreset ? credit.textContent : ''
      beamSelect.value = 'lit'
      changeView()
      activeLabel = name
      sceneStatus.textContent = [...warnings].join(' ')
      status.textContent = ''
    } catch (error) {
      if (events.signal.aborted) return
      const reason = error instanceof Error ? error.message : String(error)
      sceneStatus.textContent = `Could not load ${name}. ${reason} ` +
        (active ? `Current scene: ${activeLabel}.` : 'Select a scene or import an OBJ folder.')
      sceneSelect.value = activePreset ?? ''
      if (!active) status.textContent = 'Waiting for a scene.'
    } finally {
      candidate?.destroy()
      switching = false
      if (!events.signal.aborted) importButton.disabled = folderButton.disabled = sceneSelect.disabled = false
      if (!events.signal.aborted && active) {
        frameLoop ??= createFrameLoop(device, encodeFrame, { maxStepsPerFrame: config.maxStepsPerFrame, onError: fail })
        updateRunState()
        frameLoop.requestFrame()
      }
    }
  }

  importButton.addEventListener('click', () => objFile.click(), events)
  folderButton.addEventListener('click', () => folderInput.click(), events)
  for (const picker of [objFile, folderInput]) {
    picker.addEventListener('change', () => {
      const files = Array.from(picker.files ?? [])
      picker.value = ''
      if (!files.length) return
      const models = files.filter(file => /\.obj$/i.test(file.name))
      if (models.length !== 1) {
        sceneStatus.textContent = `Found ${models.length} OBJ files. Select one OBJ and its material files with Import OBJ. Current scene: ${activeLabel}.`
        return
      }
      void loadScene(models[0], files)
    }, events)
  }
  sceneSelect.addEventListener('change', () => { void loadScene(sceneSelect.value as BundledScene) }, events)
  await loadScene(defaultScene)
}

main().catch((error: unknown) => {
  console.error(error)
  if (lifetime.signal.aborted) return
  lifetime.abort()
  const status = document.querySelector('#status')
  if (status) status.textContent = error instanceof Error ? error.message : String(error)
})
