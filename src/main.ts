import './style.css'
import { mat4 } from 'wgpu-matrix'
import { config, incidentColumnHeight } from './config'
import { initializeGPU } from './gpu/context'
import { createDrawingPad } from './ui/drawing-pad'
import { createImageExport } from './ui/image-export'
import { bundledScenes, defaultScene, readBundledScene, type BundledScene } from './geometry/bundled-scenes'
import { readObjAsset } from './geometry/obj-materials'
import type { CoveSourceState } from './geometry/cove-source-textures'
import { placeObjOnReceiver } from './geometry/place-obj'
import { createOrbitCamera, type OrbitCameraOptions } from './rendering/orbit-camera'
import { roomViewport } from './rendering/room-viewport'
import { buildSceneGeometry, type SceneMapSlot, type SceneMaterial } from './rendering/scene-mesh'
import type { BeamView } from './rendering/beam-renderer'
import { createFrameLoop } from './frame-loop'
import { buildDesignView } from './optics/design-view'
import { createSceneSession, type SceneSession } from './scene-session'

function element<K extends keyof HTMLElementTagNameMap>(id: string, tag: K) {
  const node = document.querySelector<HTMLElementTagNameMap[K]>(`${tag}#${id}`)
  if (!node) throw new Error(`Missing application control: ${id}.`)
  return node
}

async function readImportedModel(model: File, files: File[], signal: AbortSignal) {
  if (/\.obj$/i.test(model.name)) return readObjAsset(model, files)
  const { readGltfAsset } = await import('./geometry/gltf-asset')
  return readGltfAsset(model, files, signal)
}

const lifetime = new AbortController()
if (import.meta.hot) import.meta.hot.dispose(() => lifetime.abort())

const materialDebug: Partial<Record<BeamView, { slots?: SceneMapSlot[]; legend?: string }>> = {
  albedo: { slots: ['map'] },
  normals: { slots: ['normalMap'] },
  'geometric-normals': {},
  'normal-map': { slots: ['normalMap'], legend: 'Flat = blue-purple · No map = black' },
  roughness: { slots: ['roughnessMap', 'metallicRoughnessMap'], legend: 'Black = smooth · White = rough' },
  metallic: { slots: ['metallicMap', 'metallicRoughnessMap'], legend: 'Black = dielectric · White = metal' },
  occlusion: { slots: ['occlusionMap'], legend: 'Black = occluded · White = open' },
  emissive: { slots: ['emissiveMap'] },
  'map-coverage': { legend: 'Red = normal · Green = roughness · Blue = metallic' },
}
const usesRoomFraming = (mode: BeamView) => mode === 'lit' || mode in materialDebug

async function main() {
  const canvas = element('gpu-canvas', 'canvas'), causticCanvas = element('caustic-canvas', 'canvas')
  const targetCanvas = element('target-canvas', 'canvas'), clearButton = element('clear-target', 'button')
  const status = element('status', 'p'), causticStatus = element('caustic-status', 'p')
  const sceneStatus = element('scene-status', 'p')
  const restoreButton = element('restore-textures', 'button')
  const importButton = element('import-model', 'button'), modelFiles = element('model-files', 'input')
  const folderButton = element('import-folder', 'button'), folderInput = element('asset-folder', 'input')
  const sceneSelect = element('bundled-scene', 'select'), beamSelect = element('beam-view', 'select')
  const credit = element('scene-credit', 'a')
  const topbar = element('topbar', 'header')
  const panel = element('design-panel', 'aside'), undoButton = element('undo-target', 'button')
  const brushSize = element('brush-size', 'input')
  const brushColor = element('brush-color', 'input'), swatches = element('brush-swatches', 'div')
  const recolor = element('recolor', 'button')
  const debugControl = element('debug-control', 'div')
  const debugLegend = element('debug-legend', 'output')
  const roomOptions = element('room-options', 'fieldset')
  const roomMaterials = element('room-materials', 'input')
  const lightPaths = element('light-paths', 'input')
  const viewButtons = Array.from(document.querySelectorAll<HTMLButtonElement>('#view-switch button[data-view]'))
  let currentView: BeamView = 'lit'
  for (const [id, preset] of Object.entries(bundledScenes)) sceneSelect.add(new Option(preset.name, id))
  const query = new URLSearchParams(location.search)
  const requestedScene = query.get('scene')
  const requestedView = query.get('view')
  const initialView = Array.from(beamSelect.options).some(option => option.value === requestedView)
    ? requestedView as BeamView : 'lit'
  const initialScene = requestedScene && Object.hasOwn(bundledScenes, requestedScene)
    ? requestedScene as BundledScene : defaultScene
  sceneSelect.value = initialScene
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
  let originalMaps: { source: CoveSourceState; materials: Map<string, SceneMaterial>; missing: number; upgrades: number } | undefined
  let restoreRequest: AbortController | undefined
  let materialUpload: Promise<void> | undefined
  let exportJob: Promise<ImageData> | undefined
  let exportStarted: number | undefined

  function updateMaterialLegend() {
    const info = materialDebug[currentView]
    const coverage = info?.slots && active?.mapCoverage(info.slots)
    const summary = coverage?.materials ? `Mapped materials ${coverage.mapped}/${coverage.materials}` : ''
    const text = [info?.legend, summary].filter(Boolean).join('\n')
    debugLegend.textContent = text
    debugLegend.hidden = !text
    debugLegend.title = coverage?.triangles ? `${(100 * coverage.mappedTriangles / coverage.triangles).toFixed(1)}% of triangles assigned a material sample this channel from a texture.` : ''
  }

  function updateRestoreButton() {
    const pending = originalMaps ? originalMaps.missing + originalMaps.upgrades : 0
    restoreButton.hidden = !pending
    restoreButton.disabled = switching || !!restoreRequest || events.signal.aborted
    restoreButton.textContent = restoreRequest ? 'Loading original maps…' : `Load original maps (${pending})`
  }

  async function restoreOriginalMaps() {
    const owner = active, maps = originalMaps
    if (!owner || !maps || (!maps.missing && !maps.upgrades) || switching || restoreRequest) return
    const request = new AbortController()
    restoreRequest = request
    updateRestoreButton()
    try {
      const result = await maps.source.restore(maps.materials, request.signal)
      request.signal.throwIfAborted()
      const restored = (maps.missing - result.missing) + (maps.upgrades - result.upgrades)
      if (restored > 0) {
        // Keep all tiles of an export on the same material set.
        while (exportJob) {
          await exportJob.catch(() => {})
          request.signal.throwIfAborted()
        }
        const upload = owner.replaceMaterials(result.materials, request.signal)
        materialUpload = upload
        try { await upload }
        finally { if (materialUpload === upload) materialUpload = undefined }
        request.signal.throwIfAborted()
        maps.materials = result.materials
        updateMaterialLegend()
        frameLoop?.requestFrame()
      }
      maps.missing = result.missing
      maps.upgrades = result.upgrades
      const message = restored > 0 ? `Loaded ${restored} original maps.` : ''
      const missing = result.missing && result.warnings.length
        ? `${result.missing} original maps unavailable. Showing local materials.` : ''
      const upgrades = result.upgrades && result.warnings.length
        ? `Using the bundled resolution for ${result.upgrades} maps.` : ''
      sceneStatus.textContent = [message, missing, upgrades, !missing && !upgrades ? result.warnings[0] : ''].filter(Boolean).join(' ')
      sceneStatus.title = result.warnings.join('\n')
    } catch (error) {
      if (!request.signal.aborted) sceneStatus.textContent = error instanceof Error ? error.message : String(error)
    } finally {
      if (restoreRequest === request) {
        restoreRequest = undefined
        updateRestoreButton()
      }
    }
  }
  restoreButton.addEventListener('click', () => { void restoreOriginalMaps() }, events)

  lifetime.signal.addEventListener('abort', () => {
    restoreRequest?.abort()
    importButton.disabled = folderButton.disabled = true
    sceneSelect.disabled = roomOptions.disabled = true
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
  const defaultFraming: OrbitCameraOptions = {
    target: [0, (receiver.z - lens.thickness) / 2, 0],
    size: [lens.sizeX, receiver.z, lens.sizeY],
  }
  let roomFraming = defaultFraming, sceneFraming = defaultFraming
  let currentFraming = defaultFraming
  camera = createOrbitCamera(canvas, defaultFraming, () => frameLoop?.requestFrame())
  const viewportTransform = mat4.identity()

  function composition() {
    const panelRect = panel.getBoundingClientRect()
    return roomViewport(Math.max(1, canvas.clientWidth), Math.max(1, canvas.clientHeight),
      panelRect.right, panelRect.top, topbar.getBoundingClientRect().bottom)
  }

  const imageExport = createImageExport({
    signal: events.signal,
    getScene() {
      if (!active || switching || events.signal.aborted) return
      const rect = composition()
      return {
        name: activeLabel, aspect: rect.width / rect.height,
        materialMode: roomMaterials.checked ? 'pbr' : 'basecolor',
      }
    },
    async render(request) {
      if (!active || !frameLoop || switching || exportJob) throw new Error('The scene is busy. Try again in a moment.')
      const owner = active, loop = frameLoop
      const viewProjection = mat4.copy(camera!.update(request.width / request.height, roomFraming))
      const signal = AbortSignal.any([request.signal, events.signal])
      const started = performance.now(), trainingSince = runTimer === undefined ? undefined : runStarted
      exportStarted = started
      loop.setPaused(true)
      const job = (async () => {
        if (materialUpload) await materialUpload.catch(() => {})
        await device.queue.onSubmittedWorkDone()
        signal.throwIfAborted()
        return owner.exportImage({ ...request, signal, viewProjection })
      })()
      exportJob = job
      try { return await job }
      finally {
        exportJob = undefined
        exportStarted = undefined
        if (trainingSince === runStarted && runTimer !== undefined && active === owner) {
          runStarted += performance.now() - started
        }
        if (!events.signal.aborted && active === owner) {
          loop.setPaused(false)
          loop.requestFrame()
        }
      }
    },
  })

  const seconds = () => Math.max(0, Math.round(((exportStarted ?? performance.now()) - runStarted) / 1000))
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
      causticStatus.textContent = 'Scene lighting'
    } else if (optimizationDone) {
      causticStatus.textContent = runStarted ? `Done in ${seconds()} s` : 'Done'
    } else {
      causticStatus.textContent = `Shaping the glass · ${seconds()} s`
    }
  }

  function changeView(mode: BeamView = currentView) {
    currentView = mode
    const debug = mode !== 'lit'
    debugControl.hidden = !debug
    roomOptions.hidden = debug
    updateMaterialLegend()
    const framing = usesRoomFraming(mode) ? roomFraming : sceneFraming
    if (currentFraming !== framing) {
      camera!.frame(framing)
      currentFraming = framing
    }
    for (const button of viewButtons) button.setAttribute('aria-pressed', String(button.dataset.view === (debug ? 'debug' : mode)))
    frameLoop?.requestFrame()
  }
  beamSelect.addEventListener('change', () => changeView(beamSelect.value as BeamView), events)
  roomMaterials.addEventListener('change', () => {
    active?.setMaterialsEnabled(roomMaterials.checked)
    frameLoop?.requestFrame()
  }, events)
  lightPaths.addEventListener('change', () => {
    active?.setLightPathsVisible(lightPaths.checked)
    frameLoop?.requestFrame()
  }, events)
  for (const button of viewButtons) {
    button.addEventListener('click', () => {
      changeView((button.dataset.view === 'debug' ? beamSelect.value : button.dataset.view) as BeamView)
    }, events)
  }
  clearButton.addEventListener('click', () => pad?.clear(), events)
  undoButton.addEventListener('click', () => pad?.undo(), events)
  brushSize.addEventListener('input', () => pad?.setBrush(Number(brushSize.value)), events)
  const palette = [
    ['White', '#ffffff'], ['Red', '#ff3846'], ['Amber', '#ffaf24'],
    ['Green', '#32dc85'], ['Blue', '#406dff'], ['Violet', '#b35aff'],
  ]
  function selectColor(color: string) {
    brushColor.value = color
    let custom = true
    for (const button of swatches.querySelectorAll('button')) {
      const selected = button.value === color
      button.setAttribute('aria-pressed', String(selected))
      if (selected) custom = false
    }
    brushColor.parentElement!.classList.toggle('selected', custom)
    brushColor.parentElement!.style.setProperty('--swatch', color)
    pad?.setColor(color)
  }
  const swatchButtons = document.createDocumentFragment()
  for (const [name, color] of palette) {
    const button = document.createElement('button')
    button.type = 'button'
    button.value = color
    button.title = name + ' light'
    button.setAttribute('aria-label', button.title)
    button.style.setProperty('--swatch', color)
    button.addEventListener('click', () => selectColor(color), events)
    swatchButtons.append(button)
  }
  swatches.replaceChildren(swatchButtons)
  selectColor(brushColor.value)
  brushColor.addEventListener('input', () => selectColor(brushColor.value), events)
  recolor.addEventListener('click', () => {
    const selected = recolor.getAttribute('aria-pressed') !== 'true'
    recolor.setAttribute('aria-pressed', String(selected))
    pad?.setRecolor(selected)
  }, events)
  window.addEventListener('resize', () => frameLoop?.requestFrame(), events)

  function drawSceneFrame(scene: SceneSession, steps: number, mode: BeamView) {
    const ratio = Math.min(window.devicePixelRatio, config.rendering.maxPixelRatio)
    const limit = device.limits.maxTextureDimension2D
    const cssWidth = Math.max(1, canvas.clientWidth), cssHeight = Math.max(1, canvas.clientHeight)
    const width = Math.min(limit, Math.max(1, Math.round(cssWidth * ratio)))
    const height = Math.min(limit, Math.max(1, Math.round(cssHeight * ratio)))
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width
      canvas.height = height
    }
    const encoder = device.createCommandEncoder()
    const rect = composition()
    const roomFramed = usesRoomFraming(mode)
    // Fit Room in the visible rectangle. The projection still maps to the full
    // canvas, so camera rays and G-buffer depth keep the same pixel coordinates.
    const viewProjection = camera!.update(roomFramed ? rect.width / rect.height : width / height)
    viewportTransform[0] = roomFramed ? rect.width / cssWidth : 1
    viewportTransform[5] = roomFramed ? rect.height / cssHeight : 1
    viewportTransform[12] = rect.left / cssWidth
    viewportTransform[13] = roomFramed ? 1 - (2 * rect.top + rect.height) / cssHeight : 0
    mat4.multiply(viewportTransform, viewProjection, viewProjection)
    const done = scene.render(encoder, context.getCurrentTexture(), previewContext!.getCurrentTexture(),
      viewProjection, mode, steps)
    return { done, commands: encoder.finish() }
  }

  function encodeFrame(steps: number) {
    const { done, commands } = drawSceneFrame(active!, steps, currentView)
    if (done) { optimizationDone = true; updateRunState() }
    return commands
  }

  async function loadScene(source: BundledScene | File, files: File[] = [], view: BeamView = 'lit') {
    if (switching || events.signal.aborted) return
    restoreRequest?.abort()
    restoreRequest = undefined
    switching = true
    imageExport.refresh()
    updateRestoreButton()
    roomOptions.disabled = true
    importButton.disabled = folderButton.disabled = sceneSelect.disabled = true
    const preset = typeof source === 'string' ? bundledScenes[source] : undefined
    const name = typeof source === 'string' ? bundledScenes[source].name : source.name
    const roomCamera = preset && 'roomCamera' in preset ? preset.roomCamera : undefined
    sceneStatus.title = ''
    sceneStatus.textContent = `Loading ${name}…`
    let candidate: SceneSession | undefined
    let loaded = false
    let warnings = new Set<string>()
    try {
      if (exportJob) await exportJob.catch(() => {})
      if (events.signal.aborted) return
      const asset: Awaited<ReturnType<typeof readImportedModel>> & { sourceTextures?: CoveSourceState } =
        typeof source === 'string' ? await readBundledScene(source, events.signal)
        : await readImportedModel(source, files, events.signal)
      if (events.signal.aborted) return
      warnings = asset.warnings
      const bounds = placeObjOnReceiver(asset.mesh, lens, receiver)
      const fitted = bounds.receiver
      // Place the fitted receiver on the world floor, with the lamp directly above it.
      const opticalToWorld = mat4.multiply(mat4.translation([0, fitted.z, 0]), mat4.rotationX(Math.PI / 2))
      const geometry = buildSceneGeometry(fitted, {
        mesh: asset.mesh, receiverFloor: source !== 'smugglers-cove',
      })
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
        candidate = await createSceneSession(device, format, opticalToWorld, geometry, fitted, designView,
          message => warnings.add(message))
        if (events.signal.aborted) return
        candidate.setMaterialsEnabled(roomMaterials.checked)
        candidate.setLightPathsVisible(lightPaths.checked)
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
      let lastShapeVersion: number | undefined
      const newPad = createDrawingPad(targetCanvas, fitted, target => {
        hasTarget = next.setTarget(target)
        recolor.disabled = !hasTarget
        if (!hasTarget) {
          recolor.setAttribute('aria-pressed', 'false')
          pad?.setRecolor(false)
        }
        if (lastShapeVersion !== target.version) {
          lastShapeVersion = target.version
          optimizationDone = false
          if (runTimer !== undefined) runStarted = performance.now()
          updateRunState()
        }
        frameLoop?.requestFrame()
      }, designView, canUndo => { undoButton.disabled = !canUndo })
      newPad.setBrush(Number(brushSize.value))
      newPad.setColor(brushColor.value)
      pad?.destroy()
      active?.destroy()
      pad = newPad
      active = next
      originalMaps = asset.sourceTextures ? {
        source: asset.sourceTextures, materials: asset.materials,
        missing: asset.sourceTextures.missing, upgrades: asset.sourceTextures.upgrades,
      } : undefined
      candidate = undefined
      sceneFraming = framing
      const aperture = next.state.lens
      const roomHeight = fitted.z + incidentColumnHeight(fitted.z, Math.max(aperture.sizeX, aperture.sizeY))
      // Frame the shore itself so the overhead lamp does not push the
      // starting view away from the surfaces the artist wants to paint.
      roomFraming = roomCamera ? { ...sceneFraming, ...roomCamera } : {
        ...sceneFraming, azimuth: 0.65,
        lampTop: { y: roomHeight, width: aperture.sizeX, depth: aperture.sizeY },
      }
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
        const { attribution, credit: link } = bundledScenes[activePreset]
        credit.href = link
        credit.textContent = attribution
      }
      // The model credit lives in the scene picker's tooltip.
      sceneSelect.title = activePreset ? credit.textContent : ''
      if (view !== 'lit') beamSelect.value = view
      changeView(view)
      activeLabel = name
      sceneStatus.textContent = [...warnings].join(' ')
      status.textContent = ''
      loaded = true
    } catch (error) {
      if (events.signal.aborted) return
      const reason = error instanceof Error ? error.message : String(error)
      sceneStatus.textContent = `Could not load ${name}. ${reason} ` +
        (active ? `Current scene: ${activeLabel}.` : 'Select a scene or open a model.')
      sceneSelect.value = activePreset ?? ''
      if (!active) status.textContent = 'Waiting for a scene.'
    } finally {
      candidate?.destroy()
      switching = false
      updateRestoreButton()
      imageExport.refresh()
      if (!events.signal.aborted) importButton.disabled = folderButton.disabled = sceneSelect.disabled = false
      if (!events.signal.aborted && active) {
        roomOptions.disabled = false
        frameLoop ??= createFrameLoop(device, encodeFrame, { maxStepsPerFrame: config.maxStepsPerFrame, onError: fail })
        updateRunState()
        frameLoop.requestFrame()
        if (loaded && originalMaps) void restoreOriginalMaps()
      }
    }
  }

  importButton.addEventListener('click', () => modelFiles.click(), events)
  folderButton.addEventListener('click', () => folderInput.click(), events)
  for (const picker of [modelFiles, folderInput]) {
    picker.addEventListener('change', () => {
      const files = Array.from(picker.files ?? [])
      picker.value = ''
      if (!files.length) return
      const models = files.filter(file => /\.(glb|gltf|obj)$/i.test(file.name))
      if (models.length !== 1) {
        sceneStatus.textContent = `Found ${models.length} models. Choose one GLB, glTF or OBJ together with its companion files.`
        return
      }
      void loadScene(models[0], files)
    }, events)
  }
  sceneSelect.addEventListener('change', () => { void loadScene(sceneSelect.value as BundledScene) }, events)
  await loadScene(initialScene, [], initialView)
}

main().catch((error: unknown) => {
  console.error(error)
  if (lifetime.signal.aborted) return
  lifetime.abort()
  const status = document.querySelector('#status')
  if (status) status.textContent = error instanceof Error ? error.message : String(error)
})
