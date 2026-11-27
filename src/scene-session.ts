import type { Mat4 } from 'wgpu-matrix'
import { config, sceneLamp } from './config'
import type { AppState, ForwardModel, ForwardResult, LensOptimizer, ReceiverPlane, TargetImage } from './contracts'
import { createFlatLens, createLens } from './geometry/lens-mesh'
import type { DesignView } from './optics/design-view'
import { createForwardModel } from './optics/forward-model'
import { createApertureColor } from './optics/aperture-color'
import { createLensOptimizer } from './optimization/lens-optimizer'
import { uniformIlluminationHeights } from './optimization/uniform-illumination'
import { createBeamRenderer, type BeamView } from './rendering/beam-renderer'
import { createCausticDisplay } from './rendering/caustic-display'
import type { SceneGeometry } from './rendering/scene-mesh'
import { createSceneRenderer } from './rendering/scene-renderer'
import { createSimulation } from './simulation'

// A scene owns its lens, training state and draw resources. A failed replacement
// can be discarded before the active scene or its drawing pad is touched.
export async function createSceneSession(
  device: GPUDevice, format: GPUTextureFormat, opticalToWorld: Mat4, geometry: SceneGeometry, receiver: ReceiverPlane, designView: DesignView,
  warn: (message: string) => void = console.warn,
) {
  const lamp = sceneLamp(receiver)
  // Training writes to optimizer-owned buffers. The original lens is also the
  // reusable all-scene illumination when the artist clears the drawing.
  const initialLens = createLens(device, lamp.lens,
    uniformIlluminationHeights(lamp.lens, lamp.optics, receiver, designView))
  const state: AppState = { target: null, lens: initialLens, optics: lamp.optics, receiver }
  let forward: ForwardModel | undefined
  let coarseForward: ForwardModel | undefined
  let coarseHeights: GPUBuffer | undefined
  let optimizer: LensOptimizer | null = null
  let scene: Awaited<ReturnType<typeof createSceneRenderer>> | undefined
  let beams: ReturnType<typeof createBeamRenderer> | undefined
  let display: ReturnType<typeof createCausticDisplay> | undefined
  let apertureColor: ReturnType<typeof createApertureColor> | undefined

  function destroy() {
    display?.destroy()
    beams?.destroy()
    apertureColor?.destroy()
    scene?.destroy()
    optimizer?.destroy()
    forward?.destroy()
    coarseForward?.destroy()
    coarseHeights?.destroy()
    initialLens.heightBuffer.destroy()
  }

  try {
    scene = await createSceneRenderer(device, geometry.mesh, opticalToWorld, geometry.materials, warn)
    beams = createBeamRenderer(device, format, opticalToWorld,
      config.rendering.causticExposure, geometry, lamp.optics.lensIOR / lamp.optics.outsideIOR)
    beams.setEnvironmentLight(config.rendering.environmentLight)
    display = createCausticDisplay(device, format)
    display.setExposure(config.rendering.causticExposure)
    forward = await createForwardModel(device, state, { designView })
    // Coarse-to-fine training traces the same scene on a coarse copy of the lens.
    const coarseCells = lamp.optimizer.coarseGridCells
    if (coarseCells > 0) {
      const coarseLens = createFlatLens(device, { ...lamp.lens,
        columns: coarseCells + 1, rows: coarseCells + 1 })
      coarseHeights = coarseLens.heightBuffer
      coarseForward = await createForwardModel(device, { ...state, lens: coarseLens }, { designView })
    }
    optimizer = await createLensOptimizer(device, state, forward, lamp.optimizer, { coarseForward })
    const simulation = createSimulation(state, forward, optimizer ?? undefined)
    const initial = device.createCommandEncoder({ label: 'Initial scene evaluation' })
    let result = simulation.encode(initial).result
    await Promise.all([display.prepare(result.caustic.kind), display.prepare('rgb')])
    display.setImage(result.caustic)
    device.queue.submit([initial.finish()])

    let calibration: { result: ForwardResult; target: TargetImage; shapeVersion: number; colorVersion?: number } | undefined
    let filterRevision = 0, lastCalibration = -Infinity
    // The forward buffers may be reused. Recompute from the current evaluation,
    // and keep both color textures on the GPU between updates and during camera motion.
    function updateColor(encoder: GPUCommandEncoder, target: TargetImage | null, training: boolean, evaluated: boolean) {
      if (!target?.colors) {
        if (calibration) beams!.setColorFilter(undefined)
        if (calibration || evaluated) display!.setImage(result.caustic)
        calibration = undefined
        return
      }
      const now = performance.now()
      if (calibration?.target === target && calibration.shapeVersion === target.version &&
          calibration.colorVersion === target.colorVersion) {
        if (calibration.result === result || (training && now - lastCalibration < 100)) return
      }
      apertureColor ??= createApertureColor(device)
      const { filter, image } = apertureColor.update(encoder, result, target, designView)
      beams!.setColorFilter(filter, ++filterRevision)
      display!.setImage({ kind: 'rgb', view: image, width: receiver.width, height: receiver.height })
      calibration = { result, target, shapeVersion: target.version, colorVersion: target.colorVersion }
      lastCalibration = now
    }

    // The panel starts exposed for the floodlit scene. Once there is a drawing,
    // expose for the irradiance a fully painted pixel would get if all light
    // landed on the drawing, so the drawing reads white whatever its area and
    // stray light stays as dark as its share of the power.
    let exposedVersion: number | null = null
    const exposeFor = (target: TargetImage | null) => {
      const version = target?.version ?? null
      if (version === exposedVersion) return
      exposedVersion = version
      let paintedArea = 0
      if (target) {
        for (let i = 0; i < target.values.length; i++) paintedArea += target.values[i] * (designView.surfaceAreas[i] ?? 0)
      }
      const power = state.optics.incidentIrradiance * state.lens.sizeX * state.lens.sizeY
      const drawing = config.rendering.drawingExposure > 0 && paintedArea > 0 && power > 0
      display!.setExposure(drawing ? config.rendering.drawingExposure * paintedArea / power : config.rendering.causticExposure)
      // In the room, keep the brightest strokes just below white so the
      // surface under the light still shows; the initial illumination uses 1.
      beams!.setExposure(drawing ? Math.min(1, config.rendering.roomExposure * paintedArea / power) : 1)
    }

    return {
      state, canOptimize: !!optimizer, destroy,
      setTarget(target: TargetImage | null) {
        if (target?.colors && target.colors.length !== 3 * target.values.length) {
          throw new Error('Brush colors must have one RGB triplet per target pixel.')
        }
        state.target = target?.values.some((value, i) => value > 0 && designView.surfaceAreas[i] > 0) ? target : null
        return state.target !== null
      },
      setLightPathsVisible(value: boolean) { beams!.setLightPathsVisible(value) },
      render(encoder: GPUCommandEncoder, output: GPUTexture, preview: GPUTexture, vp: Mat4,
        mode: BeamView, steps: number) {
        optimizer?.setTarget(state.target)
        exposeFor(state.target)
        let done = false
        const restoreInitialLens = !state.target && state.lens.heightBuffer !== initialLens.heightBuffer
        if (restoreInitialLens) state.lens = initialLens
        const evaluated = restoreInitialLens || (!!state.target && steps > 0)
        if (evaluated) {
          const batch = simulation.encode(encoder, state.target ? steps : 0)
          result = batch.result
          done = batch.done
        }
        const training = !!state.target && steps > 0 && !done
        updateColor(encoder, state.target, training, evaluated)
        beams!.setReferenceIrradiance(state.optics.incidentIrradiance)
        scene!.update(vp)
        beams!.render(encoder, output, vp, mode, result, scene!.draw, training)
        const pass = encoder.beginRenderPass({ label: 'Caustic preview', colorAttachments: [{
          view: preview.createView(), clearValue: [0, 0, 0, 1], loadOp: 'clear', storeOp: 'store',
        }] })
        display!.draw(pass)
        pass.end()
        return done
      },
    }
  } catch (error) {
    destroy()
    throw error
  }
}

export type SceneSession = Awaited<ReturnType<typeof createSceneSession>>
