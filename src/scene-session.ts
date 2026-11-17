import type { Mat4 } from 'wgpu-matrix'
import { config } from './config'
import type { AppState, ForwardModel, LensOptimizer, ReceiverPlane, TargetImage } from './contracts'
import { createFlatLens } from './geometry/lens-mesh'
import type { DesignView } from './optics/design-view'
import { createForwardModel } from './optics/forward-model'
import { createLensOptimizer } from './optimization/lens-optimizer'
import { createBeamRenderer, type BeamView } from './rendering/beam-renderer'
import { createCausticDisplay } from './rendering/caustic-display'
import { createLensRenderer } from './rendering/lens-renderer'
import type { SceneGeometry } from './rendering/scene-mesh'
import { createSceneRenderer } from './rendering/scene-renderer'
import { createSimulation } from './simulation'

// A scene owns its lens, training state and draw resources. A failed replacement
// can be discarded before the active scene or its drawing pad is touched.
export async function createSceneSession(
  device: GPUDevice, format: GPUTextureFormat, opticalToWorld: Mat4, geometry: SceneGeometry, receiver: ReceiverPlane, designView: DesignView,
  warn: (message: string) => void = console.warn,
) {
  const lens = createFlatLens(device, { ...config.lens, sizeX: receiver.sizeX, sizeY: receiver.sizeY })
  const state: AppState = { target: null, lens, optics: { ...config.optics }, receiver }
  let forward: ForwardModel | undefined
  let coarseForward: ForwardModel | undefined
  let coarseHeights: GPUBuffer | undefined
  let optimizer: LensOptimizer | null = null
  let scene: Awaited<ReturnType<typeof createSceneRenderer>> | undefined
  let beams: ReturnType<typeof createBeamRenderer> | undefined
  let display: ReturnType<typeof createCausticDisplay> | undefined
  let lensPreview: ReturnType<typeof createLensRenderer> | undefined

  function destroy() {
    lensPreview?.destroy()
    display?.destroy()
    beams?.destroy()
    scene?.destroy()
    optimizer?.destroy()
    forward?.destroy()
    coarseForward?.destroy()
    coarseHeights?.destroy()
    lens.heightBuffer.destroy()
  }

  try {
    scene = await createSceneRenderer(device, geometry.mesh, opticalToWorld, geometry.materials, warn)
    beams = createBeamRenderer(device, format, opticalToWorld,
      config.rendering.causticExposure, geometry)
    beams.setEnvironmentLight(config.rendering.environmentLight)
    display = createCausticDisplay(device, format)
    display.setExposure(config.rendering.causticExposure)
    forward = await createForwardModel(device, state, { designView })
    // Coarse-to-fine training traces the same scene on a coarse copy of the lens.
    const coarseCells = config.optimizer.coarseGridCells
    if (coarseCells > 0) {
      const coarseLens = createFlatLens(device, { ...config.lens, sizeX: receiver.sizeX, sizeY: receiver.sizeY,
        columns: coarseCells + 1, rows: coarseCells + 1 })
      coarseHeights = coarseLens.heightBuffer
      coarseForward = await createForwardModel(device, { ...state, lens: coarseLens }, { designView })
    }
    optimizer = await createLensOptimizer(device, state, forward, config.optimizer, { coarseForward })
    const simulation = createSimulation(state, forward, optimizer ?? undefined)
    const initial = device.createCommandEncoder({ label: 'Initial scene evaluation' })
    let result = simulation.encode(initial).result
    await display.prepare(result.caustic.kind)
    display.setImage(result.caustic)
    device.queue.submit([initial.finish()])

    // The panel starts exposed for the unlit scene. Once there is a drawing,
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
      // surface under the light still shows; plain glass stays at 1.
      beams!.setExposure(drawing ? Math.min(1, config.rendering.roomExposure * paintedArea / power) : 1)
    }

    return {
      state, canOptimize: !!optimizer, destroy,
      render(encoder: GPUCommandEncoder, output: GPUTexture, preview: GPUTexture, vp: Mat4,
        mode: BeamView, steps: number) {
        optimizer?.setTarget(state.target)
        exposeFor(state.target)
        let done = false
        if (steps > 0) {
          const batch = simulation.encode(encoder, steps)
          result = batch.result
          done = batch.done
          display!.setImage(result.caustic)
        }
        if (mode === 'lens') {
          lensPreview ??= createLensRenderer(device, format, state.lens, opticalToWorld, config.lens.thickness)
          lensPreview.update(state.lens, vp, config.rendering.lensRelief)
          lensPreview.prepare(encoder)
        }
        beams!.setReferenceIrradiance(state.optics.incidentIrradiance)
        scene!.update(vp)
        beams!.render(encoder, output, vp, mode, result, scene!.draw, pass => lensPreview?.draw(pass))
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
