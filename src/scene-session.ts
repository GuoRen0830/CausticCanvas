import type { Mat4 } from 'wgpu-matrix'
import { config } from './config'
import type { AppState, ForwardModel, LensOptimizer } from './contracts'
import { createFlatLens } from './geometry/lens-mesh'
import type { DesignView } from './optics/design-view'
import { createForwardModel } from './optics/forward-model'
import { createLensOptimizer } from './optimization/lens-optimizer'
import { createBeamRenderer, type BeamSource, type BeamView, type SceneMethod } from './rendering/beam-renderer'
import { createCausticDisplay } from './rendering/caustic-display'
import { createLensRenderer } from './rendering/lens-renderer'
import type { SceneGeometry } from './rendering/scene-mesh'
import { createSceneRenderer } from './rendering/scene-renderer'
import { createSimulation } from './simulation'

// A scene owns its lens, training state and draw resources. A failed replacement
// can be discarded before the active scene or its drawing pad is touched.
export async function createSceneSession(
  device: GPUDevice, format: GPUTextureFormat, opticalToWorld: Mat4, geometry: SceneGeometry, designView: DesignView,
) {
  const lens = createFlatLens(device, config.lens)
  const state: AppState = { target: null, lens, optics: { ...config.optics }, receiver: { ...config.receiver } }
  let forward: ForwardModel | undefined
  let optimizer: LensOptimizer | null = null
  let scene: ReturnType<typeof createSceneRenderer> | undefined
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
    lens.heightBuffer.destroy()
  }

  try {
    scene = createSceneRenderer(device, geometry.mesh, opticalToWorld)
    beams = createBeamRenderer(device, format, config.lens, state.receiver, opticalToWorld,
      config.rendering.causticExposure, geometry)
    display = createCausticDisplay(device, format)
    display.setExposure(config.rendering.causticExposure)
    forward = await createForwardModel(device, state, { designView })
    optimizer = await createLensOptimizer(device, state, forward)
    const simulation = createSimulation(state, forward, optimizer ?? undefined)
    const initial = device.createCommandEncoder({ label: 'Initial scene evaluation' })
    let result = simulation.encode(initial).result
    await display.prepare(result.caustic.kind)
    display.setImage(result.caustic)
    device.queue.submit([initial.finish()])

    return {
      state, canOptimize: !!optimizer, destroy,
      render(encoder: GPUCommandEncoder, output: GPUTexture, preview: GPUTexture, vp: Mat4,
        source: BeamSource, mode: BeamView, method: SceneMethod, steps: number) {
        optimizer?.setTarget(state.target)
        let done = false
        if (steps > 0) {
          const batch = simulation.encode(encoder, steps)
          result = batch.result
          done = batch.done
          display!.setImage(result.caustic)
        }
        if (mode === 'lens') {
          lensPreview ??= createLensRenderer(device, format, state.lens, opticalToWorld)
          lensPreview.update(state.lens, vp)
        }
        scene!.update(vp)
        beams!.render(encoder, output, vp, source, mode, result, scene!.draw, pass => lensPreview?.draw(pass), method)
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
