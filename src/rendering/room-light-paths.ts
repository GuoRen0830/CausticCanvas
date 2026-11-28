import type { Mat4 } from 'wgpu-matrix'
import type { ForwardResult } from '../contracts'
import { beamVolumeBounds, createBeamVolume } from './beam-volume'
import { createRoomGlass } from './room-glass'
import { createVolumeMarch } from './volume-march'

// Each orbit re-integrates the same refracted light field along the new camera rays.
export function createRoomLightPaths(device: GPUDevice, layout: GPUBindGroupLayout, fullResolution = false) {
  const volume = createBeamVolume(device, fullResolution)
  const march = createVolumeMarch(device, fullResolution)
  const glass = createRoomGlass(device)
  let sampledResult: ForwardResult | undefined
  let sampledFilter: GPUTextureView | undefined
  let sampledColorRevision = 0
  let sampledDuringTraining = false
  let lastUpdate = -Infinity
  let overlay: GPUTexture | undefined
  let overlayView: GPUTextureView
  let lastFog: GPUTextureView | undefined
  let group: GPUBindGroup | undefined

  return {
    prepare(encoder: GPUCommandEncoder, output: GPUTexture, depth: GPUTextureView,
      vp: Mat4, inverseVP: Mat4, result: ForwardResult, offsets: GPUBuffer,
      incidentIrradiance: number, relativeIOR: number, training: boolean,
      colorFilter?: GPUTextureView, colorRevision = 0) {
      const { lens } = result.beams
      const z = result.receiver.z
      const now = performance.now()
      // New evaluations may reuse the same GPU buffers. Limit field rebuilds
      // during optimization, then resolve the settled lens at full density.
      const needsField = sampledResult !== result || sampledFilter !== colorFilter ||
        sampledColorRevision !== colorRevision || (sampledDuringTraining && !training)
      if (needsField && (!training || now - lastUpdate >= 100)) {
        volume.update(encoder, result, offsets, z, training, colorFilter)
        sampledResult = result
        sampledFilter = colorFilter
        sampledColorRevision = colorRevision
        sampledDuringTraining = training
        lastUpdate = now
      }
      const fog = march.render(encoder, output.width, output.height, depth, volume.textureView,
        volume.sampler, inverseVP, beamVolumeBounds(result, z), [lens.sizeX, lens.sizeY], z, incidentIrradiance,
        relativeIOR, [result.receiver.sizeX, result.receiver.sizeY])
      if (!overlay || overlay.width !== output.width || overlay.height !== output.height) {
        overlay?.destroy()
        overlay = device.createTexture({
          label: 'Room glass overlay', size: [output.width, output.height],
          format: 'rgba16float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        })
        overlayView = overlay.createView()
        group = undefined
      }
      const pass = encoder.beginRenderPass({
        label: 'Room glass',
        colorAttachments: [{ view: overlayView, loadOp: 'clear', clearValue: [0, 0, 0, 0], storeOp: 'store' }],
        depthStencilAttachment: { view: depth, depthReadOnly: true },
      })
      glass.draw(pass, vp, lens, offsets, relativeIOR)
      pass.end()
      if (!group || fog !== lastFog) {
        group = device.createBindGroup({ layout, entries: [
          { binding: 3, resource: fog }, { binding: 4, resource: overlayView },
        ] })
        lastFog = fog
      }
      return group
    },
    destroy() {
      overlay?.destroy()
      glass.destroy()
      march.destroy()
      volume.destroy()
    },
  }
}
