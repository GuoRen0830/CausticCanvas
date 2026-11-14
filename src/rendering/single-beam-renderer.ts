import { mat4, vec3 } from 'wgpu-matrix'
import type { Mat4 } from 'wgpu-matrix'
import source from './shaders/single-beam.wgsl?raw'

export type BeamView = 'lit' | 'candidates' | 'footprint'

// An intentionally enlarged test triangle on the initial flat lens. XY values
// are fractions of the full aperture; direction is in optical coordinates.
const testBeam = {
  corners: [[-0.375, -0.325], [0.125, -0.325], [-0.375, 0.175]],
  direction: [-0.18, 0.15, 1],
  fluxWatts: 0.012,
}

export function createSingleBeamRenderer(
  device: GPUDevice,
  canvasFormat: GPUTextureFormat,
  lens: { sizeX: number; sizeY: number; thickness: number },
  receiverZ: number,
  opticalToWorld: Mat4,
) {
  // Two matrices followed by four vec4s. Their w components hold flux,
  // beam length, display mode, and padding respectively (see WGSL Beam).
  const data = new Float32Array(48)
  const inverseVP = data.subarray(16, 32)
  testBeam.corners.forEach(([x, y], i) => {
    const local = [x * lens.sizeX, y * lens.sizeY, lens.thickness]
    data.set(vec3.transformMat4(local, opticalToWorld), 32 + 4 * i)
  })
  const localDirection = vec3.normalize(testBeam.direction)
  data[35] = testBeam.fluxWatts
  // Extend slightly past the receiver to include it despite depth quantization.
  data[39] = (receiverZ - lens.thickness) / localDirection[2] + 0.001
  data.set(vec3.transformMat4Upper3x3(localDirection, opticalToWorld), 44)
  const uniform = device.createBuffer({
    label: 'Single beam and camera', size: data.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  const module = device.createShaderModule({ label: 'Single beam', code: source })
  const add: GPUBlendComponent = { srcFactor: 'one', dstFactor: 'one', operation: 'add' }
  const accumulate = device.createRenderPipeline({
    label: 'Single beam accumulation', layout: 'auto',
    vertex: { module, entryPoint: 'boundsVS' },
    fragment: { module, entryPoint: 'accumulateFS', targets: [
      { format: 'rg16float', blend: { color: add, alpha: add } },
    ] },
    primitive: { topology: 'triangle-strip' },
  })
  const composite = device.createRenderPipeline({
    label: 'Scene composite', layout: 'auto',
    vertex: { module, entryPoint: 'fullscreenVS' },
    fragment: { module, entryPoint: 'compositeFS', targets: [{ format: canvasFormat }] },
    // The lens draws next, using the floor and box depth from the first pass.
    depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'always' },
  })

  function createTargets(width: number, height: number) {
    const make = (label: string, format: GPUTextureFormat) => device.createTexture({
      label, format, size: [width, height],
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })
    const albedo = make('Scene albedo and occupancy', 'rgba8unorm')
    const normal = make('World normal and receiver mask', 'rgba8unorm')
    const depth = make('Scene depth', 'depth24plus')
    const light = make('Beam irradiance and candidate count', 'rg16float')
    const albedoView = albedo.createView()
    const normalView = normal.createView()
    const depthView = depth.createView()
    const lightView = light.createView()
    const beamBinding = { binding: 0, resource: { buffer: uniform } }
    const normalBinding = { binding: 2, resource: normalView }
    const beamGroup = device.createBindGroup({
      layout: accumulate.getBindGroupLayout(0),
      entries: [beamBinding, { binding: 1, resource: depthView }, normalBinding],
    })
    const compositeGroup = device.createBindGroup({
      layout: composite.getBindGroupLayout(0),
      entries: [beamBinding, normalBinding, { binding: 3, resource: albedoView },
        { binding: 4, resource: lightView }],
    })
    return {
      width, height, albedoView, normalView, depthView, lightView, beamGroup, compositeGroup,
      destroy() {
        for (const texture of [albedo, normal, depth, light]) texture.destroy()
      },
    }
  }
  let targets: ReturnType<typeof createTargets> | undefined
  const clear = (view: GPUTextureView): GPURenderPassColorAttachment => ({
    view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0],
  })

  return {
    render(
      encoder: GPUCommandEncoder, output: GPUTexture, viewProjection: Mat4, mode: BeamView,
      drawScene: (pass: GPURenderPassEncoder) => void,
      drawLens: (pass: GPURenderPassEncoder) => void,
    ) {
      if (!targets || targets.width !== output.width || targets.height !== output.height) {
        targets?.destroy()
        targets = createTargets(output.width, output.height)
      }
      data.set(viewProjection, 0)
      mat4.inverse(viewProjection, inverseVP)
      data[43] = { lit: 0, candidates: 1, footprint: 2 }[mode]
      device.queue.writeBuffer(uniform, 0, data)

      const geometry = encoder.beginRenderPass({
        label: 'Scene G-buffer',
        colorAttachments: [clear(targets.albedoView), clear(targets.normalView)],
        depthStencilAttachment: {
          view: targets.depthView, depthLoadOp: 'clear', depthClearValue: 1, depthStoreOp: 'store',
        },
      })
      drawScene(geometry)
      geometry.end()

      const beam = encoder.beginRenderPass({
        label: 'Accumulate test beam', colorAttachments: [clear(targets.lightView)],
      })
      beam.setPipeline(accumulate)
      beam.setBindGroup(0, targets.beamGroup)
      beam.draw(4)
      beam.end()

      const display = encoder.beginRenderPass({
        label: 'Display scene and lens', colorAttachments: [clear(output.createView())],
        depthStencilAttachment: {
          view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'discard',
        },
      })
      display.setPipeline(composite)
      display.setBindGroup(0, targets.compositeGroup)
      display.draw(3)
      // Diagnostic views expose all tested pixels, including those under the lens overlay.
      if (mode === 'lit') drawLens(display)
      display.end()
    },
    destroy() {
      targets?.destroy()
      uniform.destroy()
    },
  }
}
