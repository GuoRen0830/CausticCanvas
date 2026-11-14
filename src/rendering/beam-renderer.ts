import { mat4, vec3 } from 'wgpu-matrix'
import type { Mat4 } from 'wgpu-matrix'
import type { CausticImage, ForwardResult, ReceiverPlane } from '../contracts'
import type { SceneBox } from './scene-renderer'
import { createTestBeams } from './test-beams'
import beamShader from './shaders/beam.wgsl?raw'
import receiverShader from './shaders/receiver-compare.wgsl?raw'
import displayColor from './shaders/irradiance-display.wgsl?raw'

export type BeamSource = 'solver' | 'single' | 'pair'
export type BeamView = 'lit' | 'candidates' | 'footprint' | 'plane' | 'difference' | 'blocked'
const modeId = { lit: 0, candidates: 1, footprint: 2, plane: 3, difference: 4, blocked: 5 }

const clear = (view: GPUTextureView): GPURenderPassColorAttachment => ({
  view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0],
})

export function createBeamRenderer(
  device: GPUDevice, canvasFormat: GPUTextureFormat,
  aperture: { sizeX: number; sizeY: number; thickness: number },
  receiver: ReceiverPlane, opticalToWorld: Mat4, exposure: number, box: SceneBox,
) {
  const tests = createTestBeams(device, aperture)
  const data = new Float32Array(60)
  const words = new Uint32Array(data.buffer)
  const localVP = data.subarray(0, 16)
  const inverseLocalVP = data.subarray(16, 32)
  // Shared scene geometry, in optical-local meters. Each uniform vec3 starts
  // on a 16-byte boundary. Rotate the inspection light into this space once.
  data.set(box.center.map((c, axis) => c - 0.5 * box.size[axis]), 48)
  data.set(box.center.map((c, axis) => c + 0.5 * box.size[axis]), 52)
  data.set(vec3.normalize(vec3.transformMat4Upper3x3([-0.4, 0.8, 0.6], mat4.inverse(opticalToWorld))), 56)
  const uniform = device.createBuffer({
    label: 'Beam frame', size: data.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  const module = device.createShaderModule({
    label: 'Beam rendering and plane comparison', code: [displayColor, beamShader, receiverShader].join('\n'),
  })
  const fieldLayout = device.createBindGroupLayout({
    entries: [0, 1].map(binding => ({
      binding, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' },
    })),
  })
  function frameLayout(entries: GPUBindGroupLayoutEntry[]) {
    return device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ...entries,
    ] })
  }
  const sceneLayout = frameLayout([
    { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
    { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} },
  ])
  const planeLayout = frameLayout([])
  const add: GPUBlendComponent = { srcFactor: 'one', dstFactor: 'one', operation: 'add' }
  function beamPipeline(label: string, layout: GPUBindGroupLayout, entryPoint: string) {
    return device.createRenderPipeline({
      label, layout: device.createPipelineLayout({ bindGroupLayouts: [layout, fieldLayout] }),
      vertex: { module, entryPoint: 'boundsVS' },
      fragment: { module, entryPoint, targets: [{ format: 'rg16float', blend: { color: add, alpha: add } }] },
      primitive: { topology: 'triangle-strip' },
    })
  }
  const accumulate = beamPipeline('Accumulate visible scene beams', sceneLayout, 'accumulateFS')
  const integrate = beamPipeline('Integrate receiver texels', planeLayout, 'integrateFS')
  const composite = device.createRenderPipeline({
    label: 'Scene composite', layout: 'auto', vertex: { module, entryPoint: 'fullscreenVS' },
    fragment: { module, entryPoint: 'compositeFS', targets: [{ format: canvasFormat }] },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'always' },
  })
  const uniformEntry = { binding: 0, resource: { buffer: uniform } }
  const integrateGroup = device.createBindGroup({ layout: planeLayout, entries: [uniformEntry] })
  function texture(label: string, format: GPUTextureFormat, width: number, height: number) {
    return device.createTexture({
      label, format, size: [width, height],
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })
  }
  function screenTargets(width: number, height: number) {
    const albedo = texture('Scene albedo and occupancy', 'rgba8unorm', width, height)
    const normal = texture('Optical-local normal and floor mask', 'rgba8unorm', width, height)
    const depth = texture('Scene depth', 'depth24plus', width, height)
    const light = texture('Beam irradiance and candidate count', 'rg16float', width, height)
    const albedoView = albedo.createView(), normalView = normal.createView()
    const depthView = depth.createView(), lightView = light.createView()
    const normalEntry = { binding: 2, resource: normalView }
    const beamGroup = device.createBindGroup({
      layout: sceneLayout, entries: [uniformEntry, { binding: 1, resource: depthView }, normalEntry],
    })
    const compositeGroup = device.createBindGroup({
      layout: composite.getBindGroupLayout(0),
      entries: [uniformEntry, normalEntry, { binding: 3, resource: albedoView }, { binding: 4, resource: lightView }],
    })
    return {
      width, height, albedoView, normalView, depthView, lightView, beamGroup, compositeGroup,
      destroy() { for (const t of [albedo, normal, depth, light]) t.destroy() },
    }
  }

  function planeTargets(image: CausticImage) {
    const { width, height, kind } = image
    const light = texture('Integrated receiver irradiance', 'rg16float', width, height)
    const view = light.createView()
    const sampled = frameLayout([{ binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: {} }])
    const imageLayout = device.createBindGroupLayout({ entries: [kind === 'buffer'
      ? { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }
      : { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
    ] })
    const display = device.createRenderPipeline({
      label: 'Plane irradiance comparison',
      layout: device.createPipelineLayout({ bindGroupLayouts: [sampled, imageLayout] }),
      vertex: { module, entryPoint: 'fullscreenVS' },
      fragment: { module, entryPoint: kind === 'buffer' ? 'compareBufferFS' : 'compareTextureFS',
        targets: [{ format: canvasFormat }] },
    })
    const group = device.createBindGroup({ layout: sampled, entries: [uniformEntry, { binding: 4, resource: view }] })
    const references = new WeakMap<GPUBuffer | GPUTextureView, GPUBindGroup>()
    function reference(input: CausticImage) {
      const key = input.kind === 'buffer' ? input.buffer : input.view
      let bound = references.get(key)
      if (!bound) {
        const entry = input.kind === 'buffer'
          ? { binding: 2, resource: { buffer: input.buffer } }
          : { binding: 3, resource: input.view }
        bound = device.createBindGroup({ layout: imageLayout, entries: [entry] })
        references.set(key, bound)
      }
      return bound
    }
    return { width, height, kind, view, display, group, reference, destroy() { light.destroy() } }
  }

  let screen: ReturnType<typeof screenTargets> | undefined
  let plane: ReturnType<typeof planeTargets> | undefined
  let boundField: { heights: GPUBuffer; beams: GPUBuffer; group: GPUBindGroup } | undefined

  return {
    render(
      encoder: GPUCommandEncoder, output: GPUTexture, viewProjection: Mat4,
      source: BeamSource, mode: BeamView, result: ForwardResult | null,
      drawScene: (pass: GPURenderPassEncoder) => void, drawLens: (pass: GPURenderPassEncoder) => void,
    ) {
      const compare = mode === 'plane' || mode === 'difference'
      if ((compare || source === 'solver') && !result) throw new Error('Missing forward result for beam rendering.')
      if (compare && source !== 'solver') throw new Error('Plane comparison requires solver beams.')
      const field = source === 'solver' ? result!.beams : tests.field
      const r = result?.receiver ?? receiver
      const lens = field.lens
      const count = source === 'single' ? 1 : 2 * (lens.columns - 1) * (lens.rows - 1)
      if (!boundField || boundField.heights !== lens.heightBuffer || boundField.beams !== field.buffer) {
        boundField = {
          heights: lens.heightBuffer, beams: field.buffer,
          group: device.createBindGroup({ layout: fieldLayout, entries: [
            { binding: 0, resource: { buffer: lens.heightBuffer } }, { binding: 1, resource: { buffer: field.buffer } },
          ] }),
        }
      }
      mat4.multiply(viewProjection, opticalToWorld, localVP)
      mat4.inverse(localVP, inverseLocalVP)
      // WGSL Frame offsets in 32-bit words, after the two matrices.
      words.set([lens.columns, lens.rows], 32)
      data.set([lens.sizeX, lens.sizeY], 34)
      data.set(source === 'solver' ? [0, 0] : tests.offset, 36)
      data.set([r.sizeX, r.sizeY], 38)
      words.set([r.width, r.height], 40)
      data.set([output.width, output.height], 42)
      data[44] = r.z
      words[45] = modeId[mode]
      data[46] = exposure
      device.queue.writeBuffer(uniform, 0, data)

      if (compare) {
        const image = result!.caustic
        if (!plane || plane.width !== image.width || plane.height !== image.height || plane.kind !== image.kind) {
          plane?.destroy()
          plane = planeTargets(image)
        }
        const pass = encoder.beginRenderPass({ label: 'Receiver pixel integration', colorAttachments: [clear(plane.view)] })
        pass.setPipeline(integrate)
        pass.setBindGroup(0, integrateGroup)
        pass.setBindGroup(1, boundField.group)
        pass.draw(4, count)
        pass.end()

        const show = encoder.beginRenderPass({ label: 'Plane comparison', colorAttachments: [clear(output.createView())] })
        show.setPipeline(plane.display)
        show.setBindGroup(0, plane.group)
        show.setBindGroup(1, plane.reference(image))
        show.draw(3)
        show.end()
        return
      }

      if (!screen || screen.width !== output.width || screen.height !== output.height) {
        screen?.destroy()
        screen = screenTargets(output.width, output.height)
      }
      const geometry = encoder.beginRenderPass({
        label: 'Scene G-buffer', colorAttachments: [clear(screen.albedoView), clear(screen.normalView)],
        depthStencilAttachment: { view: screen.depthView, depthLoadOp: 'clear', depthClearValue: 1, depthStoreOp: 'store' },
      })
      drawScene(geometry)
      geometry.end()

      const pass = encoder.beginRenderPass({ label: 'Accumulate beams', colorAttachments: [clear(screen.lightView)] })
      pass.setPipeline(accumulate)
      pass.setBindGroup(0, screen.beamGroup)
      pass.setBindGroup(1, boundField.group)
      pass.draw(4, count)
      pass.end()

      const show = encoder.beginRenderPass({
        label: 'Display scene and lens', colorAttachments: [clear(output.createView())],
        depthStencilAttachment: { view: screen.depthView, depthLoadOp: 'load', depthStoreOp: 'discard' },
      })
      show.setPipeline(composite)
      show.setBindGroup(0, screen.compositeGroup)
      show.draw(3)
      if (mode === 'lit') drawLens(show)
      show.end()
    },
    destroy() {
      screen?.destroy()
      plane?.destroy()
      tests.destroy()
      uniform.destroy()
    },
  }
}
