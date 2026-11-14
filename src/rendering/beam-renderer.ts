import { mat4 } from 'wgpu-matrix'
import type { Mat4 } from 'wgpu-matrix'
import type { CausticImage, ForwardResult, ReceiverPlane } from '../contracts'
import { buildSceneMesh } from './scene-mesh'
import type { SceneBox } from './scene-mesh'
import { createHitTracer } from './hit-tracer'
import commonShader from './shaders/beam-common.wgsl?raw'
import coverageShader from './shaders/pixel-coverage.wgsl?raw'
import bvhShader from './shaders/bvh-query.wgsl?raw'
import traceShader from './shaders/first-hit.wgsl?raw'
import splatShader from './shaders/hit-splat.wgsl?raw'
import { createTestBeams } from './test-beams'
import beamShader from './shaders/beam.wgsl?raw'
import receiverShader from './shaders/receiver-compare.wgsl?raw'
import displayColor from './shaders/irradiance-display.wgsl?raw'

export type BeamSource = 'solver' | 'single' | 'pair'
export type SceneMethod = 'first-hit' | 'analytic'
export type BeamView = 'lit' | 'candidates' | 'footprint' | 'plane' | 'difference' | 'blocked' | 'lens'
const modeId = { lit: 0, candidates: 1, footprint: 2, plane: 3, difference: 4, blocked: 5 }

const clear = (view: GPUTextureView): GPURenderPassColorAttachment => ({
  view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0],
})

export function createBeamRenderer(
  device: GPUDevice, canvasFormat: GPUTextureFormat,
  aperture: { sizeX: number; sizeY: number; thickness: number },
  receiver: ReceiverPlane, opticalToWorld: Mat4, exposure: number, box: SceneBox,
  mesh = buildSceneMesh(receiver, box),
) {
  const tests = createTestBeams(device, aperture)
  const data = new Float32Array(56)
  const words = new Uint32Array(data.buffer)
  const localVP = data.subarray(0, 16)
  const inverseLocalVP = data.subarray(16, 32)
  // Shared scene geometry, in optical-local meters. Each uniform vec3 starts
  // on a 16-byte boundary.
  data.set(box.center.map((c, axis) => c - 0.5 * box.size[axis]), 48)
  data.set(box.center.map((c, axis) => c + 0.5 * box.size[axis]), 52)
  const uniform = device.createBuffer({
    label: 'Beam frame', size: data.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  const module = device.createShaderModule({
    label: 'Beam rendering and plane comparison', code: [displayColor, commonShader, coverageShader, beamShader, receiverShader, bvhShader, traceShader, splatShader].join('\n'),
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
  const splatFrame = frameLayout([
    { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
    { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} },
    { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'uint' } },
  ])
  const tracer = createHitTracer(device, module, uniform, mesh)
  const add: GPUBlendComponent = { srcFactor: 'one', dstFactor: 'one', operation: 'add' }
  // Dense subpixel footprints can contribute hundreds of small terms per pixel.
  const screenFormat = device.features.has('float32-blendable') ? 'rg32float' : 'rg16float'
  const accumulate = device.createRenderPipeline({
    label: 'Accumulate visible scene beams', layout: device.createPipelineLayout({ bindGroupLayouts: [sceneLayout, fieldLayout] }),
    vertex: { module, entryPoint: 'boundsVS' },
    fragment: { module, entryPoint: 'accumulateFS', targets: [{ format: screenFormat, blend: { color: add, alpha: add } }] },
    primitive: { topology: 'triangle-strip' },
  })
  const splat = device.createRenderPipeline({
    label: 'Local first-hit footprints',
    layout: device.createPipelineLayout({ bindGroupLayouts: [splatFrame, fieldLayout, tracer.splatLayout] }),
    vertex: { module, entryPoint: 'splatVS' },
    fragment: { module, entryPoint: 'splatFS', targets: [{ format: screenFormat, blend: { color: add, alpha: add } }] },
    primitive: { topology: 'triangle-strip' },
  })
  const compositeLayout = frameLayout([
    { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: {} },
    { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
  ])
  const composite = device.createRenderPipeline({
    label: 'Scene composite', layout: device.createPipelineLayout({ bindGroupLayouts: [compositeLayout] }), vertex: { module, entryPoint: 'fullscreenVS' },
    fragment: { module, entryPoint: 'compositeFS', targets: [{ format: canvasFormat }] },
  })
  const uniformEntry = { binding: 0, resource: { buffer: uniform } }
  function texture(label: string, format: GPUTextureFormat, width: number, height: number) {
    return device.createTexture({
      label, format, size: [width, height],
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })
  }
  function screenTargets(width: number, height: number) {
    const albedo = texture('Scene albedo and occupancy', 'rgba8unorm', width, height)
    const normal = texture('Optical-local normal and floor mask', 'rgba8unorm', width, height)
    const object = texture('Visible object and geometric normal', 'rg32uint', width, height)
    const depth = texture('Scene depth', 'depth24plus', width, height)
    const light = texture('Beam irradiance and candidate count', screenFormat, width, height)
    const albedoView = albedo.createView(), normalView = normal.createView()
    const depthView = depth.createView(), lightView = light.createView()
    const objectView = object.createView()
    const normalEntry = { binding: 2, resource: normalView }
    const beamGroup = device.createBindGroup({
      layout: sceneLayout, entries: [uniformEntry, { binding: 1, resource: depthView }, normalEntry],
    })
    const splatGroup = device.createBindGroup({ layout: splatFrame, entries: [
      uniformEntry, { binding: 1, resource: depthView }, normalEntry, { binding: 5, resource: objectView },
    ] })
    const compositeGroup = device.createBindGroup({
      layout: composite.getBindGroupLayout(0),
      entries: [uniformEntry, { binding: 3, resource: albedoView }, { binding: 4, resource: lightView }],
    })
    return {
      width, height, albedoView, normalView, depthView, lightView, objectView, beamGroup, splatGroup, compositeGroup,
      destroy() { for (const t of [albedo, normal, object, depth, light]) t.destroy() },
    }
  }

  function planeTargets(image: CausticImage) {
    const { width, height, kind } = image
    // Blend in f32. Devices without float blending sum positive contributions
    // with f32 compare/exchange, then copy the bits into the same display texture.
    const nativeBlend = device.features.has('float32-blendable')
    const rowWords = Math.ceil(width / 64) * 64 // copyBufferToTexture requires aligned rows.
    const light = device.createTexture({
      label: 'Integrated receiver irradiance', format: 'r32float', size: [width, height],
      usage: GPUTextureUsage.TEXTURE_BINDING | (nativeBlend ? GPUTextureUsage.RENDER_ATTACHMENT : GPUTextureUsage.COPY_DST),
    })
    const view = light.createView()
    const sampled = frameLayout([{ binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } }])
    const sums = nativeBlend ? null : device.createBuffer({
      label: 'Receiver f32 sums', size: rowWords * height * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    })
    const integrate = nativeBlend
      ? device.createRenderPipeline({ label: 'Integrate receiver texels', layout: 'auto',
        vertex: { module, entryPoint: 'boundsVS' },
        fragment: { module, entryPoint: 'integrateFS', targets: [{ format: 'r32float', blend: { color: add, alpha: add } }] },
        primitive: { topology: 'triangle-strip' } })
      : device.createComputePipeline({ label: 'Integrate receiver texels', layout: 'auto', compute: { module, entryPoint: 'integrateCS' } })
    const frameGroup = device.createBindGroup({ layout: integrate.getBindGroupLayout(0),
      entries: sums ? [uniformEntry, { binding: 6, resource: { buffer: sums } }] : [uniformEntry] })
    let planeField: { heights: GPUBuffer; beams: GPUBuffer; group: GPUBindGroup } | undefined
    let lastResult: ForwardResult | undefined
    function accumulate(encoder: GPUCommandEncoder, result: ForwardResult, count: number) {
      // Plane irradiance does not depend on the camera or the comparison view.
      if (lastResult === result) return
      const { buffer, lens } = result.beams
      if (planeField?.heights !== lens.heightBuffer || planeField.beams !== buffer) {
        planeField = { heights: lens.heightBuffer, beams: buffer,
          group: device.createBindGroup({ layout: integrate.getBindGroupLayout(1), entries: [
            { binding: 0, resource: { buffer: lens.heightBuffer } }, { binding: 1, resource: { buffer } },
          ] }) }
      }
      if (sums) {
        encoder.clearBuffer(sums)
        const pass = encoder.beginComputePass({ label: 'Receiver pixel integration' })
        pass.setPipeline(integrate as GPUComputePipeline)
        pass.setBindGroup(0, frameGroup)
        pass.setBindGroup(1, planeField.group)
        const groups = Math.ceil(count / 64)
        const x = Math.min(groups, device.limits.maxComputeWorkgroupsPerDimension)
        pass.dispatchWorkgroups(x, Math.ceil(groups / x))
        pass.end()
        encoder.copyBufferToTexture({ buffer: sums, bytesPerRow: rowWords * 4 }, { texture: light }, [width, height])
      } else {
        const pass = encoder.beginRenderPass({ label: 'Receiver pixel integration', colorAttachments: [clear(view)] })
        pass.setPipeline(integrate as GPURenderPipeline)
        pass.setBindGroup(0, frameGroup)
        pass.setBindGroup(1, planeField.group)
        pass.draw(4, count)
        pass.end()
      }
      lastResult = result
    }
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
    return { width, height, kind, accumulate, display, group, reference, destroy() { light.destroy(); sums?.destroy() } }
  }

  let screen: ReturnType<typeof screenTargets> | undefined
  function screenFor(output: GPUTexture) {
    if (!screen || screen.width !== output.width || screen.height !== output.height) {
      screen?.destroy()
      screen = screenTargets(output.width, output.height)
    }
    return screen
  }
  let plane: ReturnType<typeof planeTargets> | undefined
  let boundField: { heights: GPUBuffer; beams: GPUBuffer; group: GPUBindGroup } | undefined

  return {
    render(
      encoder: GPUCommandEncoder, output: GPUTexture, viewProjection: Mat4,
      source: BeamSource, mode: BeamView, result: ForwardResult | null,
      drawScene: (pass: GPURenderPassEncoder) => void, drawLens: (pass: GPURenderPassEncoder) => void,
      method: SceneMethod = 'first-hit',
    ) {
      const compare = mode === 'plane' || mode === 'difference'
      if (mode === 'lens') {
        const targets = screenFor(output)
        const pass = encoder.beginRenderPass({
          label: 'Inspect lens shape', colorAttachments: [{ view: output.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
          depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'clear', depthClearValue: 1, depthStoreOp: 'discard' },
        })
        drawLens(pass)
        pass.end()
        return
      }
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
      words[47] = count
      device.queue.writeBuffer(uniform, 0, data)

      if (compare) {
        const image = result!.caustic
        if (!plane || plane.width !== image.width || plane.height !== image.height || plane.kind !== image.kind) {
          plane?.destroy()
          plane = planeTargets(image)
        }
        plane.accumulate(encoder, result!, count)

        const show = encoder.beginRenderPass({ label: 'Plane comparison', colorAttachments: [clear(output.createView())] })
        show.setPipeline(plane.display)
        show.setBindGroup(0, plane.group)
        show.setBindGroup(1, plane.reference(image))
        show.draw(3)
        show.end()
        return
      }

      const targets = screenFor(output)
      const geometry = encoder.beginRenderPass({
        label: 'Scene G-buffer', colorAttachments: [clear(targets.albedoView), clear(targets.normalView), clear(targets.objectView)],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'clear', depthClearValue: 1, depthStoreOp: 'store' },
      })
      drawScene(geometry)
      geometry.end()

      const firstHit = method === 'first-hit' && mode !== 'blocked'
      const solverHits = source === 'solver' && result?.firstHits?.mesh === mesh ? result.firstHits : undefined
      const cached = !firstHit ? null : solverHits ? tracer.fromSolver(solverHits)
        : tracer.update(encoder, field, count, source === 'solver' ? result! : tests.field)
      const pass = encoder.beginRenderPass({
        label: cached ? 'Splat first hits' : 'Accumulate beams', colorAttachments: [clear(targets.lightView)],
      })
      pass.setPipeline(cached ? splat : accumulate)
      pass.setBindGroup(0, cached ? targets.splatGroup : targets.beamGroup)
      pass.setBindGroup(1, boundField.group)
      if (cached) pass.setBindGroup(2, cached.group)
      pass.draw(4, count)
      pass.end()

      const show = encoder.beginRenderPass({
        label: 'Display beam lighting', colorAttachments: [clear(output.createView())],
      })
      show.setPipeline(composite)
      show.setBindGroup(0, targets.compositeGroup)
      show.draw(3)
      show.end()
    },
    destroy() {
      screen?.destroy()
      plane?.destroy()
      tracer.destroy()
      tests.destroy()
      uniform.destroy()
    },
  }
}
