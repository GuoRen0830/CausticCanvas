import { mat4 } from 'wgpu-matrix'
import type { Mat4 } from 'wgpu-matrix'
import { config } from '../config'
import type { ForwardResult } from '../contracts'
import { sceneAlbedoFormat, type SceneGeometry } from './scene-mesh'
import { createRoomLightPaths } from './room-light-paths'
import apertureUV from './shaders/aperture-uv.wgsl?raw'
import commonShader from './shaders/beam-common.wgsl?raw'
import coverageShader from './shaders/pixel-coverage.wgsl?raw'
import hitTypes from './shaders/scene-hit.wgsl?raw'
import splatShader from './shaders/hit-splat.wgsl?raw'
import beamShader from './shaders/beam.wgsl?raw'
import roomCompositeShader from './shaders/room-composite.wgsl?raw'

const modeId = { lit: 0, candidates: 1, irradiance: 2, normals: 3, albedo: 4 }
export type BeamView = keyof typeof modeId

const clear = (view: GPUTextureView): GPURenderPassColorAttachment => ({
  view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0],
})

export function createBeamRenderer(
  device: GPUDevice, canvasFormat: GPUTextureFormat,
  opticalToWorld: Mat4, initialExposure: number, { mesh, materials }: SceneGeometry,
  relativeIOR = config.optics.lensIOR / config.optics.outsideIOR,
) {
  let exposure = initialExposure
  let environmentLight = 0, incidentIrradiance = 1
  let lightPathsVisible = true
  let colorFilter: GPUTextureView | undefined
  let filterRevision = 0
  const data = new Float32Array(44)
  const words = new Uint32Array(data.buffer)
  const localVP = data.subarray(0, 16)
  const inverseLocalVP = data.subarray(16, 32)
  const uniform = device.createBuffer({
    label: 'Beam frame', size: data.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  const module = device.createShaderModule({
    label: 'Scene beam rendering', code: [apertureUV, commonShader, coverageShader, beamShader, roomCompositeShader, hitTypes, splatShader].join('\n'),
  })
  // WebGPU still requires the binding when the white-light path skips the lookup.
  const emptyFilter = device.createTexture({
    label: 'Unused aperture filter', size: [1, 1], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING,
  })
  const emptyFilterView = emptyFilter.createView()
  const fieldLayout = device.createBindGroupLayout({
    entries: [
      ...[0, 1, 2].map((binding): GPUBindGroupLayoutEntry => ({
        binding, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' },
      })),
      { binding: 5, visibility: GPUShaderStage.VERTEX, texture: { sampleType: 'unfilterable-float' } },
    ],
  })
  function frameLayout(entries: GPUBindGroupLayoutEntry[]) {
    return device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ...entries,
    ] })
  }
  const splatFrame = frameLayout([
    { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
    { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} },
    { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'uint' } },
  ])
  const hitLayout = device.createBindGroupLayout({ entries: [0, 3].map(binding => ({
    binding, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' },
  })) })
  let hitGroup: GPUBindGroup | undefined
  let previousHits: ForwardResult['firstHits']
  const add: GPUBlendComponent = { srcFactor: 'one', dstFactor: 'one', operation: 'add' }
  // Dense subpixel footprints can contribute hundreds of small terms per pixel.
  const screenFormat = device.features.has('float32-blendable') ? 'rgba32float' : 'rgba16float'
  const splat = device.createRenderPipeline({
    label: 'Local first-hit footprints',
    layout: device.createPipelineLayout({ bindGroupLayouts: [splatFrame, fieldLayout, hitLayout] }),
    vertex: { module, entryPoint: 'splatVS' },
    fragment: { module, entryPoint: 'splatFS', targets: [{ format: screenFormat, blend: { color: add, alpha: add } }] },
    primitive: { topology: 'triangle-strip' },
  })
  const compositeLayout = frameLayout([
    { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
    { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} },
    { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: {} },
    { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
  ])
  const pathLayout = device.createBindGroupLayout({
    entries: [3, 4].map(binding => ({ binding, visibility: GPUShaderStage.FRAGMENT, texture: {} })),
  })
  function compositePipeline(entryPoint: string, layouts: GPUBindGroupLayout[]) {
    return device.createRenderPipeline({
      label: entryPoint, layout: device.createPipelineLayout({ bindGroupLayouts: layouts }),
      vertex: { module, entryPoint: 'fullscreenVS' },
      fragment: { module, entryPoint, targets: [{ format: canvasFormat }], constants: {
        materialLighting: Number(!!materials?.size),
        encodeCanvas: Number(canvasFormat === 'rgba8unorm' || canvasFormat === 'bgra8unorm'),
      } },
    })
  }
  const composite = compositePipeline('compositeFS', [compositeLayout])
  const roomComposite = compositePipeline('roomPathsFS', [compositeLayout, pathLayout])
  let paths: ReturnType<typeof createRoomLightPaths> | undefined
  const uniformEntry = { binding: 0, resource: { buffer: uniform } }
  function texture(label: string, format: GPUTextureFormat, width: number, height: number) {
    return device.createTexture({
      label, format, size: [width, height],
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })
  }
  function screenTargets(width: number, height: number) {
    const albedo = texture('Scene albedo and occupancy', sceneAlbedoFormat(materials), width, height)
    const normal = texture('Optical-local normal and floor mask', 'rgba8unorm', width, height)
    const object = texture('Visible object and geometric normal', 'rg32uint', width, height)
    const depth = texture('Scene depth', 'depth24plus', width, height)
    const light = texture('Beam irradiance and candidate count', screenFormat, width, height)
    const albedoView = albedo.createView(), normalView = normal.createView()
    const depthView = depth.createView(), lightView = light.createView()
    const objectView = object.createView()
    const normalEntry = { binding: 2, resource: normalView }
    const splatGroup = device.createBindGroup({ layout: splatFrame, entries: [
      uniformEntry, { binding: 1, resource: depthView }, normalEntry, { binding: 5, resource: objectView },
    ] })
    const compositeGroup = device.createBindGroup({
      layout: composite.getBindGroupLayout(0),
      entries: [uniformEntry, { binding: 1, resource: depthView }, normalEntry,
        { binding: 3, resource: albedoView }, { binding: 4, resource: lightView }],
    })
    return {
      width, height, albedoView, normalView, depthView, lightView, objectView, splatGroup, compositeGroup,
      destroy() { for (const t of [albedo, normal, object, depth, light]) t.destroy() },
    }
  }

  let screen: ReturnType<typeof screenTargets> | undefined
  function screenFor(output: GPUTexture) {
    if (!screen || screen.width !== output.width || screen.height !== output.height) {
      screen?.destroy()
      screen = screenTargets(output.width, output.height)
    }
    return screen
  }
  let boundField: { heights: GPUBuffer; beams: GPUBuffer; offsets: GPUBuffer; group: GPUBindGroup } | undefined
  // Lenses without vertex offsets read zeros (one vec2 per vertex).
  let zeroOffsets: GPUBuffer | undefined
  const offsetsFor = (lens: ForwardResult['beams']['lens']) => {
    if (lens.offsetBuffer) return lens.offsetBuffer
    const size = lens.columns * lens.rows * 8
    if (!zeroOffsets || zeroOffsets.size !== size) {
      zeroOffsets?.destroy()
      zeroOffsets = device.createBuffer({ label: 'Zero lens offsets', size, usage: GPUBufferUsage.STORAGE })
    }
    return zeroOffsets
  }

  return {
    // Display settings never modify the shared optical buffers.
    setExposure(value: number) {
      exposure = value
    },
    setEnvironmentLight(value: number) {
      if (!Number.isFinite(value) || value < 0) throw new Error('Environment light must be non-negative.')
      environmentLight = value
    },
    setReferenceIrradiance(value: number) {
      incidentIrradiance = value
    },
    setLightPathsVisible(value: boolean) {
      lightPathsVisible = value
    },
    setColorFilter(view: GPUTextureView | undefined, revision = 0) {
      if (colorFilter !== view) boundField = undefined
      colorFilter = view
      filterRevision = revision
    },

    render(
      encoder: GPUCommandEncoder, output: GPUTexture, viewProjection: Mat4,
      mode: BeamView, result: ForwardResult,
      drawScene: (pass: GPURenderPassEncoder) => void,
      training = false,
    ) {
      const { beams: field } = result
      const lens = field.lens
      const count = 2 * (lens.columns - 1) * (lens.rows - 1)
      const offsets = offsetsFor(lens)
      if (!boundField || boundField.heights !== lens.heightBuffer || boundField.beams !== field.buffer ||
          boundField.offsets !== offsets) {
        boundField = {
          heights: lens.heightBuffer, beams: field.buffer, offsets,
          group: device.createBindGroup({ layout: fieldLayout, entries: [
            { binding: 0, resource: { buffer: lens.heightBuffer } }, { binding: 1, resource: { buffer: field.buffer } },
            { binding: 2, resource: { buffer: offsets } },
            { binding: 5, resource: colorFilter ?? emptyFilterView },
          ] }),
        }
      }
      mat4.multiply(viewProjection, opticalToWorld, localVP)
      mat4.inverse(localVP, inverseLocalVP)
      // WGSL Frame offsets in 32-bit words, after the two matrices.
      words.set([lens.columns, lens.rows], 32)
      data.set([lens.sizeX, lens.sizeY], 34)
      data[36] = environmentLight
      data[37] = Math.max(incidentIrradiance, 1e-6)
      data.set([output.width, output.height], 38)
      words[40] = modeId[mode]
      data[41] = exposure
      words[42] = Number(!!colorFilter)
      device.queue.writeBuffer(uniform, 0, data)

      const targets = screenFor(output)
      const geometry = encoder.beginRenderPass({
        label: 'Scene G-buffer', colorAttachments: [clear(targets.albedoView), clear(targets.normalView), clear(targets.objectView)],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'clear', depthClearValue: 1, depthStoreOp: 'store' },
      })
      drawScene(geometry)
      geometry.end()

      const hits = result.firstHits
      if (!hits || hits.mesh !== mesh) throw new Error('Expected first hits for the current scene.')
      if (!hitGroup || previousHits !== hits) {
        hitGroup = device.createBindGroup({ layout: hitLayout, entries: [
          { binding: 0, resource: { buffer: hits.triangles } }, { binding: 3, resource: { buffer: hits.buffer } },
        ] })
        previousHits = hits
      }
      // G-buffer views do not need to rasterize millions of beam footprints.
      if (mode !== 'normals' && mode !== 'albedo') {
        const pass = encoder.beginRenderPass({ label: 'Splat first hits', colorAttachments: [clear(targets.lightView)] })
        pass.setPipeline(splat)
        pass.setBindGroup(0, targets.splatGroup)
        pass.setBindGroup(1, boundField.group)
        pass.setBindGroup(2, hitGroup)
        pass.draw(4, count)
        pass.end()
      }

      let pathGroup: GPUBindGroup | undefined
      if (mode === 'lit' && lightPathsVisible) {
        paths ??= createRoomLightPaths(device, pathLayout)
        pathGroup = paths.prepare(encoder, output, targets.depthView, localVP, inverseLocalVP,
          result, offsets, incidentIrradiance, relativeIOR, training, colorFilter, filterRevision)
      }
      const show = encoder.beginRenderPass({
        label: 'Display beam lighting', colorAttachments: [clear(output.createView())],
      })
      show.setPipeline(pathGroup ? roomComposite : composite)
      show.setBindGroup(0, targets.compositeGroup)
      if (pathGroup) show.setBindGroup(1, pathGroup)
      show.draw(3)
      show.end()
    },
    destroy() {
      paths?.destroy()
      screen?.destroy()
      uniform.destroy()
      emptyFilter.destroy()
      zeroOffsets?.destroy()
    },
  }
}
