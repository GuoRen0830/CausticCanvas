import { mat4, type Mat4 } from 'wgpu-matrix'
import type { ForwardResult } from '../contracts'
import { createBeamFrame } from './beam-frame'
import { createExportTransport } from './export-transport'
import { createRoomLightPaths } from './room-light-paths'
import { sceneAlbedoFormat, type SceneGeometry } from './scene-mesh'
import type { createSceneRenderer } from './scene-renderer'
import frameShader from './shaders/beam-common.wgsl?raw'
import roomRadiance from './shaders/room-radiance.wgsl?raw'
import exportResolve from './shaders/export-resolve.wgsl?raw'

interface ExportRoom {
  geometry: SceneGeometry
  opticalToWorld: Mat4
  result: ForwardResult
  scene: Pick<Awaited<ReturnType<typeof createSceneRenderer>>, 'draw' | 'update'>
  exposure: number
  environmentLight: number
  incidentIrradiance: number
  relativeIOR: number
  colorFilter?: GPUTextureView
  lightPathsVisible: boolean
  materialsEnabled?: boolean
}

interface ExportRequest {
  width: number
  height: number
  viewProjection: Mat4
  signal: AbortSignal
  onProgress?: (fraction: number, stage?: string) => void
  tileSize?: number
}

function cleared(view: GPUTextureView): GPURenderPassColorAttachment {
  return { view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }
}

export async function renderRoomExport(device: GPUDevice, room: ExportRoom, request: ExportRequest): Promise<ImageData> {
  const { width, height, signal, onProgress } = request
  const baseColorOnly = room.materialsEnabled === false
  signal.throwIfAborted()
  const edge = request.tileSize ?? 768
  if (![width, height, edge].every(v => Number.isSafeInteger(v) && v > 0)) {
    throw new Error('Export dimensions must be positive integers.')
  }
  const { lens } = room.result.beams
  const hits = room.result.firstHits
  if (!hits || hits.mesh !== room.geometry.mesh || !hits.nodes) {
    throw new Error('Export requires ray-tracing geometry for the current Room.')
  }
  // GPU allocations depend on just one tile, including for portrait and 4K images.
  const maxEdge = Math.min(768, edge, Math.floor(device.limits.maxTextureDimension2D / 2))
  // Distribute remainder pixels across tiles. Only the last row/column can
  // contain padding, even when the image is just one pixel wider than a tile.
  const tw = Math.ceil(width / Math.ceil(width / maxEdge))
  const th = Math.ceil(height / Math.ceil(height / maxEdge))
  const highWidth = 2 * tw, highHeight = 2 * th
  const pitch = Math.ceil(tw * 4 / 256) * 256
  const image = new ImageData(width, height)
  const cameraVP = mat4.copy(request.viewProjection)
  const resources: { destroy(): void }[] = []
  function own<T extends { destroy(): void }>(item: T): T {
    resources.push(item)
    return item
  }
  function buffer(label: string, size: number, usage: GPUBufferUsageFlags) {
    return own(device.createBuffer({ label, size, usage }))
  }
  function target(label: string, format: GPUTextureFormat) {
    return own(device.createTexture({ label, size: [highWidth, highHeight], format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING }))
  }

  // The session holds the evaluated lens and material bindings for this job.
  // Surface, transport, volume and readback resources below belong to the export.
  device.pushErrorScope('out-of-memory')
  device.pushErrorScope('validation')
  async function checkGpu(continueRendering: boolean) {
    const checks = [device.popErrorScope(), device.popErrorScope()]
    if (continueRendering) {
      device.pushErrorScope('out-of-memory')
      device.pushErrorScope('validation')
    }
    const [invalid, outOfMemory] = await Promise.all(checks)
    const error = invalid ?? outOfMemory
    if (error && !signal.aborted) throw new Error(error.message)
  }
  try {
    onProgress?.(0, 'Preparing light transport')
    const transport = own(await createExportTransport(device, room.result.beams, hits.triangles, hits.nodes,
      room.colorFilter, signal, fraction => onProgress?.(0.2 * fraction, 'Preparing light transport')))
    signal.throwIfAborted()
    const materialLighting = !!room.geometry.materials?.size
    const frame = own(createBeamFrame(device, room.opticalToWorld, 'Export beam frame'))
    const tile = buffer('Export image coordinates', 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)
    const readback = buffer('Export tile readback', pitch * th, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ)
    const color = target('Export albedo and occupancy', sceneAlbedoFormat(room.geometry.materials))
    const targets = {
      width: highWidth, height: highHeight,
      albedo: color.createView(),
      normal: target('Export shading normal', 'rgba8unorm').createView(),
      object: target('Export surface identity', 'rg32uint').createView(),
      material: target('Export materials', 'rg32uint').createView(),
      depth: target('Export camera depth', 'depth24plus').createView(),
      direct: target('Export direct radiance', transport.format).createView(),
    }
    const resolved = own(device.createTexture({ label: 'Resolved sRGB export tile', size: [tw, th],
      format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC }))
    const resolvedView = resolved.createView()
    const module = device.createShaderModule({ label: 'Resolve export samples',
      code: [frameShader, roomRadiance, exportResolve].join('\n') })
    const visibility = GPUShaderStage.FRAGMENT
    const layout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility, buffer: {} }, { binding: 7, visibility, buffer: {} },
      { binding: 2, visibility, texture: {} }, { binding: 3, visibility, texture: {} },
      { binding: 4, visibility, texture: { sampleType: 'unfilterable-float' } },
      { binding: 6, visibility, texture: { sampleType: 'uint' } },
    ] })
    const pathLayout = device.createBindGroupLayout({ entries: [3, 4].map(binding => ({
      binding, visibility, texture: {},
    })) })
    const pipeline = await device.createRenderPipelineAsync({
      label: 'Resolve four HDR samples to sRGB', layout: device.createPipelineLayout({ bindGroupLayouts: [layout, pathLayout] }),
      vertex: { module, entryPoint: 'fullscreenVS' },
      fragment: { module, entryPoint: 'resolveExport', targets: [{ format: 'rgba8unorm' }], constants: {
        materialLighting: Number(materialLighting), encodeCanvas: 1, showPaths: Number(room.lightPathsVisible),
      } },
    })
    const group = device.createBindGroup({ layout, entries: [
      { binding: 0, resource: { buffer: frame.uniform } }, { binding: 7, resource: { buffer: tile } },
      { binding: 2, resource: targets.normal }, { binding: 3, resource: targets.albedo },
      { binding: 4, resource: targets.direct }, { binding: 6, resource: targets.material },
    ] })
    const paths = room.lightPathsVisible ? own(createRoomLightPaths(device, pathLayout, true)) : undefined
    // WebGPU requires bindings even when the no-paths specialization skips both reads.
    const noPaths = paths ? undefined : device.createBindGroup({ layout: pathLayout, entries: [
      { binding: 3, resource: targets.normal }, { binding: 4, resource: targets.normal },
    ] })
    await checkGpu(true)
    const clip = mat4.identity(), tileVP = mat4.identity()
    clip[0] = width / tw
    clip[5] = height / th
    const tileCount = Math.ceil(width / tw) * Math.ceil(height / th)
    let completed = 0
    for (let y = 0; y < height; y += th) {
      for (let x = 0; x < width; x += tw) {
        signal.throwIfAborted()
        const stage = `Rendering tile ${completed + 1} of ${tileCount}`
        onProgress?.(0.2 + 0.8 * completed / tileCount, stage)
        // Remap this rectangle of the complete image to clip space. A padded
        // edge tile has the same projection scale and texel footprint as its neighbours.
        clip[12] = (width - tw - 2 * x) / tw
        clip[13] = (2 * y + th - height) / th
        mat4.multiply(clip, cameraVP, tileVP)
        frame.update(tileVP, lens, highWidth, highHeight, 0, room)
        device.queue.writeBuffer(tile, 0, Float32Array.of(x, y, width, height))
        const geometry = device.createCommandEncoder({ label: 'Export tile geometry' })
        const gbuffer = geometry.beginRenderPass({
          colorAttachments: [targets.albedo, targets.normal, targets.object, targets.material].map(cleared),
          depthStencilAttachment: { view: targets.depth, depthLoadOp: 'clear', depthClearValue: 1, depthStoreOp: 'store' },
        })
        room.scene.draw(gbuffer, baseColorOnly ? 'base-color' : 'material', tileVP)
        gbuffer.end()
        device.queue.submit([geometry.finish()])
        await transport.render(frame.uniform, targets, frame.localVP, materialLighting, signal,
          fraction => onProgress?.(0.2 + 0.8 * (completed + fraction * 0.9) / tileCount, stage))
        signal.throwIfAborted()
        const resolve = device.createCommandEncoder({ label: 'Export tile resolve' })
        const pathGroup = paths ? paths.prepare(resolve, color, targets.depth, frame.localVP, frame.inverseLocalVP,
          room.result, transport.offsets, room.incidentIrradiance, room.relativeIOR, false, room.colorFilter) : noPaths!
        const pass = resolve.beginRenderPass({ colorAttachments: [cleared(resolvedView)] })
        pass.setPipeline(pipeline)
        pass.setBindGroup(0, group)
        pass.setBindGroup(1, pathGroup)
        pass.draw(3)
        pass.end()
        const validWidth = Math.min(tw, width - x), validHeight = Math.min(th, height - y)
        resolve.copyTextureToBuffer({ texture: resolved }, { buffer: readback, bytesPerRow: pitch }, [validWidth, validHeight])
        device.queue.submit([resolve.finish()])
        await readback.mapAsync(GPUMapMode.READ)
        try {
          signal.throwIfAborted()
          const pixels = new Uint8Array(readback.getMappedRange())
          for (let row = 0; row < validHeight; row++) {
            image.data.set(pixels.subarray(row * pitch, row * pitch + validWidth * 4), ((y + row) * width + x) * 4)
          }
        } finally {
          readback.unmap()
        }
        await checkGpu(true)
        completed++
        onProgress?.(0.2 + 0.8 * completed / tileCount, stage)
      }
    }
    signal.throwIfAborted()
    return image
  } finally {
    for (const resource of resources.reverse()) resource.destroy()
    await checkGpu(false)
  }
}
