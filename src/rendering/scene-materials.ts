import type { SceneMaterial, SceneDraw, SceneMapSlot } from './scene-mesh'

type MapFormat = 'rgba8unorm-srgb' | 'rgba8unorm'

// Color maps filter in linear light via sRGB views. Material data maps filter raw channels.
function createMipGenerator(device: GPUDevice, format: MapFormat) {
  const module = device.createShaderModule({ code: `
    @group(0) @binding(0) var image: texture_2d<f32>;
    @group(0) @binding(1) var linearSampler: sampler;
    struct Vertex { @builtin(position) pos: vec4f, @location(0) uv: vec2f }
    @vertex fn vs(@builtin(vertex_index) i: u32) -> Vertex {
      let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
      return Vertex(vec4f(uv * vec2f(2, -2) + vec2f(-1, 1), 0, 1), uv);
    }
    @fragment fn fs(input: Vertex) -> @location(0) vec4f {
      return textureSampleLevel(image, linearSampler, input.uv, 0);
    }` })
  const pipeline = device.createRenderPipeline({ label: 'Material mipmaps', layout: 'auto',
    vertex: { module, entryPoint: 'vs' }, fragment: { module, entryPoint: 'fs', targets: [{ format }] } })
  const sampler = device.createSampler({ minFilter: 'linear', magFilter: 'linear' })
  return (texture: GPUTexture) => {
    const encoder = device.createCommandEncoder()
    let previous = texture.createView({ baseMipLevel: 0, mipLevelCount: 1 })
    for (let level = 1; level < texture.mipLevelCount; level++) {
      const view = texture.createView({ baseMipLevel: level, mipLevelCount: 1 })
      const pass = encoder.beginRenderPass({ colorAttachments: [{ view, loadOp: 'clear', storeOp: 'store' }] })
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: previous }, { binding: 1, resource: sampler },
      ] }))
      pass.draw(3)
      pass.end()
      previous = view
    }
    device.queue.submit([encoder.finish()])
  }
}

// Order matches the sampler pairs and UV transform array in scene.wgsl.
const slots = [
  ['map', 'uvTransform', 'clamp'],
  ['metallicRoughnessMap', 'metallicRoughnessUVTransform', 'metallicRoughnessClamp'],
  ['normalMap', 'normalUVTransform', 'normalClamp'],
  ['roughnessMap', 'roughnessUVTransform', 'roughnessClamp'],
  ['metallicMap', 'metallicUVTransform', 'metallicClamp'],
  ['occlusionMap', 'occlusionUVTransform', 'occlusionClamp'],
  ['emissiveMap', 'emissiveUVTransform', 'emissiveClamp'],
] as const

const isColor = (key: typeof slots[number][0]) => key === 'map' || key === 'emissiveMap'
const coordinates = (draw: SceneDraw) => draw.uvMask ?? (draw.hasUV ? 1 : 0)

export async function createMaterialBindings(device: GPUDevice, layout: GPUBindGroupLayout, draws: SceneDraw[],
  materials: ReadonlyMap<string, SceneMaterial>, warn: (message: string) => void, signal?: AbortSignal) {
  signal?.throwIfAborted()
  // Close each scope before yielding: a concurrent load/frame shares the device's
  // scope stack but must never consume an error from these replacement resources.
  const checks: Promise<string | undefined>[] = []
  function allocate<T>(label: string, make: () => T): T {
    if (!signal) return make()
    device.pushErrorScope('out-of-memory')
    device.pushErrorScope('validation')
    try { return make() }
    finally {
      const scopes = [device.popErrorScope(), device.popErrorScope()]
      checks.push(Promise.all(scopes).then(errors => {
        const error = errors[1] ?? errors[0] // Prefer the failed allocation over a dependent invalid view.
        return error ? label + ': ' + (error.message || error.constructor.name) : undefined
      }, reason => label + ': ' + String(reason)))
    }
  }
  const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
  const white = allocate('Material fallback texture', () => device.createTexture({
    label: 'Plain Base Color', size: [1, 1], format: 'rgba8unorm-srgb', usage,
  }))
  const colorMaps = new Map<File, GPUTexture>(), dataMaps = new Map<File, GPUTexture>()
  const mips = new Map<MapFormat, ReturnType<typeof createMipGenerator>>()
  const uniforms: GPUBuffer[] = []
  function destroy() {
    white.destroy()
    for (const cache of [colorMaps, dataMaps]) for (const texture of cache.values()) texture.destroy()
    for (const buffer of uniforms) buffer.destroy()
  }
  try {
    const whiteView = allocate('Material fallback texture', () => {
      device.queue.writeTexture({ texture: white }, new Uint8Array([255, 255, 255, 255]), {}, [1, 1])
      return white.createView()
    })
    async function upload(file: File, format: MapFormat, cache: Map<File, GPUTexture>) {
      signal?.throwIfAborted()
      let bitmap: ImageBitmap
      try { bitmap = await createImageBitmap(file, { premultiplyAlpha: 'none',
        colorSpaceConversion: format === 'rgba8unorm' ? 'none' : 'default' }) }
      catch { warn('Cannot decode ' + file.name + '. Using material factors.'); return }
      try {
        signal?.throwIfAborted()
        const { width, height } = bitmap
        if (Math.max(width, height) > device.limits.maxTextureDimension2D) {
          warn(file.name + ' exceeds the GPU texture size limit. Using material factors.')
          return
        }
        const mipLevelCount = 1 + Math.floor(Math.log2(Math.max(width, height)))
        allocate(file.name, () => {
          const texture = device.createTexture({ label: file.name, size: [width, height], format, usage, mipLevelCount })
          cache.set(file, texture)
          device.queue.copyExternalImageToTexture({ source: bitmap }, { texture, colorSpace: 'srgb' }, [width, height])
          if (mipLevelCount > 1) {
            let generateMips = mips.get(format)
            if (!generateMips) { generateMips = createMipGenerator(device, format); mips.set(format, generateMips) }
            generateMips(texture)
          }
        })
      } finally { bitmap.close() }
      if (signal) {
        const failure = await checks.at(-1)
        if (failure) throw new Error(failure)
      }
    }
    const colors = new Set<File>(), data = new Set<File>()
    for (const draw of draws) {
      const material = draw.material && materials.get(draw.material)
      if (!material) continue
      for (const [key] of slots) {
        const file = material[key]
        if (file && (coordinates(draw) & (1 << (material.sampling?.[key]?.uvSet ?? 0)))) {
          (isColor(key) ? colors : data).add(file)
        }
      }
    }
    for (const file of colors) await upload(file, 'rgba8unorm-srgb', colorMaps)
    for (const file of data) await upload(file, 'rgba8unorm', dataMaps)
    const samplers = new Map<string, GPUSampler>()
    function sampler(clamp: boolean, source?: GPUSamplerDescriptor) {
      const address = clamp ? 'clamp-to-edge' : 'repeat'
      const desc: GPUSamplerDescriptor = { addressModeU: address, addressModeV: address,
        minFilter: 'linear', magFilter: 'linear', mipmapFilter: 'linear', ...source }
      if (desc.maxAnisotropy === undefined && desc.minFilter === 'linear' && desc.magFilter === 'linear' &&
        desc.mipmapFilter === 'linear' && desc.lodMaxClamp !== 0) desc.maxAnisotropy = 8
      const key = JSON.stringify(desc)
      let s = samplers.get(key)
      if (!s) {
        s = device.createSampler(desc)
        samplers.set(key, s)
      }
      return s
    }
    const groupCaches = new Map<number, Map<string | undefined, { group: GPUBindGroup; mapMask: number }>>()
    const bindings = allocate('Scene material bindings', () => draws.map(draw => {
      const material = draw.material ? materials.get(draw.material) : undefined
      const mask = coordinates(draw)
      const drawFlags = mask | (draw.doubleSided ? 4 : 0)
      let cache = groupCaches.get(drawFlags)
      if (!cache) { cache = new Map(); groupCaches.set(drawFlags, cache) }
      const key = material ? draw.material : undefined
      let binding = cache.get(key)
      if (!binding) {
        const entries: GPUBindGroupEntry[] = [], transforms: number[] = []
        let enabled = 0
        for (const [i, [map, transform, clamp]] of slots.entries()) {
          const file = material?.[map]
          const sampling = material?.sampling?.[map], set = sampling?.uvSet ?? 0
          const texture = file && (mask & (1 << set)) ? (isColor(map) ? colorMaps : dataMaps).get(file) : undefined
          if (texture) enabled |= 1 << i
          entries.push({ binding: 1 + 2*i, resource: texture?.createView() ?? whiteView },
            { binding: 2 + 2*i, resource: sampler(material?.[clamp] === true, sampling?.sampler) })
          const [sx, sy, ox, oy] = material?.[transform] ?? [1, 1, 0, 0]
          const angle = sampling?.rotation ?? 0, c = Math.cos(angle), s = Math.sin(angle)
          // glTF rotates scaled top-left UVs. Existing offsets are in OBJ V-up space.
          transforms.push(c*sx, -s*sy, ox, set, s*sx, c*sy, 1-sy-oy, 0)
        }
        const wetness = material?.vertexColor ? material.vertexWetness : undefined
        const hasPbr = wetness !== undefined || material?.roughness !== undefined || material?.metallic !== undefined ||
          material?.emissive?.some(value => value > 0) || (enabled & ~1) !== 0
        const hasMetalMap = (enabled & ((1 << 1) | (1 << 4))) !== 0
        const params = new Float32Array([...(material?.color ?? [1, 1, 1]), material ? 1 : 0,
          material?.roughness ?? 1, material?.metallic ?? Number(hasMetalMap),
          material?.normalScale ?? 1, material?.normalYSign ?? 1,
          enabled, Number(hasPbr), Number(material?.vertexColor === true), Number(draw.doubleSided === true),
          ...(material?.emissive ?? [0, 0, 0]), material?.occlusionStrength ?? 1, ...transforms,
          wetness ? 1 / (1 - wetness.saturatedRed) : 0, wetness?.saturatedRoughness ?? 0, 0, 0])
        const buffer = device.createBuffer({ size: params.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
        uniforms.push(buffer)
        device.queue.writeBuffer(buffer, 0, params)
        const group = device.createBindGroup({ layout, entries: [{ binding: 0, resource: { buffer } }, ...entries] })
        binding = { group, mapMask: enabled }
        cache.set(key, binding)
      }
      return { ...draw, ...binding }
    }))
    const error = (await Promise.all(checks)).find(message => message !== undefined)
    if (error) throw new Error(error)
    function mapCoverage(channels: readonly SceneMapSlot[]) {
      const mask = slots.reduce((bits, [slot], i) => channels.includes(slot) ? bits | (1 << i) : bits, 0)
      const used = new Set<string>(), mapped = new Set<string>()
      let triangles = 0, mappedTriangles = 0
      for (const draw of bindings) {
        if (!draw.indexCount || !draw.material || !materials.has(draw.material)) continue
        used.add(draw.material)
        triangles += draw.indexCount / 3
        if (draw.mapMask & mask) {
          mapped.add(draw.material)
          mappedTriangles += draw.indexCount / 3
        }
      }
      return { materials: used.size, mapped: mapped.size, triangles, mappedTriangles }
    }
    return { bindings, mapCoverage, destroy }
  } catch (error) {
    destroy()
    throw error
  }
}
