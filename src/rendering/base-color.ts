import type { BaseColorMaterial, SceneDraw } from './scene-mesh'

// Successive render passes filter the previous sRGB level in linear space.
function createMipGenerator(device: GPUDevice) {
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
  const pipeline = device.createRenderPipeline({ label: 'Base Color mipmaps', layout: 'auto',
    vertex: { module, entryPoint: 'vs' }, fragment: { module, entryPoint: 'fs', targets: [{ format: 'rgba8unorm-srgb' }] } })
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

export async function createBaseColorBindings(device: GPUDevice, layout: GPUBindGroupLayout, draws: SceneDraw[],
  materials: ReadonlyMap<string, BaseColorMaterial>, warn: (message: string) => void) {
  const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
  const white = device.createTexture({ label: 'Plain Base Color', size: [1, 1], format: 'rgba8unorm-srgb', usage })
  device.queue.writeTexture({ texture: white }, new Uint8Array([255, 255, 255, 255]), {}, [1, 1])
  const textures = new Map<File, GPUTexture>()
  const uniforms: GPUBuffer[] = []
  function destroy() {
    white.destroy()
    for (const texture of textures.values()) texture.destroy()
    for (const buffer of uniforms) buffer.destroy()
  }
  try {
    const images = new Set<File>()
    for (const draw of draws) {
      const file = draw.material && materials.get(draw.material)?.map
      if (file && draw.hasUV) images.add(file)
    }
    let generateMips: ReturnType<typeof createMipGenerator> | undefined
    for (const file of images) {
      let bitmap: ImageBitmap
      try { bitmap = await createImageBitmap(file, { premultiplyAlpha: 'none' }) }
      catch { warn('Cannot decode ' + file.name + '. Using Kd.'); continue }
      try {
        const { width, height } = bitmap
        if (Math.max(width, height) > device.limits.maxTextureDimension2D) {
          warn(file.name + ' exceeds the GPU texture size limit. Using Kd.')
          continue
        }
        const mipLevelCount = 1 + Math.floor(Math.log2(Math.max(width, height)))
        const texture = device.createTexture({ label: file.name, size: [width, height], format: 'rgba8unorm-srgb', usage, mipLevelCount })
        textures.set(file, texture)
        device.queue.copyExternalImageToTexture({ source: bitmap }, { texture, colorSpace: 'srgb' }, [width, height])
        if (mipLevelCount > 1) {
          generateMips ??= createMipGenerator(device)
          generateMips(texture)
        }
      } finally { bitmap.close() }
    }
    const samplers = new Map<boolean, GPUSampler>()
    const plain = new Map<string | undefined, GPUBindGroup>(), textured = new Map<string | undefined, GPUBindGroup>()
    const bindings = draws.map(draw => {
      const material = draw.material ? materials.get(draw.material) : undefined
      const texture = draw.hasUV && material?.map ? textures.get(material.map) : undefined
      const cache = texture ? textured : plain
      const key = material ? draw.material : undefined
      let group = cache.get(key)
      if (!group) {
        const clamp = material?.clamp === true
        let sampler = samplers.get(clamp)
        if (!sampler) {
          const address = clamp ? 'clamp-to-edge' : 'repeat'
          sampler = device.createSampler({ addressModeU: address, addressModeV: address,
            minFilter: 'linear', magFilter: 'linear', mipmapFilter: 'linear' })
          samplers.set(clamp, sampler)
        }
        const params = new Float32Array([...(material?.color ?? [1, 1, 1]), material ? 1 : 0,
          ...(material?.uvTransform ?? [1, 1, 0, 0])])
        const buffer = device.createBuffer({ size: params.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
        uniforms.push(buffer)
        device.queue.writeBuffer(buffer, 0, params)
        group = device.createBindGroup({ layout, entries: [
          { binding: 0, resource: { buffer } }, { binding: 1, resource: (texture ?? white).createView() },
          { binding: 2, resource: sampler },
        ] })
        cache.set(key, group)
      }
      return { ...draw, group }
    })
    return { bindings, destroy }
  } catch (error) {
    destroy()
    throw error
  }
}
