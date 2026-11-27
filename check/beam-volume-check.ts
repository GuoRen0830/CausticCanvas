import type { ForwardResult } from '../src/contracts'
import { mat4 } from 'wgpu-matrix'
import { incidentColumnHeight } from '../src/config'
import { beamVolumeBounds, createBeamVolume } from '../src/rendering/beam-volume'
import { createVolumeMarch } from '../src/rendering/volume-march'

// Known vertical beams from the presentation fixture. Beams on the left hit an
// obstacle halfway to the receiver; beams on the right miss the entire scene.
export async function checkBeamVolume(device: GPUDevice, fixture: ForwardResult, report: (line: string) => void) {
  const lens = fixture.beams.lens
  const { columns, rows, sizeX, sizeY } = lens
  const count = 2 * (columns - 1) * (rows - 1)
  const rays = new Float32Array(count * 4)
  const hitWords = new Uint32Array(count * 2)
  const hitFloats = new Float32Array(hitWords.buffer)
  const fluxPerFace = sizeX * sizeY / count // incident irradiance = 1 W/m²
  for (let face = 0; face < count; face++) {
    rays[face * 4 + 2] = 1
    rays[face * 4 + 3] = fluxPerFace
    const x = Math.floor(face / 2) % (columns - 1)
    if (x < (columns - 1) / 2) {
      hitWords[face * 2] = 1
      hitFloats[face * 2 + 1] = fixture.receiver.z * 0.5
    }
  }

  const storage = (size: number) => device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
  const rayBuffer = storage(rays.byteLength), hitBuffer = storage(hitWords.byteLength)
  const offsets = storage(columns * rows * 8)
  const totals = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST })
  const readback = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
  const volume = createBeamVolume(device)
  const result = { ...fixture, beams: { lens, buffer: rayBuffer }, firstHits: { ...fixture.firstHits!, buffer: hitBuffer } }
  device.queue.writeBuffer(rayBuffer, 0, rays)
  device.queue.writeBuffer(hitBuffer, 0, hitWords)
  try {
    // Integrate two cross sections, split at x=0, independent of the path sampler.
    const module = device.createShaderModule({ code: `
      @group(0) @binding(0) var field: texture_3d<f32>;
      @group(0) @binding(1) var<storage, read_write> sums: array<atomic<u32>, 4>;
      @compute @workgroup_size(64) fn measure(@builtin(local_invocation_index) lane: u32) {
        let size = textureDimensions(field);
        var partial = array<f32, 4>();
        for (var i = lane; i < size.x * size.y; i += 64u) {
          let xy = vec2u(i % size.x, i / size.x);
          let side = select(0u, 1u, xy.x >= size.x / 2u);
          for (var k = 0u; k < 2u; k++) {
            let z = (2u * k + 1u) * size.z / 4u;
            let value = textureLoad(field, vec3i(vec3u(xy, z)), 0).r;
            partial[k * 2u + side] += value;
          }
        }
        // Quantize once per lane to keep the integration error independent of resolution.
        for (var k = 0u; k < 4u; k++) {
          atomicAdd(&sums[k], u32(round(partial[k] * 1048576.0 / f32(size.x * size.y))));
        }
      }` })
    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'measure' } })
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: volume.textureView }, { binding: 1, resource: { buffer: totals } },
    ] })
    async function measure() {
      const encoder = device.createCommandEncoder()
      volume.update(encoder, result, offsets, fixture.receiver.z)
      encoder.clearBuffer(totals)
      const pass = encoder.beginComputePass()
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, group)
      pass.dispatchWorkgroups(1)
      pass.end()
      encoder.copyBufferToBuffer(totals, 0, readback, 0, 16)
      device.queue.submit([encoder.finish()])
      await readback.mapAsync(GPUMapMode.READ)
      const data = new Uint32Array(readback.getMappedRange().slice(0))
      readback.unmap()
      return data
    }
    const [nearLeft, nearRight, farLeft, farRight] = await measure()
    const bounds = beamVolumeBounds(result, fixture.receiver.z)
    const volumeArea = (bounds.max[0] - bounds.min[0]) * (bounds.max[1] - bounds.min[1])
    const expected = 1048576 * 0.5 * sizeX * sizeY / volumeArea
    if (Math.abs(nearLeft - expected) > expected * 0.12 || Math.abs(nearRight - expected) > expected * 0.12) {
      throw new Error(`Incorrect flux at the near cross section: ${nearLeft}, ${nearRight}; expected ${expected} per half.`)
    }
    if (farLeft > nearLeft * 0.04 || Math.abs(farRight - nearRight) > nearRight * 0.04) {
      throw new Error(`Incorrect beam termination: near ${nearLeft}/${nearRight}, far ${farLeft}/${farRight}.`)
    }
    // The same lens, now without an obstacle, emits a tilted parallel bundle.
    // Reversing its slope should carry the distant cross section to the opposite side.
    device.queue.writeBuffer(hitBuffer, 0, hitWords.fill(0))
    const slope = 0.25 * sizeX / fixture.receiver.z
    const dz = 1 / Math.hypot(1, slope)
    for (const sign of [-1, 1]) {
      for (let face = 0; face < count; face++) {
        rays[face * 4] = sign * slope * dz
        rays[face * 4 + 2] = dz
      }
      device.queue.writeBuffer(rayBuffer, 0, rays)
      const [left, right, distantLeft, distantRight] = await measure()
      const bias = (a: number, b: number) => (b - a) / (a + b)
      if (!(sign * (bias(distantLeft, distantRight) - bias(left, right)) > 0.05)) {
        throw new Error(`Volume ignored tilted beam direction ${sign}: near ${left}/${right}, far ${distantLeft}/${distantRight}.`)
      }
    }
    // A narrow entrance with no bending has an exactly bounded traversed volume.
    // Rebuild through the normal injection and resolve passes, including their filter halo.
    const smallLens = { ...lens, sizeX: sizeX / 3, sizeY: sizeY / 3 }
    const narrow = { ...result, beams: { ...result.beams, lens: smallLens } }
    for (let face = 0; face < count; face++) {
      rays[face * 4] = 0
      rays[face * 4 + 2] = 1
      rays[face * 4 + 3] = smallLens.sizeX * smallLens.sizeY / count
    }
    device.queue.writeBuffer(rayBuffer, 0, rays)
    const encoder = device.createCommandEncoder()
    volume.update(encoder, narrow, offsets, fixture.receiver.z)
    device.queue.submit([encoder.finish()])
    await checkCameraIntegral(device, totals, readback, volume.textureView, narrow)
    report('PASS scattering: all-angle visibility, thin layers, local ridge contrast, non-emitting background, camera clipping, aperture envelope and lamp off')
    device.queue.writeBuffer(rayBuffer, 0, rays.fill(0))
    if ((await measure()).some(v => v !== 0)) throw new Error('Light remained in the volume after rebuilding with zero flux.')
    report('PASS volume: injected flux, deflection, first-hit cutoff, missed rays and zero-flux rebuild')
  } finally {
    volume.destroy()
    for (const buffer of [rayBuffer, hitBuffer, offsets, totals, readback]) buffer.destroy()
  }
}

async function checkCameraIntegral(device: GPUDevice, output: GPUBuffer, readback: GPUBuffer,
  narrowField: GPUTextureView, narrow: ForwardResult) {
  const march = createVolumeMarch(device)
  const depth = device.createTexture({ size: [4, 2], format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING })
  const layers = 144
  const field = device.createTexture({ size: [1, 1, layers], dimension: '3d', format: 'rgba8unorm',
    usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING })
  function fillField(value: number, neighbourhood = value) {
    const data = new Uint8Array(4 * layers)
    for (let z = 0; z < layers; z++) {
      data[z * 4] = value
      data[z * 4 + 1] = neighbourhood
    }
    device.queue.writeTexture({ texture: field }, data, { bytesPerRow: 4, rowsPerImage: 1 }, [1, 1, layers])
  }
  const fieldView = field.createView(), depthView = depth.createView()
  const sampler = device.createSampler({ minFilter: 'linear', magFilter: 'linear' })
  // Both depth samples traverse the refracted field. Only sample 0 crosses the aperture.
  const receiverZ = 0.5, apertureSpan = 0.1
  const inlet = incidentColumnHeight(receiverZ, apertureSpan)
  const fromLens = mat4.scale(mat4.translation([0.03125, 0.025, -inlet]), [0.125, 0.05, 1])
  const towardLens = mat4.scale(mat4.translation([0.03125, 0.025, receiverZ]), [0.125, 0.05, -1])
  const module = device.createShaderModule({ code: `
    @group(0) @binding(0) var fog: texture_2d<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<vec2f, 2>;
    @compute @workgroup_size(2) fn readFog(@builtin(local_invocation_index) i: u32) {
      output[i] = textureLoad(fog, vec2i(i32(i), 0), 0).rg;
    }` })
  const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'readFog' } })
  const generatedBounds = beamVolumeBounds(narrow, narrow.receiver.z)
  const constantBounds: typeof generatedBounds = { min: [-0.2, -0.2, 0], max: [0.2, 0.2, receiverZ] }
  async function sample(inverseVP: typeof fromLens, opaqueDepth = 1, incidentIrradiance = 0,
    source = fieldView, relativeIOR?: number, receiverSize?: [number, number]) {
    const encoder = device.createCommandEncoder()
    encoder.beginRenderPass({ colorAttachments: [], depthStencilAttachment: { view: depthView,
      depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: opaqueDepth } }).end()
    const generated = source === narrowField
    const lens = narrow.beams.lens
    const fog = march.render(encoder, 4, 2, depthView, source, sampler, inverseVP,
      generated ? generatedBounds : constantBounds,
      generated ? [lens.sizeX, lens.sizeY] : [apertureSpan, apertureSpan],
      generated ? narrow.receiver.z : receiverZ, incidentIrradiance, relativeIOR, receiverSize)
    const pass = encoder.beginComputePass()
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: fog }, { binding: 1, resource: { buffer: output } },
    ] }))
    pass.dispatchWorkgroups(1)
    pass.end()
    encoder.copyBufferToBuffer(output, 0, readback, 0, 16)
    device.queue.submit([encoder.finish()])
    await readback.mapAsync(GPUMapMode.READ)
    const samples = new Float32Array(readback.getMappedRange().slice(0))
    readback.unmap()
    for (const value of samples) if (!Number.isFinite(value) || value < 0) throw new Error('Invalid scattering readback.')
    return samples
  }
  function compareEqualPaths(a: number, b: number, label: string) {
    if (!(a > 0 && b > 0) || Math.abs(a - b) > 0.06 * Math.max(a, b)) {
      throw new Error(`Equal paths through the ${label} should both be visible: ${a} and ${b}.`)
    }
  }
  try {
    // Both samples look straight across the fan. The first skirts the aperture
    // edge inside the column. The second travels parallel to it through dark air.
    const lens = narrow.beams.lens
    for (const axis of [0, 1]) {
      for (const sign of [-1, 1]) {
        const across = 1 - axis
        const view = mat4.identity()
        view.fill(0)
        const span = across === 0 ? lens.sizeX : lens.sizeY
        view[across] = sign * span
        view[6] = 0.02
        view[8 + axis] = sign
        view[12 + axis] = -sign * (0.01 + generatedBounds.max[axis])
        view[12 + across] = sign * span * (0.43 + 0.25) // first tap at NDC x=-0.25
        view[14] = narrow.receiver.z * 0.4 + 0.01 // first tap at NDC y=-0.5
        view[15] = 1
        const whole = await sample(view, 1, 0, narrowField)
        const clipped = await sample(view, 1, 0, narrowField, 1)
        if (!(whole[0] > 0 && Math.abs(whole[0] - clipped[0]) < whole[0] * 0.06) || whole[2] || clipped[2]) {
          throw new Error(`Aperture clipping changed the finite beam at axis ${axis}, sign ${sign}: ${whole} vs ${clipped}.`)
        }
      }
    }
    // A long ray crosses 48 groups of three voxels. Sampling one midpoint per
    // group locks onto the centre of each bright layer and triples its apparent energy.
    // With uniform scattering the layered field has exactly the same integral as 85/255.
    fillField(85)
    const uniform = await sample(fromLens)
    const layersRGBA = new Uint8Array(4 * layers)
    for (let z = 1; z < layers; z += 3) layersRGBA[4 * z] = 255
    device.queue.writeTexture({ texture: field }, layersRGBA, { bytesPerRow: 4, rowsPerImage: 1 }, [1, 1, layers])
    const layered = await sample(fromLens)
    if (!(uniform[0] > 0) || Math.abs(layered[0] - uniform[0]) > 0.05 * uniform[0]) {
      throw new Error(`Thin layers aliased along a long ray: ${layered[0]} versus expected ${uniform[0]}.`)
    }
    fillField(255)
    const outgoing = await sample(fromLens), returning = await sample(towardLens)
    for (const i of [0, 2]) compareEqualPaths(outgoing[i], returning[i], 'refracted volume from opposite ends')
    // Orbit an ordinary point in the fan from above the floor. Compare the same
    // length of constant-fluence medium centered on that point at either pitch.
    const segment = 0.05
    function atElevation(degrees: number) {
      const pose = mat4.rotateY(mat4.translation([0.10, 0, 0.12]), degrees * Math.PI / 180 - Math.PI / 2)
      return mat4.scale(mat4.translate(pose, [0.03125, 0.025, -segment / 2]), [0.125, 0.05, 1])
    }
    // Matching the receiver to the aperture makes the reference flood constant,
    // so these oblique segments compare only the viewing direction.
    const referenceAperture: [number, number] = [apertureSpan, apertureSpan]
    const referenceIrradiance = 0.2
    const sampleInFan = (degrees: number) => sample(atElevation(degrees), segment, referenceIrradiance, fieldView, undefined, referenceAperture)
    const nearRim = await sampleInFan(8), overhead = await sampleInFan(50)
    compareEqualPaths(nearRim[0], overhead[0], 'volume from the rim and above')
    const partial = await sample(towardLens, receiverZ / 2)
    for (const i of [0, 2]) {
      if (!(partial[i] > 0 && partial[i] < returning[i])) throw new Error('Foreground depth did not shorten the light integral.')
      if (Math.abs(partial[i + 1] - receiverZ / 2) > 0.0001) throw new Error('Wrong distance for scattering upsampling.')
    }
    const hidden = await sample(fromLens, inlet / 2)
    if (hidden[0] || hidden[2]) throw new Error('Refracted light appeared behind a closer opaque surface.')

    // Probe the same segment with fluence first below then above the reference.
    fillField(32)
    const diffuse = await sampleInFan(50)
    fillField(192)
    const focused = await sampleInFan(50)
    if (!(diffuse[0] > 0 && focused[0] > diffuse[0] * (192 / 32) * 1.1)) {
      throw new Error(`Expected extra contrast where beams concentrate: diffuse ${diffuse[0]}, focused ${focused[0]}.`)
    }
    const widerFan = await sample(atElevation(50), segment, referenceIrradiance, fieldView, undefined, [apertureSpan, 5 * apertureSpan])
    if (!(widerFan[0] > 1.05 * focused[0])) throw new Error('Air contrast ignored the area of the reference fan.')
    fillField(192, 32)
    const ridge = await sampleInFan(50)
    if (!(ridge[0] > 1.1 * focused[0])) throw new Error('A narrow ridge and a uniform sheet of equal fluence had the same brightness.')
    fillField(0, 192)
    if ((await sampleInFan(50))[0] !== 0) throw new Error('A nearby bright sheet illuminated an empty voxel.')
    fillField(0)
    if ((await sampleInFan(50))[0] !== 0) throw new Error('The reference flood illuminated empty air.')
    const inletHigh = await sample(fromLens, 1, 3), inletLow = await sample(towardLens, 1, 3)
    compareEqualPaths(inletHigh[0], inletLow[0], 'incident column from opposite ends')
    if (inletHigh[2] || inletLow[2]) throw new Error('Incident light appeared outside its aperture.')
    const occludedInlet = await sample(towardLens, receiverZ, 3)
    if (occludedInlet[0] || occludedInlet[2]) throw new Error('Incident light leaked through an opaque surface.')
    const off = await sample(towardLens)
    if (off[0] || off[2]) throw new Error('Scattering remained with zero beam flux and the lamp off.')
  } finally {
    march.destroy()
    depth.destroy()
    field.destroy()
  }
}
