import type { ForwardResult, LensState, TargetImage } from '../contracts'
import { workgroupGrid } from '../gpu/dispatch'
import type { DesignView } from './design-view'
import optics from './optics.wgsl?raw'
import sceneOptics from './scene-optics.wgsl?raw'
import hits from '../rendering/shaders/scene-hit.wgsl?raw'
import bvh from '../rendering/shaders/bvh-query.wgsl?raw'
import apertureUV from '../rendering/shaders/aperture-uv.wgsl?raw'
import color from './aperture-color.wgsl?raw'

function bufferEntry(binding: number, buffer: GPUBuffer): GPUBindGroupEntry {
  return { binding, resource: { buffer } }
}

// Allocate display-only resources on demand. Neither the scalar forward image
// nor any gradient/geometry buffer is written by the three color passes.
export function createApertureColor(device: GPUDevice) {
  const module = device.createShaderModule({
    label: 'Aperture color transport', code: [optics, hits, bvh, sceneOptics, apertureUV, color].join('\n'),
  })
  const calibrate = device.createRenderPipeline({
    label: 'Calibrate the RGB entrance filter', layout: 'auto',
    vertex: { module, entryPoint: 'apertureVS' },
    fragment: { module, entryPoint: 'calibrateAperture', targets: [{ format: 'rgba16float' }] },
    primitive: { topology: 'triangle-list' }, multisample: { count: 4 },
  })
  const compute = (entryPoint: string) => device.createComputePipeline({
    label: entryPoint, layout: 'auto', compute: { module, entryPoint },
  })
  const forward = compute('colorForward'), resolve = compute('resolveColor')

  function targets(design: DesignView, lens: Readonly<LensState>) {
    const { width, height } = design
    const pixelCount = width * height
    const longerSide = Math.max(lens.sizeX, lens.sizeY)
    const filterSize = [lens.sizeX, lens.sizeY].map(side => Math.max(1, Math.round(512 * side / longerSide)))
    const filter = device.createTexture({ label: 'Physical RGB entrance filter', size: filterSize,
      format: 'rgba16float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING })
    const samples = device.createTexture({ label: 'Entrance subpixel samples', size: filterSize,
      format: 'rgba16float', sampleCount: 4, usage: GPUTextureUsage.RENDER_ATTACHMENT })
    const output = device.createTexture({ label: 'Incident RGB irradiance in design view', size: [width, height],
      format: 'rgba32float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING })
    const view = { filter: filter.createView(), image: output.createView() }
    const coverage = samples.createView()
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    const surface = device.createBuffer({ label: 'Color design surfaces', size: design.samples.byteLength, usage: storage })
    device.queue.writeBuffer(surface, 0, design.samples)
    const paint = device.createBuffer({ label: 'Linear brush RGB, coverage and fallback',
      size: (pixelCount + 1) * 16, usage: storage })
    const sums = device.createBuffer({ label: 'RGB footprint sums', size: 12 * pixelCount, usage: storage })
    const resolveGroup = device.createBindGroup({ layout: resolve.getBindGroupLayout(0),
      entries: [bufferEntry(3, sums), { binding: 15, resource: view.image }] })
    let lastTarget: TargetImage | undefined
    let version = -1
    let colorVersion: number | undefined
    let fallback: GPUColor = [1, 1, 1, 1]

    function uploadTarget(target: TargetImage) {
      if (target === lastTarget && target.version === version && target.colorVersion === colorVersion) return fallback
      if (target.width !== width || target.height !== height || target.values.length !== pixelCount ||
          (target.colors && target.colors.length !== pixelCount * 3)) {
        throw new Error('Brush color and coverage must match the scene design view.')
      }
      const packed = new Float32Array((pixelCount + 1) * 4)
      const total = [0, 0, 0]
      let area = 0
      for (let i = 0; i < pixelCount; i++) {
        const coverage = target.values[i], weight = coverage * design.surfaceAreas[i]
        packed[4 * i + 3] = coverage
        for (let c = 0; c < 3; c++) {
          const channel = target.colors?.[3 * i + c] ?? 1
          packed[4 * i + c] = channel
          total[c] += channel * weight
        }
        area += weight
      }
      const mean = total.map(sum => area > 0 ? sum / area : 1)
      fallback = [mean[0], mean[1], mean[2], 1]
      packed.set([...mean, 1], 4 * pixelCount)
      device.queue.writeBuffer(paint, 0, packed)
      lastTarget = target
      version = target.version
      colorVersion = target.colorVersion
      return fallback
    }

    return { design, view, coverage, surface, sums, paint, resolveGroup, uploadTarget,
      destroy() {
        filter.destroy(); samples.destroy(); output.destroy()
        surface.destroy(); paint.destroy(); sums.destroy()
      },
    }
  }

  // Coarse training and the fine lens have different grid dimensions.
  // Keep each uniform immutable so either grid can be encoded without a stall.
  const uniforms = new Map<string, GPUBuffer>()
  const zeroOffsets = new Map<number, GPUBuffer>()
  function parameters(result: ForwardResult) {
    const { receiver, beams: { lens } } = result
    const sizes = [lens.columns, lens.rows, receiver.width, receiver.height,
      lens.sizeX, lens.sizeY, receiver.sizeX, receiver.sizeY]
    const key = sizes.join(',')
    let uniform = uniforms.get(key)
    if (!uniform) {
      const data = new ArrayBuffer(48)
      new Uint32Array(data, 0, 4).set(sizes.slice(0, 4))
      new Float32Array(data, 16, 4).set(sizes.slice(4))
      uniform = device.createBuffer({ label: 'Aperture color geometry', size: 48,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
      device.queue.writeBuffer(uniform, 0, data)
      uniforms.set(key, uniform)
    }
    return uniform
  }
  function offsetsOf(lens: Readonly<LensState>) {
    if (lens.offsetBuffer) return lens.offsetBuffer
    const bytes = lens.columns * lens.rows * 8
    let zero = zeroOffsets.get(bytes)
    if (!zero) {
      zero = device.createBuffer({ label: 'Color lens zero displacements', size: bytes, usage: GPUBufferUsage.STORAGE })
      zeroOffsets.set(bytes, zero)
    }
    return zero
  }
  let canvas: ReturnType<typeof targets> | undefined
  return {
    update(encoder: GPUCommandEncoder, result: ForwardResult, target: TargetImage, design: DesignView) {
      if (!result.firstHits || result.firstHits.mesh !== design.mesh) throw new Error('Color transport requires matching scene first hits.')
      const { lens, buffer } = result.beams
      if (!canvas || canvas.design !== design) { canvas?.destroy(); canvas = targets(design, lens) }
      const clearValue = canvas.uploadTarget(target)
      const geometry = [bufferEntry(0, parameters(result)), bufferEntry(1, lens.heightBuffer), bufferEntry(2, buffer),
        bufferEntry(8, result.firstHits.triangles), bufferEntry(10, result.firstHits.buffer),
        bufferEntry(11, canvas.surface), bufferEntry(12, offsetsOf(lens))]
      const filterGroup = device.createBindGroup({ layout: calibrate.getBindGroupLayout(0),
        entries: [...geometry, bufferEntry(13, canvas.paint)] })
      const faceCount = 2 * (lens.columns - 1) * (lens.rows - 1)
      const filterPass = encoder.beginRenderPass({ label: 'Backproject chroma onto the physical aperture',
        colorAttachments: [{ view: canvas.coverage, resolveTarget: canvas.view.filter, clearValue,
          loadOp: 'clear', storeOp: 'discard' }] })
      filterPass.setPipeline(calibrate)
      filterPass.setBindGroup(0, filterGroup)
      filterPass.draw(3, faceCount)
      filterPass.end()

      encoder.clearBuffer(canvas.sums)
      const rgbGroup = device.createBindGroup({ layout: forward.getBindGroupLayout(0),
        entries: [...geometry, bufferEntry(3, canvas.sums), { binding: 14, resource: canvas.view.filter }] })
      const pass = encoder.beginComputePass({ label: 'Forward RGB irradiance on visible design surfaces' })
      pass.setPipeline(forward)
      pass.setBindGroup(0, rgbGroup)
      pass.dispatchWorkgroups(...workgroupGrid(Math.ceil(faceCount / 64), device.limits.maxComputeWorkgroupsPerDimension))
      pass.setPipeline(resolve)
      pass.setBindGroup(0, canvas.resolveGroup)
      pass.dispatchWorkgroups(Math.ceil(design.width / 8), Math.ceil(design.height / 8))
      pass.end()
      return canvas.view
    },
    destroy() {
      canvas?.destroy()
      for (const buffer of uniforms.values()) buffer.destroy()
      for (const buffer of zeroOffsets.values()) buffer.destroy()
    },
  }
}
