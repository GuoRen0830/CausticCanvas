import { mat4, vec3 } from 'wgpu-matrix'
import { buildLensIndices, createFlatLens } from '../src/geometry/lens-mesh'
import { createBeamRenderer } from '../src/rendering/beam-renderer'
import { createHitTracer } from '../src/rendering/hit-tracer'
import { buildSceneBvh } from '../src/rendering/scene-bvh'
import { createSceneRenderer } from '../src/rendering/scene-renderer'
import commonShader from '../src/rendering/shaders/beam-common.wgsl?raw'
import traceShader from '../src/rendering/shaders/first-hit.wgsl?raw'

export async function checkFirstHits(device: GPUDevice, report: (line: string) => void) {
  const vertices: number[] = [], indices: number[] = []
  function quad(x0: number, x1: number, halfY: number, z: number, slope: number, object: number) {
    const first = vertices.length / 11
    // Deliberately tilted shading normals on the panel exercise imported
    // smooth normals. Its geometric normal determines incident irradiance.
    const n = object === 2 ? [1, 0, 0] : vec3.normalize([slope, 0, -1])
    for (const [x, y] of [[x0, halfY], [x1, halfY], [x1, -halfY], [x0, -halfY]]) {
      vertices.push(x, y, z + slope * x, ...n, 1, 1, 1, Number(object === 1), object)
    }
    indices.push(first, first+1, first+2, first, first+2, first+3)
  }
  // The receiver reference is at 0.5 m. One surface is beyond it, one
  // overlaps in front, and two extra surfaces test distance pruning and t > 0.
  quad(-0.085, 0.08, 0.09, 0.65, 0, 1)
  quad(-0.075, -0.015, 0.065, 0.29, 0.7, 2)
  quad(-0.055, 0.045, 0.075, 0.79, -0.12, 3)
  quad(-0.18, 0.18, 0.17, -0.1, 0, 4)
  const mesh = { vertices: new Float32Array(vertices), indices: new Uint32Array(indices) }
  const spec = { columns: 17, rows: 17, sizeX: 0.2, sizeY: 0.2, thickness: 0.01 }
  const lens = createFlatLens(device, spec)
  const ids = buildLensIndices(lens)
  const count = ids.length / 3
  const rays = new Float32Array(4 * count)
  for (let i = 0; i < count; i++) {
    rays.set(vec3.normalize([i % 3 ? 0 : 0.031, i % 7 ? 0 : -0.043, 1]), 4*i)
    rays[4*i+3] = i % 13 ? 1 : 0
  }
  const buffer = device.createBuffer({ size: rays.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
  const field = { lens, buffer }
  device.queue.writeBuffer(buffer, 0, rays)
  const frame = new Float32Array(60)
  const words = new Uint32Array(frame.buffer)
  words.set([lens.columns, lens.rows], 32)
  frame.set([lens.sizeX, lens.sizeY], 34)
  words[47] = count
  const uniform = device.createBuffer({ size: frame.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  device.queue.writeBuffer(uniform, 0, frame)
  const shader = device.createShaderModule({ code: commonShader + '\n' + traceShader })
  const tracer = createHitTracer(device, shader, uniform, mesh)
  const packedFaces = buildSceneBvh(mesh).triangles
  const receiver = { sizeX: 0.2, sizeY: 0.2, z: 0.5, width: 64, height: 64 }
  const box = { center: [3, 3, 3], size: [0.01, 0.01, 0.01] } as const
  const model = mat4.identity()
  const scene = createSceneRenderer(device, receiver, box, model, mesh)
  const beams = createBeamRenderer(device, 'rgba32float', spec, receiver, model, 1, box, mesh)
  const tex = device.createTexture({ size: [64, 64], format: 'rgba32float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC })
  const read = device.createBuffer({ size: 64 * 64 * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
  const near = (actual: number, expected: number, epsilon: number, label: string) => {
    if (!(Math.abs(actual - expected) <= epsilon)) throw new Error(label + ': got ' + actual + ', expected ' + expected)
  }
  try {
    const enc = device.createCommandEncoder()
    const hits = tracer.update(enc, field, count, {}).buffer
    enc.copyBufferToBuffer(hits, 0, read, 0, count * 8)
    device.queue.submit([enc.finish()])
    await read.mapAsync(GPUMapMode.READ)
    const data = read.getMappedRange().slice(0, count * 8)
    read.unmap()
    const faces = new Uint32Array(data), times = new Float32Array(data)
    const totals = [0, 0, 0]
    let maxError = 0
    for (let i = 0; i < count; i++) {
      const o = [0, 0, spec.thickness], d = rays.subarray(4*i, 4*i+3)
      for (const v of ids.subarray(3*i, 3*i+3)) {
        o[0] += (v % lens.columns / (lens.columns-1) - 0.5) * lens.sizeX / 3
        o[1] += (Math.floor(v / lens.columns) / (lens.rows-1) - 0.5) * lens.sizeY / 3
      }
      let tMin = Infinity
      const validFaces: number[] = []
      // CPU brute force: intersect each original face plane and check its three
      // oriented edges. Accept either adjacent triangle for an exact edge tie.
      for (let f = 0; rays[4*i+3] > 0 && f < indices.length/3; f++) {
        const corners = indices.slice(3*f, 3*f+3).map(k => mesh.vertices.subarray(11*k, 11*k+3))
        const [a, b, c] = corners
        const n = vec3.cross(vec3.subtract(b,a), vec3.subtract(c,a))
        const t = vec3.dot(vec3.subtract(a,o), n) / vec3.dot(d,n)
        if (t <= 1e-7 || t > tMin + 1e-7) continue
        const point = vec3.addScaled(o,d,t)
        if (corners.some((v,k) => vec3.dot(n, vec3.cross(vec3.subtract(corners[(k+1)%3],v), vec3.subtract(point,v))) < -1e-11)) continue
        if (t < tMin - 1e-7) validFaces.length = 0
        validFaces.push(f)
        tMin = Math.min(tMin,t)
      }
      const packed = faces[2*i]
      if (Number.isFinite(tMin)) {
        if (!packed || !validFaces.includes(packedFaces[12*(packed-1)+7])) throw new Error('Wrong first face at beam ' + i)
        const object = packedFaces[12*(packed-1)+3]
        if (object !== 1 && object !== 2) throw new Error('Unexpected illuminated object ' + object)
        totals[object]++
        maxError = Math.max(maxError, Math.abs(times[2*i+1] - tMin))
      } else {
        near(packed, 0, 0, 'Miss at beam ' + i)
        totals[0]++
      }
    }
    near(maxError, 0, 2e-6, 'Hit distance')
    if (totals.some(n => n < 10)) throw new Error('Insufficient fixture coverage')
    report('PASS ' + count + ' BVH rays vs CPU, misses/far/sloped ' + totals.join('/') + ', max distance error ' + maxError.toExponential(2) + ' m')

    // Vertical beams deliver 1 W/m2 to a flat surface. The tilted panel
    // receives its cosine-scaled share, and its shadow is seen by a sheared camera.
    const flux = spec.sizeX * spec.sizeY / count
    function setFlux(value: number) {
      for (let i = 0; i < count; i++) rays.set([0, 0, 1, value], 4*i)
      device.queue.writeBuffer(buffer, 0, rays)
      return { beams: field, receiver, caustic: { width: 64, height: 64, kind: 'buffer' as const, buffer } }
    }
    async function render(result: ReturnType<typeof setFlux>, shear: number, zoom = 1) {
      const vp = mat4.scaling([10,-10,1])
      vp[0] *= zoom
      vp[5] *= zoom
      vp[8] = shear * 10
      vp[12] = -0.65 * vp[8]
      scene.update(vp)
      const e = device.createCommandEncoder()
      beams.render(e, tex, vp, 'solver', 'lit', result, scene.draw, () => {})
      e.copyTextureToBuffer({ texture: tex }, { buffer: read, bytesPerRow: 64*16 }, [64,64])
      device.queue.submit([e.finish()])
      await read.mapAsync(GPUMapMode.READ)
      const rgba = new Float32Array(read.getMappedRange())
      const red = Float32Array.from({ length: 4096 }, (_,i) => rgba[4*i])
      read.unmap()
      return red
    }
    const full = setFlux(flux)
    const top = await render(full, 0), orbit = await render(full, 0.45)
    const dark = setFlux(0)
    const base = await render(dark, 0), orbitBase = await render(dark, 0.45)
    function irradiance(lit: Float32Array, ambient: Float32Array, x: number) {
      const i = 32 * 64 + Math.floor((x / 0.2 + 0.5) * 64)
      return (lit[i] - ambient[i]) * Math.PI
    }
    near(irradiance(top, base, 0.04), 1, 0.003, 'Floor beyond solver receiver')
    near(irradiance(top, base, -0.045), 1 / Math.hypot(1, 0.7), 0.003, 'Sloped panel')
    near(irradiance(orbit, orbitBase, -0.045), 0, 0.003, 'Offscreen panel shadows floor')
    near(irradiance(orbit, orbitBase, 0.04), 1, 0.003, 'Unshadowed floor after orbit')
    report('PASS sloped hit irradiance, offscreen blocker, floor beyond receiver, and same-buffer result refresh')

    // Average many subpixel footprints with non-integer projected edges.
    const dense = createFlatLens(device, { ...spec, columns: 257, rows: 257 })
    const denseCount = 2 * 256 * 256
    const denseBuffer = device.createBuffer({ size: 16 * denseCount, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
    try {
      const terms = new Float32Array(4 * denseCount)
      for (let i = 0; i < denseCount; i++) {
        terms[4*i+2] = 1
        terms[4*i+3] = spec.sizeX * spec.sizeY / denseCount
      }
      device.queue.writeBuffer(denseBuffer, 0, terms)
      const zoom = 0.917
      const image = await render({ ...full, beams: { lens: dense, buffer: denseBuffer } }, 0, zoom)
      const ambient = await render(dark, 0, zoom)
      let worst = 0
      for (let y = 20; y < 44; y++) for (let x = 35; x < 51; x++) {
        const i = 64*y+x
        worst = Math.max(worst, Math.abs((image[i] - ambient[i]) * Math.PI - 1))
      }
      near(worst, 0, device.features.has('float32-blendable') ? 2e-5 : 0.025, 'Subpixel energy')
      report('PASS dense subpixel splats, max interior irradiance error ' + worst.toExponential(2) + ' W/m2')
    } finally { denseBuffer.destroy(); dense.heightBuffer.destroy() }
  } finally {
    tracer.destroy(); beams.destroy(); scene.destroy(); uniform.destroy(); read.destroy(); tex.destroy()
    buffer.destroy(); lens.heightBuffer.destroy()
  }
}
