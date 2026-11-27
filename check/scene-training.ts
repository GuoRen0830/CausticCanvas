import { config } from '../src/config'
import { buildSceneGeometry } from '../src/rendering/scene-mesh'
import plate from './fixtures/base-color/plate.obj?raw'
import { parseObjMesh } from '../src/geometry/obj-mesh'
import { placeObjOnReceiver } from '../src/geometry/place-obj'
import { createFlatLens } from '../src/geometry/lens-mesh'
import { buildDesignView } from '../src/optics/design-view'
import { createForwardModel } from '../src/optics/forward-model'
import { createLensOptimizer, normalizeTarget } from '../src/optimization/lens-optimizer'
import type { ForwardModel } from '../src/contracts'

export async function checkSceneTraining(device: GPUDevice, report: (line: string) => void) {
  const receiver = { width: 57, height: 41, sizeX: 0.36, sizeY: 0.27, z: 0.5 }
  const spec = { columns: 8, rows: 7, sizeX: 0.16, sizeY: 0.14, thickness: 0.025 }
  const lens = createFlatLens(device, spec)
  const input = { receiver, lens, optics: { lensIOR: 1.5, outsideIOR: 1, incidentIrradiance: 1 } }
  const vertices: number[] = [], indices: number[] = []
  // A small sloped panel at the right, with the background sheet beyond the
  // nominal receiver. The two are distinct objects in the same design image.
  function panel(x0: number, x1: number, halfY: number, z: number, slope: number) {
    const first = vertices.length / 11
    for (const [x,y] of [[x0,halfY],[x1,halfY],[x1,-halfY],[x0,-halfY]]) {
      vertices.push(x,y,z+slope*x, 0,0,-1, 1,1,1, 0,first+1)
    }
    indices.push(first,first+1,first+2,first,first+2,first+3)
  }
  panel(-0.15,0.15,0.12,0.62,0)
  panel(0.009,0.082,0.09,0.25,0.55)
  const mesh = { vertices: new Float32Array(vertices), indices: new Uint32Array(indices) }
  const view = buildDesignView(mesh,receiver)
  const depths = new Float32Array(view.samples.buffer)
  const pixelArea = receiver.sizeX*receiver.sizeY/(receiver.width*receiver.height)
  const depthAt = (x: number, y: number) => depths[2*(y*receiver.width+x)+1]
  function near(actual: number, expected: number, tolerance: number, label: string) {
    if (!(Math.abs(actual-expected) <= tolerance)) throw new Error(`${label}: ${actual} vs ${expected}`)
  }
  near(depthAt(0,0),0,0,'Empty design pixel')
  near(depthAt(16,20),0.62,1e-6,'Far receiver depth')
  near(depthAt(35,20),0.25+0.55*((35.5/57-0.5)*0.36),1e-6,'Panel depth')
  near(view.surfaceAreas[20*57+35]/pixelArea,Math.hypot(1,0.55),1e-6,'Panel surface area')
  const all = { width: 57, height: 41, version: 1, values: new Float32Array(57*41).fill(1) }
  const target = normalizeTarget(all,receiver,0.125,view.surfaceAreas)!
  near(target.reduce((s,v,i) => s+v*view.surfaceAreas[i],0),0.125,1e-7,'Integrated target power')
  near(target[0],0,0,'Unpaintable background')

  const scene = (await createForwardModel(device,input,{designView:view}))!
  const owned: GPUBuffer[] = []
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
  function allocate(size: number, usage: number) {
    const b = device.createBuffer({size,usage}); owned.push(b); return b
  }
  const pixels = receiver.width*receiver.height, count = lens.rows*lens.columns
  const upstream = Float32Array.from({length:pixels},(_,i) =>
    Math.sin(0.08*(i%receiver.width)) + 0.5*Math.cos(0.21*Math.floor(i/receiver.width)))
  const weights = allocate(upstream.byteLength,storage), grad = allocate(count*4,storage)
  device.queue.writeBuffer(weights,0,upstream)
  const read = allocate(4*pixels,GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ)
  async function evaluate(model: ForwardModel, gradient = false) {
    const enc = device.createCommandEncoder()
    const result = model.encode(enc,input)
    if (result.caustic.kind !== 'buffer') throw new Error('Expected irradiance buffer')
    if (gradient) model.encodeBackward!(enc,input,result,weights,grad)
    const bytes = 4*(gradient ? count : pixels)
    enc.copyBufferToBuffer(gradient ? grad : result.caustic.buffer,0,read,0,bytes)
    device.queue.submit([enc.finish()])
    await read.mapAsync(GPUMapMode.READ)
    const copy = new Float32Array(read.getMappedRange(0,bytes).slice(0))
    read.unmap()
    return copy
  }
  try {
    const flat = await evaluate(scene)
    near(flat[20*57+35],1/Math.hypot(1,0.55),2e-5,'Sloped panel irradiance')
    near(flat[20*57+20],1,2e-5,'Sheet irradiance beyond nominal plane')
    near(flat[0],0,0,'Background irradiance')
    // Small non-symmetric slopes avoid testing exactly at a visibility switch.
    const original = Float32Array.from({length:count},(_,i) =>
      0.025 + 0.0002*Math.sin(0.9*i) + 0.00008*(i%lens.columns))
    device.queue.writeBuffer(lens.heightBuffer,0,original)
    const analytic = await evaluate(scene,true)
    let worst = 0
    for (const v of [18,21,34,37]) {
      const samples: number[] = []
      const step = 1e-6
      for (const sign of [-1,1]) {
        const h = original.slice()
        h[v] += sign*step
        device.queue.writeBuffer(lens.heightBuffer,0,h)
        const values = await evaluate(scene)
        samples.push(values.reduce((s,e,i) => s+e*upstream[i],0))
      }
      const fd = (samples[1]-samples[0])/(2*step)
      const relative = Math.abs(fd-analytic[v])/Math.max(Math.abs(analytic[v]),Math.abs(fd),1)
      worst = Math.max(worst,relative)
      near(relative,0,0.035,`Height VJP at vertex ${v}, FD ${fd}, AD ${analytic[v]}`)
    }
    // Vertex motion: a scene that is just a flat sheet at the receiver plane
    // must reproduce the plane model (whose x/y gradients optics-check compares
    // with CPU central differences): same image, height and x/y gradients.
    // Visibility edges in the panel scene make finite differences in x and y
    // unreliable at steps the f32 image can resolve.
    const cellX = spec.sizeX/(spec.columns-1), cellY = spec.sizeY/(spec.rows-1)
    const xy = Float32Array.from({length:2*count},(_,i) => 0.1*(i%2 ? cellY : cellX)*Math.sin(1.3*i+0.2))
    const offsets = allocate(xy.byteLength,storage)
    device.queue.writeBuffer(offsets,0,xy)
    device.queue.writeBuffer(lens.heightBuffer,0,original)
    vertices.length = 0; indices.length = 0
    panel(-0.25,0.25,0.2,receiver.z,0)
    const sheet = buildDesignView({ vertices: new Float32Array(vertices), indices: new Uint32Array(indices) },receiver)
    const moved = { ...input, lens: { ...lens, offsetBuffer: offsets } }
    const sheetModel = (await createForwardModel(device,moved,{designView:sheet}))!
    const planeModel = (await createForwardModel(device,moved))!
    async function gradients(model: ForwardModel) {
      const image = allocate(4*pixels,storage), h = allocate(count*4,storage), o = allocate(xy.byteLength,storage)
      const out = allocate(4*pixels+count*4+xy.byteLength,GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ)
      const enc = device.createCommandEncoder()
      const result = model.encode(enc,moved)
      enc.copyBufferToBuffer((result.caustic as { buffer: GPUBuffer }).buffer,0,image,0,4*pixels)
      model.encodeBackward!(enc,moved,result,weights,h,o)
      enc.copyBufferToBuffer(image,0,out,0,4*pixels)
      enc.copyBufferToBuffer(h,0,out,4*pixels,count*4)
      enc.copyBufferToBuffer(o,0,out,4*pixels+count*4,xy.byteLength)
      device.queue.submit([enc.finish()])
      await out.mapAsync(GPUMapMode.READ)
      const all = new Float32Array(out.getMappedRange().slice(0))
      out.unmap()
      return [all.subarray(0,pixels),all.subarray(pixels,pixels+count),all.subarray(pixels+count)]
    }
    const fromSheet = await gradients(sheetModel), fromPlane = await gradients(planeModel)
    if (!fromPlane.every(values => values.some(v => Math.abs(v) > 1e-3))) throw new Error('Expected nonzero sheet gradients')
    let worstXY = 0
    fromSheet.forEach((values,k) => {
      let error = 0, scale = 0
      values.forEach((v,i) => { error += (v-fromPlane[k][i])**2; scale += fromPlane[k][i]**2 })
      const relative = Math.sqrt(error/Math.max(scale,1e-30))
      worstXY = Math.max(worstXY,relative)
      near(relative,0,2e-3,`Sheet scene matches the plane model (${['image','height gradient','x/y gradient'][k]})`)
    })
    sheetModel.destroy(); planeModel.destroy()
    report(`PASS scene design depths, tilted-area target power, far receiver and finite-difference VJP (max relative error ${(100*worst).toFixed(2)}%); displaced sheet scene matches the plane model within ${(100*worstXY).toFixed(3)}%`)
  } finally {
    scene.destroy()
    lens.heightBuffer.destroy()
    for (const b of owned) b.destroy()
  }
}

// Exercises multiple rows in both face and vertex dispatches on a standard device.
// Only run on request: the beam and triangle-gradient buffers are 128 MiB each.
export async function checkLargeSceneDispatch(device: GPUDevice, report: (line: string) => void) {
  const lens = createFlatLens(device, { ...config.lens, columns: 2049, rows: 2049 })
  const mesh = parseObjMesh(plate)
  const fitted = placeObjOnReceiver(mesh, config.lens, config.receiver)
  // Keep the large dispatch's outermost lens vertices over the test floor.
  const receiver = { ...config.receiver, z: fitted.receiver.z }
  const input = { lens, receiver, optics: config.optics }
  const view = buildDesignView(buildSceneGeometry(receiver, { mesh }).mesh, receiver)
  const model = await createForwardModel(device, input, { designView: view })
  const optimizer = (await createLensOptimizer(device, input, model, { ...config.optimizer, lossLogInterval: 0 }))!
  const tailCount = 2048
  const staging = device.createBuffer({ size: tailCount * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
  try {
    const lastFace = 2 * (lens.columns - 1) * (lens.rows - 1) - 1
    let enc = device.createCommandEncoder()
    const flat = model.encode(enc, input)
    enc.copyBufferToBuffer(flat.beams.buffer, lastFace * 16, staging, 0, 16)
    enc.copyBufferToBuffer(flat.firstHits!.buffer, lastFace * 8, staging, 16, 8)
    device.queue.submit([enc.finish()])
    await staging.mapAsync(GPUMapMode.READ)
    const bytes = staging.getMappedRange()
    const flux = new Float32Array(bytes)[3]
    const trianglePlusOne = new Uint32Array(bytes)[4]
    staging.unmap()
    const expected = input.optics.incidentIrradiance * lens.sizeX * lens.sizeY / (lastFace + 1)
    if (!trianglePlusOne || !(Math.abs(flux / expected - 1) < 1e-5)) throw new Error('Last beam missing or invalid')

    const { width, height } = receiver
    const values = Float32Array.from({ length: width * height }, (_,i) => i % width < width / 2 ? 1 : 0)
    optimizer.setTarget({ width, height, values, version: 1 })
    enc = device.createCommandEncoder()
    const update = optimizer.encodeStep(enc, input)
    const offset = (lens.rows * lens.columns - tailCount) * 4
    enc.copyBufferToBuffer(update.lens.heightBuffer, offset, staging, 0, staging.size)
    device.queue.submit([enc.finish()])
    await staging.mapAsync(GPUMapMode.READ)
    const tail = new Float32Array(staging.getMappedRange())
    const changed = tail.every(Number.isFinite) && tail.some(h => Math.abs(h - config.lens.thickness) > 1e-6)
    staging.unmap()
    if (!changed) throw new Error('Vertex gradients or updates missing from last dispatch row')
    report('PASS 2048 x 2048 cells: verified final beam, first hit and updated tail heights')
  } finally {
    staging.destroy(); optimizer.destroy(); model.destroy(); lens.heightBuffer.destroy()
  }
}
