import { createFlatLens } from '../src/geometry/lens-mesh'
import { buildDesignView } from '../src/optics/design-view'
import { createForwardModel } from '../src/optics/forward-model'
import { normalizeTarget } from '../src/optimization/lens-optimizer'
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
    report(`PASS scene design depths, tilted-area target power, far receiver and finite-difference VJP (max relative error ${(100*worst).toFixed(2)}%)`)
  } finally {
    scene.destroy()
    lens.heightBuffer.destroy()
    for (const b of owned) b.destroy()
  }
}
