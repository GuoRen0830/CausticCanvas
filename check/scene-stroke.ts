import { config } from '../src/config'
import { createFlatLens } from '../src/geometry/lens-mesh'
import { buildDesignView } from '../src/optics/design-view'
import { createForwardModel } from '../src/optics/forward-model'
import { createLensOptimizer } from '../src/optimization/lens-optimizer'
import { buildSceneMesh } from '../src/rendering/scene-mesh'
import { createSimulation } from '../src/simulation'
import type { ForwardResult } from '../src/contracts'

// Opt in with /check/beam-check.html?stroke. Keep the historical debug grid
// and brush fixed even when the application's working grid changes.
export async function checkSceneStroke(device: GPUDevice, report: (line: string) => void) {
  const receiver = { width: 256, height: 256, sizeX: 0.4, sizeY: 0.4, z: 0.5 }
  const lens = createFlatLens(device,{ columns: 33, rows: 33, sizeX: 0.2, sizeY: 0.2, thickness: 0.01 })
  const state = { lens, receiver, optics: config.optics, target: null }
  const box = { center: [0.05,0.03,0.3], size: [0.08,0.08,0.1] } as const
  const view = buildDesignView(buildSceneMesh(receiver,box),receiver)
  const solver = (await createForwardModel(device,state,{designView:view}))!
  const settings = { ...config.optimizer, maxIterations: 3000, lossLogInterval: 0 }
  const optimizer = (await createLensOptimizer(device,state,solver,settings))!
  const sim = createSimulation(state,solver,optimizer)
  const weights = Float32Array.from({length:256*256},(_,i) => {
    const x = i%256+0.5-70, y = Math.floor(i/256)+0.5-140
    const t = Math.max(0,Math.min(1,(94*x+2.8*y)/(94**2+2.8**2)))
    return Math.max(0,Math.min(1,6.9-Math.hypot(x-94*t,y-2.8*t)))
  })
  const staging = device.createBuffer({ size: weights.byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
  try {
    optimizer.setTarget({version:1, width:256, height:256, values:weights})
    let result: ForwardResult
    for (let i=0; i<settings.maxIterations; i+=10) {
      const enc = device.createCommandEncoder()
      result = sim.encode(enc,10).result
      device.queue.submit([enc.finish()])
      await device.queue.onSubmittedWorkDone()
      if ((i+10)%500 === 0) report(`Scene stroke iteration ${i+10}`)
    }
    if (result!.caustic.kind !== 'buffer') throw new Error('Expected scene image buffer')
    const enc = device.createCommandEncoder()
    enc.copyBufferToBuffer(result!.caustic.buffer,0,staging,0,staging.size)
    device.queue.submit([enc.finish()])
    await staging.mapAsync(GPUMapMode.READ)
    const image = new Float32Array(staging.getMappedRange())
    let total = 0, painted = 0, top = 0, paintedTop = 0
    for (let i=0; i<weights.length; i++) {
      const object = view.bvh.triangles[(view.samples[2*i]-1)*12+3]
      const power = image[i]*view.surfaceAreas[i]
      total += power
      if (weights[i]>=0.5) painted += power
      if (object === 2) {
        top += power
        if (weights[i]>=0.5) paintedTop += power
      }
    }
    staging.unmap()
    const topFraction = paintedTop/top, sceneFraction = painted/total
    if (!(top > 0.005 && total > 0.025 && topFraction > 0.82 && sceneFraction > 0.72)) {
      throw new Error(`Horizontal stroke regression: top ${top} W, total ${total} W, fractions ${topFraction}/${sceneFraction}`)
    }
    report(`PASS horizontal scene stroke: ${(100*topFraction).toFixed(1)}% of top power and ${(100*sceneFraction).toFixed(1)}% of light-view power inside brush`)
  } finally {
    staging.destroy(); optimizer.destroy(); solver.destroy(); lens.heightBuffer.destroy()
  }
}
