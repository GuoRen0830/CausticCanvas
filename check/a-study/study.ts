import { createForwardModel, type OpticsOptions } from '../../src/optics/forward-model'
import { createLensOptimizer } from '../../src/optimization/lens-optimizer'
import { config } from '../../src/config'
import type { ForwardInput } from '../../src/contracts'
import { extendedChecks } from './extended'

type Strategy = NonNullable<OpticsOptions['strategy']>
const strategies: Strategy[] = ['baseline','cached','cooperative','striped']
const lines: string[] = []
function log(value: unknown) {
  const text = typeof value === 'string' ? value : JSON.stringify(value)
  lines.push(text)
  document.querySelector('#output')!.textContent = lines.join('\n')
  console.log(text)
}
function quantile(values: number[], fraction: number) {
  const sorted = [...values].sort((a,b)=>a-b)
  return sorted[Math.floor((sorted.length-1)*fraction)]
}
async function run() {
  const checksOnly = new URLSearchParams(location.search).has('checks-only')
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) throw new Error('WebGPU unavailable.')
  if (!checksOnly && !adapter.features.has('timestamp-query')) throw new Error('GPU timestamp queries unavailable; no CPU timing substitution.')
  const device = await adapter.requestDevice({ requiredFeatures: checksOnly ? [] : ['timestamp-query'] })
  const errors: string[] = []
  device.addEventListener('uncapturederror', e => errors.push(e.error.message))
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  function buffer(values: Float32Array) {
    const result = device.createBuffer({ size: values.byteLength, usage })
    device.queue.writeBuffer(result, 0, values)
    return result
  }
  async function read(source: GPUBuffer, count: number) {
    const staging = device.createBuffer({size:count*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST})
    const e=device.createCommandEncoder();e.copyBufferToBuffer(source,0,staging,0,count*4)
    device.queue.submit([e.finish()]);await staging.mapAsync(GPUMapMode.READ)
    const values=new Float32Array(staging.getMappedRange().slice(0));staging.unmap();staging.destroy();return values
  }
  function relative(a: Float32Array, b: Float32Array) {
    let error=0,scale=0
    for(let i=0;i<a.length;i++){error+=(a[i]-b[i])**2;scale+=b[i]**2}
    return Math.sqrt(error/Math.max(scale,1e-20))
  }
  // Capture pass-sized command buffers. Await between them to prevent Metal coalescing.
  function isolatedEncoder(pending: GPUCommandBuffer[]) {
    let current = device.createCommandEncoder()
    return new Proxy(current, {
      get(_, key) {
        if (key === 'beginComputePass') return (descriptor: GPUComputePassDescriptor) => {
          pending.push(current.finish())
          current = device.createCommandEncoder()
          const pass = current.beginComputePass(descriptor)
          return new Proxy(pass, { get(target, method) {
            if (method === 'end') return () => {
              target.end();pending.push(current.finish());current=device.createCommandEncoder()
            }
            const value=Reflect.get(target,method,target)
            return typeof value==='function'?value.bind(target):value
          } })
        }
        const value=Reflect.get(current,key,current)
        return typeof value==='function'?value.bind(current):value
      },
    })
  }
  const correctness = await extendedChecks(device)
  if (checksOnly) {
    await device.queue.onSubmittedWorkDone()
    if (errors.length) throw new Error(errors.join(';'))
    ;(window as any).studyResult = {passed:true,correctness,errors}
    document.querySelector('#status')!.textContent = 'All extended checks passed'
    device.destroy(); return
  }
  const cases = [
    { name:'flat',grid:33,image:256 },
    { name:'focus',grid:33,image:256 },
    { name:'ripple',grid:33,image:256 },
    { name:'ripple',grid:17,image:512 },
    { name:'ripple',grid:65,image:256 },
  ]
  const records: unknown[] = [], equivalence: unknown[] = []
  for(const c of cases) {
    const heights=Float32Array.from({length:c.grid*c.grid},(_,i)=>{
      const x=i%c.grid/(c.grid-1)-.5,y=Math.floor(i/c.grid)/(c.grid-1)-.5
      if(c.name==='focus')return .04-.065*(x*x+y*y)
      if(c.name==='ripple')return .02+.004*Math.sin(19*x)*Math.cos(17*y)
      return .01
    })
    const hb=buffer(heights),pg=buffer(Float32Array.from({length:c.image*c.image},(_,i)=>Math.sin(i*.013)/c.image**2)),hg=buffer(new Float32Array(heights.length))
    const input:ForwardInput={lens:{columns:c.grid,rows:c.grid,sizeX:.2,sizeY:.2,heightBuffer:hb},receiver:{width:c.image,height:c.image,sizeX:.4,sizeY:.4,z:.5},optics:{lensIOR:1.5,outsideIOR:1,incidentIrradiance:1}}
    const target={width:c.image,height:c.image,version:0,values:Float32Array.from({length:c.image*c.image},(_,i)=>{
      const x=(i%c.image+.5)/c.image-.5,y=(Math.floor(i/c.image)+.5)/c.image-.5
      return x*x+y*y<.15**2?1:0
    })}
    const queries=device.createQuerySet({type:'timestamp',count:32})
    const resolved=device.createBuffer({size:256,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC})
    const staging=device.createBuffer({size:256,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST})
    let cursor=0,recording=false,labels:string[]=[]
    const contexts=[]
    let referenceImage:Float32Array|null=null,referenceGradient:Float32Array|null=null
    for(const strategy of strategies) {
      const model=(await createForwardModel(device,input,{strategy,timestamps(stage){
        if(!recording)return undefined
        labels.push(stage);const begin=cursor;cursor+=2
        return {querySet:queries,beginningOfPassWriteIndex:begin,endOfPassWriteIndex:begin+1}
      }}))!
      const optimizer=(await createLensOptimizer(device,input,model,{...config.optimizer,lossLogInterval:0}))!
      const e=device.createCommandEncoder(),r=model.encode(e,input)
      model.encodeBackward!(e,input,r,pg,hg);device.queue.submit([e.finish()])
      if(r.caustic.kind!=='buffer')throw new Error('Expected buffer')
      const image=await read(r.caustic.buffer,c.image*c.image),gradient=await read(hg,heights.length)
      if(!referenceImage){referenceImage=image;referenceGradient=gradient}
      const imageError=relative(image,referenceImage),gradientError=relative(gradient,referenceGradient!)
      if(imageError>3e-4||gradientError>3e-4)throw new Error('Variant mismatch: '+JSON.stringify({c,strategy,imageError,gradientError}))
      equivalence.push({...c,strategy,imageError,gradientError})
      contexts.push({strategy,model,optimizer,phases:[] as Record<string,number>[],iteration:[] as Record<string,number>[]})
    }
    for(const mode of ['phases','iteration'] as const) {
      for(let round=0;round<3;round++) {
        const order=contexts.map((_,i)=>contexts[(i+round)%contexts.length])
        for(const context of order) {
          for(let sample=round===0?-10:0;sample<10;sample++) {
            recording=sample>=0;cursor=0;labels=[]
            const pending:GPUCommandBuffer[]=[]
            const e=mode==='phases'?isolatedEncoder(pending):device.createCommandEncoder()
            const start=performance.now()
            if(mode==='phases') {
              const r=context.model.encode(e,input)
              context.model.encodeBackward!(e,input,r,pg,hg)
            } else {
              // A fixed starting surface and reset Adam state keep comparisons matched.
              context.optimizer.setTarget({...target,version:++target.version})
              const update=context.optimizer.encodeStep(e,input)
              context.model.encode(e,{...input,lens:update.lens})
            }
            if(recording){e.resolveQuerySet(queries,0,cursor,resolved,0);e.copyBufferToBuffer(resolved,0,staging,0,cursor*8)}
            pending.push(e.finish())
            for(const commands of pending){device.queue.submit([commands]);await device.queue.onSubmittedWorkDone()}
            const submitAndWaitMs=performance.now()-start
            if(recording){
              await staging.mapAsync(GPUMapMode.READ)
              const ts=new BigUint64Array(staging.getMappedRange())
              const values:Record<string,number>={round,sample}
              for(let j=0;j<cursor;j+=2)if(ts[j+1]<ts[j])throw new Error('Invalid GPU timestamps')
              if(mode==='iteration'){
                values.gpuMs=Number(ts[cursor-1]-ts[0])/1e6
                values.submitAndWaitMs=submitAndWaitMs
              } else labels.forEach((label,j)=>values[label]=(values[label]||0)+Number(ts[j*2+1]-ts[j*2])/1e6)
              context[mode].push(values);staging.unmap()
            }
          }
        }
      }
    }
    for(const context of contexts) {
      const summarize=(samples:Record<string,number>[])=>Object.fromEntries(Object.keys(samples[0]).filter(k=>k!=='round'&&k!=='sample').map(k=>{
        const values=samples.map(s=>s[k]);return [k,{median:quantile(values,.5),p25:quantile(values,.25),p75:quantile(values,.75),mean:values.reduce((s,v)=>s+v,0)/values.length}]
      }))
      const record={...c,strategy:context.strategy,phaseSummary:summarize(context.phases),iterationSummary:summarize(context.iteration),phases:context.phases,iteration:context.iteration}
      records.push(record);log({...c,strategy:context.strategy,phaseSummary:record.phaseSummary,iterationSummary:record.iterationSummary})
      context.optimizer.destroy();context.model.destroy()
    }
    hb.destroy();pg.destroy();hg.destroy();queries.destroy();resolved.destroy();staging.destroy()
  }
  await device.queue.onSubmittedWorkDone()
  if(errors.length)throw new Error(errors.join(';'))
  const result={passed:true,date:new Date().toISOString(),adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture,description:adapter.info.description},userAgent:navigator.userAgent,method:{warmup:10,samples:30,blocks:3,order:'rotated between blocks',phaseTiming:'separate submissions with GPU completion between phases',iterationTiming:'first optics pass to final preview; fixed input and fresh Adam state; initial resets and final readback excluded'},correctness,equivalence,records,errors}
  ;(window as any).studyResult=result
  const link=document.createElement('a');link.textContent='Download raw results';link.download='a-study.json';link.href=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));document.body.appendChild(link)
  document.querySelector('#status')!.textContent='All checks and measurements completed'
  device.destroy()
}
run().catch(e=>{log(String(e));(window as any).studyResult={passed:false,error:String(e)}})
