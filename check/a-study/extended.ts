import { createForwardModel } from '../../src/optics/forward-model'
import { reference } from '../optics-reference'
import type { ForwardInput } from '../../src/contracts'

export async function extendedChecks(device: GPUDevice) {
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  const records: unknown[] = []
  function buffer(data: Float32Array) {
    const b = device.createBuffer({ size:data.byteLength, usage:storage })
    device.queue.writeBuffer(b,0,data); return b
  }
  async function read(source: GPUBuffer, count: number) {
    const b=device.createBuffer({size:count*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST})
    const e=device.createCommandEncoder();e.copyBufferToBuffer(source,0,b,0,count*4)
    device.queue.submit([e.finish()]);await b.mapAsync(GPUMapMode.READ)
    const result=new Float32Array(b.getMappedRange().slice(0));b.unmap();b.destroy();return result
  }
  function relative(a: ArrayLike<number>, b: ArrayLike<number>) {
    let error=0,scale=0
    for(let i=0;i<a.length;i++){error+=(a[i]-b[i])**2;scale+=b[i]**2}
    return Math.sqrt(error/Math.max(scale,1e-20))
  }
  const cases = [
    { name:'non-square ripple', columns:5,rows:4,width:31,height:27,shape:'ripple',fd:true },
    { name:'saddle',columns:4,rows:5,width:29,height:23,shape:'saddle',fd:true },
    { name:'near receiver',columns:3,rows:3,width:21,height:19,shape:'near' },
    { name:'at receiver',columns:3,rows:3,width:21,height:19,shape:'at',dark:true },
    { name:'behind receiver',columns:3,rows:3,width:21,height:19,shape:'behind',dark:true },
    { name:'near critical below',columns:3,rows:3,width:25,height:23,shape:'criticalBelow' },
    { name:'critical above',columns:3,rows:3,width:25,height:23,shape:'criticalAbove',dark:true },
    { name:'all nonfinite',columns:3,rows:3,width:21,height:19,shape:'nan',dark:true },
    { name:'all infinite',columns:3,rows:3,width:21,height:19,shape:'inf',dark:true },
    { name:'zero illumination',columns:3,rows:3,width:21,height:19,shape:'ripple',dark:true,irradiance:0 },
    { name:'partial screen clipping',columns:4,rows:3,width:25,height:23,shape:'tilt',fd:true },
    { name:'pixel boundary',columns:3,rows:3,width:16,height:16,shape:'flat' },
  ]
  for(const c of cases) {
   for (const strategy of ['baseline','cached','cooperative','striped'] as const) {
    const h=Float32Array.from({length:c.columns*c.rows},(_,i)=>{
      const x=i%c.columns/(c.columns-1)-.5,y=Math.floor(i/c.columns)/(c.rows-1)-.5
      switch(c.shape){
        case 'ripple':return .02+.0013*Math.sin(5*x+.17)*Math.cos(4*y+.23)
        case 'saddle':return .02+.004*x*y+.0003*x
        case 'near':return .4999
        case 'at':return .5
        case 'behind':return .6
        case 'criticalBelow':return .15+.2*x*.893
        case 'criticalAbove':return .15+.2*x*.896
        case 'nan':return NaN
        case 'inf':return Infinity
        case 'tilt':return .09+.12*x+.00011*y
        default:return .01
      }
    })
    const hb=buffer(h),count=c.width*c.height
    const input:ForwardInput={
      lens:{columns:c.columns,rows:c.rows,sizeX:.2,sizeY:.18,heightBuffer:hb},
      receiver:{width:c.width,height:c.height,sizeX:.4,sizeY:.4,z:.5},
      optics:{lensIOR:1.5,outsideIOR:1,incidentIrradiance:c.irradiance??1},
    }
    const model=(await createForwardModel(device,input,{strategy}))!
    const pixel=Float32Array.from({length:count},(_,i)=>Math.sin(i*.17)+Math.cos(i*.31))
    const pg=buffer(pixel),hg=buffer(new Float32Array(h.length).fill(999))
    const e=device.createCommandEncoder(),result=model.encode(e,input)
    model.encodeBackward!(e,input,result,pg,hg);device.queue.submit([e.finish()])
    if(result.caustic.kind!=='buffer')throw new Error('Expected buffer')
    const image=await read(result.caustic.buffer,count),gradient=await read(hg,h.length)
    if(!image.every(v=>Number.isFinite(v)&&v>=0)||!gradient.every(Number.isFinite))throw new Error(c.name+': nonfinite output')
    if(c.dark&&(image.some(v=>v!==0)||gradient.some(v=>v!==0)))throw new Error(c.name+': expected zero image and gradient')
    const beams=await read(result.beams.buffer,2*(c.columns-1)*(c.rows-1)*4)
    const transmittedPower=beams.reduce((sum,v,i)=>sum+(i%4===3?v:0),0)
    if(c.shape==='criticalBelow'&&transmittedPower<=0)throw new Error('Below-critical light must exit the lens')
    if(c.shape==='criticalAbove'&&transmittedPower!==0)throw new Error('Above-critical light must be inactive')
    let imageError=0,beamError=0,gradientError:number|null=null
    if(h.every(Number.isFinite)){
      const cpu=reference(input,h)
      imageError=relative(image,cpu.image)
      beamError=relative(beams,cpu.beams)
      if(beamError>5e-4)throw new Error(c.name+': beam mismatch '+beamError)
      if(imageError>5e-4)throw new Error(c.name+': CPU mismatch '+imageError)
    }
    if(c.fd){
      const objective=(values:number[])=>reference(input,values).image.reduce((sum,v,i)=>sum+v*pixel[i],0)
      const finite=Array.from(h,(_,i)=>{const a=Array.from(h),b=Array.from(h);a[i]+=1e-7;b[i]-=1e-7;return(objective(a)-objective(b))/2e-7})
      gradientError=relative(gradient,finite)
      if(gradientError>.005)throw new Error(c.name+': gradient mismatch '+gradientError)
    }
    const receivedPower=image.reduce((s,v)=>s+v,0)*input.receiver.sizeX*input.receiver.sizeY/count
    const incidentPower=input.optics.incidentIrradiance*.2*.18
    if(receivedPower>incidentPower+1e-5)throw new Error(c.name+': power gain')
    // Invalid aliases and resized metadata must be rejected before dispatch.
    let aliasRejected=false,resizeRejected=false
    try{model.encodeBackward!(device.createCommandEncoder(),input,result,pg,hb)}catch{aliasRejected=true}
    try{model.encode(device.createCommandEncoder(),{...input,receiver:{...input.receiver,width:c.width+1}})}catch{resizeRejected=true}
    if(!aliasRejected||!resizeRejected)throw new Error('Input validation failed')
    const record={name:c.name,strategy,imageError,beamError,gradientError,transmittedPower,receivedPower,incidentPower,aliasRejected,resizeRejected}
    records.push(record);console.log('PASS',JSON.stringify(record))
    model.destroy();hb.destroy();pg.destroy();hg.destroy()
  }
  }
  return records
}
