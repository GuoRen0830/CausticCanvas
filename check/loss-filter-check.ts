import shader from '../src/optimization/shaders/optimizer.wgsl?raw'
async function run() {
 const adapter=(await navigator.gpu.requestAdapter())!, device=await adapter.requestDevice()
 const errors:string[]=[];device.addEventListener('uncapturederror',e=>errors.push(e.error.message))
 const width=9,height=7,n=width*height,scale=1/n
 const residual=Float32Array.from({length:n},(_,i)=>Math.sin(i*.7)+.2)
 const make=(size:number,uniform=false)=>device.createBuffer({size,usage:(uniform?GPUBufferUsage.UNIFORM:GPUBufferUsage.STORAGE)|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC})
 const params=make(80,true),state=make(16),input=make(n*4),a=make(n*4),b=make(n*4)
 const entries=[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform' as const}},...[2,8,9,10].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage' as const}}))]
 const layout=device.createBindGroupLayout({entries})
 const group=device.createBindGroup({layout,entries:[{binding:0,resource:{buffer:params}},{binding:2,resource:{buffer:input}},{binding:8,resource:{buffer:state}},{binding:9,resource:{buffer:a}},{binding:10,resource:{buffer:b}}]})
 const module=device.createShaderModule({code:shader})
 const pipelines=await Promise.all(['filterResidualX','filterResidualY','filterGradientY','filterGradientX'].map(entryPoint=>device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint}})))
 const results=[]
 for(const sigma of [0,1.5,4]) {
 const data=new ArrayBuffer(80),view=new DataView(data);view.setUint32(8,width,true);view.setUint32(12,height,true);view.setFloat32(24,1,true);view.setFloat32(28,scale,true);view.setUint32(72,3000,true);view.setFloat32(76,sigma,true)
 device.queue.writeBuffer(params,0,data);device.queue.writeBuffer(input,0,residual)
 const dst=device.createBuffer({size:n*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST})
 const enc=device.createCommandEncoder(),pass=enc.beginComputePass();pass.setBindGroup(0,group);for(const p of pipelines){pass.setPipeline(p);pass.dispatchWorkgroups(1)}pass.end();enc.copyBufferToBuffer(input,0,dst,0,n*4);device.queue.submit([enc.finish()]);await dst.mapAsync(GPUMapMode.READ)
 const gpu=new Float32Array(dst.getMappedRange().slice(0));dst.unmap();dst.destroy()
 const radius=Math.ceil(3*sigma),weights=Array.from({length:radius*2+1},(_,i)=>Math.exp(-.5*(i-radius)**2/Math.max(sigma*sigma,1e-8))),sum=weights.reduce((a,b)=>a+b,0)
 function blur(x:number[],axis:number){return x.map((_,i)=>{let s=0;for(let j=-radius;j<=radius;j++){const px=i%width+(axis===0?j:0),py=Math.floor(i/width)+(axis===1?j:0);if(px>=0&&px<width&&py>=0&&py<height)s+=x[py*width+px]*weights[j+radius]/sum}return s})}
 function loss(x:number[]){return blur(blur(x,0),1).reduce((s,v)=>s+v*v*scale,0)}
 const x=Array.from(residual),eps=1e-4;let maxError=0
 for(let i=0;i<n;i++){x[i]+=eps;const plus=loss(x);x[i]-=2*eps;const minus=loss(x);x[i]+=eps;maxError=Math.max(maxError,Math.abs(gpu[i]-(plus-minus)/(2*eps)))}
 if(maxError>1e-6)throw new Error(`Gradient error ${maxError} at sigma ${sigma}`)
 results.push({sigma,maxError})
 }
 await device.queue.onSubmittedWorkDone();if(errors.length)throw new Error(errors.join(';'))
 ;(window as any).filterResult={passed:true,results,errors};document.querySelector('#output')!.textContent=JSON.stringify((window as any).filterResult,null,2)
 device.destroy()
}
run().catch(e=>{(window as any).filterResult={passed:false,error:String(e)};document.querySelector('#output')!.textContent=String(e)})
