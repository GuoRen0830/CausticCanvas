// Bake the difference between the published dense scans and the realtime meshes.
// Run from the project root with "node scripts/bake-cove-scan-normals.mjs".
// This is an offline geometry bake. No color image is used as height data.
import { readFile, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { crc32, deflateSync } from 'node:zlib'
import { createServer } from 'vite'
import { sceneSourceCache } from './scene-source-cache.mjs'

const root = fileURLToPath(new URL('../', import.meta.url))
const scene = join(root, 'scene', 'smugglers-cove')
const catalog = JSON.parse(await readFile(join(scene, 'source-catalog.json'), 'utf8'))
const sources = sceneSourceCache(join(root, 'node_modules', '.cache', 'caustic-source-cache'))
const resolution = 1024

async function readGeometry(id, budget, readPreviewGltf) {
  const source = catalog[id].gltf['1k'].gltf
  const response = await sources.fetch(source.url)
  if (!response.ok) throw new Error('Cannot read ' + source.url)
  const document = await response.json()
  // These two scans use one atlas on TEXCOORD_0. Retain all original vertex
  // data and nodes, but do not download images just to bake geometric normals.
  document.materials = document.materials.map(() => ({}))
  const binary = new Map()
  for (const buffer of document.buffers) {
    const file = source.include[buffer.uri]
    const result = await sources.fetch(file.url)
    if (!result.ok) throw new Error('Cannot read ' + file.url)
    binary.set(buffer.uri, await result.blob())
  }
  const model = { document, base: new URL(source.url) }
  const options = { pbr: true, readUri: async uri => binary.get(uri) }
  const low = await readPreviewGltf(model, undefined, { ...options, maxTriangles: budget, simplifyToBudget: true })
  const high = await readPreviewGltf(model, undefined, options)
  return { low: low.mesh, high: high.mesh }
}

// UVs in SceneMesh use bottom-left OBJ space. Pixel rows here use the
// top-left glTF normal-map convention, as does the image bound in Room.
function rasterize(mesh, stride, shade) {
  const { indices, uvs } = mesh
  for (let face = 0; face < indices.length; face += 3) {
    const ia = indices[face], ib = indices[face+1], ic = indices[face+2]
    const ax = uvs[2*ia] * resolution, ay = (1 - uvs[2*ia+1]) * resolution
    const bx = uvs[2*ib] * resolution, by = (1 - uvs[2*ib+1]) * resolution
    const cx = uvs[2*ic] * resolution, cy = (1 - uvs[2*ic+1]) * resolution
    const du1 = bx-ax, dv1 = by-ay, du2 = cx-ax, dv2 = cy-ay
    const det = du1*dv2 - du2*dv1
    if (Math.abs(det) < 1e-9) continue
    const left = Math.max(0, Math.ceil(Math.min(ax,bx,cx)-0.5))
    const right = Math.min(resolution-1, Math.floor(Math.max(ax,bx,cx)-0.5))
    const top = Math.max(0, Math.ceil(Math.min(ay,by,cy)-0.5))
    const bottom = Math.min(resolution-1, Math.floor(Math.max(ay,by,cy)-0.5))
    for (let y = top; y <= bottom; y++) for (let x = left; x <= right; x++) {
      const dx = x+0.5-ax, dy = y+0.5-ay
      const b = (dx*dv2-du2*dy) / det, c = (du1*dy-dx*dv1) / det, a = 1-b-c
      if (Math.min(a,b,c) >= -1e-7) {
        shade(y*resolution+x, ia*stride, ib*stride, ic*stride, a,b,c,du1,dv1,du2,dv2,Math.sign(det))
      }
    }
  }
}

function bake(high, low, stride) {
  const size = resolution ** 2
  const highFields = new Float32Array(size*6), owner = new Uint32Array(size)
  const hv = high.vertices
  rasterize(high, stride, (pixel, va,vb,vc, a,b,c) => {
    owner[pixel] = hv[va+10]
    for (let i=0; i<6; i++) highFields[6*pixel+i] = a*hv[va+i]+b*hv[vb+i]+c*hv[vc+i]
  })
  const rgb = Buffer.alloc(size*3), covered = new Uint8Array(size)
  for (let i=0; i<size; i++) { rgb[3*i]=128; rgb[3*i+1]=128; rgb[3*i+2]=255 }
  const lv = low.vertices, bounds = [0,1,2].map(() => [Infinity,-Infinity])
  for (let i=0; i<lv.length; i+=stride) for (let k=0; k<3; k++) {
    bounds[k][0] = Math.min(bounds[k][0],lv[i+k])
    bounds[k][1] = Math.max(bounds[k][1],lv[i+k])
  }
  // A collapsed seam or another chart can cover the same texel. Reject
  // correspondences far away in source space or on the opposite-facing sheet.
  const maxDistance = 0.025 * Math.max(...bounds.map(([a,b])=>b-a))
  rasterize(low, stride, (pixel, va,vb,vc, a,b,c,du1,dv1,du2,dv2,sign) => {
    if (owner[pixel] !== lv[va+10]) return
    const dense = 6*pixel
    let dist2 = 0
    for (let k=0; k<3; k++) dist2 += (a*lv[va+k]+b*lv[vb+k]+c*lv[vc+k]-highFields[dense+k])**2
    if (dist2 > maxDistance**2) return
    let nx=a*lv[va+3]+b*lv[vb+3]+c*lv[vc+3]
    let ny=a*lv[va+4]+b*lv[vb+4]+c*lv[vc+4]
    let nz=a*lv[va+5]+b*lv[vb+5]+c*lv[vc+5]
    const length = Math.hypot(nx,ny,nz)
    if (!length) return
    nx/=length; ny/=length; nz/=length
    const hx=highFields[dense+3], hy=highFields[dense+4], hz=highFields[dense+5]
    const hLength = Math.hypot(hx,hy,hz)
    const cosine = (nx*hx+ny*hy+nz*hz)/hLength
    if (!(cosine >= 0.42)) return
    const abx=lv[vb]-lv[va], aby=lv[vb+1]-lv[va+1], abz=lv[vb+2]-lv[va+2]
    const acx=lv[vc]-lv[va], acy=lv[vc+1]-lv[va+1], acz=lv[vc+2]-lv[va+2]
    // Use the same orthonormal derivative basis as scene.wgsl. Here V
    // already follows the glTF direction, so the installed MTL carries Y=-1.
    let tx=(abx*dv2-acx*dv1)*sign, ty=(aby*dv2-acy*dv1)*sign, tz=(abz*dv2-acz*dv1)*sign
    const nt=nx*tx+ny*ty+nz*tz
    tx-=nx*nt; ty-=ny*nt; tz-=nz*nt
    const tLength=Math.hypot(tx,ty,tz)
    if (!tLength) return
    tx/=tLength; ty/=tLength; tz/=tLength
    const bx=ny*tz-nz*ty, by=nz*tx-nx*tz, bz=nx*ty-ny*tx
    const handedness = Math.sign(sign*(bx*(du1*acx-du2*abx)+by*(du1*acy-du2*aby)+bz*(du1*acz-du2*abz)))
    const byte = value => Math.round(127.5*(1+Math.min(1, Math.max(-1, value))))
    rgb[3*pixel]=byte((tx*hx+ty*hy+tz*hz)/hLength)
    rgb[3*pixel+1]=byte(handedness*(bx*hx+by*hy+bz*hz)/hLength)
    rgb[3*pixel+2]=byte(cosine)
    covered[pixel]=1
  })
  const valid = covered.reduce((sum,value)=>sum+value,0)
  // Extend island borders for bilinear/mip filtering. Unmatched regions
  // beyond this margin retain the neutral tangent normal.
  for (let pass=1; pass<=8; pass++) {
    for (let y=0; y<resolution; y++) for (let x=0; x<resolution; x++) {
      const pixel=y*resolution+x
      if (covered[pixel]) continue
      const neighbors = [x>0 ? pixel-1 : -1, x+1<resolution ? pixel+1 : -1,
        y>0 ? pixel-resolution : -1, y+1<resolution ? pixel+resolution : -1]
      const neighbor = neighbors.find(n => n>=0 && covered[n]>0 && covered[n]<=pass)
      if (neighbor === undefined) continue
      for (let k=0;k<3;k++) rgb[3*pixel+k]=rgb[3*neighbor+k]
      covered[pixel] = pass+1
    }
  }
  return { rgb, valid }
}

function writePng(rgb) {
  const chunk = (name, bytes) => {
    const data = Buffer.concat([Buffer.from(name), bytes]), size = Buffer.alloc(4), checksum=Buffer.alloc(4)
    size.writeUInt32BE(bytes.length); checksum.writeUInt32BE(crc32(data))
    return Buffer.concat([size,data,checksum])
  }
  const header=Buffer.alloc(13)
  header.writeUInt32BE(resolution,0); header.writeUInt32BE(resolution,4); header[8]=8; header[9]=2
  const row=3*resolution, scanlines=Buffer.alloc((row+1)*resolution)
  for (let y=0; y<resolution; y++) rgb.copy(scanlines,y*(row+1)+1,y*row,(y+1)*row)
  return Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]),chunk('IHDR',header),
    chunk('IDAT',deflateSync(scanlines,{level:9})),chunk('IEND',Buffer.alloc(0))])
}

const vite = await createServer({ root, configFile:false, server:{middlewareMode:true,hmr:false,watch:null}, appType:'custom' })
try {
  const { readPreviewGltf } = await vite.ssrLoadModule('/src/geometry/preview-gltf.ts')
  const { sceneVertexStride } = await vite.ssrLoadModule('/src/rendering/scene-mesh.ts')
  const { coveScanBudgets } = await vite.ssrLoadModule('/src/geometry/cove-scan-budgets.ts')
  for (const [id, filename] of [['treasure_chest','chest-scan-normal.png'], ['coast_land_rocks_02','reef-scan-normal.png']]) {
    const { high,low } = await readGeometry(id,coveScanBudgets[id],readPreviewGltf)
    const { rgb, valid } = bake(high,low,sceneVertexStride)
    if (valid < 0.2*resolution**2) throw new Error('Low UV coverage on ' + id + ': ' + valid)
    await writeFile(join(scene,filename), writePng(rgb))
    console.log(filename + ': ' + high.indices.length/3 + ' source triangles, ' + low.indices.length/3 +
      ' preview triangles, ' + valid + ' projected texels')
  }
} finally { await vite.close() }
