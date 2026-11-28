// Offline authoring tool. Set FFMPEG to the executable path if it is not on PATH.
// All generated files are local estimates or artist-authored surfaces. None are
// registered as Poly Haven originals. The application ships the baked PNGs.
import { createHash } from 'node:crypto'
import { readdirSync, readFileSync, mkdirSync } from 'node:fs'
import { basename, join } from 'node:path'
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('../scene/smugglers-cove/', import.meta.url))
const target = join(root, 'detail')
const tau = 2 * Math.PI
const clamp = (n, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, n))
const frac = x => x - Math.floor(x)
const mix = (a, b, t) => a + (b - a) * t
function smooth(a, b, x) {
  const t = clamp((x - a) / (b - a))
  return t * t * (3 - 2 * t)
}

function ffmpeg(args, input) {
  const out = spawnSync(process.env.FFMPEG || 'ffmpeg', ['-v', 'error', ...args], {
    input, maxBuffer: 64 * 1024 * 1024,
  })
  if (out.error || out.status !== 0) throw new Error(out.error?.message || out.stderr.toString())
  return out.stdout
}

function readImage(file, w, h = w) {
  const bytes = ffmpeg(['-i', file, '-vf', `scale=${w}:${h}:flags=lanczos`,
    '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'])
  if (bytes.length !== 3 * w * h) throw new Error('Invalid image extent for ' + file)
  return bytes
}

function write(name, w, h, channels, bytes) {
  ffmpeg(['-y', '-f', 'rawvideo', '-pixel_format', channels === 1 ? 'gray' : 'rgb24',
    '-video_size', `${w}x${h}`, '-i', 'pipe:0', '-frames:v', '1',
    '-compression_level', '9', '-pred', 'mixed', join(target, name + '.png')], bytes)
}

const byte = x => Math.round(255 * clamp(x))
const mod = (i, n) => (i + n) % n
// A separable, periodic low-pass removes the photographed light gradient.
function blur(input, w, h, radius) {
  const temp = new Float32Array(input.length), out = new Float32Array(input.length)
  const span = 2 * radius + 1
  for (let y = 0; y < h; y++) {
    let sum = 0
    for (let i = -radius; i <= radius; i++) sum += input[y*w + mod(i,w)]
    for (let x = 0; x < w; x++) {
      temp[y*w+x] = sum / span
      sum += input[y*w+mod(x+radius+1,w)] - input[y*w+mod(x-radius,w)]
    }
  }
  for (let x = 0; x < w; x++) {
    let sum = 0
    for (let i = -radius; i <= radius; i++) sum += temp[mod(i,h)*w+x]
    for (let y = 0; y < h; y++) {
      out[y*w+x] = sum / span
      sum += temp[mod(y+radius+1,h)*w+x] - temp[mod(y-radius,h)*w+x]
    }
  }
  return out
}

function heightNormals(height, w, h, strength) {
  // The engine uses a bottom-left OBJ tangent basis. The vertical derivative
  // has the opposite sign to U because the decoded image has a top-left origin.
  const rgb = Buffer.alloc(3*w*h)
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const dx = (height[y*w+mod(x-1,w)] - height[y*w+mod(x+1,w)]) * strength
    const dy = (height[mod(y+1,h)*w+x] - height[mod(y-1,h)*w+x]) * strength
    const len = Math.hypot(1, dx, dy), i = 3*(y*w+x)
    rgb[i] = byte(0.5 + 0.5*dx/len)
    rgb[i+1] = byte(0.5 + 0.5*dy/len)
    rgb[i+2] = byte(0.5 + 0.5/len)
  }
  return rgb
}

function sandAO(height, w) {
  const ao = Buffer.alloc(w*w)
  const directions = [[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,1],[1,-1],[-1,-1]]
  for (let y = 0; y < w; y++) for (let x = 0; x < w; x++) {
    let occlusion = 0
    for (const [dx,dy] of directions) {
      let horizon = 0
      for (const distance of [2,5,12]) {
        const rim = height[mod(y+dy*distance,w)*w+mod(x+dx*distance,w)]
        horizon = Math.max(horizon, (rim - height[y*w+x]) / (distance * Math.hypot(dx,dy)))
      }
      occlusion += horizon / Math.sqrt(1 + horizon*horizon)
    }
    ao[y*w+x] = byte(1 - Math.min(0.30, occlusion/directions.length))
  }
  return ao
}

function photoDetail(stem, file, w, slope, { normals = true, roughness = true } = {}) {
  const rgb = readImage(file,w), luma = new Float32Array(w*w)
  for (let i = 0; i < luma.length; i++) {
    // Log intensity suppresses slowly varying baked illumination in these
    // relief estimates. Broad color changes never directly become height.
    luma[i] = Math.log(24 + 0.23*rgb[3*i] + 0.68*rgb[3*i+1] + 0.09*rgb[3*i+2])
  }
  const fine = blur(luma,w,w,1), mid = blur(fine,w,w,4), broad = blur(mid,w,w,17)
  const height = new Float32Array(w*w)
  let meanAbs = 0
  for (let i = 0; i < height.length; i++) {
    height[i] = 0.24*(luma[i]-fine[i]) + (fine[i]-mid[i]) + 0.60*(mid[i]-broad[i])
    meanAbs += Math.abs(height[i])
  }
  meanAbs = Math.max(0.001, meanAbs / height.length)
  for (let i = 0; i < height.length; i++) height[i] = 2 * Math.tanh(height[i] / (2*meanAbs))
  if (normals) write(stem+'-normal', w,w,3, heightNormals(height,w,w,slope))
  if (roughness) {
    const values = Buffer.from(height.map(h => byte(0.85 - 0.125*Math.tanh(h))))
    write(stem+'-roughness',w,w,1,values)
  }
  if (stem === 'sand') write('sand-ao',w,w,1,sandAO(height,w))
  return { rgb, height }
}

// Stable identification across rebakes that renumber the OBJ's map_Kd files.
function localScans() {
  const catalog = JSON.parse(readFileSync(join(root,'source-catalog.json'),'utf8'))
  const signatures = new Map()
  function visit(node) {
    if (!node || typeof node !== 'object') return
    if (node.url && /_diff_1k\.jpg$/i.test(node.url)) signatures.set(node.md5, basename(new URL(node.url).pathname))
    for (const child of Object.values(node)) visit(child)
  }
  visit(catalog)
  const scans = new Map(), directory = join(root,'preview','textures')
  for (const name of readdirSync(directory).filter(n => /\.jpg$/i.test(n))) {
    const file = join(directory,name), id = createHash('md5').update(readFileSync(file)).digest('hex')
    if (signatures.has(id)) scans.set(signatures.get(id), file)
  }
  return name => {
    const file = scans.get(name+'_diff_1k.jpg')
    if (!file) throw new Error('Missing original 1K base color for '+name)
    return file
  }
}

function chestMetal(file) {
  const w = 1024
  const { rgb, height } = photoDetail('chest',file,w,0,{normals:false,roughness:false})
  const mr = Buffer.alloc(3*w*w)
  for (let i = 0; i < height.length; i++) {
    const r = rgb[3*i], g = rgb[3*i+1], b = rgb[3*i+2]
    // On this atlas the timber is warm brown; the straps, hinges, and lock
    // use grey iron, interrupted by ochre rust. Limit the mask to those regions.
    const warm = (r - b) / Math.max(r, 28)
    const iron = 1 - smooth(0.26, 0.43, warm)
    const polished = smooth(52, 135, (r + 2*g + b)/4)
    mr[3*i] = 255
    mr[3*i+1] = byte(mix(0.79 - 0.075*Math.tanh(height[i]), 0.69 - 0.36*polished, iron))
    mr[3*i+2] = byte(iron * (0.69 + 0.29*polished))
  }
  write('chest-mr',w,w,3,mr)
}

function pierMetal(rgb) {
  // The scanned plank atlas dedicates a small L-shaped island to grey iron
  // hardware. The four brown patches in this island are oxidation. The poles
  // and the timber outside this island are dielectrics.
  const w=1024, mask=Buffer.alloc(w*w)
  const island = [[811,168],[849,164],[920,165],[950,177],[952,216],
    [943,230],[927,249],[902,249],[902,206],[811,206]]
  for (let y=164; y<=249; y++) for (let x=811; x<=952; x++) {
    let inside=false, edgeDistance=Infinity
    for (let i=0; i<island.length; i++) {
      const [ax,ay]=island[i], [bx,by]=island[(i+1)%island.length]
      if ((ay>y)!==(by>y) && x<(bx-ax)*(y-ay)/(by-ay)+ax) inside=!inside
      const dx=bx-ax, dy=by-ay
      const t=clamp(((x-ax)*dx+(y-ay)*dy)/(dx*dx+dy*dy))
      edgeDistance=Math.min(edgeDistance,Math.hypot(x-ax-t*dx,y-ay-t*dy))
    }
    if (!inside) continue
    const i=y*w+x, r=rgb[3*i], b=rgb[3*i+2]
    const rust=smooth(0.07,0.34,(r-b)/Math.max(24,r))
    mask[i]=byte(mix(0.90,0.08,rust)*smooth(0,2,edgeDistance))
  }
  write('pier-planks-metallic',w,w,1,mask)
}

function lattice(x,y) {
  let n = Math.imul(x+37, 0x45d9f3b) ^ Math.imul(y+11, 0x119de1f3)
  n = Math.imul(n ^ (n >>> 16), 0x45d9f3b)
  return ((n ^ (n >>> 16)) >>> 0) / 2**32
}

function noise(u,v,nx,ny) {
  const x = frac(u)*nx, y = frac(v)*ny, ix = Math.floor(x), iy = Math.floor(y)
  const a = smooth(0,1,frac(x)), b = smooth(0,1,frac(y))
  const xn = (ix+1)%nx, yn=(iy+1)%ny
  return mix(mix(lattice(ix,iy),lattice(xn,iy),a),mix(lattice(ix,yn),lattice(xn,yn),a),b)
}

// Each prototype describes a surface the project modeled. UVs on ropes and
// beams run along U. Shells and blades have their own radial/longitudinal UVs.
const surfaces = {
  rope(u,v) {
    const twist = tau*(8*u-24*v), strands = (Math.cos(twist)+1)*0.5
    const fibre = Math.sin(5*twist+tau*u) * (0.25 + 0.75*strands)
    return [0.65*strands + 0.055*fibre, 0.83+0.10*(1-strands)]
  },
  grass(u,v) {
    const spine = Math.exp(-((u-0.5)**2)/0.004) * (0.65+0.35*Math.sin(Math.PI*v))
    const ridges = Math.cos(tau*(6*u + 0.15*Math.sin(Math.PI*v)))
    return [0.86*spine+0.10*ridges, 0.92-0.22*spine-0.035*ridges]
  },
  glass(u,v) {
    const flow = Math.sin(tau*(2*v+0.23*Math.sin(tau*u)))
    return [0.18*flow+0.035*Math.sin(tau*(3*u+v)), 0.91+0.065*flow]
  },
  enamel(u,v) {
    const peel = noise(u,v,40,40), undercoat = noise(u,v,8,8)
    return [0.026*peel+0.085*undercoat, 0.79+0.10*peel+0.065*undercoat]
  },
  fish(u,v) {
    const row = Math.floor(v*56), along = frac(38*u + 0.5*(row%2)) - 0.5
    const edge = frac(v*56) - (0.88 - 1.65*along*along)
    const ridge = Math.exp(-edge*edge/0.037)
    return [0.23*ridge, 0.71+0.22*ridge]
  },
  cork(u,v) {
    const pores = smooth(0.56,0.89,noise(u,v,48,54)), flecks = noise(u,v,15,17)
    return [0.20*flecks - 0.27*pores, 0.77+0.18*pores+0.04*flecks]
  },
  'fired-clay'(u,v) {
    const pores = smooth(0.63,0.91,noise(u,v,49,53))
    const grain = noise(u,v,90,86), tooling = noise(u,v,8,31)
    return [0.14*tooling + 0.09*grain - 0.26*pores, 0.80+0.10*tooling+0.09*pores]
  },
  'sea-star'(u,v) {
    const ossicles = smooth(0.38,0.85,noise(u,v,34,32))
    const skin = noise(u,v,79,80)
    return [0.16*ossicles+0.045*skin, 0.74+0.14*ossicles+0.07*skin]
  },
  shell(u,v) {
    const x = u-0.5, y = v-0.5, r = Math.hypot(x,y)
    const ribs = Math.cos(18*Math.atan2(y,x))
    const ring = 0.5 + 0.5*Math.cos(tau*(18*r + 0.12*ribs))
    return [smooth(0.025,0.12,r)*(0.11*ring+0.08*ribs), 0.79+0.16*ring]
  },
  fittings(u,v) {
    const patina = smooth(0.38,0.83,0.7*noise(u,v,9,9)+0.3*noise(u,v,28,28))
    const brush = noise(u,v,8,96)
    return [0.085*patina+0.058*brush, 0.64+0.27*patina+0.065*brush, 0.98-0.69*patina]
  },
}

function makeSurface(stem, scale) {
  const w = 256, height = new Float32Array(w*w), packed = stem === 'fittings'
  const channels = packed ? 3 : 1, surface = Buffer.alloc(channels*w*w)
  for (let y=0; y<w; y++) for (let x=0; x<w; x++) {
    const i=y*w+x, [h, rough, metal] = surfaces[stem]((x+0.5)/w,1-(y+0.5)/w)
    height[i]=h
    if (packed) {
      surface[3*i]=255; surface[3*i+1]=byte(rough); surface[3*i+2]=byte(metal)
    } else surface[i]=byte(rough)
  }
  write(stem+'-normal',w,w,3,heightNormals(height,w,w,scale))
  write(stem+(packed ? '-mr' : '-roughness'),w,w,channels,surface)
}

mkdirSync(target,{recursive:true})
const scan = localScans()
photoDetail('sand',scan('coast_sand_01'),1024,0.44)
const planks = photoDetail('pier-planks',scan('modular_wooden_pier_planks'),1024,0.48)
pierMetal(planks.rgb)
photoDetail('pier-poles',scan('modular_wooden_pier_poles'),1024,0.54)
photoDetail('reef',join(root,'rock-shore-color.jpg'),1024,0,{normals:false})
chestMetal(scan('treasure_chest'))
photoDetail('timber',join(root,'quay-timber.png'),256,0.36)
photoDetail('sailcloth',join(root,'lug-sailcloth.png'),256,0.29)
photoDetail('linen',join(root,'market-linen.png'),192,0.32)
photoDetail('stone',join(root,'quay-limestone.png'),256,0.36,{roughness:false})
for (const [name,scale] of Object.entries({
  rope:0.86, grass:2.5, glass:0.8, enamel:1.65, fish:0.9, cork:1.45, shell:1.9, fittings:1.9,
  'fired-clay':1.65, 'sea-star':1.1,
})) makeSurface(name,scale)

// The existing water normal atlas already describes the direction of capillary
// ripples. Accumulated unresolved slopes broaden the lobe in rougher patches.
const waterPath = join(root,'tide-normal.png')
const waterHeader = readFileSync(waterPath)
const waterWidth = waterHeader.readUInt32BE(16), waterHeight = waterHeader.readUInt32BE(20)
const water = readImage(waterPath,waterWidth,waterHeight)
const slopes = new Float32Array(waterWidth*waterHeight)
for (let i=0; i<slopes.length; i++) slopes[i] = Math.hypot(water[3*i]-127.5,water[3*i+1]-127.5)/127.5
const envelope = blur(slopes,waterWidth,waterHeight,Math.max(1, Math.round(waterWidth*5/512)))
write('water-roughness',waterWidth,waterHeight,1,Buffer.from(envelope.map(s => byte(0.68+0.30*Math.tanh(6*s)))))
