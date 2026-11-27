import { HarborMesh } from './cove-quayside'
import { shoreHeight } from './cove-shore'

type Point = [number, number, number]
type Plan = [number, number]
const innerRadius = 4.68, outerRadius = 4.95, meanRadius = (innerRadius + outerRadius) / 2
const startAngle = -0.57, length = (1.158 - startAngle) * meanRadius

// Stations run along the curved walk. Positive offsets point out to sea.
function at(station: number, offset: number, y: number): Point {
  const angle = startAngle + station / meanRadius, c = Math.cos(angle), s = Math.sin(angle)
  const n = Math.hypot(innerRadius*c, outerRadius*s)
  return [1.70 + outerRadius*c + offset*innerRadius*c/n, y,
    2.20 + innerRadius*s + offset*outerRadius*s/n]
}

const start = at(0, 0, 0), landLevel = shoreHeight(start[0], start[2]) + 0.026
const walkLevel = (station: number) => landLevel + (0.51 - landLevel)*Math.min(station/1.8, 1)
const freestone = ['sandstone', 'weathered limestone', 'capstone', 'warm freestone', 'weathered limestone']

function noise(seed: number) {
  const value = Math.sin(seed*127.1 + 311.7)*43758.5453
  return value-Math.floor(value)
}

// Use straight chords for each quarried block. A clipped corner gives a highlight
// in cross-light, and a center fan keeps the cap valid across the access ramp.
function block(mesh: HarborMesh, mat: string, s0: number, s1: number, l0: number, l1: number,
  base: (s: number) => number, top: (s: number) => number, chip = 0.019) {
  const cutS = Math.min(0.20, chip/(s1-s0)), cutL = Math.min(0.20, chip/(l1-l0))
  const outline: Plan[] = chip ? [[cutS,0],[1-cutS,0],[1,cutL],[1,1-cutL],
    [1-cutS,1],[cutS,1],[0,1-cutL],[0,cutL]] : [[0,0],[1,0],[1,1],[0,1]]
  function point(u: number, v: number, heights: (s: number) => number): Point {
    const off = l0 + (l1-l0)*v, a = at(s0, off, heights(s0)), b = at(s1, off, heights(s1))
    return [a[0]+u*(b[0]-a[0]), a[1]+u*(b[1]-a[1]), a[2]+u*(b[2]-a[2])]
  }
  const lower = outline.map(([u,v]) => point(u,v,base)), upper = outline.map(([u,v]) => point(u,v,top))
  const center = point(0.5,0.5,top), bottom = point(0.5,0.5,base)
  for (let i = 0; i < outline.length; i++) {
    const j = (i+1)%outline.length, a = upper[i], b = upper[j]
    mesh.face(mat, [center,a,b], [0,1,0])
    mesh.face(mat, [bottom,lower[j],lower[i]], [0,-1,0])
    mesh.face(mat, [lower[i],lower[j],b,a], [a[0]+b[0]-2*center[0],0,a[2]+b[2]-2*center[2]])
  }
}

function stonework(mesh: HarborMesh) {
  const foundation = -0.48
  for (let i=0; i<22; i++) {
    block(mesh,'lime mortar',length*i/22,length*(i+1)/22,-0.398,0.398,
      () => foundation, s => walkLevel(s)-0.018, 0)
  }
  for (const side of [-1,1]) for (let row=0; row<3; row++) {
    const width = 0.61 - 0.071*row
    const elevation = (s: number, n: number) => foundation + (walkLevel(s)-0.115-foundation)*n/3
    let s0 = 0
    for (let i=0; s0<length; i++) {
      const h = noise(i+row*51+side*19), span = (i===0 && row%2 ? 0.26 : 0.46) + 0.18*h
      const s1 = length-s0-span < 0.19 ? length : s0+span
      const edge = width + 0.023*h
      const offsets = side<0 ? [-edge,-0.37] : [0.37,edge]
      block(mesh, row===0 ? 'tidal stone' : freestone[Math.floor(h*freestone.length)],
        s0+0.007,s1-0.007,offsets[0],offsets[1],
        s => elevation(s,row)+0.005, s => elevation(s,row+1)-0.005)
      s0=s1
    }
  }
  // Two staggered ranks of flagstones, rising from the beach to the level walk.
  for (let row=0; row<2; row++) {
    let s0=0
    for (let i=0; s0<length; i++) {
      const h=noise(200+row*27+i), span=(i===0 && row===1 ? 0.25 : 0.52)+h*0.21
      const s1=length-s0-span<0.21 ? length : s0+span
      block(mesh, i%5===2 ? 'weathered limestone' : 'capstone',s0+0.009,s1-0.009,
        row===0 ? -0.482 : 0.007,row===0 ? -0.007 : 0.482,
        s=>walkLevel(s)-0.12,s=>walkLevel(s)+0.008*h,0.028)
      s0=s1
    }
  }
  // Seaward curb is low and periodically pierced by scuppers.
  for (let i=0; i<16; i++) {
    const s0=0.54+i*0.47, s1=Math.min(s0+0.448,length-0.35)
    if (s1-s0<0.15 || i%6===3) continue
    block(mesh, 'weathered limestone',s0,s1,0.328,0.47,walkLevel,s=>walkLevel(s)+0.105,0.016)
  }
}

function armorRock(mesh: HarborMesh, station: number, offset: number, seed: number) {
  const [x,,z] = at(station,offset,0), size=0.16+0.16*noise(seed), facets=7
  const crown: Point = [x+0.05*noise(seed+2),0.05+size*0.46,z-0.06]
  const bottom: Point = [x,-0.33,z], outline: Point[] = []
  for (let i=0; i<facets; i++) {
    const angle=(i+noise(seed+3)*0.7)*2*Math.PI/facets
    const r=size*(0.74+0.36*noise(seed*13+i))
    outline.push([x+r*Math.cos(angle),-0.06+0.13*noise(seed*7+i),z+r*0.80*Math.sin(angle)])
  }
  for (let i=0; i<facets; i++) {
    const a=outline[i], b=outline[(i+1)%facets]
    mesh.face(i%4===0 ? 'weathered limestone' : 'armor stone',[a,b,crown],[0,1,0])
    mesh.face('tidal stone',[a,b,bottom],[a[0]+b[0]-2*x,-0.1,a[2]+b[2]-2*z])
  }
}

function dockFittings(mesh: HarborMesh) {
  for (const s of [2.6,4.5,6.4]) {
    const [x,y,z]=at(s,-0.285,walkLevel(s)+0.009)
    mesh.column('wrought iron',x,z,[[y,0.095],[y+0.025,0.096],[y+0.045,0.049],
      [y+0.145,0.05],[y+0.168,0.079],[y+0.184,0.073]],10)
  }
  // Timber rubbing piles shield the inward stone face. The ladder lands at a gap.
  for (const s of [2.24,3.02,4.25,5.89,6.61]) {
    const [x,,z]=at(s,-0.52,0)
    mesh.column('dock oak',x,z,[[-0.42,0.076],[0.29,0.068],[0.34,0.058]],8)
  }
  const s=5.11
  for (const shift of [-0.106,0.106]) {
    const foot=at(s+shift,-0.649,-0.20), shoulder=at(s+shift,-0.649,0.64)
    mesh.beam('wrought iron',foot,shoulder,0.022,0.026)
    mesh.beam('wrought iron',shoulder,at(s+shift,-0.43,0.54),0.022,0.026)
  }
  for (let i=0;i<6;i++) mesh.beam('aged zinc',at(s-0.106,-0.649,-0.13+i*0.133),
    at(s+0.106,-0.649,-0.13+i*0.133),0.025,0.025)
}

function pierhead(mesh: HarborMesh) {
  const [x,,z]=at(length,0,0)
  mesh.column('tidal stone',x,z,[[-0.48,0.64],[0.08,0.61],[0.40,0.535]],12,
    ['tidal stone','armor stone','weathered limestone'])
  mesh.column('capstone',x,z,[[0.395,0.54],[0.46,0.558],[0.523,0.548]],12)
  const drum=(mat: string,rings: [number,number][],sides=12)=>mesh.column(mat,x,z,rings,sides)
  drum('capstone',[[0.523,0.344],[0.575,0.346],[0.637,0.300]])
  drum('beacon ivory',[[0.637,0.297],[0.688,0.303],[0.725,0.260],[0.89,0.249]])
  drum('beacon oxblood',[[0.89,0.249],[1.113,0.223]])
  drum('beacon ivory',[[1.113,0.223],[1.324,0.195]])
  drum('beacon oxblood',[[1.324,0.195],[1.395,0.194],[1.433,0.238]])
  drum('aged zinc',[[1.433,0.237],[1.463,0.354],[1.493,0.354]])
  function around(angle: number,r: number,y: number): Point {
    return [x+r*Math.cos(angle),y,z+r*Math.sin(angle)]
  }
  for (let i=0; i<12; i++) {
    const ang=i*Math.PI/6, next=(i+1)*Math.PI/6
    mesh.beam('wrought iron',around(ang,0.326,1.49),around(ang,0.326,1.704),0.016,0.016)
    for (const y of [1.600,1.700]) mesh.beam('aged zinc',around(ang,0.326,y),around(next,0.326,y),0.015,0.015)
  }
  drum('aged zinc',[[1.492,0.202],[1.551,0.198],[1.573,0.164]],8)
  drum('lantern amber',[[1.569,0.087],[1.612,0.117],[1.808,0.117],[1.854,0.084]],12)
  for (const y of [1.627,1.676,1.725,1.774]) {
    drum('lens brass',[[y,0.119],[y+0.008,0.128],[y+0.015,0.119]],12)
  }
  for (let i=0;i<8;i++) {
    const a=i*Math.PI/4
    mesh.beam('wrought iron',around(a,0.172,1.568),around(a,0.178,1.882),0.022,0.022)
  }
  drum('verdigris',[[1.873,0.190],[1.896,0.252],[1.919,0.252],[1.990,0.172],[2.057,0.069]],8)
  drum('lens brass',[[2.056,0.071],[2.088,0.066],[2.101,0.030],[2.167,0.014]],10)
  // A small service hatch faces the harbor and breaks the painted silhouette.
  mesh.box('wrought iron',x-0.245,0.806,z,0.028,0.218,0.14)
  mesh.box('verdigris',x-0.262,0.808,z,0.014,0.187,0.116)
}

export function makeCoveBreakwater() {
  const mesh=new HarborMesh()
  stonework(mesh)
  ++mesh.object
  for (let i=0; i<30; i++) armorRock(mesh,0.42+0.262*i,0.53+noise(i+9)*0.15,i+32)
  ++mesh.object
  dockFittings(mesh)
  ++mesh.object
  pierhead(mesh)
  const result=mesh.finish(), stone=result.materials.get('weathered limestone')!.map
  result.materials.set('tidal stone',{color:[0.31,0.35,0.29],map:stone})
  result.materials.set('armor stone',{color:[0.44,0.45,0.41],map:stone})
  result.materials.set('beacon ivory',{color:[0.92,0.85,0.68]})
  result.materials.set('beacon oxblood',{color:[0.59,0.105,0.063]})
  result.materials.set('verdigris',{color:[0.17,0.39,0.34]})
  result.materials.set('lantern amber',{color:[1.00,0.78,0.38]})
  result.materials.set('lens brass',{color:[0.66,0.46,0.21]})
  return result
}
