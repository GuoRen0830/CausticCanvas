import { HarborMesh, dressedStone, harborSailcloth, stoneCramp } from './cove-quayside'
import { shoreHeight } from './cove-shore'
import { makePierhead } from './cove-pierhead'

type Point = [number, number, number]
type Plan = [number, number]
const innerRadius = 4.68, outerRadius = 4.95, meanRadius = (innerRadius + outerRadius) / 2
const startAngle = -0.57, length = (1.158 - startAngle) * meanRadius
const turn = 2*Math.PI

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
  function paving(mat: string, s0: number, s1: number, inner: number, outer: number,
    rise: number, thickness: number, wear: number) {
    const y0=walkLevel(s0)+rise, y1=walkLevel(s1)+rise
    dressedStone(mesh,mat,[at(s0,inner,y0),at(s1,inner,y1),at(s1,outer,y1),at(s0,outer,y0)],thickness,wear)
  }
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
      const rise=0.006*h
      paving(i%5===2 ? 'weathered limestone' : 'capstone',s0+0.006,s1-0.006,
        row===0 ? -0.482 : 0.006,row===0 ? -0.006 : 0.482,rise,0.12+rise,0.016+0.007*h)
      // Selected inner coping joints have been pinned back against wave action.
      if (row===0 && (i===4 || i===9 || i===11)) {
        const nextRise=0.006*noise(201+i)
        stoneCramp(mesh,at(s1-0.114,-0.177,walkLevel(s1-0.114)+rise+0.003),
          at(s1+0.114,-0.177,walkLevel(s1+0.114)+nextRise+0.003))
      }
      s0=s1
    }
  }
  // Seaward curb is low and periodically pierced by scuppers.
  for (let i=0; i<16; i++) {
    const s0=0.54+i*0.47, s1=Math.min(s0+0.448,length-0.35)
    if (s1-s0<0.15 || i%6===3) continue
    paving('weathered limestone',s0,s1,0.328,0.47,0.105,0.105,0.010)
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
  // Timber rubbing piles shield the inward stone face on either side of the landing.
  for (const s of [2.24,3.02,4.16,6.18,6.61]) {
    const [x,,z]=at(s,-0.52,0)
    mesh.column('dock oak',x,z,[[-0.42,0.076],[0.29,0.068],[0.34,0.058]],8)
  }
}

function pilotLanding(mesh: HarborMesh) {
  const start = 4.47, end = 5.91, deck = 0.285
  // Joists carry the planking between two pile bents. The fenders on the
  // exposed face are separate from the structural piles and take the impact.
  for (const s of [start+0.09,end-0.09]) {
    for (const offset of [-0.61,-1.02]) {
      const [x,,z]=at(s,offset,0)
      const bed=shoreHeight(x,z)
      mesh.column('dock oak',x,z,[[bed-0.08,0.052],[deck-0.103,0.052]],8)
    }
    mesh.beam('dock oak',at(s,-0.54,deck-0.125),at(s,-1.11,deck-0.125),0.107,0.082)
    mesh.beam('sawn cedar',at(s,-0.61,-0.16),at(s,-1.02,deck-0.142),0.054,0.070)
  }
  for (const offset of [-0.61,-1.02]) {
    mesh.beam('dock oak',at(start,offset,deck-0.081),at(end,offset,deck-0.081),0.078,0.082)
  }
  for (let i=0; i<11; i++) {
    const a=start+(end-start)*i/11+0.005, b=start+(end-start)*(i+1)/11-0.005
    block(mesh,i%4===1 ? 'dock oak' : 'sawn cedar',a,b,-1.11,-0.535,()=>deck-0.044,()=>deck,0)
  }
  block(mesh,'capstone',5.02,5.37,-0.718,-0.44,()=>deck,()=>0.397,0.014)
  for (const s of [start+0.15,end-0.15]) {
    const [x,,z]=at(s,-1.164,0)
    mesh.column('dock oak',x,z,[[-0.22,0.06],[0.3,0.071],[0.402,0.062]],10)
    for (const y of [0.068,0.092,0.307,0.331]) {
      mesh.column('hemp rope',x,z,[[y-0.010,0.067],[y,0.077],[y+0.010,0.068]],10)
    }
    mesh.beam('wrought iron',at(s,-1.185,0.223),at(s,-1.02,0.223),0.034,0.054)
  }
  const s=5.19
  for (const shift of [-0.106,0.106]) {
    const foot=at(s+shift,-1.19,-0.24), shoulder=at(s+shift,-1.19,deck+0.24)
    mesh.beam('aged zinc',foot,shoulder,0.024,0.026)
    mesh.beam('aged zinc',shoulder,at(s+shift,-1.09,deck+0.27),0.024,0.026)
    mesh.beam('aged zinc',at(s+shift,-1.09,deck+0.27),at(s+shift,-1.015,deck),0.024,0.026)
    mesh.beam('wrought iron',at(s+shift,-1.19,deck-0.076),at(s+shift,-1.02,deck-0.076),0.028,0.03)
  }
  for (let i=0;i<4;i++) mesh.beam('aged zinc',at(s-0.106,-1.19,-0.16+i*0.134),
    at(s+0.106,-1.19,-0.16+i*0.134),0.027,0.029)
}

function rescueRing(mesh: HarborMesh) {
  const s=6.59, off=0.312, centerY=0.92, radius=0.156, tube=0.044
  for (const shift of [-0.096,0.096]) {
    mesh.beam('harbor blue',at(s+shift,0.222,walkLevel(s)),at(s+shift,0.222,1.105),0.039,0.046)
    mesh.beam('aged zinc',at(s+shift,0.222,1.057),at(s+shift,off+0.011,1.057),0.023,0.025)
  }
  mesh.beam('harbor blue',at(s-0.135,0.222,1.06),at(s+0.135,0.222,1.06),0.051,0.049)
  function ring(angle: number, rim: number, thickness: number): Point {
    const r=radius+thickness*Math.cos(rim)
    return at(s+r*Math.cos(angle),off+thickness*Math.sin(rim),centerY+r*Math.sin(angle))
  }
  for (let i=0; i<24; i++) for (let j=0; j<8; j++) {
    const a=i*Math.PI/12, b=(i+1)*Math.PI/12, c=j*Math.PI/4, d=(j+1)*Math.PI/4
    const p=ring((a+b)/2,(c+d)/2,tube), center=ring((a+b)/2,0,0)
    mesh.face(i%6<2 ? 'beacon ivory' : 'beacon oxblood',
      [ring(a,c,tube),ring(b,c,tube),ring(b,d,tube),ring(a,d,tube)],
      [p[0]-center[0],p[1]-center[1],p[2]-center[2]])
  }
}

function signalMast(mesh: HarborMesh) {
  const s=3.16, off=0.207, [x,,z]=at(s,off,0), base=walkLevel(s)
  mesh.column('wrought iron',x,z,[[base,0.084],[base+0.036,0.084],[base+0.072,0.047]],10)
  mesh.column('sawn cedar',x,z,[[base+0.04,0.039],[1.07,0.034],[1.62,0.024]],10)
  mesh.column('aged zinc',x,z,[[1.562,0.035],[1.636,0.036],[1.662,0.022]],10)
  // The mouth stays spread by a rigid hoop and three bridle lines. Both ends
  // are open; an inner skin gives the woven sleeve thickness viewed end-on.
  const mouth=0.21, span=0.88, sides=12, sections=10
  function sleeve(t: number, angle: number, inset=0): Point {
    const r=(0.141-0.083*t)*(1+t*0.075*Math.cos(3*angle+t*6))-inset
    const drift=0.086*Math.sin(Math.PI*t)-0.118*t
    const y=1.585-0.147*t*t+0.024*Math.sin(2*Math.PI*t)
    return at(s+mouth+span*t,off+drift+r*Math.cos(angle),y+r*Math.sin(angle))
  }
  for (let i=0; i<3; i++) mesh.beam('hemp rope',at(s,off,1.613),
    sleeve(0,(i+0.25)*2*Math.PI/3),0.009,0.009)
  for (let j=0; j<sides; j++) {
    const a=j*2*Math.PI/sides, b=(j+1)*2*Math.PI/sides
    mesh.beam('aged zinc',sleeve(0,a),sleeve(0,b),0.012,0.014)
    for (let i=0; i<sections; i++) {
      const t=i/sections, u=(i+1)/sections
      const mat=Math.floor(i/2)%2 ? 'pilot signal cream' : 'pilot signal indigo'
      const p=sleeve(t,a), q=sleeve(t,a,0.003)
      const outward: Point=[p[0]-q[0],p[1]-q[1],p[2]-q[2]]
      mesh.face(mat,[p,sleeve(u,a),sleeve(u,b),sleeve(t,b)],outward)
      mesh.face(mat,[q,sleeve(u,a,0.003),sleeve(u,b,0.003),sleeve(t,b,0.003)],
        [-outward[0],-outward[1],-outward[2]])
    }
  }
}

// A suspended fog bell and a tide staff are readable from the sea as well as
// the walking deck. The bell has a hollow skirt around its separate clapper.
function fogBell(mesh: HarborMesh) {
  const station=1.38, offset=0.271, deck=walkLevel(station)
  const p=(s: number,o: number,h: number) => at(station+s,offset+o,deck+h)
  for (const s of [-0.25,0.25]) {
    mesh.beam('dock oak',p(s,0,0.006),p(s,0,0.968),0.076,0.085)
    mesh.beam('wrought iron',p(s,0,0.008),p(s,0,0.147),0.093,0.098)
    mesh.beam('sawn cedar',p(s,0,0.652),p(s*0.30,0,0.915),0.041,0.041)
  }
  mesh.beam('sawn cedar',p(-0.314,0,0.935),p(0.314,0,0.935),0.083,0.098)
  mesh.beam('wrought iron',p(0,-0.055,0.87),p(0,0.055,0.87),0.055,0.055)
  mesh.beam('wrought iron',p(0,0,0.869),p(0,0,0.783),0.039,0.036)
  const [cx,,cz]=p(0,0,0)
  const atBell=(a: number,r: number,h: number): Point => [cx+r*Math.cos(a),deck+h,cz+r*Math.sin(a)]
  const profile=[[0.477,0.159],[0.489,0.158],[0.513,0.135],[0.564,0.115],
    [0.639,0.100],[0.707,0.091],[0.745,0.076],[0.774,0.047]]
  for (let i=0; i<24; i++) {
    const a=i*turn/24, b=(i+1)*turn/24, mid=(a+b)/2
    for (let j=1; j<profile.length; j++) {
      const [h0,r0]=profile[j-1],[h1,r1]=profile[j]
      mesh.face(j===1 ? 'lens brass' : 'verdigris',
        [atBell(a,r0,h0),atBell(b,r0,h0),atBell(b,r1,h1),atBell(a,r1,h1)],[Math.cos(mid),0,Math.sin(mid)])
      mesh.face('verdigris',[atBell(a,r0-0.009,h0),atBell(b,r0-0.009,h0),
        atBell(b,r1-0.009,h1),atBell(a,r1-0.009,h1)],[-Math.cos(mid),0,-Math.sin(mid)])
    }
    mesh.face('lens brass',[atBell(a,0.159,0.477),atBell(b,0.159,0.477),
      atBell(b,0.150,0.477),atBell(a,0.150,0.477)],[0,-1,0])
  }
  mesh.column('lens brass',cx,cz,[[deck+0.766,0.048],[deck+0.79,0.039],[deck+0.797,0.025]],16)
  mesh.column('wrought iron',cx,cz,[[deck+0.432,0.015],[deck+0.46,0.035],[deck+0.48,0.023],
    [deck+0.485,0.012],[deck+0.769,0.012]],12)
  mesh.beam('hemp rope',p(0,0,0.437),p(-0.159,-0.069,0.313),0.012,0.012)
  mesh.beam('hemp rope',p(-0.159,-0.069,0.313),p(-0.248,-0.022,0.325),0.012,0.012)
}

function tideStaff(mesh: HarborMesh) {
  const station=5.48
  const p=(s: number,y: number,extra=0) => at(station+s,0.597-0.105*y+extra,y)
  for (const y of [0.17,0.455]) {
    mesh.beam('wrought iron',at(station,0.411,y),p(0,y),0.054,0.063)
  }
  const board=(offset: number) => [p(-0.093,-0.27,offset),p(0.093,-0.27,offset),
    p(0.093,0.80,offset),p(-0.093,0.80,offset)]
  const face=board(0.038), back=board(-0.024), center=p(0,0.265)
  const frontNormal: Point=[face[0][0]+face[2][0]-back[0][0]-back[2][0],0,
    face[0][2]+face[2][2]-back[0][2]-back[2][2]]
  mesh.face('beacon ivory',face,frontNormal)
  mesh.face('dock oak',back,[-frontNormal[0],0,-frontNormal[2]])
  for (let i=0;i<4;i++) {
    const j=(i+1)%4, edge=face[i].map((v,k)=>v+face[j][k]-2*center[k]) as Point
    mesh.face('dock oak',[face[i],face[j],back[j],back[i]],edge)
  }
  for (let i=0;i<12;i++) {
    const y=-0.235+0.086*i, major=i%3===0
    mesh.beam(major ? 'beacon oxblood' : 'wrought iron',p(-0.091,y,0.043),p(major ? 0.060 : -0.016,y,0.043),
      0.010,major ? 0.029 : 0.013)
  }
}

export function makeCoveBreakwater() {
  const mesh=new HarborMesh()
  stonework(mesh)
  ++mesh.object
  for (let i=0; i<30; i++) armorRock(mesh,0.42+0.262*i,0.53+noise(i+9)*0.15,i+32)
  ++mesh.object
  dockFittings(mesh)
  ++mesh.object
  pilotLanding(mesh)
  ++mesh.object
  rescueRing(mesh)
  ++mesh.object
  signalMast(mesh)
  ++mesh.object
  fogBell(mesh)
  ++mesh.object
  tideStaff(mesh)
  ++mesh.object
  const [headX,,headZ] = at(length,0,0)
  makePierhead(mesh,headX,headZ)
  const result=mesh.finish(), stone=result.materials.get('weathered limestone')!
  result.materials.set('tidal stone',{...stone,color:[0.31,0.35,0.29],roughness:0.40})
  result.materials.set('armor stone',{...stone,color:[0.44,0.45,0.41],roughness:0.81})
  result.materials.set('beacon ivory',{color:[0.92,0.85,0.68],roughness:0.41})
  result.materials.set('beacon oxblood',{color:[0.59,0.105,0.063],roughness:0.39})
  result.materials.set('verdigris',{color:[0.17,0.39,0.34],metallic:0.31,roughness:0.61})
  result.materials.set('lantern amber',{color:[1.00,0.78,0.38],roughness:0.15})
  result.materials.set('lens brass',{color:[0.79,0.58,0.25],metallic:0.96,roughness:0.24})
  result.materials.set('pilot signal cream',{color:[0.96,0.82,0.52],map:harborSailcloth,roughness:0.86})
  result.materials.set('pilot signal indigo',{color:[0.105,0.29,0.55],map:harborSailcloth,roughness:0.83})
  return result
}
