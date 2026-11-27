import canvasTile from '../../scene/smugglers-cove/lug-sailcloth.png?inline'
import { sceneVertexStride } from '../rendering/scene-mesh'
import { HarborMesh } from './cove-quayside'

type Point = [number, number, number]
const up: Point = [0, 1, 0]
const mix = (a: number, b: number, t: number) => a + (b-a)*t
function between(a: Point, b: Point, t: number): Point {
  return [mix(a[0],b[0],t), mix(a[1],b[1],t), mix(a[2],b[2],t)]
}
const red = 'tanbark linen', copper = 'sun-faded tanbark', reef = 'reefed oxblood linen'
const sailPatch = 'lug sail corner doubler'
const ivory = 'flax jib', patched = 'waxed flax', piping = 'sail bolt rope'
const blue = 'lugger petrol', teal = 'lugger lapstrake', gold = 'lugger ivory stripe'

// Local waterline at Y=0. Station 0 is the transom, station 1 the bow (+X).
function hull(t: number, h: number, side: number): Point {
  const breadth = 0.037 + 0.762*Math.sin(Math.PI*t)**0.82 + 0.30*(1-t)**3
  const keel = -0.42 + 0.18*(2*t-1)**2 + 0.075*t**5
  const sheer = 0.40 + 0.155*(2*t-1)**2 + 0.092*t**5
  return [mix(-2.63,2.72,t) + (2*t-1)*(h-1)*(t > 0.5 ? 0.32 : 0.19),
    mix(keel,sheer,h), side*breadth*(0.082 + 0.937*h**0.58 - 0.019*h**6)]
}

function buildHull(mesh: HarborMesh) {
  const rings = [0,0.20,0.39,0.52,0.64,0.75,0.79,0.855,0.92,1]
  const colors = ['dock oak',blue,teal,blue,teal,'ochre boards',gold,blue,'dock oak']
  const stations = 24
  function deck(t: number, across: number): Point {
    const edge = hull(t,1,1)
    return [edge[0], edge[1]-0.035 + 0.050*(1-across*across), edge[2]*across]
  }
  for (let i = 0; i < stations; i++) {
    const t = i/stations, end = (i+1)/stations
    for (let row = 1; row < rings.length; row++) {
      const a = rings[row-1], b = rings[row]
      for (const side of [-1,1]) {
        mesh.face(colors[row-1], [hull(t,a,side),hull(end,a,side),hull(end,b,side),hull(t,b,side)], [0,0,side])
      }
      for (const k of [0,1]) if (i === (k ? stations-1 : 0)) {
        mesh.face(colors[row-1], [hull(k,a,-1),hull(k,a,1),hull(k,b,1),hull(k,b,-1)], [2*k-1,0,0])
      }
    }
    mesh.face('dock oak', [hull(t,0,-1),hull(end,0,-1),hull(end,0,1),hull(t,0,1)], [0,-1,0])
    for (const side of [-1,1]) {
      mesh.face('sawn cedar', [hull(t,1,side),hull(end,1,side),deck(end,0.935*side),deck(t,0.935*side)], up)
    }
    // Swept deck planks taper continuously into the stem.
    for (let plank = 0; plank < 12; plank++) {
      const a = mix(-0.935,0.935,plank/12), b = mix(-0.935,0.935,(plank+1)/12)
      const caulking = a+0.008
      mesh.face('dock oak', [deck(t,a),deck(end,a),deck(end,caulking),deck(t,caulking)], up)
      mesh.face(plank%4 === 1 ? 'lugger deck' : 'sawn cedar',
        [deck(t,caulking),deck(end,caulking),deck(end,b),deck(t,b)], up)
    }
  }
}

function cord(mesh: HarborMesh, a: Point, b: Point, sag: number, width: number, material = 'hemp rope') {
  let previous = a
  const count = sag ? 6 : 1
  for (let i = 1; i <= count; i++) {
    const t = i/count, next = between(a,b,t)
    next[1] -= 4*sag*t*(1-t)
    mesh.beam(material,previous,next,width,width)
    previous = next
  }
}

// The yard and the foot twist in different directions. The draft is fullest
// near the luff, then flattens toward the trailing edge of each sewn panel.
function mainsail(u: number, v: number): Point {
  const foot = between([-2.08,1.03,0.36],[0.57,0.84,0.13],u)
  const head = between([-1.51,2.80,-0.32],[0.79,2.15,-0.18],u)
  const p = between(foot,head,v), chord = Math.sin(Math.PI*u)
  p[0] -= 0.078*(1-u)*Math.sin(Math.PI*v)
  p[1] -= 0.105*chord*(1-v)
  p[2] += chord*(0.29*(1-v) + 0.43*Math.sin(Math.PI*v))*(0.65+0.7*u)
    + 0.014*Math.sin(14*Math.PI*u)*Math.sin(Math.PI*v)
  return p
}

function headsail(u: number, v: number): Point {
  const foot = between([1.01,1.05,0.42],[2.65,0.78,0.03],u)
  const head = between([0.305,2.63,-0.20],[0.352,2.65,-0.205],u)
  const p = between(foot,head,v)
  p[0] += 0.10*Math.sin(Math.PI*v)*(1-u)
  p[1] -= 0.101*Math.sin(Math.PI*u)*(1-v)
  p[2] += (1-v)*Math.sin(Math.PI*u)*(0.20 + 0.13*Math.sin(Math.PI*v))
  return p
}

type Cut = (u: number, v: number) => Point
type FabricVertex = { normal: Point; uv: [number,number] }
const key = ([x,y,z]: Point) => [x,y,z].map(Math.fround).join(',')
function stitchSail(mesh: HarborMesh, cut: Cut, columns: number, rows: number, jib: boolean,
  fabricVertices: Map<string,FabricVertex>) {
  function vertex(i: number, j: number, side: number): Point {
    const u = i/columns, v = j/rows, p = cut(u,v)
    // Smooth normals include the wind-filled belly on both cloth faces.
    const h = 0.0001
    const a = cut(Math.max(0,u-h),v), b = cut(Math.min(1,u+h),v)
    const c = cut(u,Math.max(0,v-h)), d = cut(u,Math.min(1,v+h))
    const ux = b[0]-a[0], uy = b[1]-a[1], uz = b[2]-a[2]
    const vx = d[0]-c[0], vy = d[1]-c[1], vz = d[2]-c[2]
    const nx = uy*vz-uz*vy, ny = uz*vx-ux*vz, nz = ux*vy-uy*vx
    const norm = Math.hypot(nx,ny,nz)*Math.sign(nz)*side
    p[2] += side*0.009
    fabricVertices.set(key(p), {normal:[nx/norm,ny/norm,nz/norm], uv:[p[0]*1.12,p[1]*1.3]})
    return p
  }
  const grid = (side: number) => Array.from({length:rows+1}, (_,j) =>
    Array.from({length:columns+1}, (_,i) => vertex(i,j,side)))
  const front = grid(1), back = grid(-1)
  // These bias-cut reinforcement patches follow a mesh diagonal exactly.
  const reinforced = (x: number, y: number) => x+rows-y < 3 || columns-x+y < 2
  const patch = jib ? patched : sailPatch
  for (let j = 0; j < rows; j++) for (let i = 0; i < columns; i++) {
    const panel = Math.floor(i/3)
    const mat = jib ? (panel%3 === 1 ? patched : ivory)
      : j < 2 ? reef : [copper,red,copper,red,red,copper][panel%6]
    for (const [side, g] of [[1,front],[-1,back]] as const) {
      const a = g[j][i], b = g[j][i+1], c = g[j+1][i+1], d = g[j+1][i]
      mesh.face(reinforced(i+2/3,j+1/3) ? patch : mat,[a,b,c],[0,0,side])
      mesh.face(reinforced(i+1/3,j+2/3) ? patch : mat,[a,c,d],[0,0,side])
    }
    if (j === 0 || j === rows-1) {
      const k = j === 0 ? 0 : rows
      mesh.face(piping,[front[k][i],front[k][i+1],back[k][i+1],back[k][i]], [0,j === 0 ? -1 : 1,0])
    }
    if (i === 0 || i === columns-1) {
      const k = i === 0 ? 0 : columns
      mesh.face(piping,[front[j][k],front[j+1][k],back[j+1][k],back[j][k]], [i === 0 ? -1 : 1,0,0])
    }
  }
  // Flat-felled seams and the reinforced reef band follow the actual surface.
  function seam(u: number, v: number, du: number, dv: number, n: number, radius: number, mat: string) {
    let a = cut(u,v)
    a[2] += 0.012
    for (let i = 1; i <= n; i++) {
      const b = cut(u+du*i/n,v+dv*i/n)
      b[2] += 0.012
      mesh.beam(mat,a,b,radius,radius)
      a = b
    }
  }
  for (let i = 0; i <= columns; i += 3) {
    seam(i/columns,0,0,1,5,i === 0 || i === columns ? 0.016 : 0.009, jib ? piping : 'tanbark seam')
  }
  seam(0,0,1,0,columns,0.017,piping)
  if (!jib) {
    seam(0,2/rows,1,0,columns,0.015,'tanbark seam')
    for (let i = 1; i < columns; i += 3) {
      const a = cut(i/columns,2/rows)
      a[2] += 0.018
      cord(mesh,a,[a[0]-0.038,a[1]-0.085,a[2]+0.035],0,0.012,piping)
    }
  }
}

function deckFittings(mesh: HarborMesh) {
  // Rudder blade, gudgeons, and curved tiller at the raised transom.
  const outline: Point[] = [[-2.62,0.60,0],[-2.68,0.54,0],[-2.91,-0.20,0],
    [-2.82,-0.32,0],[-2.51,-0.29,0],[-2.53,0.22,0]]
  const faces = [-1,1].map(side => outline.map(([x,y]):Point => [x,y,side*0.041]))
  mesh.face(teal,faces[0],[0,0,-1]); mesh.face(teal,faces[1],[0,0,1])
  for (let i = 0; i < outline.length; i++) {
    const j = (i+1)%outline.length, a = outline[i], b = outline[j]
    mesh.face('dock oak',[faces[0][i],faces[0][j],faces[1][j],faces[1][i]], [b[1]-a[1],a[0]-b[0],0])
  }
  for (const y of [-0.06,0.23]) mesh.box('wrought iron',-2.60,y,0,0.23,0.041,0.11)
  mesh.beam('sawn cedar',[-2.66,0.55,0],[-2.28,0.68,-0.09],0.071,0.059)
  mesh.beam('sawn cedar',[-2.28,0.68,-0.09],[-1.64,0.71,-0.17],0.047,0.048)
  // Raised fish hold with removable covers. The hinged forehatch and grated
  // cover sit in open portions of the deck on either side of the mast.
  for (const [x,w,d] of [[-0.72,0.88,0.68],[1.41,0.49,0.50]]) {
    mesh.box(blue,x,0.485,0,w,0.12,d)
    for (let i = 0; i < 6; i++) {
      mesh.box('lugger deck',x-w/2+w*(i+0.5)/6,0.561,0,w/6-0.009,0.045,d+0.025)
    }
    for (const z of [-0.32,0.32]) {
      mesh.box('wrought iron',x-w*0.28,0.589,z*d,0.043,0.011,d*0.26)
    }
  }
  // Mast partners, transverse thwarts and towing bitts.
  mesh.box('dock oak',0.47,0.47,0,0.20,0.115,1.45)
  mesh.box('sawn cedar',-1.68,0.52,0,0.24,0.072,1.07)
  for (const [x,z] of [[-2.12,0.32],[-2.12,-0.32],[1.99,0.23],[1.99,-0.23]]) {
    mesh.column('dock oak',x,z,[[0.48,0.037],[0.63,0.037],[0.651,0.051],[0.673,0.045]],8)
  }
  let previous: Point = [0.88,0.48,0.22]
  for (let i = 1; i <= 51; i++) {
    const a = 2*Math.PI*i/16, radius = 0.059+0.112*i/51
    const next: Point = [0.821+Math.cos(a)*radius,0.48,0.22+Math.sin(a)*radius*0.76]
    mesh.beam('hemp rope',previous,next,0.024,0.022)
    previous = next
  }
}

const clothBytes = atob(canvasTile.slice(canvasTile.indexOf(',')+1))
const clothMap = new File([Uint8Array.from(clothBytes,char => char.charCodeAt(0))], 'lug-sailcloth.png', {type:'image/png'})

export function makeCoveLugger() {
  const mesh = new HarborMesh()
  buildHull(mesh)
  ++mesh.object
  deckFittings(mesh)
  ++mesh.object
  const mastFoot: Point = [0.47,0.45,0], mastHead: Point = [0.27,2.91,-0.22]
  mesh.beam('dock oak',mastFoot,mastHead,0.087,0.086)
  const partners = between(mastFoot,mastHead,0.15)
  mesh.beam('wrought iron',between(mastFoot,mastHead,0.132),partners,0.097,0.097)
  mesh.beam('sawn cedar',[1.97,0.54,0],[2.87,0.77,0],0.066,0.071)
  mesh.beam('sawn cedar',[-1.60,2.837,-0.342],[0.92,2.127,-0.19],0.057,0.053)
  const fabricVertices = new Map<string,FabricVertex>()
  stitchSail(mesh,mainsail,18,10,false,fabricVertices)
  stitchSail(mesh,headsail,9,8,true,fabricVertices)
  ++mesh.object
  // Standing rigging is tarred. Running sheets and halyards retain pale hemp.
  for (const side of [-1,1]) {
    cord(mesh,mastHead,[-0.18,0.45,0.70*side],0.018,0.016,'tarred shroud')
  }
  cord(mesh,mastHead,[2.77,0.75,0],0.013,0.017,'tarred shroud')
  cord(mesh,mastHead,[-0.49,2.525,-0.27],0.026,0.019)
  cord(mesh,mainsail(0,0),[-2.09,0.653,0.32],0.054,0.024)
  cord(mesh,mainsail(1,0),[0.48,0.53,-0.05],0.02,0.021)
  cord(mesh,headsail(0,0),[0.40,0.49,0.68],0.075,0.019)
  cord(mesh,headsail(1,0),[2.80,0.751,0],0.011,0.018)
  cord(mesh,[1.99,0.623,0.23],[2.65,0.593,0.075],0.027,0.022)
  cord(mesh,[2.65,0.593,0.075],[3.27,-0.13,0.23],0.105,0.022)
  // A short split ivory pennon flies clear of the yard.
  const flag: Point[] = [[0.274,2.91,-0.22],[0.279,2.77,-0.21],[-0.045,2.802,-0.13],
    [-0.19,2.78,-0.12],[-0.086,2.841,-0.152],[-0.21,2.881,-0.16],[-0.046,2.886,-0.16]]
  for (const side of [-1,1]) {
    const points = flag.map(([x,y,z]):Point => [x,y,z+side*0.003])
    for (const indices of [[0,1,2],[0,2,6],[6,2,4],[4,5,6],[2,3,4]]) {
      mesh.face(gold,indices.map(i => points[i]),[0,0,side])
    }
  }
  const result = mesh.finish()
  const timber = result.materials.get('sawn cedar')?.map
  for (const [name,color] of [[blue,[0.055,0.22,0.235]],[teal,[0.080,0.30,0.29]],
    ['lugger deck',[0.71,0.51,0.30]]] as [string,Point][]) {
    result.materials.set(name,{color,map:timber})
  }
  result.materials.set(gold,{color:[0.90,0.77,0.52]})
  result.materials.set('tarred shroud',{color:[0.17,0.13,0.084]})
  result.materials.set(piping,{color:[0.69,0.55,0.32]})
  result.materials.set('tanbark seam',{color:[0.46,0.215,0.099]})
  const cloth = new Map<string,Point>([[red,[0.83,0.32,0.151]],[copper,[0.91,0.392,0.18]],[sailPatch,[0.73,0.337,0.182]],
    [reef,[0.66,0.215,0.121]],[ivory,[0.92,0.82,0.62]],[patched,[0.86,0.745,0.54]]])
  for (const [name,color] of cloth) result.materials.set(name,{color,map:clothMap})
  // HarborMesh batches by material. Only the sewn panels use continuous fabric
  // coordinates and smooth shading; the deck, booms and bolt ropes stay faceted.
  const { vertices, indices, uvs, draws } = result.mesh
  for (const draw of draws!) {
    if (!cloth.has(draw.material!)) continue
    for (let i = draw.firstIndex; i < draw.firstIndex+draw.indexCount; i++) {
      const index = indices[i], start = index*sceneVertexStride
      const fabric = fabricVertices.get(key([vertices[start],vertices[start+1],vertices[start+2]]))!
      vertices.set(fabric.normal,start+3)
      uvs!.set(fabric.uv,index*2)
    }
  }
  return result
}
