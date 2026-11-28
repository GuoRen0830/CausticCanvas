import limestone from '../../scene/smugglers-cove/quay-limestone.png?inline'
import stoneRoughness from '../../scene/smugglers-cove/quay-stone-roughness.png?inline'
import timber from '../../scene/smugglers-cove/quay-timber.png?inline'
import sailcloth from '../../scene/smugglers-cove/lug-sailcloth.png?inline'
import { sceneVertexStride, type SceneMaterial, type SceneMesh } from '../rendering/scene-mesh'
import { shoreHeight } from './cove-shore'

type Point = [number, number, number]
const subtract = (a: Point, b: Point): Point => [a[0]-b[0], a[1]-b[1], a[2]-b[2]]
const cross = (a: Point, b: Point): Point => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
const dot = (a: Point, b: Point) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
function unit(v: Point): Point {
  const length = Math.hypot(...v)
  return [v[0]/length, v[1]/length, v[2]/length]
}

export const quayLevel = 0.80, quayFront = -1.47
const stairDepth = 0.26, stairCount = 4
export const pierApproach = { x: 2.57, z: quayFront + stairCount * stairDepth }

// Planar dressed faces share one UV frame across the entire cap. A narrow
// sloping shoulder and four differently clipped corners take real highlights.
export function dressedStone(mesh: HarborMesh, material: string, corners: Point[], thickness: number, bevel: number) {
  const [a,b,c,d] = corners
  const span = Math.hypot(b[0]-a[0],b[2]-a[2]), depth = Math.hypot(d[0]-a[0],d[2]-a[2])
  const seed = a[0]*17.7 + a[2]*39.3
  const clips = corners.map((_,i) => bevel*(1.0+2.6*Math.sin(seed+1.9*i)**6))
  function ring(inset: number, drop: number): Point[] {
    const lo = inset/span, near = inset/depth, hi = 1-lo, far = 1-near
    const ux = clips.map(size => Math.min(0.14, size/span)), vz = clips.map(size => Math.min(0.16, size/depth))
    const plan = [[lo+ux[0],near], [hi-ux[1],near], [hi,near+vz[1]], [hi,far-vz[2]],
      [hi-ux[2],far], [lo+ux[3],far], [lo,far-vz[3]], [lo,near+vz[0]]]
    return plan.map(([u,v]): Point => [0,1,2].map(k =>
      (1-v)*((1-u)*a[k]+u*b[k])+v*((1-u)*d[k]+u*c[k]) - (k===1 ? drop : 0)) as Point)
  }
  const top = ring(bevel,0), shoulder = ring(0,bevel*0.8), bottom = ring(0,thickness)
  mesh.face(material,top,[0,1,0])
  mesh.face(material,bottom,[0,-1,0])
  const center = corners.reduce((sum,p) => [sum[0]+0.25*p[0],0,sum[2]+0.25*p[2]] as Point,[0,0,0] as Point)
  for (let i=0; i<top.length; i++) {
    const j=(i+1)%top.length
    const out: Point = [shoulder[i][0]+shoulder[j][0]-2*center[0],0,shoulder[i][2]+shoulder[j][2]-2*center[2]]
    mesh.face(material,[bottom[i],bottom[j],shoulder[j],shoulder[i]],out)
    mesh.face(material,[shoulder[i],shoulder[j],top[j],top[i]],[0,1,0])
  }
}

// Flush iron cramps tie adjacent coping stones together. Their splayed ends
// seat in the stone and only a slim iron bridge is exposed over the joint.
export function stoneCramp(mesh: HarborMesh, a: Point, b: Point) {
  const dx=b[0]-a[0], dz=b[2]-a[2], length=Math.hypot(dx,dz)
  const edge=(t: number, across: number, below=0): Point =>
    [a[0]+t*dx-across*dz/length,a[1]+t*(b[1]-a[1])-below,a[2]+t*dz+across*dx/length]
  const profile = [[0,0.027],[0.16,0.027],[0.26,0.014],[0.74,0.014],[0.84,0.027],[1,0.027]]
  for (let i=1; i<profile.length; i++) {
    const [t0,w0]=profile[i-1], [t1,w1]=profile[i]
    mesh.face('wrought iron',[edge(t0,-w0),edge(t1,-w1),edge(t1,w1),edge(t0,w0)],[0,1,0])
    mesh.face('wrought iron',[edge(t0,-w0,0.014),edge(t1,-w1,0.014),edge(t1,w1,0.014),edge(t0,w0,0.014)],[0,-1,0])
    for (const s of [-1,1]) mesh.face('wrought iron',
      [edge(t0,s*w0),edge(t1,s*w1),edge(t1,s*w1,0.014),edge(t0,s*w0,0.014)],[-s*dz,0,s*dx])
  }
  for (const t of [0,1]) mesh.face('wrought iron',
    [edge(t,-0.027),edge(t,0.027),edge(t,0.027,0.014),edge(t,-0.027,0.014)],[(2*t-1)*dx,0,(2*t-1)*dz])
}

export function makeQuayside(deckY: number) {
  const mesh = new HarborMesh()
  const left = 0.39, right = 5.84, back = -3.87
  const cx = (left+right)/2, cz = (back+quayFront)/2, wallTop = quayLevel - 0.13, foundation = -0.16
  function flag(mat: string, x0: number, x1: number, z0: number, z1: number, top: number, thickness: number, wear: number) {
    dressedStone(mesh,mat,[[x0,top,z0],[x1,top,z0],[x1,top,z1],[x0,top,z1]],thickness,wear)
  }
  mesh.box('lime mortar', cx, (wallTop+foundation)/2, cz, right-left-0.035, wallTop-foundation, quayFront-back-0.025)
  const stones = ['sandstone', 'weathered limestone', 'warm freestone', 'weathered limestone']
  // An unbroken core sits behind the mortar joints where the beach falls away.
  for (let row = 0; row < 4; row++) {
    const h = (wallTop-foundation)/4
    for (const edge of ['front','back','left','right']) {
      const alongX = edge === 'front' || edge === 'back'
      const start = alongX ? left : back+0.105, end = alongX ? right : quayFront-0.105
      const cuts = [start]
      for (let t = start+0.48+(row%2)*0.32; t < end-0.17; t += 0.71+0.077*Math.sin(t*3.9+row)) cuts.push(t)
      cuts.push(end)
      for (let k = 1; k < cuts.length; k++) {
        const mid = (cuts[k]+cuts[k-1])/2, width = cuts[k]-cuts[k-1]-0.017
        const y = foundation+(row+0.5)*h, mat = stones[(row*3+k)%stones.length]
        if (alongX) mesh.box(mat, mid, y, edge==='front' ? quayFront-0.036 : back+0.036, width, h-0.012, 0.13)
        else mesh.box(mat, edge === 'left' ? left+0.037 : right-0.037, y, mid, 0.13, h-0.012, width)
      }
    }
  }
  mesh.box('lime mortar', cx, (wallTop+quayLevel-0.021)/2, cz, right-left-0.056, quayLevel-0.021-wallTop, quayFront-back-0.056)
  for (let row = 0; row < 5; row++) {
    const depth = (quayFront-back)/5, z = quayFront-(row+0.5)*depth, ends = [left]
    for (let x = left+0.49+(row%2)*0.42; x < right-0.18; x += 0.93) ends.push(x)
    ends.push(right)
    for (let i = 1; i < ends.length; i++) {
      flag(row === 0 ? 'capstone' : stones[(row+i*3)%stones.length],
        ends[i-1]+0.006,ends[i]-0.006,z-depth/2+0.006,z+depth/2-0.006,quayLevel,0.132,row===0 ? 0.017 : 0.010)
      if (row===0 && (i===1 || i===4 || i===5)) {
        stoneCramp(mesh,[ends[i]-0.114,quayLevel+0.003,z+0.11],[ends[i]+0.114,quayLevel+0.003,z+0.11])
      }
    }
  }
  ++mesh.object
  for (let i = 0; i < stairCount; i++) {
    const top = quayLevel + (deckY-quayLevel)*(i+1)/stairCount
    const height = top-0.036-foundation, z = quayFront+(i+0.5)*stairDepth
    mesh.box('weathered limestone', pierApproach.x, foundation+height/2, z, 1.38, height, stairDepth+0.001)
    const x0=pierApproach.x-0.7125, x1=pierApproach.x+0.7125
    const joint=pierApproach.x + [-0.20,0.15,-0.09,0.23][i]
    for (const [a,b] of [[x0,joint-0.004],[joint+0.004,x1]]) {
      flag('capstone',a,b,z-stairDepth/2-0.004,z+stairDepth/2+0.022,top,0.053,0.018)
    }
  }
  return mesh.finish()
}

// A-frame pairs straddle the slope. The fishing net hangs from a level ridge
// and bows between its supports, with open diamonds that cast real shadows.
export function makeDryingRack() {
  const mesh = new HarborMesh()
  const left = -3.06, right = -0.40, z = -2.89
  const ridge = Math.max(shoreHeight(left,z), shoreHeight(right,z)) + 1.17
  for (const x of [left, right]) {
    for (const side of [-1,1]) {
      const foot = z + side*0.36
      mesh.beam('dock oak', [x,shoreHeight(x,foot)-0.047,foot], [x,ridge+0.058,z], 0.096, 0.085)
    }
    mesh.beam('dock oak', [x,ridge-0.64,z-0.195], [x,ridge-0.64,z+0.195], 0.054, 0.068)
  }
  mesh.beam('sawn cedar', [left-0.15,ridge,z], [right+0.15,ridge,z], 0.082, 0.10)
  const span = right-left-0.22, columns = 13, rows = 5
  function knot(col: number, row: number): Point {
    const u = (col+(row%2)*0.5)/columns, v = row/rows
    return [left+0.11 + u*span,
      ridge+0.055 - 0.79*v - 0.17*Math.sin(Math.PI*u)*(0.30+v),
      z + 0.075 + 0.24*v + 0.13*Math.sin(Math.PI*v)]
  }
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col <= columns-(row%2); col++) {
      for (const target of [col,col+(row%2 ? 1 : -1)]) {
        if (target < 0 || target > columns-((row+1)%2)) continue
        mesh.beam('hemp rope', knot(col,row), knot(target,row+1), 0.014,0.014)
      }
    }
  }
  for (let col = 0; col < columns; col++) {
    const a = knot(col,0), b = knot(col+1,0)
    mesh.beam('hemp rope', a,b,0.021,0.021)
    if (col%2 === 0) mesh.beam('ochre boards', a, [a[0]+0.082,a[1],a[2]], 0.054,0.054)
  }
  return mesh.finish()
}

export function makeLocalCargo(kind: 'crate' | 'barrel' | 'lantern') {
  const mesh = new HarborMesh()
  if (kind === 'crate') {
    const width = 0.88, height = 0.65, depth = 0.72
    mesh.box('dock oak', 0, height/2, 0, width-0.059, height-0.04, depth-0.059)
    for (const side of [-1,1]) {
      for (let row = 0; row < 5; row++) {
        const y = (row+0.5)*height/5
        mesh.box('sawn cedar', 0, y, side*depth/2, width, height/5-0.009, 0.037)
        mesh.box('sawn cedar', side*width/2, y, 0, 0.037, height/5-0.009, depth)
      }
      for (const x of [-0.38,0.38]) mesh.box('dock oak', x, height/2, side*(depth/2+0.026), 0.072, height+0.01, 0.039)
      mesh.beam('ochre boards', [-0.34,0.043,side*(depth/2+0.026)],
        [0.34,height-0.043,side*(depth/2+0.026)], 0.032, 0.075)
    }
    for (let i = 0; i < 5; i++) mesh.box('sawn cedar', 0, height+0.012, -depth/2+(i+0.5)*depth/5, width, 0.032, depth/5-0.005)
  } else if (kind === 'barrel') {
    mesh.column('dock oak', 0, 0, [[0,0.26], [0.16,0.32], [0.43,0.35], [0.70,0.32], [0.87,0.26]],
      18, ['sawn cedar','dock oak','ochre boards','dock oak'])
    for (const [y,r] of [[0.09,0.305],[0.22,0.336],[0.63,0.34],[0.778,0.308]]) {
      mesh.column('wrought iron', 0, 0, [[y-0.023,r], [y+0.023,r]], 18)
    }
    for (let j = -2; j <= 2; j++) {
      const z = j*0.092, span = 2*Math.sqrt(0.248**2-(Math.abs(z)+0.044)**2)
      mesh.box('sawn cedar', 0, 0.874, z, span, 0.018, 0.087)
    }
  } else {
    mesh.column('wrought iron', 0, 0, [[0,0.115], [0.036,0.12], [0.07,0.086]])
    mesh.column('lamp glass', 0, 0, [[0.065,0.081], [0.205,0.099], [0.228,0.086]], 8)
    for (const x of [-0.065,0.065]) for (const z of [-0.065,0.065]) {
      mesh.beam('wrought iron', [x,0.056,z], [1.18*x,0.232,1.18*z], 0.014, 0.014)
    }
    mesh.column('wrought iron', 0, 0, [[0.224,0.118], [0.242,0.118], [0.279,0.043], [0.30,0.038]], 12)
    let last: Point = [-0.07,0.263,0]
    for (let i = 1; i <= 10; i++) {
      const angle = Math.PI*i/10, p: Point = [-0.07*Math.cos(angle), 0.263+0.127*Math.sin(angle), 0]
      mesh.beam('wrought iron', last, p, 0.011, 0.011)
      last = p
    }
  }
  return mesh.finish()
}

export function makeMoorings(points: { post: Point; boat: Point }[]) {
  const mesh = new HarborMesh()
  for (const { post: [x,y,z], boat } of points) {
    ++mesh.object
    mesh.column('wrought iron', x, z, [[y,0.105], [y+0.030,0.105], [y+0.058,0.058],
      [y+0.16,0.044], [y+0.18,0.074], [y+0.21,0.074]])
    const from: Point = [x,y+0.112,z]
    let last = from
    for (let i = 1; i <= 11; i++) {
      const t = i/11
      const p: Point = [from[0]+(boat[0]-from[0])*t, from[1]+(boat[1]-from[1])*t-0.14*4*t*(1-t),
        from[2]+(boat[2]-from[2])*t]
      mesh.beam('hemp rope', last, p, 0.018, 0.018)
      last = p
    }
  }
  return mesh.finish()
}

function image(data: string, name: string) {
  const binary = atob(data.slice(data.indexOf(',') + 1))
  return new File([Uint8Array.from(binary, c => c.charCodeAt(0))], name, { type: 'image/png' })
}
const stoneMap = image(limestone, 'quay-limestone.png'), woodMap = image(timber, 'quay-timber.png')
export const harborSailcloth = image(sailcloth, 'lug-sailcloth.png')
const roughnessMap = image(stoneRoughness, 'quay-stone-roughness.png')
const finishes = new Map<string, SceneMaterial>([
  ['lime mortar', { color: [0.28, 0.27, 0.23], map: stoneMap, roughness: 0.96 }],
  ['sandstone', { color: [0.59, 0.52, 0.39], map: stoneMap, roughness: 0.97, roughnessMap }],
  ['weathered limestone', { color: [0.53, 0.53, 0.46], map: stoneMap, roughness: 0.90, roughnessMap }],
  ['warm freestone', { color: [0.62, 0.53, 0.43], map: stoneMap, roughness: 0.93, roughnessMap }],
  ['capstone', { color: [0.67, 0.64, 0.57], map: stoneMap, roughness: 0.70, roughnessMap }],
  ['dock oak', { color: [0.39, 0.26, 0.13], map: woodMap, roughness: 0.72 }],
  ['sawn cedar', { color: [0.63, 0.45, 0.26], map: woodMap, roughness: 0.86 }],
  ['ochre boards', { color: [0.62, 0.31, 0.13], map: woodMap, roughness: 0.60 }],
  ['wrought iron', { color: [0.19, 0.20, 0.19], metallic: 0.82, roughness: 0.41 }],
  ['hemp rope', { color: [0.53, 0.40, 0.24], roughness: 0.95 }],
  ['lamp glass', { color: [0.76, 0.60, 0.30], roughness: 0.16 }],
  ['fish market silver', { color: [0.78,0.83,0.76], roughness: 0.24 }],
  ['fish market mackerel', { color: [0.22,0.38,0.42], roughness: 0.27 }],
  ['aged zinc', { color: [0.27, 0.35, 0.35], metallic: 0.70, roughness: 0.49 }],
  ['weathered zinc', { color: [0.245, 0.325, 0.31], metallic: 0.41, roughness: 0.63 }],
  ['zinc flashing', { color: [0.41, 0.47, 0.46], metallic: 0.92, roughness: 0.31 }],
  ['harbor blue', { color: [0.105, 0.255, 0.275], map: woodMap, roughness: 0.53 }],
])

// Build the fittings as indexed surfaces with physical thickness so the camera
// and the top-light solver agree on steps, rope shadows, and exposed cargo.
export class HarborMesh {
  private vertices: number[] = []
  private uvs: number[] = []
  private batches = new Map<string, number[]>()
  object = 2

  face(material: string, corners: Point[], outward: Point) {
    const n = unit(cross(subtract(corners[1], corners[0]), subtract(corners[2], corners[0])))
    const flip = dot(n, outward) < 0, sign = flip ? -1 : 1
    let batch = this.batches.get(material)
    if (!batch) { batch = []; this.batches.set(material, batch) }
    const edge = subtract(corners[1], corners[0]), other = subtract(corners.at(-1)!, corners[0])
    const grain = unit(Math.hypot(...edge) >= Math.hypot(...other) ? edge : other)
    const across = cross(n, grain), first = this.vertices.length / sceneVertexStride
    for (const p of corners) {
      this.vertices.push(...p, sign*n[0], sign*n[1], sign*n[2], 1,1,1,0,this.object)
      this.uvs.push(dot(p, grain)*0.92, dot(p, across)*1.34)
    }
    for (let i = 2; i < corners.length; i++) batch.push(first, first+(flip ? i : i-1), first+(flip ? i-1 : i))
  }

  beam(material: string, a: Point, b: Point, width: number, depth: number) {
    const axis = unit(subtract(b, a)), guide: Point = Math.abs(axis[1]) > 0.97 ? [0,0,1] : [0,1,0]
    const side = unit(cross(axis, guide)), vertical = cross(side, axis)
    const section = (origin: Point) => [[-1,-1], [1,-1], [1,1], [-1,1]].map(([i,j]): Point => [
      origin[0] + 0.5*(i*width*side[0]+j*depth*vertical[0]),
      origin[1] + 0.5*(i*width*side[1]+j*depth*vertical[1]),
      origin[2] + 0.5*(i*width*side[2]+j*depth*vertical[2])])
    const start = section(a), end = section(b)
    this.face(material, start, subtract(a,b))
    this.face(material, end, axis)
    for (let i = 0; i < 4; i++) {
      const j = (i+1) % 4
      const outward = start[i].map((v,k) => v + start[j][k] - 2*a[k]) as Point
      this.face(material, [start[i], end[i], end[j], start[j]], outward)
    }
  }

  box(mat: string, x: number, y: number, z: number, w: number, h: number, d: number) {
    this.beam(mat, [x-w/2,y,z], [x+w/2,y,z], d,h)
  }

  column(mat: string, x: number, z: number, rings: [number, number][], sides = 14, staves?: string[]) {
    const sections = rings.map(([y,r]) => Array.from({length:sides}, (_,i): Point => [
      x+r*Math.cos(2*Math.PI*i/sides), y, z+r*Math.sin(2*Math.PI*i/sides)]))
    this.face(mat, sections[0], [0,-1,0])
    this.face(mat, sections.at(-1)!, [0,1,0])
    for (let row = 1; row < sections.length; row++) for (let i = 0; i < sides; i++) {
      const j = (i+1) % sides, a = sections[row-1], b = sections[row]
      this.face(staves?.[i%staves.length] ?? mat, [a[i], a[j], b[j], b[i]],
        [a[i][0]+a[j][0]-2*x, 0, a[i][2]+a[j][2]-2*z])
    }
  }

  finish() {
    const indices: number[] = [], materials = new Map<string, SceneMaterial>()
    const draws = Array.from(this.batches, ([material, batch]) => {
      const firstIndex = indices.length
      for (const index of batch) indices.push(index)
      materials.set(material, finishes.get(material)!)
      return { material, hasUV: true, indexCount: batch.length, firstIndex }
    })
    const mesh: SceneMesh = { vertices: new Float32Array(this.vertices), indices: new Uint32Array(indices),
      uvs: new Float32Array(this.uvs), draws }
    return { mesh, materials, warnings: new Set<string>() }
  }
}
