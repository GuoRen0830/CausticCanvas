import { shoreHeight, tideUV } from './cove-shore'
import { addIntertidalLife } from './cove-intertidal-life'
import { sceneVertexStride, type SceneMaterial, type SceneDraw, type SceneMesh } from '../rendering/scene-mesh'

type Plan = [number, number]
type Point = [number, number, number]
type Sample = { position: Point; index: number; edge: number }
type Pool = { outline: Plan[]; level: number }

// Surveyed in the harbor's XZ plane. The long shared fractures divide the
// two basins into shelves instead of giving each pool a circular stone curb.
const shelf: Plan[] = [
  [-5.68,0.32], [-5.10,0.09], [-4.64,0.25], [-4.27,0.51], [-3.91,0.44], [-3.57,0.63],
  [-3.67,0.99], [-3.13,1.23], [-3.18,1.51], [-2.94,1.76], [-3.20,1.91], [-3.04,2.21],
  [-3.41,2.50], [-3.99,2.63], [-4.27,2.42], [-4.62,2.48], [-4.86,2.25], [-5.20,2.20],
  [-5.12,1.91], [-5.49,1.70], [-5.40,1.42], [-5.73,1.23], [-5.62,0.90], [-5.82,0.75],
]
const pools: Pool[] = [
  { level: 0.075, outline: [[-5.30,0.91],[-4.99,0.68],[-4.70,0.78],[-4.61,0.92],
    [-4.34,1.01],[-4.38,1.18],[-4.62,1.24],[-4.80,1.45],[-5.09,1.38],[-5.35,1.17]] },
  { level: 0.039, outline: [[-4.37,1.78],[-4.17,1.62],[-3.89,1.68],[-3.79,1.84],
    [-3.50,1.92],[-3.55,2.07],[-3.84,2.16],[-4.04,2.07],[-4.29,2.10],[-4.42,1.95]] },
]
const faults: [Plan, Plan][] = [
  [[-5.75,0.80],[-5.23,0.56]], [[-5.45,1.62],[-4.67,1.51]], [[-4.67,1.51],[-4.08,1.32]],
  [[-4.08,1.32],[-3.43,1.39]], [[-4.33,0.49],[-4.13,0.91]], [[-3.64,1.51],[-3.49,1.71]],
  [[-4.85,2.25],[-4.48,2.10]], [[-3.98,2.60],[-3.81,2.34]],
]

const saturate = (v: number) => Math.max(0, Math.min(1, v))
function segment(x: number, z: number, a: Plan, b: Plan) {
  const dx = b[0] - a[0], dz = b[1] - a[1]
  const t = saturate(((x-a[0])*dx + (z-a[1])*dz)/(dx*dx + dz*dz))
  return Math.hypot(x-a[0]-t*dx, z-a[1]-t*dz)
}
function signedDistance(x: number, z: number, outline: Plan[]) {
  let inside = false, distance = Infinity
  for (let i = 0; i < outline.length; i++) {
    const a = outline[i], b = outline[(i+1) % outline.length]
    distance = Math.min(distance, segment(x, z, a, b))
    if ((a[1] > z) !== (b[1] > z) && x < a[0] + (z-a[1])*(b[0]-a[0])/(b[1]-a[1])) inside = !inside
  }
  return inside ? distance : -distance
}
function relief(x: number, z: number) {
  const grain = 0.016*Math.sin(17*x + 4*z) + 0.011*Math.sin(11*z - 3*x)
  const edge = signedDistance(x, z, shelf) + grain
  // Nearly parallel raised strata, breached by a few cross-cutting joints.
  const bed = 0.5 + 0.5*Math.sin(10.5*(z + 0.30*x) + 0.42*Math.sin(2.2*x))
  let top = 0.179 - 0.046*(z-0.5) + 0.026*Math.sin(3.6*x + z) + 0.042*saturate(3*bed-1.1) + 0.55*grain
  const joint = Math.min(...faults.map(([a,b]) => segment(x, z, a, b)))
  top -= 0.095*saturate(1 - joint/0.09)
  const bedrock = Math.max(-0.13, shoreHeight(x,z)-0.055)
  let y = bedrock + saturate((edge+0.036)/0.23)*(top-bedrock)
  for (const pool of pools) {
    const d = signedDistance(x, z, pool.outline) + 0.50*grain
    const incision = saturate((d+0.060)/0.13)
    y += incision*(pool.level-0.055-y)
  }
  return { y, edge }
}

// Both the lens solver and the camera get the same rock mesh and the same
// waterline, clipped to the rock faces which actually lie below each pool.
export function makeCoveTidePools(stone: SceneMaterial, water: SceneMaterial) {
  const materials = new Map<string, SceneMaterial>([
    ['bedrock', { ...stone, color: [0.295,0.275,0.217], roughness: 0.90 }],
    ['wet strata', { ...stone, color: [0.148,0.164,0.138], roughness: 0.34 }],
    ['algal rock', { ...stone, color: [0.142,0.175,0.102], roughness: 0.46 }],
    ['rock pool', { color: [0.78,0.90,0.75], map: water.map, clamp: true, roughness: 0.18 }],
    ['limpet lip', { color: [0.255,0.225,0.166], roughness: 0.72 }],
    ['limpet shell', { color: [0.62,0.56,0.42], roughness: 0.37 }],
    ['limpet rib', { color: [0.45,0.42,0.32], roughness: 0.51 }],
    ['tidepool mussel', { color: [0.037,0.063,0.099], roughness: 0.27 }],
    ['tidepool nacre', { color: [0.35,0.43,0.49], roughness: 0.31 }],
    ['olive kelp', { color: [0.164,0.191,0.047], roughness: 0.32, doubleSided: true }],
    ['amber kelp', { color: [0.304,0.164,0.042], roughness: 0.31, doubleSided: true }],
    ['ochre sea star', { color: [0.61,0.162,0.049], roughness: 0.49 }],
    ['sea star ossicles', { color: [0.84,0.45,0.21], roughness: 0.58 }],
    ['acorn barnacle', { color: [0.54,0.47,0.37], roughness: 0.61 }],
    ['barnacle aperture', { color: [0.135,0.105,0.084], roughness: 0.37 }],
  ])
  const vertices: number[] = [], uvs: number[] = [], batches = new Map<string, number[]>()
  function vertex(p: Point, n: Point, uv: Plan, object: number) {
    const index = vertices.length / sceneVertexStride, length = Math.hypot(...n)
    vertices.push(...p, n[0]/length, n[1]/length, n[2]/length, 1,1,1,0,object)
    uvs.push(...uv)
    return index
  }
  function tri(material: string, a: number, b: number, c: number) {
    let batch = batches.get(material)
    if (!batch) { batch = []; batches.set(material, batch) }
    batch.push(a,b,c)
  }
  function rockNormal(x: number, z: number): Point {
    const d = 0.021
    return [relief(x-d,z).y-relief(x+d,z).y, 2*d, relief(x,z-d).y-relief(x,z+d).y]
  }
  function waterline(corners: Point[], pool: Pool, object: number) {
    const clipped: Point[] = []
    for (let i = 0; i < 3; i++) {
      const a = corners[i], b = corners[(i+1)%3]
      if (a[1] <= pool.level) clipped.push([a[0],pool.level,a[2]])
      if ((a[1] < pool.level && b[1] > pool.level) || (b[1] < pool.level && a[1] > pool.level)) {
        const t = (pool.level-a[1])/(b[1]-a[1])
        clipped.push([a[0]+t*(b[0]-a[0]),pool.level,a[2]+t*(b[2]-a[2])])
      }
    }
    if (clipped.length < 3) return
    // The lagoon image remains continuous over each pool in the OBJ bake.
    const ids = clipped.map(p => vertex(p,[0,1,0],tideUV(p[0],p[2]),object))
    for (let j = 2; j < ids.length; j++) tri('rock pool',ids[0],ids[j-1],ids[j])
  }
  const cols = 52, rows = 44, grid: Sample[] = []
  for (let j=0;j<=rows;j++) for (let i=0;i<=cols;i++) {
    const x=-5.91+3.10*i/cols, z=-0.02+2.77*j/rows
    const { y, edge } = relief(x,z), p: Point = [x,y,z]
    grid.push({ position:p, edge, index:vertex(p,rockNormal(x,z),[0.58*x,-0.58*z],2) })
  }
  function rockFace(a: Sample,b: Sample,c: Sample) {
    if (Math.max(a.edge,b.edge,c.edge) < -0.035) return
    const points = [a.position,b.position,c.position]
    const x=(points[0][0]+points[1][0]+points[2][0])/3, z=(points[0][2]+points[1][2]+points[2][2])/3
    const y=(points[0][1]+points[1][1]+points[2][1])/3
    const growth=0.024*Math.sin(6*x+1.7*z) + 0.023*Math.sin(8.8*z-2.1*x)
    const finish=y+growth < 0.095 ? 'algal rock' : y+0.35*growth < 0.15 ? 'wet strata' : 'bedrock'
    tri(finish,a.index,b.index,c.index)
    for (const [i,pool] of pools.entries()) if (signedDistance(x,z,pool.outline) > -0.078) {
      waterline(points,pool,3+i)
    }
  }
  for (let j=0;j<rows;j++) for (let i=0;i<cols;i++) {
    const a = j*(cols+1)+i, b=a+cols+1
    rockFace(grid[a],grid[b],grid[a+1]); rockFace(grid[a+1],grid[b],grid[b+1])
  }
  // Clusters occupy an exposed shoulder above the waterline. Each shell has
  // a low, off-center apex and alternating ribs rather than a painted dot.
  const colonies: [number,number,number][] = [
    [-5.35,0.72,0.052],[-5.27,0.63,0.039],[-5.19,0.62,0.045],
    [-4.26,1.48,0.052],[-4.35,1.45,0.037],[-4.23,1.57,0.035],
    [-3.40,2.15,0.041],[-3.36,2.24,0.034],
  ]
  for (const [s,[x,z,r]] of colonies.entries()) {
    const rim: number[] = [], lips: number[] = [], object = 5+s
    const normal=rockNormal(x,z), ground=relief(x,z).y
    const angle=s*1.17, tangent=(dx:number,dz:number) => ground-(normal[0]*dx+normal[2]*dz)/normal[1]
    const peak=vertex([x+r*0.22*Math.cos(angle),ground+r*0.58,z+r*0.22*Math.sin(angle)],normal,[0.5,0.5],object)
    for (let k=0;k<12;k++) {
      const u=2*Math.PI*k/12, radial=(k%2 ? 0.9 : 1)*r
      const dx=radial*(Math.cos(u)*Math.cos(angle)-0.75*Math.sin(u)*Math.sin(angle))
      const dz=radial*(Math.cos(u)*Math.sin(angle)+0.75*Math.sin(u)*Math.cos(angle))
      const h=tangent(dx,dz), n: Point = [0.65*dx/r,1,0.65*dz/r]
      rim.push(vertex([x+dx,h+0.007,z+dz],n,[0.5+0.43*Math.cos(u),0.5+0.43*Math.sin(u)],object))
      lips.push(vertex([x+1.1*dx,h-0.014,z+1.1*dz],n,[0.5+0.49*Math.cos(u),0.5+0.49*Math.sin(u)],object))
    }
    for (let k=0;k<rim.length;k++) {
      const next=(k+1)%rim.length
      tri(k%3 ? 'limpet shell' : 'limpet rib',peak,rim[next],rim[k])
      tri('limpet lip',rim[k],rim[next],lips[k]); tri('limpet lip',lips[k],rim[next],lips[next])
    }
  }
  // Drape the wrack and shell margins on the triangles actually exported.
  // The analytic relief has sharper fissures than this sampled shelf.
  function shelfHeight(x: number, z: number) {
    const u = Math.max(0, Math.min(cols, (x+5.91)*cols/3.10))
    const v = Math.max(0, Math.min(rows, (z+0.02)*rows/2.77))
    const i = Math.min(Math.floor(u),cols-1), j = Math.min(Math.floor(v),rows-1)
    const a = j*(cols+1)+i, b = a+cols+1, s = u-i, t = v-j
    const y0 = grid[a].position[1], y1 = grid[a+1].position[1]
    const y2 = grid[b].position[1], y3 = grid[b+1].position[1]
    return s+t <= 1 ? y0+s*(y1-y0)+t*(y2-y0) : y3+(1-s)*(y2-y3)+(1-t)*(y1-y3)
  }
  let object = 5 + colonies.length
  // Blue mussels gather in creases on several sides of the pools. The wider
  // end of a valve has a pale worn edge; most of its surface stays inky.
  const musselBeds = [[-5.38,1.00,0.8],[-4.62,1.47,-0.36],[-3.48,1.63,1.87],[-4.34,2.24,-0.76]]
  for (const [bed,[ox,oz,heading]] of musselBeds.entries()) {
    for (let i=0;i<8;i++,object++) {
      const radius = 0.081*Math.sqrt(i), spiral = heading + 2.4*i
      const x = ox + radius*Math.cos(spiral), z = oz + 0.62*radius*Math.sin(spiral)
      const angle = heading + 0.57*Math.sin(1.8*i+bed)
      const cos = Math.cos(angle), sin = Math.sin(angle)
      const length = 0.052 + 0.013*Math.cos(2.3*i-bed)
      const ring: number[] = [], shoulder: number[] = []
      const peak = vertex([x+0.15*length*cos,shelfHeight(x,z)+0.45*length,z+0.15*length*sin],
        [0,1,0],[0.56,0.5],object)
      for (let j=0;j<10;j++) {
        const phi = j*Math.PI/5, c = Math.cos(phi), s = Math.sin(phi)
        const dx = length*(c*cos-s*(0.48+0.14*c)*sin)
        const dz = length*(c*sin+s*(0.48+0.14*c)*cos)
        const n: Point = [0.62*c*cos-s*sin,1,0.62*c*sin+s*cos]
        ring.push(vertex([x+dx,shelfHeight(x+dx,z+dz)-0.003,z+dz],n,[0.5+0.49*c,0.5+0.49*s],object))
        const sx = x+0.88*dx, sz = z+0.88*dz
        shoulder.push(vertex([sx,shelfHeight(sx,sz)+0.16*length,sz],n,[0.5+0.42*c,0.5+0.42*s],object))
      }
      for (let j=0;j<ring.length;j++) {
        const k = (j+1)%ring.length
        const lip = (j === 0 || j === 9) && i%3 !== 2 ? 'tidepool nacre' : 'tidepool mussel'
        tri(lip,ring[j],shoulder[j],ring[k])
        tri(lip,ring[k],shoulder[j],shoulder[k])
        tri('tidepool mussel',shoulder[j],peak,shoulder[k])
      }
    }
  }
  // Blades fan out from a common holdfast. A raised midrib, curled edges and
  // a lifted bow give grazing light a different highlight on each blade.
  function frond(x: number,z: number,angle: number,length: number,halfWidth: number,phase: number,finish: string) {
    const count = 11, sections: Point[][] = [], strip: number[][] = []
    const cos = Math.cos(angle), sin = Math.sin(angle)
    const bend = (t: number) => length*(0.14*Math.sin(Math.PI*t+0.4*phase) -
      0.14*Math.sin(0.4*phase) + 0.052*Math.sin(2*Math.PI*t))
    for (let i=0;i<=count;i++) {
      const t = i/count, arch = Math.sin(Math.PI*t)
      const spread = halfWidth*(0.035 + 0.965*arch**0.77)*(0.92+0.08*Math.sin(16*t+phase))
      const side = bend(t), tangent = (bend(t+0.001)-bend(t-0.001))/(0.002*length)
      const norm = Math.hypot(1,tangent), nx = (-sin-tangent*cos)/norm, nz = (cos-tangent*sin)/norm
      const centerX = x+t*length*cos-side*sin, centerZ = z+t*length*sin+side*cos
      const across: Point[] = []
      for (const lane of [-1,0,1]) {
        const px = centerX+nx*spread*lane, pz = centerZ+nz*spread*lane
        let floor = shelfHeight(px,pz)
        for (const pool of pools) if (signedDistance(px,pz,pool.outline)>0) floor = Math.max(floor,pool.level)
        const curl = 0.16*halfWidth*lane*Math.sin(11*t+phase) + (lane === 0 ? 0.17*halfWidth : 0)
        across.push([px,floor+0.002+arch*(0.037+curl),pz])
      }
      sections.push(across)
    }
    for (let i=0;i<=count;i++) {
      const lo = sections[Math.max(i-1,0)], hi = sections[Math.min(i+1,count)], row = sections[i]
      const ids: number[] = []
      for (let lane=0;lane<3;lane++) {
        const l = row[Math.max(lane-1,0)], r = row[Math.min(lane+1,2)]
        const ax = r[0]-l[0], ay = r[1]-l[1], az = r[2]-l[2]
        const bx = hi[lane][0]-lo[lane][0], by = hi[lane][1]-lo[lane][1], bz = hi[lane][2]-lo[lane][2]
        const n: Point = [ay*bz-az*by,az*bx-ax*bz,ax*by-ay*bx]
        ids.push(vertex(row[lane],n,[lane/2,i/count],object))
      }
      strip.push(ids)
      if (i > 0) for (let lane=0;lane<2;lane++) {
        tri(finish,strip[i-1][lane],strip[i-1][lane+1],ids[lane])
        tri(finish,ids[lane],strip[i-1][lane+1],ids[lane+1])
      }
    }
    object++
  }
  for (const [fan,[x,z,angle,scale]] of [[-4.86,1.63,-0.26,1],[-3.65,2.22,3.15,0.80],[-5.17,0.60,0.42,0.65]].entries()) {
    for (let i=0;i<4;i++) {
      const phase = i*1.8+fan
      frond(x,z,angle+0.39*(i-1.5),scale*(0.54+0.17*Math.sin(phase)),scale*(0.042+0.011*Math.cos(phase)),
        phase,i === 1 || (fan === 2 && i === 3) ? 'amber kelp' : 'olive kelp')
    }
  }
  addIntertidalLife({ height:shelfHeight, vertex, tri },object)
  const draws: SceneDraw[] = [], indices: number[] = []
  for (const [material,batch] of batches) {
    draws.push({material, firstIndex:indices.length,indexCount:batch.length,hasUV:true,
      doubleSided: materials.get(material)?.doubleSided})
    for (const id of batch) indices.push(id)
  }
  const mesh: SceneMesh = { vertices:new Float32Array(vertices), uvs:new Float32Array(uvs), indices:new Uint32Array(indices), draws }
  return { mesh, materials, warnings:new Set<string>() }
}
