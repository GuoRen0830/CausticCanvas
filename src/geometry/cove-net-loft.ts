import { HarborMesh, harborSailcloth } from './cove-quayside'
import { makeBellLoftRoof } from './cove-bell-loft'
import { sceneVertexStride } from '../rendering/scene-mesh'

type Point = [number, number, number]
const floats = [[-1.002,0.885,0.123],[-0.757,0.901,0.088],[-0.525,0.904,0.113]]
const benchTop = 0.4315

function dressFloatBench(loft: HarborMesh) {
  // A salvaged three-cornered sail, laced to a headrail and a single projecting timber knee.
  // All three corners are fixed; the bias folds deepen between them and die away at the clew.
  const clew: Point = [-1.04, 1.025, 1.385]
  const fabric = (u: number, v: number): Point => {
    const bow = Math.sin(Math.PI*v), fold = Math.sin(5*Math.PI*u) * bow
    return [(-1.21 + 1.02*u)*(1-v) + clew[0]*v,
      (1.295+0.027*u)*(1-v) + clew[1]*v - 0.046*bow + 0.026*(1-v)*fold,
      0.704*(1-v) + clew[2]*v - 0.045*4*u*(1-u)*(1-v)*bow]
  }
  loft.beam('dock oak', [-1.23,1.31,0.669], [-0.16,1.34,0.669], 0.075,0.052)
  loft.beam('sawn cedar', [-1.20,1.19,0.67], clew, 0.041,0.045)
  loft.beam('dock oak', [-1.20,0.90,0.66], [-1.055,1.04,1.32], 0.035,0.039)
  const panels = ['sun-faded tanbark', 'tanbark linen', 'tanbark linen', 'sun-faded tanbark', 'tanbark linen', 'waxed flax']
  const stops = [0,0.2,0.44,0.71,1]
  for (let i = 0; i < panels.length; i++) {
    const u = i/panels.length, u1 = (i+1)/panels.length
    for (let j = 1; j < stops.length; j++) {
      const a = fabric(u,stops[j-1]), b = fabric(u1,stops[j-1]), d = fabric(u,stops[j])
      const corners = stops[j] === 1 ? [a,b,clew] : [a,b,fabric(u1,stops[j]),d]
      loft.face(panels[i], corners, [0,1,0])
      loft.face(panels[i], corners.map(([x,y,z]) => [x,y-0.008,z]), [0,-1,0])
      if (i === 0) loft.beam('hemp rope', a, d, 0.016,0.016)
      if (i === panels.length-1) loft.beam('hemp rope', b, fabric(u1,stops[j]), 0.016,0.016)
    }
    const [x,y,z] = fabric(u,0)
    loft.beam('hemp rope', [x,y-0.002,z+0.007],[x+0.012,y+0.019,0.663], 0.014,0.014)
  }
  loft.box('lens brass', clew[0],clew[1],clew[2],0.040,0.027,0.044)

  // Two slats and through-wall brackets support the glass floats at the window sill.
  for (const z of [0.782,0.956]) loft.box('sawn cedar', -0.768,0.41,z,0.85,0.043,0.170)
  for (const x of [-1.088,-0.444]) {
    loft.beam('wrought iron', [x,0.24,0.665],[x,0.391,1.008], 0.031,0.032)
    loft.box('dock oak', x,0.38,0.851,0.055,0.025,0.378)
  }
  loft.box('harbor blue', -0.768,0.439,1.035,0.85,0.034,0.023)
  for (const [x,z,r] of floats) {
    const y = benchTop + 0.91*r
    const latitudes = [-0.91,-0.63,-0.27,0.15,0.52,0.83,0.94]
    const radius = (v: number) => r*Math.sqrt(1-v*v)
    loft.column('lamp glass', x,z,latitudes.map(v => [y+r*v,radius(v)]), 12)
    loft.column('lamp glass', x,z,[[y+0.94*r,0.28*r],[y+1.08*r,0.25*r]], 10)
    // Four strands cross the equator under the glass lip, like the carrying net of a fishing float.
    for (let strand = 0; strand < 4; strand++) {
      const onNet = (lat: number): Point => {
        const angle = Math.PI*strand/2 + 0.24 + 0.30*lat
        return [x+(radius(lat)+0.003)*Math.cos(angle), y+lat*r, z+(radius(lat)+0.003)*Math.sin(angle)]
      }
      for (let k = 1; k < latitudes.length; k++) {
        loft.beam('hemp rope', onNet(latitudes[k-1]), onNet(latitudes[k]), 0.010,0.010)
      }
    }
  }
}

function dressGearRack(loft: HarborMesh) {
  // Each rear-wall bracket carries two oars. The blades have a real bevel and a painted tip.
  for (const x of [-0.64,0.62]) {
    loft.box('dock oak', x,0.73,-0.743,0.081,0.71,0.072)
    for (const [rail,dir] of [[0.53,1],[0.88,-1]]) {
      const y = rail + (dir*x < 0 ? 0.008 : -0.009) - 0.0175
      loft.box('lens brass', x,y,-0.797,0.035,0.035,0.157)
      loft.box('lens brass', x,y+0.035,-0.866,0.035,0.085,0.027)
    }
  }
  for (const [y,dir] of [[0.53,1],[0.88,-1]]) {
    const profiles = [[-1.05,0.020],[-0.96,0.026],[-0.88,0.019],[0.38,0.026],
      [0.64,0.080],[0.84,0.103],[0.91,0.102],[1.05,0.080],[1.10,0.046]]
    const sections = profiles.map(([d,w]): Point[] => {
      const h = y + 0.048 + d*0.031
      return [[dir*d,h+w,-0.810],[dir*d,h,-0.827],[dir*d,h-w,-0.810],[dir*d,h,-0.791]]
    })
    loft.face('sawn cedar', sections[0], [-dir,0,0])
    for (let i = 1; i < sections.length; i++) {
      const prev = sections[i-1], next = sections[i]
      const mat = i === 6 ? 'cream band' : i > 6 ? 'harbor blue' : 'sawn cedar'
      for (let edge = 0; edge < 4; edge++) {
        const j = (edge+1)%4
        loft.face(mat, [prev[edge],prev[j],next[j],next[edge]], [0,edge === 0 || edge === 3 ? 1 : -1, edge < 2 ? -1 : 1])
      }
    }
    loft.face('harbor blue', sections.at(-1)!, [dir,0,0])
  }

  // A rope hanging over a cleat on the gable returns the work area around the corner.
  loft.box('dock oak', 1.283,0.87,-0.23,0.059,0.19,0.22)
  loft.box('lens brass', 1.326,0.902,-0.23,0.091,0.047,0.23)
  const loop = (t: number): Point => [1.334+0.033*t, 0.665+0.24*Math.cos(t*4*Math.PI),
    -0.23+0.126*Math.sin(t*4*Math.PI)]
  for (let i = 0; i < 34; i++) loft.beam('hemp rope', loop(i/34),loop((i+1)/34),0.022,0.022)
  loft.beam('hemp rope',loop(1),[1.375,0.54,-0.103],0.021,0.021)
}

// A louvred opening through each gable lets the loft dry its nets.
function ventedGable(loft: HarborMesh, x: number, back: number, front: number, eave: number, peak: number) {
  const sign = Math.sign(x), z0 = -0.21, z1 = 0.21, bottom = eave+0.045, top = eave+0.205
  const p = (z: number, y: number, offset = 0): Point => [x+sign*offset,y,z]
  const leftRoof = peak+(eave-peak)*z0/back, rightRoof = peak+(eave-peak)*z1/front
  const outward: Point = [sign,0,0]
  loft.face('sawn cedar',[p(back,eave),p(z0,eave),p(z0,leftRoof)],outward)
  loft.face('sawn cedar',[p(z1,eave),p(front,eave),p(z1,rightRoof)],outward)
  loft.face('sawn cedar',[p(z0,eave),p(z1,eave),p(z1,bottom),p(z0,bottom)],outward)
  loft.face('sawn cedar',[p(z0,top),p(z1,top),p(z1,rightRoof),p(0,peak),p(z0,leftRoof)],outward)
  for (const z of [z0,z1]) loft.box('harbor blue',x,(top+bottom)/2,z,0.067,top-bottom,0.024)
  for (const y of [bottom,top]) loft.box('harbor blue',x,y,0,0.076,0.025,z1-z0+0.024)
  for (let slat = 0; slat < 3; slat++) {
    const y = bottom+0.034+0.045*slat
    const corners = [p(z0,y+0.022,-0.022),p(z1,y+0.022,-0.022),p(z1,y-0.010,0.022),p(z0,y-0.010,0.022)]
    loft.face('weathered zinc',corners,[0,1,0])
    loft.face('aged zinc',corners.map(([px,py,pz]) => [px,py-0.006,pz]),[0,-1,0])
    loft.beam('zinc flashing',corners[2],corners[3],0.008,0.008)
  }
}

function dressRearLights(loft: HarborMesh, back: number, bottom: number, top: number) {
  const bay=0.64, height=top-bottom, z=back-0.031
  loft.box('cream band',0,top,z,1.98,0.044,0.085)
  loft.box('sawn cedar',0,bottom-0.007,z-0.012,1.99,0.043,0.115)
  for (const x of [-0.96,-0.32,0.32,0.96]) loft.box('cream band',x,(bottom+top)/2,z,0.036,height,0.076)
  // Two timber shutters are held out on iron stays; the middle light opens into the loft.
  for (const side of [-1,1]) {
    const x=side*bay, angle=side<0 ? 0.83 : 0.53
    const point=(dx: number,t: number,offset=0):Point=>[x+dx,top-t*height*Math.cos(angle),z-0.044-t*height*Math.sin(angle)-offset]
    for (let i=0; i<6; i++) {
      const dx=(i-2.5)*(bay-0.05)/6
      loft.beam('harbor blue',point(dx,0),point(dx,0.92),(bay-0.05)/6-0.004,0.018)
    }
    for (const t of [0.10,0.78]) loft.beam('sawn cedar',point(-0.293,t,0.012),point(0.293,t,0.012),0.029,0.021)
    for (const dx of [-0.245,0.245]) {
      const hinge=point(dx,0)
      loft.beam('wrought iron',hinge,point(dx,0.30,0.019),0.033,0.012)
      loft.beam('wrought iron',[x+dx,bottom+0.019,z],point(dx,0.81),0.012,0.012)
    }
  }
  for (const x of [-0.097,0.097]) loft.box('harbor blue',x,(bottom+top)/2,z,0.018,height,0.031)
}

export function makeCoveNetLoft() {
  const loft = new HarborMesh()
  const left = -1.24, right = 1.24, back = -0.69, front = 0.64
  const ridge = 1.85, pitch = 0.59, roofY = (z: number) => ridge - Math.abs(z) * pitch
  const eave = roofY(back)
  const siding = ['ochre boards', 'sawn cedar', 'ochre boards', 'ochre boards']
  const doorLeft = -0.11, doorRight = 0.54, windowLeft = -1.06, windowRight = -0.48
  loft.box('weathered limestone', 0, 0.055, 0, 2.50, 0.11, 1.42)
  loft.box('dock oak', 0, 0.128, 0, 2.46, 0.04, 1.36)

  // The door opens into a furnished interior. The opening and recessed
  // window both pass through the siding, including its shadow-casting edges.
  const rows = 10, sill = 0.10, rowHeight = (eave - sill) / rows
  for (let row = 0; row < rows; row++) {
    const y = sill + (row + 0.5) * rowHeight
    const mat = siding[row % siding.length], lip = (row % 2) * 0.004
    const board = (x0: number, x1: number) => loft.box(mat, (x0+x1)/2, y, front+lip,
      x1-x0, rowHeight, 0.057)
    if (row < 8) {
      if (row >= 4 && row < 7) {
        board(left, windowLeft)
        board(windowRight, doorLeft)
      } else board(left, doorLeft)
      board(doorRight, right)
    } else board(left, right)
    if (row === 7 || row === 8) {
      for (const [a,b] of [[left,-0.96],[0.96,right]]) loft.box(mat,(a+b)/2,y,back-lip,b-a,rowHeight,0.058)
    } else loft.box(mat,0,y,back-lip,right-left,rowHeight,0.058)
    for (const x of [left, right]) {
      loft.box(siding[(row+1)%siding.length], x, y, (front+back)/2, 0.06+lip, rowHeight, front-back)
    }
  }
  for (const x of [left,right]) {
    ventedGable(loft,x,back,front,eave,ridge-0.02)
    for (const z of [front,back]) {
      loft.box('dock oak', x, (eave+0.10)/2, z, 0.087, eave-0.10, 0.082)
      loft.beam('dock oak', [x,eave-0.032,z],[x,ridge-0.045,0], 0.061,0.064)
    }
  }

  dressRearLights(loft,back,sill+7*rowHeight,sill+9*rowHeight)
  const windowBottom = sill+4*rowHeight, windowTop = sill+7*rowHeight
  loft.box('weathered zinc', (windowLeft+windowRight)/2, (windowTop+windowBottom)/2, front-0.042,
    windowRight-windowLeft, windowTop-windowBottom, 0.015)
  for (const x of [windowLeft, (windowLeft+windowRight)/2, windowRight]) {
    loft.box('harbor blue', x, (windowBottom+windowTop)/2, front+0.018, 0.034, windowTop-windowBottom+0.04, 0.086)
  }
  for (const y of [windowBottom, (windowBottom+windowTop)/2, windowTop]) {
    loft.box('harbor blue', (windowLeft+windowRight)/2, y, front+0.030, windowRight-windowLeft+0.08, 0.038, 0.086)
  }
  loft.box('sawn cedar', (windowLeft+windowRight)/2, windowBottom-0.010, front+0.05, 0.71, 0.045, 0.15)
  const doorTop = sill+8*rowHeight
  for (const x of [doorLeft, doorRight]) {
    loft.box('dock oak', x, (doorTop+sill)/2, front+0.016, 0.060, doorTop-sill, 0.10)
  }
  loft.box('dock oak', (doorLeft+doorRight)/2, doorTop+0.018, front+0.018, 0.78,0.086,0.11)
  loft.box('wrought iron', 0.54, doorTop+0.093, front+0.063, 1.44,0.028,0.035)
  for (let p = 0; p < 7; p++) {
    loft.box('harbor blue', 0.589+(p+0.5)*0.088, (sill+doorTop)/2+0.006,
      front+0.08, 0.085, doorTop-sill, 0.039)
  }
  for (const y of [sill+0.12,doorTop-0.11]) {
    loft.box('sawn cedar', 0.897, y, front+0.115, 0.61, 0.059, 0.033)
  }
  loft.beam('sawn cedar', [0.627,sill+0.15,front+0.116],[1.167,doorTop-0.142,front+0.116], 0.038,0.053)
  for (const x of [0.68,1.10]) {
    loft.box('wrought iron', x, doorTop+0.039, front+0.089, 0.027,0.13,0.025)
  }
  for (const y of [0.48,0.60]) loft.beam('wrought iron',[0.618,y,front+0.099],[0.618,y,front+0.157],0.024,0.022)
  loft.beam('aged zinc',[0.618,0.48,front+0.157],[0.618,0.60,front+0.157],0.020,0.022)

  // A laced hemp net hangs from two pins against the open door. Alternating
  // knot rows give the diamonds real openings and the slack a catenary-like sag.
  const netWidth = 0.52, knots = 5
  function knot(i: number, j: number): [number,number,number] {
    const t = (i + (j % 2) * 0.5) / knots
    return [0.63+t*netWidth, 0.94 - j*0.065 - 0.16*t*(1-t), front+0.150+0.009*Math.sin(j*0.65+t*2.5)]
  }
  for (let j = 0; j < 7; j++) {
    for (let i = 0; i < knots; i++) {
      const offset = j % 2
      loft.beam('hemp rope', knot(i, j+offset), knot(i,j+1-offset), 0.009,0.009)
      loft.beam('hemp rope', knot(i+1,j+offset), knot(i,j+1-offset), 0.009,0.009)
    }
  }
  loft.box('sawn cedar', 0.23, 0.51, -0.30, 0.87,0.058,0.35)
  for (const x of [-0.13,0.58]) loft.box('dock oak', x,0.31,-0.30,0.055,0.36,0.30)
  loft.column('sawn cedar', 0.23,-0.30, [[0.54,0.124],[0.74,0.148]], 12)
  loft.column('wrought iron', 0.23,-0.30, [[0.696,0.145],[0.718,0.149]], 12)

  makeBellLoftRoof(loft)
  loft.object++
  dressFloatBench(loft)
  loft.object++
  dressGearRack(loft)
  const result = loft.finish()
  result.materials.set('verdigris', {color:[0.16,0.40,0.34],metallic:0.63,roughness:0.47})
  result.materials.set('tanbark linen', {color:[0.76,0.31,0.14],map:harborSailcloth,roughness:0.89})
  result.materials.set('sun-faded tanbark', {color:[0.86,0.405,0.205],map:harborSailcloth,roughness:0.93})
  result.materials.set('waxed flax', {color:[0.88,0.76,0.52],map:harborSailcloth,roughness:0.79})
  result.materials.set('cream band', {color:[0.89,0.78,0.54],map:result.materials.get('sawn cedar')!.map,roughness:0.49})
  result.materials.set('lamp glass', {color:[0.105,0.52,0.425],roughness:0.16})
  result.materials.set('lens brass', {color:[0.78,0.59,0.28],metallic:0.95,roughness:0.27})
  // Only smooth the blown bulbs. Their flattened bases, sealed necks and fittings keep a crisp rim.
  const {vertices,indices,draws} = result.mesh
  const glass = draws!.find(draw => draw.material === 'lamp glass')!
  const glassVertices = new Set(indices.subarray(glass.firstIndex,glass.firstIndex+glass.indexCount))
  for (const vertex of glassVertices) {
    const i = vertex*sceneVertexStride, x = vertices[i], y = vertices[i+1], z = vertices[i+2]
    if (Math.abs(vertices[i+4]) > 0.999 || Math.abs(vertices[i+4]) < 0.001) continue
    const [cx,cz,r] = floats.find(([fx,fz,fr]) => Math.hypot(x-fx,z-fz) <= fr+0.00001)!
    const dx = x-cx, dy = y-benchTop-0.91*r, dz = z-cz
    if (dy >= 0.94*r-1e-6 && Math.hypot(dx,dz) < 0.30*r) continue
    const len = Math.hypot(dx,dy,dz)
    vertices.set([dx/len,dy/len,dz/len],i+3)
  }
  return result
}
