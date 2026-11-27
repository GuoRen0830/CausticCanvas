import { sceneVertexStride, type BaseColorMaterial, type SceneDraw, type SceneMesh } from '../rendering/scene-mesh'

type Point = [number, number, number]
type UV = [number, number]

const length = 3.55, halfBeam = 0.636
const sub = (a: Point, b: Point): Point => [a[0]-b[0], a[1]-b[1], a[2]-b[2]]
const cross = (a: Point, b: Point): Point => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
const dot = (a: Point, b: Point) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
function unit(a: Point): Point {
  const d = Math.hypot(...a)
  return [a[0]/d, a[1]/d, a[2]/d]
}

export function makeCoveSkiff(planks?: BaseColorMaterial) {
  const materials = new Map<string, BaseColorMaterial>()
  function material(name: string, color: Point, grain = false) {
    materials.set(name, { color, map: grain ? planks?.map : undefined })
  }
  material('deep teal paint', [0.036, 0.154, 0.155])
  material('faded teal paint', [0.059, 0.207, 0.188])
  material('plank edges', [0.028, 0.079, 0.071])
  material('interior oak', planks?.map ? [0.94, 0.68, 0.43] : [0.33, 0.165, 0.074], true)
  material('interior cedar', planks?.map ? [0.82, 0.55, 0.33] : [0.24, 0.115, 0.054], true)
  material('varnished trim', [0.39, 0.21, 0.091])
  material('ash oars', [0.55, 0.35, 0.16])
  material('cream band', [0.75, 0.65, 0.42])
  const vertices: number[] = [], uvs: number[] = [], groups = new Map<string, number[]>()
  const up: Point = [0, 1, 0], down: Point = [0, -1, 0]
  const defaultUV: UV[] = [[0,0], [1,0], [1,1], [0,1]]
  // Orient each face from its physical side. BVH picking and top-light baking
  // use the polygon winding independently of its shading normal.
  function quad(mat: string, corners: Point[], outward: Point, texcoords: UV[] = defaultUV) {
    const [a, b, c, d] = corners
    const normal = unit(cross(sub(c,a), sub(d,b)))
    const flip = dot(normal, outward) < 0
    const sign = flip ? -1 : 1, start = vertices.length / sceneVertexStride
    for (let i = 0; i < 4; i++) {
      vertices.push(...corners[i], sign*normal[0], sign*normal[1], sign*normal[2], 1,1,1,0,2)
      uvs.push(...texcoords[i])
    }
    let batch = groups.get(mat)
    if (!batch) { batch = []; groups.set(mat, batch) }
    for (const local of flip ? [0,2,1,0,3,2] : [0,1,2,0,2,3]) batch.push(start + local)
  }

  // t runs from the broad transom to the narrow stem. The raised ends of
  // both keel and sheer line give the skiff its longitudinal curvature.
  function hull(t: number, height: number, side: number, offset = 0): Point {
    const wave = Math.sin(Math.PI*t), ends = (2*t-1)**2
    const width = halfBeam*(0.032 + 0.94*wave**0.89 + 0.47*(1-t)**3)
    const keelY = -0.24 + 0.11*ends + 0.035*t**4
    const rimY = 0.215 + 0.045*ends + 0.06*t**5
    const x = width * (0.145 + 0.855*height**0.57)
    return [side*Math.max(0.002, x + offset*Math.min(1,width / 0.2)), keelY + height*(rimY-keelY), (t-0.5)*length]
  }
  const segments = 20, bands = 5
  const stripUV = (z0: number, z1: number, h: number): UV[] =>
    [[z0/1.5,h], [z1/1.5,h], [z1/1.5,h+0.22], [z0/1.5,h+0.22]]
  for (let i = 0; i < segments; i++) {
    const t = i/segments, end = (i+1)/segments
    for (const side of [-1,1]) {
      for (let row = 0; row < bands; row++) {
        const h0 = row/bands, h1 = (row+1)/bands
        const outside = row%2 ? 'deep teal paint' : 'faded teal paint'
        const lip = row ? 0.012 : 0
        const outer = [hull(t,h0,side,lip),hull(end,h0,side,lip),hull(end,h1,side),hull(t,h1,side)]
        const inner = [hull(t,h0,side,-0.026),hull(end,h0,side,-0.026),hull(end,h1,side,-0.026),hull(t,h1,side,-0.026)]
        const tex = stripUV(outer[0][2], outer[1][2], row*0.29)
        quad(outside, outer, [side,0,0], tex)
        quad(row%2 ? 'interior oak' : 'interior cedar', inner, [-side,0.5,0], tex)
        if (row) quad('plank edges', [hull(t,h0,side),hull(end,h0,side),outer[1],outer[0]], down)
      }
      const a = hull(t,1,side,0.014), b = hull(end,1,side,0.014)
      quad('varnished trim', [a,b,hull(end,1,side,-0.033),hull(t,1,side,-0.033)], up)
      const aLow: Point = [a[0], a[1]-0.022, a[2]], bLow: Point = [b[0],b[1]-0.022,b[2]]
      quad('cream band', [a,b,bLow,aLow], [side,0,0])
    }
    quad('plank edges', [hull(t,0,-1),hull(end,0,-1),hull(end,0,1),hull(t,0,1)], down)
    quad('interior cedar', [hull(t,0.10,-1,-0.028),hull(end,0.10,-1,-0.028),
      hull(end,0.10,1,-0.028),hull(t,0.10,1,-0.028)], up)
  }
  for (const t of [0,1]) {
    const direction = 2*t-1
    for (let row = 0; row < bands; row++) {
      const lo = row/bands, hi = (row+1)/bands
      quad('deep teal paint', [hull(t,lo,-1),hull(t,lo,1),hull(t,hi,1),hull(t,hi,-1)], [0,0,direction])
      const inside = t - direction*0.004
      quad('interior oak', [hull(inside,lo,-1,-0.026),hull(inside,lo,1,-0.026),
        hull(inside,hi,1,-0.026),hull(inside,hi,-1,-0.026)], [0,0,-direction])
    }
  }
  // The raised sole keeps the floorboards above the waterline. Each strip
  // follows the bilge as the hull narrows toward the bow and stern.
  for (let j = 0; j < 5; j++) for (let i = 1; i < segments-1; i++) {
    const t = i/segments, end = (i+1)/segments
    function sole(u: number, edge: number): Point {
      const p = hull(u,0.61,1,-0.05)
      return [p[0]*edge, 0.031 + 0.008*(2*u-1)**2, p[2]]
    }
    const left = (j+0.016)*0.4-1, right = (j+0.984)*0.4-1
    quad(j%2 ? 'interior oak':'interior cedar',
      [sole(t,left),sole(end,left),sole(end,right),sole(t,right)], up, stripUV(t*length,end*length,j*0.45))
  }
  // Visible frames curve up each side between the planks and the thwarts.
  for (const t of [0.10,0.24,0.38,0.52,0.66,0.80,0.91]) for (const side of [-1,1]) {
    for (let row = 2; row < bands; row++) {
      const lo = row/bands, hi = (row+1)/bands
      quad('varnished trim',[hull(t-0.005,lo,side,-0.037),hull(t+0.005,lo,side,-0.037),
        hull(t+0.005,hi,side,-0.037),hull(t-0.005,hi,side,-0.037)],[-side,1,0])
    }
  }
  for (const center of [0.19,0.49,0.77]) {
    const t0 = center-0.026, t1 = center+0.026
    const top = [hull(t0,0.81,-1,-0.024),hull(t1,0.81,-1,-0.024),hull(t1,0.81,1,-0.024),hull(t0,0.81,1,-0.024)]
    const bottom = top.map(([x,y,z]): Point => [x, y-0.035, z])
    quad('interior oak', top, up, top.map(([x,,z]) => [x/1.5,z]))
    quad('interior cedar', bottom, down)
    for (const [a,b] of [[0,1],[1,2],[2,3],[3,0]]) {
      const middleZ = (top[0][2]+top[1][2])/2
      quad('varnished trim',[top[a],top[b],bottom[b],bottom[a]],
        [top[a][0]+top[b][0],0,top[a][2]+top[b][2]-2*middleZ])
    }
  }
  // One elongated, flattened blade and a slender round loom with a thicker
  // grip share the same section curve. The oars lie across the thwarts.
  function oar(start: Point, finish: Point) {
    const axis = sub(finish,start), horizontal = unit(cross(up, axis))
    const vertical = unit(cross(axis,horizontal))
    const sections = [[0,0.017,0.016],[0.026,0.024,0.023],[0.10,0.022,0.021],
      [0.12,0.016,0.016],[0.64,0.021,0.019],[0.70,0.036,0.016],
      [0.80,0.071,0.013],[0.95,0.082,0.010],[1,0.054,0.007]]
    const corners = 8
    function point(section: number, facet: number): Point {
      const [t,w,h] = sections[section], angle = facet*2*Math.PI/corners
      const a = Math.cos(angle)*w, b = Math.sin(angle)*h
      return [start[0]+axis[0]*t+horizontal[0]*a+vertical[0]*b,
        start[1]+axis[1]*t+horizontal[1]*a+vertical[1]*b,
        start[2]+axis[2]*t+horizontal[2]*a+vertical[2]*b]
    }
    for (let section = 0; section < sections.length-1; section++) for (let k = 0; k < corners; k++) {
      const face = [point(section,k),point(section+1,k),point(section+1,k+1),point(section,k+1)]
      const t = sections[section][0]
      const outward = sub(face[0],[start[0]+axis[0]*t,start[1]+axis[1]*t,start[2]+axis[2]*t])
      quad(section === 6 ? 'cream band' : 'ash oars', face, outward)
    }
  }
  oar([-0.36,0.217,-1.07],[0.26,0.231,1.28])
  oar([-0.25,0.231,-1.22],[0.40,0.238,1.02])
  const indices: number[] = [], draws: SceneDraw[] = []
  for (const [name, batch] of groups) {
    draws.push({ material: name, hasUV: true, firstIndex: indices.length, indexCount: batch.length })
    indices.push(...batch)
  }
  const mesh: SceneMesh = { vertices: new Float32Array(vertices), uvs: new Float32Array(uvs),
    indices: new Uint32Array(indices), draws }
  return { mesh, materials, warnings: new Set<string>() }
}
