import type { HarborMesh } from './cove-quayside'

type Point = [number, number, number]
const ochre = 'fish market marigold', silver = 'fish market silver', mackerel = 'fish market mackerel'

// The catch has a dark dorsal half, pale belly and a forked, pinched tail.
// The same little model is turned upright to hang as the fishmonger's sign.
export function fish(mesh: HarborMesh, origin: Point, length: number, heading: number, sign = false) {
  const [x,y,z] = origin, co = Math.cos(heading), si = Math.sin(heading)
  const transform = (s: number, w: number, h: number): Point => sign
    ? [x+length*s,y+length*w,z+length*h]
    : [x+length*(s*co-w*si),y+length*h,z+length*(s*si+w*co)]
  const normal = (s: number, w: number, h: number): Point => sign ? [s,w,h] : [s*co-w*si,h,s*si+w*co]
  const sections = [[-0.30,0.032,0.023],[-0.07,0.104,0.061],[0.16,0.112,0.071],[0.33,0.065,0.047],[0.44,0.013,0.012]]
  const rings = sections.map(([s,w,h]) => Array.from({length:8},(_,j)=>transform(s,w*Math.cos(j*Math.PI/4),h*Math.sin(j*Math.PI/4))))
  for (let i = 1; i < rings.length; i++) for (let j = 0; j < 8; j++) {
    const k = (j+1)%8, a = (j+0.5)*Math.PI/4
    mesh.face(sign ? ochre : Math.cos(a) > 0 ? mackerel : silver,
      [rings[i-1][j],rings[i][j],rings[i][k],rings[i-1][k]],normal(0,Math.cos(a),Math.sin(a)))
  }
  mesh.face(sign ? ochre : mackerel,rings[0],normal(-1,0,0))
  mesh.face(sign ? ochre : silver,rings.at(-1)!,normal(1,0,0))
  for (const side of [-1,1]) {
    const base = transform(-0.28,0.008*side,0), tip = transform(-0.52,0.145*side,0)
    const notch = transform(-0.46,0,0), top = transform(-0.415,0.04*side,0.018), low = transform(-0.415,0.04*side,-0.018)
    const outline = [base,tip,notch]
    for (let i = 0; i < outline.length; i++) {
      const j = (i+1)%outline.length
      mesh.face(sign ? ochre : silver,[outline[i],outline[j],top],normal(0,0,1))
      mesh.face(sign ? ochre : silver,[outline[i],outline[j],low],normal(0,0,-1))
    }
  }
}
