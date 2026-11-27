import linen from '../../scene/smugglers-cove/market-linen.png?inline'
import { HarborMesh, quayLevel } from './cove-quayside'
import { shoreHeight } from './cove-shore'

type Point = [number, number, number]
const left = -7.14, span = 1.16, deck = 0.79
const front = [-2.49, -2.26, -2.25, -2.45]
const rear = [-4.10, -4.03, -4.02, -4.05]
const crest = [2.34, 2.45, 2.40, 2.32]
const edge = [2.05, 2.01, 2.01, 2.05]
const indigo = 'fish market indigo', ivory = 'fish market sailcloth', ochre = 'fish market marigold'
const enamel = 'fish market enamel', silver = 'fish market silver', mackerel = 'fish market mackerel'
const saltwood = 'salt-bleached slats'
const mix = (a: number, b: number, t: number) => a + (b-a)*t
const linenBytes = Uint8Array.from(atob(linen.slice(linen.indexOf(',')+1)), c => c.charCodeAt(0))
const linenMap = new File([linenBytes], 'market-linen.png', { type:'image/png' })

// Each pair of posts follows the curved frontage. The counter projects beyond
// the eaves, so its catch receives direct overhead light.
function point(bay: number, u: number, v: number, y: number): Point {
  return [left+(bay+u)*span, y,
    mix(mix(rear[bay], rear[bay+1], u), mix(front[bay], front[bay+1], u), v)]
}

function cloth(bay: number, u: number, v: number): Point {
  const ridge = mix(crest[bay], crest[bay+1], u), eave = mix(edge[bay], edge[bay+1], u)
  const belly = Math.sin(Math.PI*v) * (0.072+0.048*Math.sin(Math.PI*u))
  return point(bay, u, 0.06+0.76*v, mix(ridge, eave, v)-belly-0.037*Math.sin(Math.PI*u))
}

function makeAwning(mesh: HarborMesh, bay: number) {
  const under = (p: Point, drop = 0.012): Point => [p[0], p[1]-drop, p[2]]
  const stripes = bay === 0 ? [indigo,ivory,indigo,ivory,indigo] : [ivory,indigo,ivory,indigo,ivory]
  for (let i = 0; i < stripes.length; i++) {
    const l = i/stripes.length, r = (i+1)/stripes.length, mat = stripes[i]
    for (let j = 0; j < 5; j++) {
      const a = cloth(bay,l,j/5), b = cloth(bay,r,j/5), c = cloth(bay,r,(j+1)/5), d = cloth(bay,l,(j+1)/5)
      mesh.face(mat,[a,b,c],[0,1,0])
      mesh.face(mat,[a,c,d],[0,1,0])
      mesh.face(mat,[under(a),under(b),under(c)],[0,-1,0])
      mesh.face(mat,[under(a),under(c),under(d)],[0,-1,0])
      if (j === 0) mesh.face(mat,[a,b,under(b),under(a)],[0,0,-1])
      if (j === 4) mesh.face(mat,[d,c,under(c),under(d)],[0,0,1])
      if (i === 0) mesh.face(mat,[a,d,under(d),under(a)],[-1,0,0])
      if (i === stripes.length-1) mesh.face(mat,[b,c,under(c),under(b)],[1,0,0])
    }
    // The cut scallops have a contrasting sewn binding and real thickness.
    for (let j = 0; j < 4; j++) {
      const u = mix(l,r,j/4), w = mix(l,r,(j+1)/4)
      const a = cloth(bay,u,1), b = cloth(bay,w,1)
      const bottom = (p: Point, t: number): Point => [p[0],p[1]-0.058-0.066*Math.sin(Math.PI*t),p[2]+0.008]
      const c = bottom(b,(j+1)/4), d = bottom(a,j/4)
      const inner = (p: Point): Point => [p[0],p[1],p[2]-0.012]
      mesh.face(mat,[a,b,c,d],[0,0,1])
      mesh.face(mat,[inner(a),inner(b),inner(c),inner(d)],[0,0,-1])
      if (i === 0 && j === 0) mesh.face(mat,[a,d,inner(d),inner(a)],[-1,0,0])
      if (i === stripes.length-1 && j === 3) mesh.face(mat,[b,c,inner(c),inner(b)],[1,0,0])
      mesh.beam(ochre,d,c,0.014,0.017)
    }
  }
}

// A horizontal roll with changing radii gives the middle bay an unmistakable
// open silhouette. The gathering cords wrap over the cloth in two places.
function rolledAwning(mesh: HarborMesh) {
  const rings = [0,0.06,0.18,0.23,0.28,0.39,0.58,0.70,0.75,0.80,0.94,1]
  function ring(u: number, angle: number): Point {
    const t = point(1,u,0.11,mix(crest[1],crest[2],u)-0.046*Math.sin(Math.PI*u))
    const r = 0.078 - 0.025*(Math.exp(-((u-0.23)**2)/0.0012)+Math.exp(-((u-0.75)**2)/0.0012))
    return [t[0],t[1]+r*Math.cos(angle),t[2]+r*Math.sin(angle)]
  }
  for (let j = 0; j < rings.length-1; j++) {
    const mat = j === 2 || j === 7 ? 'hemp rope' : j%2 === 0 ? indigo : ivory
    for (let k = 0; k < 10; k++) {
      const a = 2*Math.PI*k/10, b = 2*Math.PI*(k+1)/10
      mesh.face(mat,[ring(rings[j],a),ring(rings[j+1],a),ring(rings[j+1],b),ring(rings[j],b)],
        [0,Math.cos((a+b)/2),Math.sin((a+b)/2)])
    }
  }
  for (const end of [0,1]) mesh.face(indigo,Array.from({length:10},(_,k)=>ring(end,k*2*Math.PI/10)),[end ? 1 : -1,0,0])
  for (const u of [0.23,0.75]) {
    const p = ring(u,Math.PI/2)
    mesh.beam('hemp rope',p,[p[0]+0.018,p[1]-0.18,p[2]+0.015],0.016,0.016)
  }
}

// The catch has a dark dorsal half, pale belly and a forked, pinched tail.
// The same little model is turned upright to hang as the fishmonger's sign.
function fish(mesh: HarborMesh, origin: Point, length: number, heading: number, sign = false) {
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
    const base = transform(-0.32,0.008*side,0), tip = transform(-0.52,0.145*side,0)
    const notch = transform(-0.46,0,0), top = transform(-0.415,0.04*side,0.018), low = transform(-0.415,0.04*side,-0.018)
    const outline = [base,tip,notch]
    for (let i = 0; i < outline.length; i++) {
      const j = (i+1)%outline.length
      mesh.face(sign ? ochre : silver,[outline[i],outline[j],top],normal(0,0,1))
      mesh.face(sign ? ochre : silver,[outline[i],outline[j],low],normal(0,0,-1))
    }
  }
}

function creel(mesh: HarborMesh,x: number,y: number,z: number,r: number) {
  const band = (h: number, radius: number, k: number): Point => {
    const a = 2*Math.PI*k/14
    return [x+radius*Math.cos(a),y+h,z+0.78*radius*Math.sin(a)]
  }
  for (let k = 0; k < 14; k++) {
    const normal: Point = [Math.cos((k+0.5)*Math.PI/7),0,Math.sin((k+0.5)*Math.PI/7)]
    const inner: Point = [-normal[0],0,-normal[2]]
    mesh.face('hemp rope',[band(0,0.72*r,k),band(0,0.72*r,k+1),band(0.64*r,r,k+1),band(0.64*r,r,k)],normal)
    mesh.face('ochre boards',[band(0.1*r,0.67*r,k),band(0.1*r,0.67*r,k+1),band(0.64*r,r-0.009,k+1),band(0.64*r,r-0.009,k)],inner)
    mesh.beam('sawn cedar',band(0,0.72*r,k),band(0.64*r,r,k),0.013,0.012)
    for (let row = 1; row <= 4; row++) {
      const radius = (0.72+0.28*row/4)*r+0.004,h = 0.16*r*row
      mesh.beam(row === 4 ? 'sawn cedar' : 'hemp rope',band(h,radius,k),band(h,radius,k+1),0.014,0.011)
    }
  }
  mesh.face('dock oak',Array.from({length:14},(_,k)=>band(0.10*r,0.67*r,k)),[0,1,0])
  mesh.face('dock oak',Array.from({length:14},(_,k)=>band(0,0.72*r,k)),[0,-1,0])
}

function scale(mesh: HarborMesh) {
  const end=point(1,0.82,0.44,2.28)
  mesh.beam('sawn cedar',point(1,0.82,0.07,2.38),end,0.04,0.058)
  const [x,,z]=end, y=1.72
  mesh.beam('wrought iron',end,[x,y+0.24,z],0.015,0.015)
  for (let k=0; k<3; k++) {
    const a=2*Math.PI*k/3
    mesh.beam('hemp rope',[x,y+0.24,z],[x+0.13*Math.cos(a),y,z+0.13*Math.sin(a)],0.009,0.009)
  }
  mesh.column('aged zinc',x,z,[[y-0.055,0.059],[y-0.012,0.144],[y,0.141]],14)
}

function counter(mesh: HarborMesh,bay: number) {
  const [x,,z] = point(bay,0.5,0.89,deck), top = deck+0.56
  for (const dx of [-0.44,0.44]) for (const dz of [-0.15,0.15]) {
    mesh.box('dock oak',x+dx,deck+0.26,z+dz,0.062,0.52,0.068)
  }
  mesh.box('sawn cedar',x,top-0.035,z,1.04,0.07,0.48)
  mesh.box('harbor blue',x,top-0.15,z+0.205,0.96,0.13,0.025)
  for (const dx of [-0.253,0.253]) {
    if (bay === 0 && dx < 0) {
      creel(mesh,x+dx,top,z,0.208)
      for (let j=0; j<3; j++) fish(mesh,[x+dx+(j-1)*0.067,top+0.099,z],0.22,1.48+(j-1)*0.18)
      continue
    }
    mesh.box(enamel,x+dx,top+0.025,z,0.465,0.034,0.346)
    for (const side of [-1,1]) {
      mesh.box(enamel,x+dx,top+0.058,z+side*0.17,0.473,0.055,0.012)
      mesh.box(enamel,x+dx+side*0.230,top+0.058,z,0.012,0.055,0.334)
    }
    for (let j = 0; j < 3; j++) fish(mesh,[x+dx+(j-1)*0.123,top+0.067,z],0.30,
      (j%2 ? -1 : 1)*(Math.PI/2+(j-1)*0.11))
  }
}

// A short apron turns the walk along the market. The final stringers ramp up
// to the west coping of the existing stone quay, at the same finished height.
function boardwalk(mesh: HarborMesh) {
  const x = left+1.5*span, startZ = point(1,0.5,1.04,deck)[2], landingZ = -1.82, landingY = deck-0.055
  const width = 0.42, half = width/2
  const route: Point[] = [[x+half,landingY,landingZ],[-3.66,0.59,-1.98],[-2.70,0.48,-2.02],
    [-1.50,0.43,-1.84],[-0.85,0.52,-1.78],[-0.12,0.70,-1.90],[0.399,quayLevel,-1.90]]
  mesh.box(saltwood,x,landingY-0.04,landingZ,width,0.08,width)
  const gap = landingZ-half-startZ, tread = gap/2
  for (let i=0; i<2; i++) mesh.box(saltwood,x,deck-0.02-(i+1)*0.017,startZ+(i+0.5)*tread,width,0.054,tread-0.004)
  for (const side of [-1,1]) {
    const postX=x+side*(half-0.06), ground=shoreHeight(postX,landingZ)-0.035
    mesh.box('dock oak',postX,(ground+landingY-0.055)/2,landingZ,0.065,landingY-0.055-ground,0.085)
    mesh.beam('dock oak',[postX,deck-0.074,startZ-0.02],[postX,landingY-0.064,landingZ+half],0.061,0.066)
  }
  const directions = route.slice(1).map((p,i)=>{
    const dx=p[0]-route[i][0],dz=p[2]-route[i][2],length=Math.hypot(dx,dz)
    return [-dz/length,dx/length]
  })
  // Miter adjacent cross-sections so the transverse planks meet at bends.
  const offsets = route.map((_,i)=>{
    if (i===0 || i===route.length-1) return [0,1]
    const a=directions[i-1],b=directions[i],divisor=1+a[0]*b[0]+a[1]*b[1]
    return [(a[0]+b[0])/divisor,(a[1]+b[1])/divisor]
  })
  for (let i=1; i<route.length; i++) {
    const a=route[i-1], b=route[i], na=offsets[i-1], nb=offsets[i]
    const at = (t: number, offset: number, dy=0): Point => [
      mix(a[0]+na[0]*offset,b[0]+nb[0]*offset,t), mix(a[1],b[1],t)+dy, mix(a[2]+na[1]*offset,b[2]+nb[1]*offset,t)]
    const length=Math.hypot(b[0]-a[0],b[2]-a[2]), count=Math.ceil(length/0.144)
    for (let k=0; k<count; k++) {
      const t0=(k+0.025)/count,t1=i===route.length-1 && k===count-1 ? 1 : (k+0.975)/count
      const board = (dy: number) => [at(t0,-half,dy),at(t0,half,dy),at(t1,half,dy),at(t1,-half,dy)]
      const top=board(0), bottom=board(-0.045), mat=(i+3*k)%7===1 ? 'sawn cedar' : saltwood
      mesh.face(mat,top,[0,1,0])
      mesh.face(mat,bottom,[0,-1,0])
      for (let j=0; j<4; j++) {
        const next=(j+1)%4,dx=top[next][0]-top[j][0],dz=top[next][2]-top[j][2]
        mesh.face(mat,[top[j],top[next],bottom[next],bottom[j]],[-dz,0,dx])
      }
    }
    const supports=Math.ceil(length/0.58)
    for (const side of [-0.135,0.135]) {
      mesh.beam('dock oak',at(0,side,-0.072),at(1,side,-0.072),0.056,0.068)
      for (let k=0; k<supports; k++) {
        const head=at((k+0.20)/supports,side,-0.072)
        mesh.beam('dock oak',[head[0],shoreHeight(head[0],head[2])-0.045,head[2]],head,0.062,0.072)
      }
    }
  }
}

export function makeCoveFishMarket() {
  const mesh = new HarborMesh()
  for (let bay = 0; bay < 3; bay++) {
    ++mesh.object
    for (let i = 0; i < 8; i++) {
      const u = (i+0.5)/8
      mesh.beam(i%3 === 2 ? 'dock oak' : 'sawn cedar',point(bay,u,0,deck-0.035),point(bay,u,1.04,deck-0.035),span/8-0.006,0.07)
    }
    for (const v of [0.12,0.79]) {
      mesh.beam('dock oak',point(bay,0,v,deck-0.087),point(bay,1,v,deck-0.087),0.076,0.116)
    }
    mesh.beam('dock oak',cloth(bay,0,0),cloth(bay,1,0),0.06,0.052)
    if (bay === 1) rolledAwning(mesh)
    else makeAwning(mesh,bay)
    counter(mesh,bay)
  }
  ++mesh.object
  for (let i = 0; i <= 3; i++) {
    const bay = Math.min(i,2), u = i === 3 ? 1 : 0
    for (const v of [0.08,0.78]) {
      const end = cloth(bay,u,v < 0.5 ? 0 : 1), foot = point(bay,u,v,0)
      foot[1] = shoreHeight(foot[0],foot[2])-0.055
      mesh.beam('dock oak',foot,end,0.073,0.08)
      const brace = point(bay,u,v,end[1]-0.30)
      mesh.beam('sawn cedar',brace,point(bay,u,v < 0.5 ? 0.25 : 0.60,end[1]-0.04),0.045,0.054)
    }
    mesh.beam('sawn cedar',cloth(bay,u,0),cloth(bay,u,1),0.06,0.065)
  }
  const sign = point(1,0.51,0.79,1.82)
  const a = cloth(1,0,1), b = cloth(1,1,1)
  mesh.beam('dock oak',a,b,0.047,0.058)
  for (const dx of [-0.13,0.13]) {
    mesh.beam('hemp rope',[sign[0]+dx,2.00,sign[2]+0.045],[sign[0]+dx,sign[1]+0.035,sign[2]],0.011,0.011)
  }
  fish(mesh,sign,0.78,0,true)
  scale(mesh)
  ++mesh.object
  boardwalk(mesh)
  const result = mesh.finish()
  result.materials.set(indigo,{color:[0.17,0.35,0.56],map:linenMap})
  result.materials.set(ivory,{color:[0.94,0.82,0.60],map:linenMap})
  result.materials.set(ochre,{color:[0.86,0.43,0.105]})
  result.materials.set(enamel,{color:[0.15,0.41,0.43]})
  result.materials.set(mackerel,{color:[0.22,0.38,0.42]})
  result.materials.set(silver,{color:[0.78,0.83,0.76]})
  result.materials.set(saltwood,{color:[0.67,0.57,0.42],map:result.materials.get('sawn cedar')!.map})
  return result
}
