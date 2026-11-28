import { HarborMesh } from './cove-quayside'
import { shoreHeight } from './cove-shore'

type Point = [number, number, number]
const deckX = -6.6, deckY = 0.69, netX = -6.47, netZ = 2.68
const oak = 'dock oak', cedar = 'sawn cedar', iron = 'wrought iron', hemp = 'hemp rope'
const tarred = 'tarred seine', tau = Math.PI*2
const diff = (a: Point, b: Point): Point => [a[0]-b[0], a[1]-b[1], a[2]-b[2]]
const cross = (a: Point, b: Point): Point => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
function unit(p: Point): Point {
  const d = Math.hypot(...p)
  return [p[0]/d,p[1]/d,p[2]/d]
}
function interpolate(a: Point, b: Point, t: number): Point {
  return [a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t,a[2]+(b[2]-a[2])*t]
}

// Each strand shares a ring through each bend, so the tarred line has
// thickness and real openings but no overlapping end caps at every knot.
function strand(mesh: HarborMesh, mat: string, path: Point[], radius: number, sides = 6, endRadius = radius) {
  let previous: Point[] = []
  let frame: Point = [0,1,0]
  for (let k=0; k<path.length; k++) {
    const center = path[k], t = k/(path.length-1)
    const axis = unit(diff(path[Math.min(k+1,path.length-1)],path[Math.max(0,k-1)]))
    if (k === 0 && Math.abs(axis[1]) > .90) frame = [0,0,1]
    const across = unit(cross(axis,frame))
    frame = cross(across,axis)
    const r = radius+(endRadius-radius)*t
    const ring = Array.from({length:sides}, (_,i): Point => {
      const angle = tau*i/sides, a = r*Math.cos(angle), b = r*Math.sin(angle)
      return [center[0]+a*frame[0]+b*across[0],center[1]+a*frame[1]+b*across[1],center[2]+a*frame[2]+b*across[2]]
    })
    if (k === 0) mesh.face(mat,ring,diff(path[0],path[1]))
    else for (let i=0; i<sides; i++) {
      const j = (i+1)%sides
      const out = diff(interpolate(ring[i],ring[j],.5),center)
      mesh.face(mat,[previous[i],previous[j],ring[j],ring[i]],out)
    }
    if (k === path.length-1) mesh.face(mat,ring,axis)
    previous = ring
  }
}

function landing(mesh: HarborMesh) {
  for (const side of [-1,1]) {
    const x = deckX+side*.43
    for (const z of [-.42,.31,.97]) {
      const ground = shoreHeight(x,z)
      mesh.column(oak,x,z,[[ground-.13,.064],[deckY-.045,.057]],9)
    }
    mesh.beam(oak,[x,deckY-.09,-.51],[x,deckY-.09,1.07],.10,.13)
    mesh.beam(cedar,[x,shoreHeight(x,.31)+.08,.31],[x,deckY-.09,.93],.061,.071)
    const xRamp = deckX+side*.33
    mesh.beam(oak,[xRamp,shoreHeight(xRamp,-1.70)-.024,-1.70],[xRamp,deckY-.078,-.51],.063,.087)
  }
  for (let i=0;i<13;i++) {
    const z = -.50+(i+.5)*1.55/13
    mesh.box(i%4 === 1 ? oak : cedar,deckX,deckY-.026,z,1.06,.052,1.55/13-.006)
  }
  for (let i=0;i<10;i++) {
    const t = (i+.5)/10, z = -1.70+1.20*t
    const high = shoreHeight(deckX,-1.70)+.020
    mesh.box(cedar,deckX,high+(deckY-high)*t-.024,z,.79,.048,.116)
  }
}

// The drum and the boom sheave turn around horizontal axles across the wharf.
function gear(mesh: HarborMesh) {
  const pivot: Point = [deckX,1.94,.61], head: Point = [netX,2.55,netZ-.078]
  const heel: Point = [deckX-.055,1.69,-.24]
  for (const side of [-1,1]) {
    const foot: Point = [deckX+side*.43,deckY,.86]
    mesh.beam(oak,foot,[deckX+side*.084,pivot[1]+.055,pivot[2]],.103,.11)
    mesh.beam(oak,[deckX+side*.43,deckY,-.35],[deckX+side*.18,1.66,.65],.075,.075)
    mesh.beam(iron,[deckX+side*.43,deckY+.08,.62],[deckX+side*.41,deckY+.12,.87],.029,.05)
  }
  mesh.beam(cedar,heel,head,.090,.118)
  strand(mesh,iron,[[deckX-.16,pivot[1],pivot[2]],[deckX+.16,pivot[1],pivot[2]]],.052,10)
  mesh.beam(oak,[deckX,deckY,-.40],heel,.078,.081)
  const shaftY = 1.075, shaftZ = -.20
  const axle = (mat: string, x0: number,x1: number,y: number,z: number,r: number) =>
    strand(mesh,mat,[[x0,y,z],[x1,y,z]],r,14)
  for (const side of [-1,1]) {
    const x = deckX+side*.205
    mesh.beam('harbor blue',[x,deckY,shaftZ],[x,shaftY+.048,shaftZ],.090,.105)
    axle(iron,x-.017,x+.017,shaftY,shaftZ,.059)
    const lip = deckX+side*.143
    axle(iron,lip-.012,lip+.012,shaftY,shaftZ,.115)
  }
  axle(iron,deckX-.26,deckX+.30,shaftY,shaftZ,.029)
  axle(cedar,deckX-.14,deckX+.14,shaftY,shaftZ,.074)
  const winding: Point[] = []
  for (let i=0;i<=72;i++) {
    const a = tau*i/12
    winding.push([deckX-.125+.25*i/72,shaftY+.084*Math.cos(a),shaftZ+.084*Math.sin(a)])
  }
  strand(mesh,hemp,winding,.012,6)
  mesh.beam(iron,[deckX+.28,shaftY,shaftZ],[deckX+.28,shaftY+.085,shaftZ-.123],.028,.032)
  axle(cedar,deckX+.28,deckX+.389,shaftY+.085,shaftZ-.123,.026)
  for (const side of [-1,1]) {
    mesh.beam(iron,[head[0]+side*.053,head[1]-.013,head[2]-.19],
      [head[0]+side*.053,head[1],head[2]],.023,.063)
  }
  axle(iron,head[0]-.044,head[0]+.044,head[1],head[2],.072)
  axle('weathered zinc',head[0]-.079,head[0]+.079,head[1],head[2],.022)
  // Route up the seaward face of a metal fairlead and over the head sheave.
  const fair: Point = [deckX+.125,2.08,.60]
  mesh.beam(iron,[deckX,1.945,.61],[fair[0],fair[1]-.022,fair[2]],.034,.034)
  axle(iron,fair[0]-.028,fair[0]+.028,fair[1]-.030,fair[2],.033)
  const hoist: Point[] = [winding[winding.length-1],fair]
  for (let i=0;i<=7;i++) {
    const a = 1.88*(1-i/7)
    hoist.push([netX,head[1]+.078*Math.sin(a),head[2]+.078*Math.cos(a)])
  }
  hoist.push([netX,1.67,netZ])
  strand(mesh,hemp,hoist,.012)
}

// In net coordinates the four tied corners sit at the same height. Between
// them the bolt rope bows under the sinkers and the belly drops toward the sea.
function netPoint(u: number,v: number): Point {
  const angle=.07, x=.91*u, z=v
  return [netX+x*Math.cos(angle)-z*Math.sin(angle),
    .16+.17*(u*u+v*v)+.26*u*u*v*v,
    netZ+x*Math.sin(angle)+z*Math.cos(angle)]
}

function squareNet(mesh: HarborMesh) {
  const hub: Point = [netX,1.65,netZ], grids = 14
  mesh.column(iron,netX,netZ,[[1.615,.066],[1.662,.074],[1.69,.038]],10)
  for (const u of [-1,1]) for (const v of [-1,1]) {
    const tip = netPoint(u,v), arch: Point[] = []
    for (let i=0;i<=9;i++) {
      const t=i/9, p=interpolate(hub,tip,t)
      p[1]+=.31*Math.sin(Math.PI*t)
      arch.push(p)
    }
    strand(mesh,cedar,arch,.032,8,.015)
    // Tow lashings at the spring and where each pole meets the edge rope.
    for (const k of [1,7,8]) strand(mesh,hemp,[arch[k],interpolate(arch[k],arch[k+1],.22)],k===1 ? .034 : .021,8)
  }
  for (const side of [-1,1]) for (const axis of [0,1]) {
    const along = (t: number) => axis===0 ? netPoint(side,t) : netPoint(t,side)
    strand(mesh,hemp,Array.from({length:grids+1},(_,i)=>along(-1+2*i/grids)),.015,7)
    for (const knot of [-.57,0,.57]) {
      const [x,y,z] = along(knot)
      strand(mesh,tarred,[[x,y,z],[x,y-.050,z]],.007,5)
      mesh.column('aged zinc',x,z,[[y-.097,.012],[y-.083,.023],[y-.059,.024],[y-.045,.010]],8)
    }
  }
  ++mesh.object
  // Clip both diagonal families to the edge of the square. Half-cell steps
  // place a shared point at every crossing, even along the steep corner walls.
  for (const sign of [-1,1]) {
    for (let intercept = 1-grids; intercept<grids; intercept++) {
      const lo=Math.max(0,-intercept), hi=Math.min(grids,grids-intercept)
      const line: Point[] = []
      for (let i=2*lo;i<=2*hi;i++) {
        const u=i/grids-1, v=(i+2*intercept)/grids-1
        line.push(netPoint(u,sign*v))
      }
      strand(mesh,tarred,line,.006,5)
    }
  }
}

export function makeCoveLiftNet() {
  const mesh = new HarborMesh()
  landing(mesh)
  ++mesh.object
  gear(mesh)
  ++mesh.object
  squareNet(mesh)
  const result = mesh.finish()
  result.materials.set(tarred,{color:[.27,.258,.15],roughness:.84})
  return result
}
