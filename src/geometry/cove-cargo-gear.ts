import { HarborMesh, quayLevel } from './cove-quayside'
import { shoreHeight } from './cove-shore'

type Point = [number, number, number]
type WheelPlane = (horizontal: number, vertical: number, axial: number) => Point
const tau = 2 * Math.PI
const diff = (a: Point, b: Point): Point => [a[0]-b[0], a[1]-b[1], a[2]-b[2]]

// A hollow rim has real sidewalls. The openings between the spokes remain
// visible from either bank and are also open to the optical solver.
function rim(mesh: HarborMesh, material: string, plane: WheelPlane, outer: number, inner: number,
  thickness: number, sides = 16) {
  const point = (angle: number, radius: number, axial: number) =>
    plane(radius*Math.cos(angle), radius*Math.sin(angle), axial)
  for (let i = 0; i < sides; i++) {
    const a = tau*i/sides, b = tau*(i+1)/sides, half = thickness/2
    const radial = diff(point((a+b)/2, 1, 0), plane(0,0,0))
    mesh.face(material, [point(a,outer,-half),point(b,outer,-half),point(b,outer,half),point(a,outer,half)], radial)
    mesh.face(material, [point(a,inner,half),point(b,inner,half),point(b,inner,-half),point(a,inner,-half)],
      [-radial[0],-radial[1],-radial[2]])
    for (const side of [-1,1]) {
      const z = side*half
      mesh.face(material, [point(a,inner,z),point(a,outer,z),point(b,outer,z),point(b,inner,z)],
        diff(plane(0,0,side),plane(0,0,0)))
    }
  }
}

function spokedWheel(mesh: HarborMesh, plane: WheelPlane, radius: number, spokes: number) {
  rim(mesh,'dock oak',plane,radius-0.006,0.86*radius,0.045)
  rim(mesh,'wrought iron',plane,radius,radius-0.006,0.048)
  for (let i = 0; i < spokes; i++) {
    const angle = tau*i/spokes, x = radius*Math.cos(angle), y = radius*Math.sin(angle)
    mesh.beam('sawn cedar',plane(0.13*x,0.13*y,0),plane(0.91*x,0.91*y,0),0.024,0.025)
  }
  rim(mesh,'wrought iron',plane,0.19*radius,0.045*radius,0.082,10)
}

function cargoCrane(mesh: HarborMesh, deckY: number) {
  // In jib coordinates, u is forward and v runs along both sheave axles.
  // The base plate fits on the capstones between the top step and the barrels.
  const x = 3.67, z = -1.77, dx = -0.19, dz = Math.sqrt(1-dx*dx)
  const p = (u: number, y: number, v = 0): Point => [x+dx*u+dz*v,quayLevel+y,z+dz*u-dx*v]
  mesh.column('wrought iron',x,z,[[quayLevel,0.235],[quayLevel+0.055,0.235],
    [quayLevel+0.065,0.155],[quayLevel+0.135,0.146]],16)
  for (let i = 0; i < 6; i++) {
    const a = i*tau/6
    mesh.column('weathered zinc',x+0.193*Math.cos(a),z+0.193*Math.sin(a),
      [[quayLevel+0.055,0.018],[quayLevel+0.073,0.018]],6)
  }
  mesh.beam('dock oak',p(0,0.11),p(0,1.54),0.155,0.165)
  mesh.beam('sawn cedar',p(-0.035,0.38),p(1.80,1.34),0.14,0.135)
  mesh.beam('dock oak',p(0,0.15),p(0.87,0.858),0.095,0.095)
  // Metal straps on both cheeks spread the sheave load into the timber.
  for (const side of [-1,1]) {
    mesh.beam('wrought iron',p(-0.11,0.32,side*0.089),p(0.06,0.52,side*0.089),0.022,0.075)
    mesh.beam('wrought iron',p(1.54,1.22,side*0.075),p(1.87,1.39,side*0.075),0.025,0.060)
    mesh.beam('hemp rope',p(0,1.515,side*0.058),p(1.67,1.30,side*0.065),0.018,0.018)
  }
  const tip = (u: number,h: number,v: number) => p(1.83+u,1.35+h,v)
  rim(mesh,'wrought iron',tip,0.103,0.018,0.071,16)
  mesh.beam('weathered zinc',tip(0,0,-0.115),tip(0,0,0.115),0.036,0.036)

  // Hand cranks wind the lifting line onto a drum at a distinct axle height.
  for (const side of [-1,1]) {
    mesh.beam('dock oak',p(-0.29,0.18,side*0.15),p(-0.15,0.63,side*0.15),0.065,0.06)
    mesh.beam('wrought iron',p(-0.15,0.57,side*0.15),p(0,0.50,side*0.12),0.048,0.048)
  }
  const drum = (u: number,h: number,v: number) => p(-0.16+u,0.59+h,v)
  rim(mesh,'sawn cedar',drum,0.101,0.028,0.238)
  for (const side of [-1,1]) {
    rim(mesh,'wrought iron',(u,h,v)=>drum(u,h,v+side*0.132),0.126,0.026,0.024)
  }
  let rope = drum(-0.105,0,-0.098)
  for (let j = 1; j <= 40; j++) {
    const angle = Math.PI + tau*j/8, next = drum(0.105*Math.cos(angle),0.105*Math.sin(angle),-0.098+0.193*j/40)
    mesh.beam('hemp rope',rope,next,0.018,0.018)
    rope = next
  }
  const fairlead = p(-0.093,1.49,0.084)
  mesh.beam('hemp rope',rope,fairlead,0.020,0.020)
  mesh.beam('hemp rope',fairlead,tip(-0.008,0.108,0),0.020,0.020)
  mesh.beam('wrought iron',drum(0,0,-0.21),drum(0,0,0.22),0.044,0.044)
  mesh.beam('wrought iron',drum(0,0,0.212),drum(-0.067,-0.14,0.212),0.035,0.035)
  mesh.beam('ochre boards',drum(-0.067,-0.14,0.212),drum(-0.067,-0.14,0.31),0.043,0.043)
  let last = tip(0,0.108,0)
  for (let i = 1; i <= 5; i++) {
    const angle = Math.PI*(1-i/5)/2
    const next = tip(0.108*Math.cos(angle),0.108*Math.sin(angle),0)
    mesh.beam('hemp rope',last,next,0.020,0.020)
    last = next
  }
  const hook = deckY+0.71-quayLevel
  mesh.beam('hemp rope',last,p(1.938,hook+0.10),0.020,0.020)
  // The forged J-hook has an open throat and is visibly free of the deck.
  last = p(1.938,hook+0.119)
  for (const [u,h] of [[1.938,0.018],[1.925,-0.012],[1.894,-0.028],[1.865,-0.021],[1.85,0.006],[1.85,0.037]]) {
    const next = p(u,hook+h)
    mesh.beam('wrought iron',last,next,0.025,0.025)
    last=next
  }
}

function fishCart(mesh: HarborMesh) {
  const x = 4.85, z = -0.24, track = 0.35, radius = 0.24
  const near = shoreHeight(x,z-track), far = shoreHeight(x,z+track)
  const y = (near+far)/2, roll = (far-near)/(2*track)
  const p = (u: number,h: number,v: number): Point => [x+u,y+h+v*roll,z+v]
  for (const side of [-1,1]) {
    const wheel = (u: number,h: number,v: number) => p(u,radius+h,side*track+v)
    spokedWheel(mesh,wheel,radius,8)
    const foot = p(-0.565,0.05,side*0.205)
    mesh.beam('dock oak',[foot[0],shoreHeight(foot[0],foot[2])-0.012,foot[2]],p(-0.51,0.305,side*0.205),0.042,0.045)
    mesh.beam('dock oak',p(0.44,0.32,side*0.205),p(-1.17,0.405,side*0.215),0.059,0.057)
    mesh.beam('ochre boards',p(-1.17,0.405,side*0.215),p(-1.04,0.398,side*0.214),0.061,0.06)
  }
  mesh.beam('wrought iron',p(0,radius,-0.41),p(0,radius,0.41),0.043,0.043)
  for (let i = 0; i < 6; i++) {
    const u=-0.49+i*0.169
    mesh.beam('sawn cedar',p(u,0.366,-0.289),p(u,0.366,0.289),0.162,0.043)
  }
  // Two open fish trays share a bottom but preserve the gaps between slats.
  for (const side of [-1,1]) for (const height of [0.42,0.494]) {
    mesh.beam('harbor blue',p(-0.564,height,side*0.274),p(0.435,height,side*0.274),0.027,0.061)
  }
  for (const u of [-0.554,-0.06,0.425]) {
    mesh.beam('dock oak',p(u,0.45,-0.26),p(u,0.45,0.26),0.030,0.13)
  }
  // Split each fish into dark dorsal and silver lateral surfaces. The tail
  // and head retain their own silhouette at close range in the exported OBJ.
  for (let row=0; row<2; row++) for (let i=0; i<3; i++) {
    const u=-0.45+row*0.49+i*0.12, v=(i%2)*0.026-0.022
    const tail=p(u,0.42,v-0.060), nose=p(u+0.035,0.425,v+0.174)
    const left=p(u-0.022,0.41,v+0.055), right=p(u+0.067,0.413,v+0.046)
    const back=p(u+0.019,0.472,v+0.083)
    mesh.face('aged zinc',[tail,left,back],[0,1,0])
    mesh.face('aged zinc',[left,nose,back],[0,1,0])
    mesh.face('zinc flashing',[tail,back,right],[0,1,0])
    mesh.face('zinc flashing',[right,back,nose],[0,1,0])
    mesh.face('weathered zinc',[tail,right,nose,left],[0,-1,0])
    mesh.face('aged zinc',[tail,p(u-0.048,0.405,v-0.141),p(u+0.015,0.418,v-0.127)],[0,1,0])
  }
}

export function makeCoveCargoGear(deckY: number) {
  const mesh = new HarborMesh()
  cargoCrane(mesh,deckY)
  mesh.object++
  fishCart(mesh)
  return mesh.finish()
}
