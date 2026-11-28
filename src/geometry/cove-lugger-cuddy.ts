import type { HarborMesh } from './cove-quayside'

type Point = [number, number, number]
const aft = -1.53, fore = -0.32, center = -0.13, eave = 1.025
const roofAft = aft-0.065, roofFore = fore+0.068, halfRoof = 0.443
const cream = 'cream band', brass = 'lens brass', glass = 'lamp glass'
const up: Point = [0,1,0]

function sidePoint(x: number, y: number, side: number, depth = 0): Point {
  return [x,y,center+side*(0.41-0.066*(y-0.41)+depth)]
}

function cabinSide(mesh: HarborMesh, side: number) {
  const surface = (x: number, y: number, d = 0) => sidePoint(x,y,side,d)
  const normal: Point = [0,0,side], y0 = 0.675, y1 = 0.968, cy = (y0+y1)/2
  function board(x0: number,x1: number,h0: number,h1: number,mat: string) {
    mesh.face(mat,[surface(x0,h0),surface(x1,h0),surface(x1,h1),surface(x0,h1)],normal)
  }
  for (let i = 0; i < 12; i++) {
    const x0 = aft+(fore-aft)*i/12, x1 = aft+(fore-aft)*(i+1)/12
    board(x0,x1,0.40,y0,cream)
    if (i > 0) mesh.beam('sawn cedar',surface(x0,0.46,0.004),surface(x0,y0-0.025,0.004),0.009,0.009)
  }
  board(aft,fore,y1,eave,cream)
  for (const y of [0.48,y0,eave-0.017]) mesh.beam('sawn cedar',surface(aft,y),surface(fore,y),0.033,0.032)
  // The bronze portlights have rolled outer lips, a recessed pane and a
  // square-to-round opening in the surrounding timber. There is no wall across the glass.
  for (let bay = 0; bay < 3; bay++) {
    const half = (fore-aft)/6, x = aft+(2*bay+1)*half
    const point = (angle: number, radius: number, depth: number) =>
      surface(x+radius*Math.cos(angle),cy+radius*Math.sin(angle),depth)
    const boundary = (angle: number) => {
      const c = Math.cos(angle), s = Math.sin(angle), r = 1/Math.max(Math.abs(c),Math.abs(s))
      return surface(x+half*c*r,cy+(y1-y0)*0.5*s*r)
    }
    const sides = 24, glassFace: Point[] = []
    for (let i = 0; i < sides; i++) {
      const a = 2*Math.PI*i/sides, b = 2*Math.PI*(i+1)/sides
      mesh.face(cream,[boundary(a),boundary(b),point(b,.117,0),point(a,.117,0)],normal)
      const profile = [[.117,0],[.116,.014],[.110,.022],[.092,.022],[.085,.015],[.085,-.012]]
      for (let j=1; j<profile.length; j++) {
        const [r,d] = profile[j-1], [s,e] = profile[j]
        const m = (a+b)/2, radial = Math.sign(e-d)
        mesh.face(brass,[point(a,r,d),point(b,r,d),point(b,s,e),point(a,s,e)],
          [radial*Math.cos(m),radial*Math.sin(m),side*Math.sign(r-s)])
      }
      glassFace.push(point(a,.085,-.013))
    }
    mesh.face(glass,glassFace,normal)
  }
}

function roofPoint(x: number,z: number,offset = 0): Point {
  const u = z/halfRoof
  return [x,eave+0.064*(1-u*u)+offset,center+z]
}

function curvedRoof(mesh: HarborMesh) {
  const skylightAft = -1.32, skylightFore = -0.88, skylightHalf = .145
  const span = [-halfRoof,-.31,-.22,-skylightHalf,0,skylightHalf,.22,.31,halfRoof]
  const stations = [roofAft,skylightAft,skylightFore,roofFore]
  for (let j=0; j<span.length-1; j++) {
    const a = span[j], b = span[j+1]
    for (let i=0; i<stations.length-1; i++) {
      if (i===1 && a >= -skylightHalf && b <= skylightHalf) continue
      const x = stations[i], y = stations[i+1], p = roofPoint
      mesh.face('verdigris',[p(x,a),p(y,a),p(y,b),p(x,b)],up)
      mesh.face('sawn cedar',[p(x,a,-.025),p(y,a,-.025),p(y,b,-.025),p(x,b,-.025)],[0,-1,0])
    }
    for (const x of [roofAft,roofFore]) {
      mesh.face('sawn cedar',[roofPoint(x,a),roofPoint(x,b),roofPoint(x,b,-.025),roofPoint(x,a,-.025)],
        [x===roofAft ? -1 : 1,0,0])
    }
  }
  for (const side of [-1,1]) {
    const z = halfRoof*side
    mesh.beam('sawn cedar',roofPoint(roofAft,z,-.012),roofPoint(roofFore,z,-.012),.027,.034)
    const rail = side*.30
    for (const x of [aft+.10,fore-.10]) mesh.beam(brass,roofPoint(x,rail),roofPoint(x,rail,.066),.026,.027)
    mesh.beam('sawn cedar',roofPoint(aft+.065,rail,.066),roofPoint(fore-.065,rail,.066),.038,.030)
    const zEdge = skylightHalf*side
    const baseA = roofPoint(skylightAft,zEdge), baseB = roofPoint(skylightFore,zEdge)
    const lipA = roofPoint(skylightAft,zEdge,.034), lipB = roofPoint(skylightFore,zEdge,.034)
    mesh.face('sawn cedar',[baseA,baseB,lipB,lipA],[0,0,side])
    const peakA = roofPoint(skylightAft,0,.125), peakB = roofPoint(skylightFore,0,.125)
    mesh.face(glass,[lipA,lipB,peakB,peakA],up)
    mesh.beam(brass,lipA,lipB,.024,.024)
    for (const x of [skylightAft,(skylightAft+skylightFore)/2,skylightFore]) {
      mesh.beam('sawn cedar',roofPoint(x,zEdge,.042),roofPoint(x,0,.131),.027,.025)
    }
    for (const x of [skylightAft,skylightFore]) {
      mesh.face('sawn cedar',[roofPoint(x,zEdge),roofPoint(x,0),roofPoint(x,0,.125),roofPoint(x,zEdge,.034)],
        [x===skylightAft ? -1 : 1,0,0])
    }
  }
  mesh.beam(brass,roofPoint(skylightAft,0,.13),roofPoint(skylightFore,0,.13),.028,.024)
}

function companionway(mesh: HarborMesh) {
  // The sliding companionway faces the helm. Outside each end frame the
  // planked walls follow the slight inward slope of the cabin sides.
  for (const x of [aft,fore]) {
    const dir = x===aft ? -1 : 1, normal: Point = [dir,0,0]
    const p = (z:number,y:number,d=0):Point => [x+dir*d,y,center+z]
    for (const side of [-1,1]) {
      mesh.face(cream,[sidePoint(x,.40,side),p(side*.22,.40),p(side*.22,eave),sidePoint(x,eave,side)],normal)
      mesh.beam('sawn cedar',sidePoint(x,.416,side),sidePoint(x,eave,side),.041,.040)
    }
    if (x===aft) {
      for (let i=0; i<6; i++) {
        const z = -.22 + i*.44/6
        mesh.box(i%2===0 ? 'ochre boards' : 'sawn cedar',x,.694,center+z+.44/12,.031,.585,.44/6)
      }
      for (const y of [.63,.96]) mesh.beam('dock oak',p(-.215,y,.022),p(.215,y,.022),.028,.028)
      for (const z of [-.22,.22]) mesh.beam('sawn cedar',p(z,.407),p(z,eave),.041,.045)
      mesh.beam('aged zinc',p(.14,.676,.033),p(.14,.768,.033),.027,.029)
    } else {
      mesh.face(cream,[p(-.22,.40),p(.22,.40),p(.22,.704),p(-.22,.704)],normal)
      mesh.face(glass,[p(-.22,.704,-.012),p(.22,.704,-.012),p(.22,.94,-.012),p(-.22,.94,-.012)],normal)
      mesh.face(cream,[p(-.22,.94),p(.22,.94),p(.22,eave),p(-.22,eave)],normal)
      for (const y of [.704,.94]) mesh.beam('sawn cedar',p(-.23,y),p(.23,y),.040,.034)
      for (const z of [-.22,0,.22]) mesh.beam(brass,p(z,.70),p(z,.95),.022,.023)
    }
    for (const z of [-.32,.32]) mesh.beam('sawn cedar',p(z,.45),p(z,eave),.022,.022)
    const fascia = [-.367,-.24,0,.24,.367]
    for (let i=1; i<fascia.length; i++) {
      const a=fascia[i-1], b=fascia[i]
      mesh.face(cream,[p(a,eave),p(b,eave),roofPoint(x,b),roofPoint(x,a)],normal)
    }
  }
}

export function makeLuggerCuddy(mesh: HarborMesh) {
  for (const side of [-1,1]) cabinSide(mesh,side)
  companionway(mesh)
  curvedRoof(mesh)
}
