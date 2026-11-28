import { HarborMesh } from './cove-quayside'

type Point = [number, number, number]
const turn = 2*Math.PI

// The access door faces the path back along the breakwater. Keeping the tower
// self-contained also keeps the rest of the inner harbor free for painting.
export function makePierhead(mesh: HarborMesh, x: number, z: number) {
  const entry = -0.40, facets = 8
  const at = (angle: number, r: number, y: number): Point =>
    [x+r*Math.cos(angle),y,z+r*Math.sin(angle)]
  const radiusAt = (y: number) => 0.422 - 0.083*(y-0.627)
  const drum = (mat: string, rings: [number,number][], sides=24) => mesh.column(mat,x,z,rings,sides)

  function sector(mat: string, a: number, b: number, inner: number, outer: number,
    y: number, depth: number) {
    const p = [at(a,inner,y),at(a,outer,y),at(b,outer,y),at(b,inner,y)]
    const low = p.map(([px,py,pz]):Point => [px,py-depth,pz])
    mesh.face(mat,p,[0,1,0])
    mesh.face(mat,low,[0,-1,0])
    const angle = (a+b)/2, radial: Point = [Math.cos(angle),0,Math.sin(angle)]
    const normals: Point[] = [[Math.sin(a),0,-Math.cos(a)],radial,
      [-Math.sin(b),0,Math.cos(b)],[-radial[0],0,-radial[2]]]
    for (let i=0; i<4; i++) {
      const j = (i+1)%4
      mesh.face(mat,[p[i],low[i],low[j],p[j]],normals[i])
    }
  }

  // Interlocking courses step back towards the watchtower. The continuous core
  // is visible down the narrow mortar joints and closes the foundation at low tide.
  drum('tidal stone',[[-0.48,0.65],[0.35,0.58],[0.50,0.56]],24)
  for (let course=0; course<5; course++) {
    const top = -0.46+(course+1)*0.194, r = 0.70-course*0.018
    for (let i=0; i<12; i++) {
      const a = (i+0.5*(course%2))*turn/12
      const mat = course < 2 ? 'tidal stone' :
        ['weathered limestone','warm freestone','armor stone','weathered limestone'][(i+course)%4]
      sector(mat,a+0.008,a+turn/12-0.008,r-0.11,r,top,0.188)
    }
  }
  drum('capstone',[[0.509,0.628],[0.529,0.644],[0.555,0.644],[0.568,0.624]])
  drum('capstone',[[0.568,0.479],[0.592,0.489],[0.615,0.472],[0.637,0.418]],16)

  // Each octagonal face can be cut independently. The door and the two
  // high-set windows have real reveals and a recessed, arched infill.
  const halfFace = Math.PI/facets
  for (let i=0; i<facets; i++) {
    const a = entry+i*turn/facets, n: Point = [Math.cos(a),0,Math.sin(a)]
    const t: Point = [-n[2],0,n[0]]
    const face = (u: number, y: number, relief=0): Point => {
      const radius = radiusAt(y), d = radius*Math.cos(halfFace)+relief, lateral = u*radius*Math.sin(halfFace)
      return [x+n[0]*d+t[0]*lateral,y,z+n[2]*d+t[2]*lateral]
    }
    function wall(mat: string, lo: number, hi: number, l=-1, r=1) {
      mesh.face(mat,[face(l,lo),face(r,lo),face(r,hi),face(l,hi)],n)
    }
    function opening(lo: number, hi: number, base: number, spring: number, arch: number,
      u: number, doorway: boolean) {
      const mat = 'beacon ivory', depth = -0.019
      wall(mat,lo,hi,-1,-u)
      wall(mat,lo,hi,u,1)
      wall(mat,lo,base,-u,u)
      const head = (s: number) => spring + arch*Math.sqrt(Math.max(0,1-s*s/(u*u)))
      const pieces = 10
      for (let k=0; k<pieces; k++) {
        const l=-u*Math.cos(Math.PI*k/pieces), r=-u*Math.cos(Math.PI*(k+1)/pieces)
        const left = face(l,head(l)), right = face(r,head(r))
        mesh.face(mat,[left,right,face(r,hi),face(l,hi)],n)
        mesh.face('wrought iron',[left,face(l,head(l),depth),face(r,head(r),depth),right],[0,-1,0])
        mesh.face(doorway ? 'harbor blue' : 'lamp glass',
          [face(l,base,depth),face(r,base,depth),face(r,head(r),depth),face(l,head(l),depth)],n)
        mesh.beam(doorway ? 'capstone' : 'lens brass',left,right,doorway ? 0.025 : 0.015,doorway ? 0.029 : 0.017)
      }
      for (const side of [-1,1]) {
        mesh.face('wrought iron',[face(side*u,base),face(side*u,spring),
          face(side*u,spring,depth),face(side*u,base,depth)],[-side*t[0],0,-side*t[2]])
        mesh.beam(doorway ? 'capstone' : 'lens brass',face(side*u,base),face(side*u,spring),
          doorway ? 0.026 : 0.015,doorway ? 0.03 : 0.017)
      }
      mesh.face('weathered limestone',[face(-u,base),face(u,base),face(u,base,depth),face(-u,base,depth)],[0,1,0])
      mesh.beam(doorway ? 'sawn cedar' : 'lens brass',face(-u,base),face(u,base),0.019,0.021)
      if (doorway) {
        for (const y of [base+0.09,spring-0.04]) {
          mesh.beam('wrought iron',face(-u,y,-0.009),face(u*0.45,y,-0.009),0.017,0.030)
        }
        mesh.beam('lens brass',face(u*0.72,base+0.18,0),face(u*0.72,base+0.23,0),0.024,0.018)
      } else {
        mesh.beam('lens brass',face(0,base,-0.003),face(0,spring+arch,-0.003),0.010,0.014)
        mesh.beam('lens brass',face(-u,spring-0.030,-0.003),face(u,spring-0.030,-0.003),0.010,0.014)
      }
    }
    if (i===0) opening(0.627,1.137,0.643,1.005,0.082,0.74,true)
    else wall('beacon ivory',0.627,1.137)
    wall('beacon oxblood',1.137,1.465)
    if (i===2 || i===6) opening(1.465,1.817,1.503,1.649,0.051,0.44,false)
    else wall('beacon ivory',1.465,1.817)
  }
  const footing = at(entry,0.492,0.591)
  sector('weathered limestone',entry-0.22,entry+0.22,0.463,0.596,footing[1],0.031)

  // A cast spiral starts on the coping and ends at the gallery. Its treads
  // are cantilevered from the tower with an open space between each riser.
  const steps = 16, start = entry+0.56, sweep = 1.42*Math.PI, rise=(1.856-0.568)/steps
  const rail: Point[] = []
  for (let i=0; i<steps; i++) {
    const a=start+i*sweep/steps, b=start+(i+1)*sweep/steps, y=0.568+(i+1)*rise
    const inner=radiusAt(y)*Math.cos(halfFace)-0.012, r=inner+0.205
    sector('weathered zinc',a,b,inner,r,y,0.023)
    mesh.beam('wrought iron',at((a+b)/2,inner,y-0.076),at((a+b)/2,r-0.035,y-0.027),0.019,0.024)
    if (i%2===0) {
      const head = at(a,r-0.013,y+0.217)
      mesh.beam('wrought iron',at(a,r-0.013,y-0.014),head,0.018,0.018)
      rail.push(head)
    }
    if (i===steps-1) rail.push(at(b,r-0.013,y+0.217))
  }
  for (let i=1; i<rail.length; i++) mesh.beam('aged zinc',rail[i-1],rail[i],0.023,0.023)

  drum('beacon oxblood',[[1.807,0.324],[1.838,0.339],[1.859,0.339]],16)
  drum('aged zinc',[[1.814,0.348],[1.849,0.468],[1.862,0.468]],20)
  const gate=start+sweep
  for (let i=0; i<16; i++) {
    const a=gate+i*turn/16, b=a+turn/16
    // One segment stays open at the head of the spiral stair.
    if (i!==15) for (const y of [1.977,2.096]) {
      mesh.beam(y>2 ? 'lens brass' : 'aged zinc',at(a,0.451,y),at(b,0.451,y),0.017,0.018)
    }
    mesh.beam('wrought iron',at(a,0.451,1.851),at(a,0.451,2.099),0.018,0.018)
    if (i%2===0) mesh.beam('wrought iron',at(a,0.288,1.700),at(a,0.447,1.838),0.032,0.029)
  }
  const railEnd=rail.at(-1)!
  mesh.beam('aged zinc',railEnd,at(gate,0.451,2.096),0.023,0.023)

  // Clear gaps between mullions let the fluted amber optic read from either bank.
  drum('verdigris',[[1.860,0.272],[1.900,0.284],[1.944,0.260],[2.002,0.256]],12)
  drum('lens brass',[[2.000,0.189],[2.017,0.196],[2.042,0.149]],16)
  const lens: [number,number][] = [[2.042,0.126]]
  for (let i=0; i<8; i++) {
    const t=i/7, y=2.05+i*0.034, r=0.127+0.043*Math.sin(t*Math.PI)
    lens.push([y,r],[y+0.008,r+0.016],[y+0.027,r])
  }
  lens.push([2.331,0.112])
  drum('lantern amber',lens,32)
  drum('lens brass',[[2.324,0.13],[2.339,0.129],[2.353,0.077]],16)
  for (let i=0; i<8; i++) {
    const a=entry+halfFace+i*turn/8, b=a+turn/8
    mesh.beam('wrought iron',at(a,0.244,1.992),at(a,0.244,2.390),0.023,0.025)
    mesh.beam('lens brass',at(a,0.245,2.371),at(b,0.245,2.371),0.018,0.015)
    mesh.beam('aged zinc',at(a,0.244,2.288),at(b,0.244,2.377),0.011,0.012)
  }
  // The ogee copper roof is spun in gores. Raised seams follow the change of
  // profile into the vented crown and catch oblique light.
  const roof: [number,number][] = [[2.382,0.259],[2.396,0.336],[2.414,0.345],[2.433,0.335],
    [2.477,0.275],[2.529,0.250],[2.581,0.207],[2.621,0.139],[2.639,0.074]]
  drum('verdigris',roof,24)
  for (let i=0; i<12; i++) {
    const a=i*turn/12
    for (let j=3; j<roof.length; j++) {
      const [h0,r0]=roof[j-1],[h1,r1]=roof[j]
      mesh.beam('weathered zinc',at(a,r0+0.004,h0+0.002),at(a,r1+0.004,h1+0.002),0.009,0.008)
    }
  }
  drum('lens brass',[[2.638,0.079],[2.656,0.081],[2.666,0.063]],16)
  for (let i=0; i<8; i++) mesh.beam('wrought iron',at(i*turn/8,0.058,2.661),at(i*turn/8,0.058,2.717),0.012,0.013)
  drum('verdigris',[[2.710,0.076],[2.721,0.084],[2.752,0.048],[2.766,0.020]],16)
  drum('lens brass',[[2.761,0.019],[2.795,0.022],[2.818,0.009],[2.862,0.004]],12)
}
