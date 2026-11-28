import type { HarborMesh } from './cove-quayside'

type Point = [number, number, number]
const ridge = 1.85, pitch = 0.59
const roofY = (z: number) => ridge - pitch*Math.abs(z)
const halfSpan = 1.40, run = 0.827, panels = 10, panelWidth = 2*halfSpan/panels
// The curb follows entire standing-seam panels so the roof can be flashed on all four sides.
const firstPanel = 2, lastPanel = 7, left = -halfSpan+firstPanel*panelWidth, right = -halfSpan+lastPanel*panelWidth
const center = (left+right)/2, curb = 0.30, sill = 1.88, eave = 2.34

function pitchedRoof(mesh: HarborMesh) {
  for (const side of [-1,1]) {
    const z = side*run
    for (let i = 0; i < panels; i++) {
      const a = -halfSpan+i*panelWidth, b = a+panelWidth
      const upper = i >= firstPanel && i < lastPanel ? side*curb : 0
      const sheet: Point[] = [[a,roofY(upper),upper],[b,roofY(upper),upper],[b,roofY(z),z],[a,roofY(z),z]]
      mesh.face((i*3+side)%5 === 0 ? 'weathered zinc' : 'aged zinc',sheet,[0,1,side])
      mesh.face('dock oak',sheet.map(([x,y,z]) => [x,y-0.040,z]),[0,-1,-side])
      // A stepped flashing laps onto the trimmed sheet and turns up the curb.
      if (upper !== 0) {
        const drip = side*(curb+0.059)
        mesh.face('zinc flashing',[[a,roofY(drip)+0.004,drip],[b,roofY(drip)+0.004,drip],
          [b,roofY(upper)+0.008,upper],[a,roofY(upper)+0.008,upper]],[0,1,side])
        mesh.face('aged zinc',[[a,roofY(upper)+0.008,upper],[b,roofY(upper)+0.008,upper],
          [b,roofY(upper)+0.076,upper],[a,roofY(upper)+0.076,upper]],[0,0,side])
      }
    }
    for (let i=0; i<=panels; i++) {
      const x=-halfSpan+i*panelWidth, upper=i>firstPanel && i<lastPanel ? side*curb : 0
      mesh.beam('zinc flashing',[x,roofY(upper)+0.004,upper],[x,roofY(z)+0.004,z],0.012,0.018)
    }
    for (const [a,b] of [[-halfSpan,left],[right,halfSpan]]) {
      mesh.face('zinc flashing',[[a,ridge+0.017,0],[b,ridge+0.017,0],
        [b,roofY(0.05)+0.017,side*0.05],[a,roofY(0.05)+0.017,side*0.05]],[0,1,side])
    }
    mesh.box('dock oak',0,roofY(z)-0.025,z,2*halfSpan,0.048,0.032)
    for (const x of [-halfSpan,halfSpan]) mesh.beam('sawn cedar',[x,ridge-0.025,0],[x,roofY(z)-0.025,z],0.036,0.055)
  }
}

function soundWindows(mesh: HarborMesh) {
  const lower = roofY(curb), width = right-left
  // Seat both long walls on the cut edge. The upper chamber is open through
  // its center bay and has angled sound louvres at each end.
  for (const side of [-1,1]) {
    const z = side*(curb-0.022)
    mesh.box('harbor blue',center,(lower+sill)/2,z,width,sill-lower,0.044)
    mesh.box('cream band',center,sill,z,width+0.035,0.048,0.080)
    mesh.box('dock oak',center,eave-0.033,z,width+0.035,0.067,0.072)
    for (const x of [left+0.033,center-0.385,center+0.385,right-0.033]) {
      mesh.box('cream band',x,(sill+eave)/2,z,0.051,eave-sill,0.064)
    }
    for (const x of [(left+0.033+center-0.385)/2,(right-0.033+center+0.385)/2]) {
      const half = (width/2-0.385-0.033-0.04)/2
      for (let row=0; row<4; row++) {
        const y = sill+0.073+row*0.076
        const a: Point=[x-half,y+0.026,z-side*0.045], b: Point=[x+half,y+0.026,z-side*0.045]
        const c: Point=[x+half,y-0.012,z+side*0.037], d: Point=[x-half,y-0.012,z+side*0.037]
        mesh.face('harbor blue',[a,b,c,d],[0,1,side])
        mesh.face('sawn cedar',[a,b,c,d].map(([px,py,pz])=>[px,py-0.014,pz]),[0,-1,-side])
        mesh.beam('cream band',c,d,0.013,0.017)
      }
    }
    for (const x of [center-0.385,center+0.385]) {
      mesh.beam('sawn cedar',[x,eave-0.195,z],[x+Math.sign(center-x)*0.124,eave-0.062,z],0.044,0.053)
    }
  }
  for (const side of [-1,1]) {
    const x=side<0 ? left : right
    // This end follows the original gable slopes down to each eave.
    for (const sz of [-1,1]) {
      mesh.face('harbor blue',[[x,ridge,0],[x,lower,sz*curb],[x,sill,sz*curb],[x,sill,0]],[side,0,0])
      mesh.beam('zinc flashing',[x,ridge+0.009,0],[x,lower+0.009,sz*curb],0.052,0.020)
    }
    mesh.box('cream band',x,sill,0,0.079,0.047,curb*2)
    mesh.box('dock oak',x,eave-0.030,0,0.074,0.062,curb*2)
    // Cross-braced side openings leave the bell silhouetted from either gable.
    for (const sz of [-1,1]) mesh.beam('harbor blue',[x,sill+0.020,sz*0.254],[x,eave-0.047,-sz*0.254],0.037,0.039)
  }
}

function castBell(mesh: HarborMesh) {
  const y = sill+0.004, step = Math.PI*2/32
  const profile = [[0.043,0.161],[0.048,0.178],[0.061,0.184],[0.077,0.179],[0.100,0.153],
    [0.132,0.137],[0.190,0.119],[0.241,0.109],[0.263,0.104],[0.285,0.077],[0.295,0.035]]
  const point = (angle: number,radius: number,h: number): Point => [center+radius*Math.cos(angle),y+h,radius*Math.sin(angle)]
  // Spin the inside and outside of the casting independently. The thick lip
  // and clapper remain visible when looking up into the chamber from the dock.
  for (let i=0; i<32; i++) {
    const a=i*step,b=(i+1)*step,n:Point=[Math.cos(a+step/2),0,Math.sin(a+step/2)]
    for (let row=1; row<profile.length; row++) {
      const [h0,r0]=profile[row-1], [h1,r1]=profile[row]
      mesh.face('lens brass',[point(a,r0,h0),point(b,r0,h0),point(b,r1,h1),point(a,r1,h1)],n)
      mesh.face('lens brass',[point(a,r0-0.013,h0),point(b,r0-0.013,h0),point(b,r1-0.013,h1),point(a,r1-0.013,h1)],[-n[0],0,-n[2]])
    }
    mesh.face('lens brass',[point(a,0.148,0.043),point(b,0.148,0.043),point(b,0.161,0.043),point(a,0.161,0.043)],[0,-1,0])
    mesh.face('lens brass',[point(a,0.035,0.295),point(b,0.035,0.295),[center,y+0.299,0]],[0,1,0])
  }
  mesh.column('wrought iron',center,0,[[y+0.013,0.018],[y+0.026,0.034],[y+0.047,0.037],[y+0.066,0.014],[y+0.281,0.010]],14)
  mesh.box('dock oak',center,y+0.347,0,0.128,0.089,0.35)
  for (const z of [-0.063,0.063]) {
    mesh.beam('wrought iron',[center,y+0.281,z*0.50],[center,y+0.359,z],0.025,0.026)
  }
  mesh.beam('lens brass',[center,y+0.347,-curb-0.018],[center,y+0.347,curb+0.018],0.030,0.030)
  for (const z of [-curb,curb]) {
    mesh.box('dock oak',center,eave-0.075,z,0.080,0.130,0.061)
    mesh.box('wrought iron',center,y+0.347,z,0.079,0.047,0.082)
  }
  mesh.beam('wrought iron',[center,y+0.352,0.126],[center+0.19,y+0.325,0.126],0.022,0.026)
  mesh.beam('hemp rope',[center+0.19,y+0.325,0.126],[center+0.204,sill-0.46,0.139],0.013,0.013)
  mesh.beam('hemp rope',[center+0.204,sill-0.46,0.139],[center+0.22,0.70,0.13],0.013,0.013)
}

function copperRoof(mesh: HarborMesh) {
  const span=0.805, depth=0.383, ridgeHalf=0.405, crown=2.59
  const ring=(sx: number,sz: number,h:number):Point=>[center+sx*span,h,sz*depth]
  const ridgePoint=(sx:number):Point=>[center+sx*ridgeHalf,crown,0]
  const sheet=(points: Point[], out: Point, underside: string) => {
    const thickness=0.014/Math.hypot(...out)
    mesh.face('verdigris',points,out)
    mesh.face(underside,points.map(([x,y,z])=>[x-thickness*out[0],y-thickness*out[1],z-thickness*out[2]]),
      [-out[0],-out[1],-out[2]])
  }
  for (const side of [-1,1]) {
    const edgeB=ring(1,side,eave)
    mesh.box('cream band',center,eave-0.009,side*depth,2*span,0.029,0.025)
    mesh.box('cream band',center+side*span,eave-0.009,0,0.026,0.029,depth*2)
    const iron=(u: number,v: number):Point=>[center+(ridgeHalf+(span-ridgeHalf)*v)*(2*u-1),
      crown+(eave-crown)*v,side*depth*v]
    // The three central sheets on the sea side form one propped hatch. Its
    // clear aperture lets overhead light reach the bell's flared mouth.
    const hinge=0.12, apron=0.80, angle=2.02, c=Math.cos(angle), s=Math.sin(angle)
    const lifted=(u:number,v:number):Point=>{
      const p=iron(u,v), axis=iron(u,hinge), dy=p[1]-axis[1], dz=p[2]-axis[2]
      return [p[0],axis[1]+c*dy+s*dz,axis[2]-s*dy+c*dz]
    }
    const normal:Point=[0,1,side*(crown-eave)/depth]
    const tilted:Point=[0,c*normal[1]+s*normal[2],-s*normal[1]+c*normal[2]]
    for (let i=0; i<7; i++) {
      const t0=i/7,t1=(i+1)/7
      const open = side>0 && i>=2 && i<=4
      for (const [v0,v1] of open ? [[0,hinge],[apron,1]] : [[0,1]]) {
        sheet([iron(t0,v1),iron(t1,v1),iron(t1,v0),iron(t0,v0)],normal,'dock oak')
        mesh.beam('verdigris',iron(t0,v1),iron(t0,v0),0.015,0.016)
      }
      if (open) {
        sheet([lifted(t0,apron),lifted(t1,apron),lifted(t1,hinge),lifted(t0,hinge)],tilted,'sawn cedar')
        mesh.beam('verdigris',lifted(t0,apron),lifted(t0,hinge),0.016,0.015)
      }
    }
    if (side>0) {
      for (const u of [2/7,5/7]) {
        mesh.beam('sawn cedar',iron(u,hinge),iron(u,apron),0.025,0.025)
        mesh.beam('wrought iron',iron(u,0.74),lifted(u,0.62),0.014,0.014)
      }
      mesh.beam('verdigris',lifted(2/7,apron),lifted(5/7,apron),0.022,0.020)
      mesh.beam('lens brass',iron(2/7,hinge),iron(5/7,hinge),0.025,0.022)
    }
    mesh.beam('verdigris',edgeB,ridgePoint(1),0.015,0.016)
    sheet([ring(side,-1,eave),ring(side,1,eave),ridgePoint(side)],[side*(crown-eave)/(span-ridgeHalf),1,0],'dock oak')
    mesh.beam('verdigris',ring(side,0,eave),ridgePoint(side),0.015,0.016)
  }
  mesh.beam('verdigris',ridgePoint(-1),ridgePoint(1),0.036,0.027)
}

export function makeBellLoftRoof(mesh: HarborMesh) {
  pitchedRoof(mesh)
  ++mesh.object
  soundWindows(mesh)
  castBell(mesh)
  copperRoof(mesh)
}
