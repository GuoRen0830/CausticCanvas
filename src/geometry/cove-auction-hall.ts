import { HarborMesh } from './cove-quayside'
import { shoreHeight } from './cove-shore'

type Point = [number, number, number]
const blue = 'harbor blue', cedar = 'sawn cedar', oak = 'dock oak', white = 'salt-bleached slats'
const clay = 'fish market terracotta', fired = 'fish market fired clay'
const ivory = 'ivory buoy paint', brass = 'lens brass'
const back = -4.36
export const auctionFront = -3.65

// A roof pan and the ridge over its joint are cut from one S-shaped section.
// The courses overlap uphill. Thickness is visible at the eaves and verges.
function pantiles(mesh: HarborMesh, surface: (u: number, v: number) => Point, columns: number, rows: number) {
  const profile = [0.003,0.001,0,0.005,0.022,0.033,0.023,0.004]
  for (let j=0; j<rows; j++) for (let i=0; i<columns; i++) {
    const mat = (7*i+3*j+i*j)%9 < 3 ? fired : clay
    const lo = (j-(j ? 0.072 : 0))/rows, hi = (j+1)/rows
    const at = (k: number, t: number, lower=false): Point => {
      const u = (i+(k/(profile.length-1))*0.993)/columns
      const [x,y,z] = surface(u,t)
      return [x,y+profile[k]+0.012*(hi-t)/(hi-lo)-(lower ? 0.010 : 0),z]
    }
    for (let k=0; k+1<profile.length; k++) {
      const a=at(k,lo), b=at(k+1,lo), c=at(k+1,hi), d=at(k,hi)
      mesh.face(mat,[a,b,c,d],[0,1,0])
      mesh.face(mat,[a,b,at(k+1,lo,true),at(k,lo,true)],[a[0]-d[0],0,a[2]-d[2]])
      if (i===0 && k===0 || i===columns-1 && k===profile.length-2) {
        const side = i===0 ? k : k+1, opposite=i===0 ? b : a, outside=at(side,lo)
        mesh.face(mat,[outside,at(side,hi),at(side,hi,true),at(side,lo,true)],
          [outside[0]-opposite[0],0,outside[2]-opposite[2]])
      }
    }
  }
}

function receivingShutter(mesh: HarborMesh, x: number, z: number, y: number, width: number, side: number) {
  const project = (u: number,h: number,offset: number): Point => [x+u,y+h,z+side*offset]
  for (let i=0; i<5; i++) mesh.box(i%3===1 ? cedar : blue,x+width*((i+0.5)/5-0.5),y+0.46,z,width/5-0.007,0.918,0.037)
  for (const h of [0.13,0.78]) {
    mesh.box(white,x,y+h,z+side*0.038,width*0.94,0.044,0.040)
    mesh.beam('wrought iron',project(-0.48*width,h,0.065),project(-0.09*width,h,0.065),0.018,0.033)
  }
  mesh.beam(white,project(-0.43*width,0.18,0.040),project(0.43*width,0.74,0.040),0.038,0.046)
  mesh.box('wrought iron',x+0.33*width,y+0.47,z+side*0.047,0.035,0.07,0.045)
}

// Geometric letters follow the pitch of the sign so both the ivory and its
// surround catch light from above. They remain legible in an exported mesh.
function pedimentSign(mesh: HarborMesh, x: number, y: number, z: number) {
  const p = (u: number,h: number,d: number): Point => [x+u,y+h,z-0.31*h+d]
  const a=p(-0.405,-0.066,0), b=p(0.405,-0.066,0), c=p(0.405,0.066,0), d=p(-0.405,0.066,0)
  mesh.face(blue,[a,b,c,d],[0,0.31,1])
  for(const [from,to] of [[a,b],[b,c],[c,d],[d,a]]) mesh.beam(brass,from,to,0.010,0.013)
  // Polylines are in letter space, from the lower left corner of each character.
  const letters = [
    [[[0,0],[0,1],[0.77,1]],[[0,0.53],[0.62,0.53]]],
    [[[0.14,1],[0.74,1]],[[0.44,1],[0.44,0]],[[0.13,0],[0.75,0]]],
    [[[0.78,0.94],[0.65,1],[0.14,1],[0,0.85],[0,0.64],[0.15,0.52],
      [0.64,0.48],[0.78,0.34],[0.78,0.14],[0.65,0],[0.13,0],[0,0.083]]],
    [[[0,0],[0,1]],[[0,0.5],[0.77,0.5]],[[0.77,1],[0.77,0]]],
  ]
  for (const [i,letter] of letters.entries()) for(const path of letter) {
    const points=path.map(([u,h])=>p(-0.272+i*0.145+u*0.14,(h-0.5)*0.092,0.008))
    for(let j=1;j<points.length;j++) mesh.beam(ivory,points[j-1],points[j],0.011,0.016)
  }
}

function clock(mesh: HarborMesh, x: number, y: number, z: number) {
  const at=(t: number,r: number,d: number): Point=>[x+r*Math.sin(t),y+r*Math.cos(t),z-0.20*r*Math.cos(t)+d]
  const rim: Point[]=[]
  for (let i=0;i<32;i++) {
    const a=i*Math.PI/16,b=(i+1)*Math.PI/16
    rim.push(at(a,0.122,0.014))
    mesh.face(brass,[at(a,0.139,0),at(b,0.139,0),at(b,0.122,0.018),at(a,0.122,0.018)],[0,0.2,1])
  }
  mesh.face(ivory,rim,[0,0.2,1])
  for(let i=0;i<12;i++){
    const a=i*Math.PI/6
    mesh.beam(blue,at(a,0.109,0.018),at(a,i%3 ? 0.092 : 0.084,0.018),0.008,0.009)
  }
  mesh.beam('wrought iron',at(0,0.007,0.022),at(-1.00,0.061,0.022),0.011,0.015)
  mesh.beam('wrought iron',at(0,0.004,0.026),at(1.08,0.088,0.026),0.008,0.012)
}

// This is a shallow timber fish exchange behind the three existing counters.
// The open center window looks right through to the receiving hatch at the rear.
export function addAuctionHall(mesh: HarborMesh, left: number, span: number, floor: number) {
  const center=left+1.5*span, front=auctionFront, depth=front-back
  const right=left+span*3, midZ=(front+back)/2, high=2.99, eave=2.48
  for (const z of [back,front]) {
    for(let i=0;i<=3;i++){
      const x=left+i*span, base=shoreHeight(x,z)-0.020
      mesh.box(oak,x,(eave+base)/2,z,0.076,eave-base,0.072)
    }
  }
  for(let bay=0;bay<3;bay++){
    ++mesh.object
    const x=left+(bay+0.5)*span, width=span-0.078
    for(let i=0;i<9;i++){
      mesh.box(cedar,x-span/2+(i+0.5)*span/9,floor-0.042,(back-4.07)/2,span/9-0.004,0.064,-4.07-back)
    }
    for(const side of [-1,1]){
      const z=side>0 ? front : back, sill=bay===1 ? 1.48 : 1.32
      const baseWidth=width/8
      for(let i=0;i<8;i++){
        const paint=side>0 || bay===1 ? blue : i%3===1 ? 'ochre boards' : cedar
        mesh.box(paint,x-width/2+(i+0.5)*baseWidth,(floor+sill)/2,z,baseWidth-0.003,sill-floor,0.042)
      }
      mesh.box(white,x,sill,z+side*0.010,width+0.020,0.045,0.094)
      mesh.box(blue,x,2.381,z,width,0.16,0.065)
      mesh.box(white,x,2.302,z+side*0.018,width,0.037,0.071)
      if (bay===1) {
        for (const dir of [-1,1]) {
          const jamb=x+dir*0.468
          mesh.box(white,jamb,1.88,z+side*0.010,0.066,0.80,0.075)
          const spring: Point=[jamb,2.07,z+side*0.01]
          mesh.beam(white,spring,[jamb-dir*0.19,2.286,z+side*0.01],0.064,0.061)
        }
        // A short shelf on each side serves the auction ledger and delivery slips.
        mesh.box(cedar,x,1.49,z+side*0.037,0.85,0.048,0.15)
      } else if (side<0) {
        receivingShutter(mesh,x+(bay===0 ? -0.21 : 0.21),z-0.034,sill+0.030,0.46,-1)
        const openX=x+(bay===0 ? 0.27 : -0.27)
        for(let k=0;k<5;k++) {
          const y=sill+0.49+0.090*k
          mesh.face(white,[[openX-0.19,y-0.030,z-0.045],[openX+0.19,y-0.030,z-0.045],
            [openX+0.19,y+0.023,z+0.020],[openX-0.19,y+0.023,z+0.020]],[0,1,-1])
        }
      }
    }
    if (bay===1) continue
    const leftEdge=bay===0 ? x-span/2-0.068 : x-span/2+0.015
    const rightEdge=bay===0 ? x+span/2-0.015 : x+span/2+0.068
    const roof=(u: number,v: number): Point=>[mix(leftEdge,rightEdge,u),2.47+0.16*v,front+0.112-v*(depth+0.186)]
    mesh.face(cedar,[roof(0,0),roof(0,1),roof(1,1),roof(1,0)],[0,-1,0])
    pantiles(mesh,roof,9,5)
    mesh.beam(white,roof(0,0),roof(1,0),0.033,0.063)
    mesh.beam(cedar,roof(0,1),roof(1,1),0.033,0.075)
    const endX=bay===0 ? left : right
    for(let i=0;i<7;i++){
      const z=back+depth*(i+0.5)/7, height=2.47+0.16*(front-z)/depth
      mesh.box(i%3===2 ? cedar : blue,endX,(floor+height)/2,z,0.051,height-floor,depth/7-0.003)
    }
    mesh.beam(white,[endX,floor+0.09,back],[endX,1.46,front],0.058,0.077)
    // Cured fish on the west curing rail leave the counter free for today's catch.
    if(bay===0){
      mesh.beam(oak,[x-0.48,2.06,midZ],[x+0.48,2.06,midZ],0.041,0.05)
      for(const u of [-0.44,0.44]) mesh.beam(oak,[x+u,2.23,back],[x+u,2.06,midZ],0.038,0.045)
      for(let i=0;i<5;i++){
        const cx=x-0.32+i*0.16, top=1.98-0.035*(i%3), dz=0.014*(i%2)
        mesh.beam('hemp rope',[cx,2.08,midZ],[cx,top,midZ+dz],0.010,0.011)
        const tail: Point=[cx,top,midZ+dz], head: Point=[cx-0.020,top-0.30,midZ+dz+0.006]
        const a: Point=[cx-0.063,top-0.176,midZ+dz], b: Point=[cx+0.046,top-0.179,midZ+dz]
        for(const sign of [-1,1]){
          const spine: Point=[cx-0.009,top-0.190,midZ+dz+sign*0.026]
          for(const [p,q,skin] of [[tail,b,'fish market mackerel'],[b,head,'fish market mackerel'],
            [head,a,'fish market silver'],[a,tail,'fish market silver']] as const) mesh.face(skin,[p,q,spine],[0,0,sign])
        }
      }
    }
  }

  ++mesh.object
  const half=span/2+0.095
  for(const side of [-1,1]) {
    const z=side>0 ? front : back
    mesh.face(blue,[[center-half,eave,z],[center+half,eave,z],[center,high,z]],[0,0,side])
    for(const dir of [-1,1]){
      mesh.beam(white,[center+dir*half,eave,z+side*0.045],[center,high,z+side*0.045],0.065,0.059)
    }
    if(side>0){
      pedimentSign(mesh,center,2.553,z+0.054)
      clock(mesh,center,2.759,z+0.058)
    }else{
      for(let k=-2;k<=2;k++){
        const dz=Math.abs(k), h=0.24-0.06*dz
        mesh.box(white,center+k*0.105,2.57+h/2,z-0.015,0.037,h,0.047)
      }
    }
  }
  for(const side of [-1,1]) {
    const roof=(u:number,v:number):Point=>[center+side*half*(1-v),mix(eave,high,v),mix(back-0.08,front+0.115,u)]
    mesh.face(cedar,[roof(0,0),roof(1,0),roof(1,1),roof(0,1)],[0,-1,0])
    pantiles(mesh,roof,7,4)
    mesh.beam(white,roof(0,0),roof(1,0),0.034,0.068)
  }
  mesh.beam(clay,[center,high+0.017,back-0.083],[center,high+0.017,front+0.118],0.087,0.056)
}

const mix=(a:number,b:number,t:number)=>a+(b-a)*t
