type Point = [number, number, number]
type UV = [number, number]
type Skin = {
  height: (x: number, z: number) => number
  vertex: (p: Point, n: Point, uv: UV, object: number) => number
  tri: (material: string, a: number, b: number, c: number) => void
}

function normal(a: Point, b: Point, c: Point): Point {
  const x=b[0]-a[0], y=b[1]-a[1], z=b[2]-a[2]
  const u=c[0]-a[0], v=c[1]-a[1], w=c[2]-a[2]
  return [y*w-z*v,z*u-x*w,x*v-y*u]
}

// The outline has five broad armpits and narrowing, individually bent rays.
// Sample the very same shelf triangles as the export so the underside seats
// on the stratum even where an arm crosses a fissure.
function star(skin: Skin, object: number, cx: number, cz: number, size: number, phase: number) {
  const perArm=20, count=5*perArm, bands=10
  const center: Point=[cx,skin.height(cx,cz)+0.166*size,cz]
  const uv=(p:Point):UV=>[0.5+(p[0]-cx)/(2.4*size),0.5+(p[2]-cz)/(2.4*size)]
  const outline=Array.from({length:count},(_,k)=>{
    const arm=Math.floor((k+perArm/2)/perArm)%5, angle=2*Math.PI*k/count
    const offset=((k+perArm/2)%perArm-perArm/2)*2*Math.PI/count
    const peak=(0.92+0.09*Math.cos(phase+2.1*arm))/(1+(offset/0.19)**2)
    return {angle:phase+angle, reach:0.245+0.755*peak, bend:0.195*Math.sin(phase+1.6*arm), peak}
  })
  const rows:Point[][]=[]
  for(let j=1;j<=bands;j++) {
    // Tighter rings at the margin retain the rounded lip against sloped rock.
    const t=Math.sin(Math.PI*j/(2*bands))
    rows.push(outline.map(({angle,reach,bend,peak}):Point=>{
      const radius=t*reach, direction=angle+bend*radius**1.5
      const x=cx+size*radius*Math.cos(direction), z=cz+size*radius*Math.sin(direction)
      const profile=(0.166-0.05*t+0.035*t*peak)*Math.sqrt(Math.max(0,1-t*t))
      return [x,skin.height(x,z)+size*(profile-0.009*t**6),z]
    }))
  }
  const next=(k:number)=>(k+1)%count
  const surfaceNormal=(row:number,k:number)=>{
    const prev=(k+count-1)%count, out=rows[Math.min(bands-1,row+1)][k]
    const inside=row===0 ? center : rows[row-1][k]
    const tangent:Point=[inside[0]+rows[row][next(k)][0]-rows[row][prev][0],
      inside[1]+rows[row][next(k)][1]-rows[row][prev][1],inside[2]+rows[row][next(k)][2]-rows[row][prev][2]]
    return normal(inside,tangent,out)
  }
  const apex=skin.vertex(center,[0,1,0],uv(center),object)
  let inner:number[]=[]
  for(let j=0;j<bands;j++) {
    const outer=rows[j].map((p,k)=>skin.vertex(p,surfaceNormal(j,k),uv(p),object))
    for(let k=0;k<count;k++) {
      const n=next(k)
      if(!j) skin.tri('ochre sea star',apex,outer[n],outer[k])
      else {
        // Where a curved arm crosses a rock joint, choose the shorter
        // diagonal so a skinny triangle cannot cut across the fold.
        const a=rows[j-1][k], b=rows[j-1][n], c=rows[j][k], d=rows[j][n]
        const distance=(p:Point,q:Point)=>Math.hypot(p[0]-q[0],p[1]-q[1],p[2]-q[2])
        if(distance(a,d)<distance(b,c) && normal(a,b,d)[1]>0 && normal(a,d,c)[1]>0) {
          skin.tri('ochre sea star',inner[k],inner[n],outer[n])
          skin.tri('ochre sea star',inner[k],outer[n],outer[k])
        } else {
          skin.tri('ochre sea star',inner[k],inner[n],outer[k])
          skin.tri('ochre sea star',outer[k],inner[n],outer[n])
        }
      }
    }
    inner=outer
  }
  // Two broken chains of raised calcite ossicles follow each ray's ridge.
  for(let arm=0;arm<5;arm++) for(let row=1;row<7;row++) for(const side of [-1,1]) {
    const k=(arm*perArm+side*(1+row%2)+count)%count
    const p=rows[row][k], n=surfaceNormal(row,k)
    const r=size*(0.018+0.005*Math.cos(arm+row-side))
    const top=skin.vertex([p[0],p[1]+r*.55,p[2]],n,[0.5,0.5],object)
    const rim:number[]=[]
    for(let i=0;i<5;i++) {
      const angle=2*Math.PI*i/5+phase, dx=r*Math.cos(angle), dz=r*Math.sin(angle)
      const y=p[1]-(n[0]*dx+n[2]*dz)/n[1]-r*.23
      rim.push(skin.vertex([p[0]+dx,y,p[2]+dz],n,[0.5+0.48*Math.cos(angle),0.5+0.48*Math.sin(angle)],object))
    }
    for(let i=0;i<5;i++) skin.tri('sea star ossicles',top,rim[(i+1)%5],rim[i])
  }
}

// Six folded wall plates lead to a pinched oval aperture. Its dark opercular
// plates sit below the lip, so grazing light also reads the recess in closeup.
function barnacle(skin: Skin, object: number, x: number, z: number, r: number, angle: number) {
  const count=12, cos=Math.cos(angle), sin=Math.sin(angle)
  function ring(radius:number,rise:number,squeeze:number) {
    return Array.from({length:count},(_,i):Point=>{
      const a=2*Math.PI*i/count, rib=i%2 ? 0.88 : 1
      const u=r*(radius*rib*Math.cos(a)+rise*.20), v=r*radius*rib*Math.sin(a)*squeeze
      const px=x+cos*u-sin*v, pz=z+sin*u+cos*v
      return [px,skin.height(px,pz)+r*rise*(0.94+0.06*Math.cos(3*a+angle)),pz]
    })
  }
  const tiers=[ring(1,-0.065,.83),ring(.92,.23,.82),ring(.63,.69,.75),ring(.47,.80,.58),ring(.34,.58,.37)]
  for(let j=0;j<tiers.length-1;j++) for(let i=0;i<count;i++) {
    const k=(i+1)%count, lower=tiers[j], upper=tiers[j+1]
    const n=normal(lower[i],upper[i],lower[k]), material=j===3 ? 'barnacle aperture' : 'acorn barnacle'
    // Unroll the mantle UV at its seam so ridges keep the same texel scale.
    const ids=[skin.vertex(lower[i],n,[i/count,j/4],object),skin.vertex(upper[i],n,[i/count,(j+1)/4],object),
      skin.vertex(lower[k],n,[(i+1)/count,j/4],object),skin.vertex(upper[k],n,[(i+1)/count,(j+1)/4],object)]
    skin.tri(material,ids[0],ids[1],ids[2]); skin.tri(material,ids[2],ids[1],ids[3])
  }
  const rim=tiers[tiers.length-1]
  const lid:Point=[x+r*.116*cos,skin.height(x+r*.116*cos,z+r*.116*sin)+r*.51,z+r*.116*sin]
  const center=skin.vertex(lid,[0,1,0],[.5,.5],object)
  const edge=rim.map((p,i)=>skin.vertex(p,[0,1,0],[.5+.45*Math.cos(2*Math.PI*i/count),.5+.45*Math.sin(2*Math.PI*i/count)],object))
  for(let i=0;i<count;i++) skin.tri('barnacle aperture',center,edge[(i+1)%count],edge[i])
}

export function addIntertidalLife(skin: Skin, object: number) {
  for(const [x,z,r,heading] of [[-4.70,.53,.225,.28],[-4.25,1.32,.136,2.17],[-4.55,2.13,.175,4.91]]) {
    star(skin,object++,x,z,r,heading)
  }
  const colonies=[[-5.48,.66,13,.87],[-4.25,.79,6,1.36],[-4.96,1.77,11,2.06],[-3.83,1.47,8,3.97]]
  for(const [cx,cz,count,phase] of colonies) for(let i=0;i<count;i++) {
    const angle=i*2.39996+phase, d=.044*Math.sqrt(i)
    const x=cx+d*Math.cos(angle), z=cz+d*.82*Math.sin(angle)
    const r=(.019+.011*(.5+.5*Math.sin(2.56*i+phase)))*(1-.16*i/count)
    barnacle(skin,object++,x,z,r,angle)
  }
}
