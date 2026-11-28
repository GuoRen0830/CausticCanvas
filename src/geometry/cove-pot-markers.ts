import { HarborMesh } from './cove-quayside'
import { shoreHeight } from './cove-shore'

type Point = [number, number, number]
type Ring = [height: number, radius: number, finish: string]
const red = 'vermilion buoy paint', cream = 'ivory buoy paint', cork = 'oiled cork'

// Rings share a tilted axis. The larger pear-shaped float sits upright, while
// the smaller corks are threaded lengthwise onto the floating ground line.
function lathe(mesh: HarborMesh, center: Point, axis: Point, rings: Ring[], sides: number) {
  const length = Math.hypot(...axis), [a,b,c] = axis.map(v => v / length)
  const h = Math.hypot(a,c)
  const u: Point = h > 0 ? [c / h, 0, -a / h] : [1,0,0]
  const v: Point = [b*u[2], c*u[0]-a*u[2], -b*u[0]]
  const sections = rings.map(([t,r]) => Array.from({ length: sides }, (_, j): Point => {
    const theta = 2*Math.PI*j/sides, x = r*Math.cos(theta), y = r*Math.sin(theta)
    return [center[0]+t*a+x*u[0]+y*v[0], center[1]+t*b+x*u[1]+y*v[1],
      center[2]+t*c+x*u[2]+y*v[2]]
  }))
  mesh.face(rings[0][2], sections[0], [-a,-b,-c])
  for (let i = 1; i < sections.length; i++) {
    const bottom = sections[i-1], top = sections[i], t = rings[i][0]
    for (let j = 0; j < sides; j++) {
      const k = (j+1)%sides
      mesh.face(rings[i][2], [bottom[j], bottom[k], top[k], top[j]], [
        top[j][0]-center[0]-t*a, top[j][1]-center[1]-t*b, top[j][2]-center[2]-t*c])
    }
  }
  mesh.face(rings.at(-1)![2], sections.at(-1)!, [a,b,c])
}

export function makeCovePotMarkers() {
  const mesh = new HarborMesh()
  const pear: Ring[] = [[-0.16,0.075,red],[-0.10,0.166,red],[0.02,0.214,red],
    [0.095,0.208,red],[0.165,0.185,cream],[0.23,0.148,red],[0.33,0.075,red],
    [0.374,0.045,cream],[0.414,0.044,cream]]
  lathe(mesh, [-2.42,0,3.19], [0.14,1,-0.09], pear, 20)
  mesh.beam('hemp rope', [-2.38,0.39,3.15], [-2.37,0.464,3.13], 0.018,0.018)
  mesh.beam('hemp rope', [-2.37,0.464,3.13], [-2.33,0.40,3.15], 0.018,0.018)

  const line: Point[] = [[-2.42,0.015,3.20],[-2.71,0.012,3.27],[-2.93,0.015,3.34],
    [-3.16,0.016,3.36],[-3.33,0.013,3.35],[-3.58,0.015,3.31],[-3.79,-0.013,3.26]]
  for (let i = 1; i < line.length; i++) mesh.beam('hemp rope', line[i-1], line[i], 0.015,0.015)
  for (const [i, size] of [[2,1],[5,0.77]]) {
    ++mesh.object
    const p = line[i], prev = line[i-1], axis: Point = [p[0]-prev[0],0,p[2]-prev[2]]
    const rings: Ring[] = [[-0.115,0.035,cork],[-0.08,0.078,cork],[-0.035,0.085,cream],
      [0.055,0.083,cork],[0.096,0.057,cork],[0.11,0.034,cork]]
    lathe(mesh, p, axis, rings.map(([t,r,m]) => [t*size,r*size,m]), 12)
  }

  // Spare gear hangs on the free end of the net rack's ridge.
  ++mesh.object
  const ridge = Math.max(shoreHeight(-3.06,-2.89), shoreHeight(-0.40,-2.89)) + 1.17
  const hang: Point = [-3.17,ridge-0.45,-2.815]
  lathe(mesh, hang, [0,1,0], pear.map(([t,r,m]) => [t*0.68,r*0.68,m]), 14)
  mesh.beam('hemp rope', [-3.17,ridge-0.17,-2.815], [-3.17,ridge+0.031,-2.89], 0.016,0.016)
  const result = mesh.finish()
  result.materials.set(red, { color: [0.59,0.091,0.040], roughness: 0.36 })
  result.materials.set(cream, { color: [0.82,0.70,0.48], roughness: 0.45 })
  result.materials.set(cork, { color: [0.43,0.237,0.099], roughness: 0.67 })
  return result
}
