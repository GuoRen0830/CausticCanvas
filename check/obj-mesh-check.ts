import { parseObjMesh, readObjMesh } from '../src/geometry/obj-mesh'
import { placeObjOnReceiver } from '../src/geometry/place-obj'
import wedgeObj from './fixtures/wedge.obj?raw'
import { buildSceneGeometry, sceneVertexStride } from '../src/rendering/scene-mesh'
import { buildDesignView } from '../src/optics/design-view'
import { config, sceneLamp } from '../src/config'
import { transportHeights, transportTargets } from '../src/optimization/transport-init'
import { reference } from './optics-reference'

function expect(value: boolean, message: string) {
  if (!value) throw new Error('OBJ check: ' + message)
}

export async function checkObjMesh(report: (line: string) => void) {
  // Two triangles sharing a diagonal, with optical +Z depth increasing to the right.
  const panel = [
    'v -0.0625 0.0625 0.25', 'v 0.0625 0.0625 0.3125',
    'v 0.0625 -0.0625 0.3125', 'v -0.0625 -0.0625 0.25',
    'vt 0 0', 'vt 1 0', 'vt 1 1', 'vn 1 0 -2',
    'g Panel', 'f 1/1/1 2/2/1 3/3/1', 'usemtl matte',
    'f -4/1/-1 -2/3/-1 -1/2/-1 # same diagonal',
  ].join('\r\n')
  const model = await readObjMesh(new Blob([panel]))
  expect(model.vertices.length === 4 * sceneVertexStride && model.indices.join() === '0,1,2,0,2,3',
    'identical position/normal/UV corners should be shared across materials')
  const seam = parseObjMesh(panel + '\nf 1/2/1 3/3/1 4//1')
  expect(seam.indices.slice(-3).join() === '4,2,5' && seam.uvs?.[8] === 1 && !seam.draws?.at(-1)?.hasUV,
    'UV seams split vertices, incomplete UV triangles use the solid material')
  expect(Math.abs(model.vertices[3] - 1 / Math.sqrt(5)) < 1e-6 && model.vertices[5] < 0,
    'vertex normals must be normalized')

  const hardEdge = parseObjMesh(panel + '\nvn 0 1 0\nf 1//2 3//2 4//2')
  expect(hardEdge.indices.slice(-3).join() === '4,5,6' && hardEdge.vertices[4 * sceneVertexStride + 4] === 1,
    'the same position with a different normal must keep the hard edge')
  const grouped = parseObjMesh(panel + '\ng Back\nf 1/1 3/2 2/3\no Cap\nf 1 2 3')
  const offset = 4 * sceneVertexStride
  expect(grouped.vertices[offset + 5] > 0 && grouped.vertices[offset + 10] === 3 &&
    grouped.vertices[7 * sceneVertexStride + 10] === 4, 'flat winding normals and separate group/object IDs')

  const receiver = { sizeX: 0.25, sizeY: 0.25, z: 0.5, width: 32, height: 32 }
  const geometry = buildSceneGeometry(receiver, { mesh: model })
  expect(geometry.mesh.draws?.[2].firstIndex === 9 && geometry.mesh.uvs?.[10] === 1,
    'rebase material ranges and UVs past the receiver floor')
  expect(geometry.mesh.indices.slice(6).join() === '4,5,6,4,6,7',
    'custom scene must rebase indices past the floor')
  const view = buildDesignView(geometry.mesh, receiver)
  const terrain = buildSceneGeometry(receiver, { mesh: model, receiverFloor: false })
  const terrainView = buildDesignView(terrain.mesh, receiver)
  const emptyPixel = 2 * receiver.width + 2
  const insidePixel = 16 * receiver.width + 10
  const terrainFace = terrainView.samples[2 * insidePixel] - 1
  expect(terrain.mesh === model && terrainView.bvh.triangles[12 * terrainFace + 3] === 2 &&
    terrainView.samples[2 * emptyPixel] === 0 && terrainView.surfaceAreas[emptyPixel] === 0,
    'authored terrain supplies both the visible mesh and lamp hits, and only its footprint can be painted')
  const sample = (x: number, y: number) => {
    const i = y * receiver.width + x
    const triangle = view.samples[2 * i] - 1
    return { object: view.bvh.triangles[12 * triangle + 3], depth: new Float32Array(view.samples.buffer)[2 * i + 1] }
  }
  const left = sample(10, 16), right = sample(21, 16), outside = sample(2, 2)
  expect(view.mesh === geometry.mesh && left.object === 2 && right.object === 2 && outside.object === 1,
    'the training view must see the imported panel in front of the floor')
  expect(left.depth < right.depth && right.depth < outside.depth && outside.depth === receiver.z,
    'import must preserve the depth slope and the exposed floor')
  expect(model.indices[0] === 0 && model.vertices[9] === 0, 'scene assembly must leave the source mesh intact')

  // Offset Y-up asset: the roof rises toward +X, and its base is at Maya Y=0.
  const lens = { sizeX: 0.2, sizeY: 0.2, thickness: 0.01 }
  const wedge = parseObjMesh(wedgeObj)
  const { receiver: fitted, height } = placeObjOnReceiver(wedge, lens, receiver)
  const v = wedge.vertices
  expect(Math.abs(v[0] + 0.1) < 1e-6 && Math.abs(v[1] + 0.06) < 1e-6,
    'center and uniformly scale Maya X/Z inside the fitted receiver')
  expect(v[2] < fitted.z && fitted.z - v[2] < 1e-4 && v[5] === 1,
    'place the base just above the floor and rotate its downward normal')
  expect(fitted.width === 32 && fitted.height === 19 && Math.abs(fitted.sizeY - 0.12) < 1e-7,
    'the canvas and receiver must fit both dimensions of the model')
  const lamp = sceneLamp(fitted)
  const fraction = config.sceneApertureFraction
  const tilt = Math.atan(lamp.optimizer.maxSurfaceSlope)
  const deflection = Math.asin((lamp.optics.lensIOR / lamp.optics.outsideIOR) * Math.sin(tilt)) - tilt
  const cornerShift = 0.5 * (1 - fraction) * Math.hypot(fitted.sizeX, fitted.sizeY)
  expect(fitted.z < 0.2 && Math.atan2(cornerShift, v[2] - height - lamp.lens.thickness) < deflection,
    'fit the lamp close to the wedge while leaving enough throw to reach the raised corner')
  expect(fraction <= 0.12 && Math.abs(lamp.lens.sizeX / fitted.sizeX - fraction) < 1e-6 &&
    Math.abs(lamp.lens.sizeY / fitted.sizeY - fraction) < 1e-6 &&
    Math.abs(lamp.lens.thickness / config.lens.thickness - fraction) < 1e-6,
    'fit a physically smaller, thinner lens over the full rectangular canvas')
  const lampPower = lamp.optics.incidentIrradiance * lamp.lens.sizeX * lamp.lens.sizeY
  expect(Math.abs(lampPower - config.optics.incidentIrradiance * fitted.sizeX * fitted.sizeY) < 1e-7,
    'concentrate the lamp power into the smaller source aperture')

  // A vertical stroke beyond the flat lens footprint still receives refracted light.
  const grid = 9
  const strokeStart = 0.23 * fitted.sizeX
  const target = Float32Array.from({ length: fitted.width * fitted.height }, (_, i) => {
    const x = ((i % fitted.width + 0.5) / fitted.width - 0.5) * fitted.sizeX
    return x >= strokeStart && x < strokeStart + 0.12 * fitted.sizeX ? 1 : 0
  })
  const transport = { target, width: fitted.width, height: fitted.height,
    receiverSizeX: fitted.sizeX, receiverSizeY: fitted.sizeY, receiverZ: fitted.z,
    lensSizeX: lamp.lens.sizeX, lensSizeY: lamp.lens.sizeY, baseHeight: lamp.lens.thickness,
    columns: grid, rows: grid, eta: lamp.optics.lensIOR / lamp.optics.outsideIOR,
    maxSlope: lamp.optimizer.maxSurfaceSlope, iterations: 40 }
  const assigned = transportTargets(transport)
  const smaller = transportTargets({ ...transport, lensSizeX: lamp.lens.sizeX / 2, lensSizeY: lamp.lens.sizeY / 2 })
  expect(assigned.x.every((x, i) => Math.hypot(x - smaller.x[i], assigned.y[i] - smaller.y[i]) < 1e-8),
    'changing the source scale preserves the transport pairing in the drawing')
  const heights = transportHeights(transport)
  const { image } = reference({ receiver: fitted, optics: lamp.optics,
    lens: { ...lamp.lens, columns: grid, rows: grid } }, heights)
  const strokePower = image.reduce((sum, value, i) => sum + value * target[i], 0) *
    fitted.sizeX * fitted.sizeY / target.length
  expect(strokePower > lampPower * 0.25, 'steer light onto a painted strip outside the source footprint')
  const placed = buildDesignView(buildSceneGeometry(fitted, { mesh: wedge }).mesh, fitted)
  const depths = new Float32Array(placed.samples.buffer)
  const roofDepth = (x: number) => depths[2 * (Math.floor(fitted.height / 2) * fitted.width + x) + 1]
  expect(roofDepth(22) < roofDepth(10) && roofDepth(10) < roofDepth(1) && roofDepth(1) < fitted.z,
    'the transformed roof must be front-facing and rise to the right in the training view')
  const tall = parseObjMesh('v 0 0 0\nv 0 1000 0\nv 1 0 0\nf 1 2 3')
  const tower = placeObjOnReceiver(tall, lens, receiver)
  expect(tower.receiver.z <= receiver.z && tower.height > 0 && tower.receiver.height >= 1 &&
    tall.vertices[sceneVertexStride + 2] > lamp.lens.thickness, 'fit a tall asset with zero depth below the lens')
  const sheet = parseObjMesh('v 0 0 0\nv 0 0 1\nv 1 0 0\nf 1 2 3')
  const flat = placeObjOnReceiver(sheet, lens, receiver)
  const flatShift = 0.5 * (1 - fraction) * Math.hypot(flat.width, flat.depth)
  expect(sheet.vertices[2] === sheet.vertices[sceneVertexStride + 2] && sheet.vertices[5] === -1 &&
    Math.atan2(flatShift, sheet.vertices[2] - lamp.lens.thickness) < deflection && flat.receiver.z < fitted.z,
    'a horizontal sheet retains corner reach and needs less clearance than the elevated roof')

  const vertices = 'v 0 0 0\nv 1 0 0\nv 0 1 0\n'
  for (const [text, message] of [
    [vertices + 'f 1 2 3 1', 'OBJ line 4: Expected a triangle'],
    [vertices + 'f 1 0 3', 'OBJ line 4: Invalid position index'],
    [vertices + 'f -4 -2 -1', 'OBJ line 4: position index'],
    [vertices + 'f 1/2 2 3', 'OBJ line 4: texture coordinate index'],
    [vertices + 'f 1//2 2 3', 'OBJ line 4: normal index'],
    [vertices + 'f 1 1 3', 'OBJ line 4: Degenerate'],
    ['v 1e50 0 0', 'OBJ line 1: Invalid coordinate'],
    [vertices, 'OBJ contains no triangle faces'],
  ]) {
    let error = ''
    try { parseObjMesh(text) } catch (e) { error = String(e) }
    expect(error.includes(message), 'expected diagnostic ' + message)
  }
  report('PASS OBJ geometry, fitted optical clearance, roof/floor depths and compact-aperture steering')
}
