import { parseObjMesh, readObjMesh } from '../src/geometry/obj-mesh'
import { placeObjOnReceiver } from '../src/geometry/place-obj'
import wedgeObj from './fixtures/wedge.obj?raw'
import { buildSceneGeometry, sceneVertexStride } from '../src/rendering/scene-mesh'
import { buildDesignView } from '../src/optics/design-view'

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
    'f -4//-1 -2/2/-1 -1/3/-1 # same diagonal',
  ].join('\r\n')
  const model = await readObjMesh(new Blob([panel]))
  expect(model.vertices.length === 4 * sceneVertexStride && model.indices.join() === '0,1,2,0,2,3',
    'position/normal pairs should be shared across faces and UV seams')
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
  expect(!geometry.referenceBox && geometry.mesh.indices.slice(6).join() === '4,5,6,4,6,7',
    'custom scene must rebase indices past the floor and omit the analytic box')
  const view = buildDesignView(geometry.mesh, receiver)
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
  placeObjOnReceiver(wedge, lens, receiver)
  const v = wedge.vertices
  expect(Math.abs(v[0] + 0.08) < 1e-6 && Math.abs(v[1] + 0.048) < 1e-6,
    'center and uniformly scale Maya X/Z inside the aperture')
  expect(v[2] < receiver.z && receiver.z - v[2] < 1e-4 && v[5] === 1,
    'place the base just above the floor and rotate its downward normal')
  const placed = buildDesignView(buildSceneGeometry(receiver, { mesh: wedge }).mesh, receiver)
  const depths = new Float32Array(placed.samples.buffer)
  const roofDepth = (x: number) => depths[2 * (16 * receiver.width + x) + 1]
  expect(roofDepth(22) < roofDepth(10) && roofDepth(10) < roofDepth(1) && roofDepth(1) === receiver.z,
    'the transformed roof must be front-facing and rise to the right in the training view')
  const tall = parseObjMesh('v 0 0 0\nv 0 1000 0\nv 1 0 0\nf 1 2 3')
  placeObjOnReceiver(tall, lens, receiver)
  expect(tall.vertices[sceneVertexStride + 2] > lens.thickness, 'limit tall assets below the lens')
  const sheet = parseObjMesh('v 0 0 0\nv 0 0 1\nv 1 0 0\nf 1 2 3')
  placeObjOnReceiver(sheet, lens, receiver)
  expect(sheet.vertices[2] === sheet.vertices[sceneVertexStride + 2] && sheet.vertices[5] === -1,
    'a horizontal sheet with zero height retains its upward-facing normal')

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
  report('PASS OBJ indices, normals, groups, malformed input, Maya placement and imported roof/floor depths')
}
