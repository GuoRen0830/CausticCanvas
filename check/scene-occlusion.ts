import { mat4, vec3 } from 'wgpu-matrix'
import type { Mat4 } from 'wgpu-matrix'
import type { ReceiverPlane, TriangleBeamField } from '../src/contracts'
import { createFlatLens } from '../src/geometry/lens-mesh'
import type { BeamView } from '../src/rendering/beam-renderer'

// Two 0.02 m² triangles from a flat 0.2 m aperture. Each transmits 0.02 W.
// Face 0 travels straight down. Face 1 translates by +0.1 m in optical X
// for each meter of depth. This creates a narrow, known partial-shadow strip.
export async function checkSceneOcclusion(
  device: GPUDevice, receiver: ReceiverPlane, model: Mat4,
  draw: (field: TriangleBeamField, mode: BeamView, camera: Mat4) => Promise<Float32Array>,
  report: (message: string) => void,
) {
  const lens = createFlatLens(device, { columns: 2, rows: 2, sizeX: 0.2, sizeY: 0.2, thickness: 0.01 })
  const buffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
  const field = { lens, buffer }
  const rays = new Float32Array(8)
  rays.set([0, 0, 1, 0.02])
  rays.set(vec3.normalize([0.1, 0, 1]), 4)
  rays[7] = 0.02
  device.queue.writeBuffer(buffer, 0, rays)

  // Put the entire visible floor on one side of the box's camera silhouette.
  // For a floor point (x,y) this affine camera always gives NDC (2x/sx,-2y/sy,z).
  function camera(shear: number, centerX = 0, spanX = receiver.sizeX) {
    const clip = mat4.scaling([2 / spanX, -2 / receiver.sizeY, 1])
    clip[8] = shear * clip[0]
    clip[12] = (-centerX - shear * receiver.z) * clip[0]
    return mat4.multiply(clip, mat4.inverse(model))
  }
  const { width, height } = receiver
  function pixel(x: number, y: number, centerX = 0, spanX = receiver.sizeX) {
    return Math.floor((y / receiver.sizeY + 0.5) * height) * width
         + Math.floor(((x - centerX) / spanX + 0.5) * width)
  }
  function expect(actual: number, expected: number, label: string, tolerance = 0.003) {
    if (!(Math.abs(actual - expected) < tolerance)) throw new Error(label + ': ' + actual + ', expected ' + expected)
  }

  try {
    const view = camera(0.9)
    const floor = await draw(field, 'footprint', view)
    const blocked = await draw(field, 'blocked', view)
    let checked = 0, partial = 0, full = 0
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const px = ((x + 0.5) / width - 0.5) * receiver.sizeX
      const py = ((y + 0.5) / height - 0.5) * receiver.sizeY
      // This X range avoids the projected box itself. All these pixels see floor.
      if (px < 0.011 || px > 0.098 || py < -0.08 || py > 0.08) continue
      const shifted = px - 0.049
      if (Math.min(Math.abs(px + py), Math.abs(shifted + py)) < 0.0001) continue
      const inA = px + py < 0
      const inB = shifted + py > 0
      const boxY = py > -0.01 && py < 0.07
      const hitsA = inA && boxY && px > 0.01 && px < 0.09
      const hitsB = inB && boxY && px > 0.025 && px < 0.115
      const lost = Number(hitsA) + Number(hitsB)
      const delivered = Number(inA) + Number(inB) - lost
      expect(floor[y * width + x], delivered, 'Visible floor')
      expect(blocked[y * width + x], lost, 'Intercepted floor')
      checked++
      if (lost > 0 && !delivered) full++
    }
    if (checked < 500 || full < 10) throw new Error('Incomplete analytical shadow samples')

    // Reverse face 0's tilt so the two disjoint source triangles overlap at
    // the receiver: face 0 translates +0.049 m, face 1 travels vertically.
    rays.set(vec3.normalize([0.1, 0, 1]), 0)
    rays.set([0, 0, 1], 4)
    device.queue.writeBuffer(buffer, 0, rays)
    const overlap = await draw(field, 'footprint', view)
    const intercepted = await draw(field, 'blocked', view)
    for (let j = 0; j < height; j++) for (let i = 0; i < width; i++) {
      const x = ((i + 0.5) / width - 0.5) * receiver.sizeX
      const y = ((j + 0.5) / height - 0.5) * receiver.sizeY
      if (x < 0.012 || x > 0.024 || y < -0.008 || y > 0.068 || x+y < 0.001 || x+y > 0.048) continue
      expect(overlap[j * width + i], 1, 'Remaining oblique beam')
      expect(intercepted[j * width + i], 1, 'Blocked vertical beam')
      partial++
    }
    if (partial < 15) throw new Error('Incomplete partial-shadow samples')
    report('PASS ' + checked + ' analytical floor samples (' + full + ' fully shadowed), plus '
      + partial + ' two-beam partial-shadow samples (1 W/m² received + 1 W/m² blocked)')

    // Crop to a floor patch: the rightmost projected box x is -0.045 m,
    // outside the camera's [0.01, 0.05] interval. The physical ray is unchanged.
    const close = camera(0.9, 0.03, 0.04)
    const offscreen = await draw(field, 'blocked', close)
    expect(offscreen[pixel(0.018, 0.01, 0.03, 0.04)], 1, 'Off-camera blocker')
    const otherView = await draw(field, 'blocked', camera(-0.9))
    expect(otherView[pixel(0.018, 0.01)], 1, 'Shadow from the opposite camera')
    report('PASS shadow remains on the same floor patch from the opposite view and with the box offscreen')

    // Recover direct irradiance by subtracting a zero-flux rendering of the
    // same face. R of the albedo texture is round(0.85*255)/255 for the box.
    const left = camera(-0.4)
    const right = camera(0.4)
    const topIndex = pixel(0.05 - 0.4 * (0.25 - receiver.z), 0.02)
    const leftIndex = pixel(0.01 - 0.4 * (0.30 - receiver.z), -0.005)
    const rightIndex = pixel(0.09 + 0.4 * (0.30 - receiver.z), -0.005)
    const litLeft = await draw(field, 'lit', left)
    const litRight = await draw(field, 'lit', right)
    rays[3] = rays[7] = 0
    device.queue.writeBuffer(buffer, 0, rays)
    const baseLeft = await draw(field, 'lit', left)
    const baseRight = await draw(field, 'lit', right)
    const toIrradiance = Math.PI / (217 / 255)
    expect((litLeft[topIndex] - baseLeft[topIndex]) * toIrradiance, 1, 'Box top')
    expect((litLeft[leftIndex] - baseLeft[leftIndex]) * toIrradiance, 0.1, 'Box illuminated side')
    expect((litRight[rightIndex] - baseRight[rightIndex]) * toIrradiance, 0, 'Box back-facing side')
    report('PASS box top 1 W/m², inclined light on the side 0.1 W/m², opposite side 0 W/m²')

    // Slide the source behind the box but before the floor. Any intersection
    // beyond the emitting triangle must be ignored by the finite segment.
    device.queue.writeBuffer(lens.heightBuffer, 0, new Float32Array(4).fill(0.40))
    rays[7] = 0.02
    device.queue.writeBuffer(buffer, 0, rays)
    const afterBox = await draw(field, 'footprint', view)
    const noShadow = await draw(field, 'blocked', view)
    expect(afterBox[pixel(0.05, 0.03)], 1, 'Box behind source')
    expect(noShadow[pixel(0.05, 0.03)], 0, 'No blocker on finite segment')
    const upstreamFace = await draw(field, 'lit', left)
    expect((upstreamFace[topIndex] - baseLeft[topIndex]) * toIrradiance, 0, 'Box face upstream of source')
    report('PASS box behind the emitting triangle casts no shadow')
  } finally {
    buffer.destroy()
    lens.heightBuffer.destroy()
  }
}
