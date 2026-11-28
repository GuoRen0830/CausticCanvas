import assert from 'node:assert/strict'
import { test } from 'node:test'
import { vec4 } from 'wgpu-matrix'
import { createFrameLoop } from '../src/frame-loop.ts'
import { createOrbitCamera } from '../src/rendering/orbit-camera.ts'
import { roomViewport } from '../src/rendering/room-viewport.ts'
import { getExportSize } from '../src/ui/image-export.ts'

test('export frames the Room area on desktop and mobile and retains the artist’s saved Room orbit', t => {
  const layouts = [
    [[1440, 900, 378, 74, 61], { left: 378, top: 61, width: 1062, height: 839 }],
    [[390, 844, 390, 523, 60], { left: 0, top: 60, width: 390, height: 463 }],
  ]
  const canvas = Object.assign(new EventTarget(), {
    clientHeight: 844, classList: { add() {}, remove() {} },
    setPointerCapture() {}, hasPointerCapture() { return false },
  })
  globalThis.window = new EventTarget()
  t.after(() => { delete globalThis.window })
  const room = { target: [0, 0, 0], size: [2, 0.32, 1.2], lampTop: { y: 0.8, width: 0.2, depth: 0.2 } }
  const camera = createOrbitCamera(canvas, room, () => {})
  t.after(() => camera.destroy())
  const initial = camera.update(1.2).slice()
  function drag(type, x, y) {
    canvas.dispatchEvent(Object.assign(new Event(type), { pointerId: 1, isPrimary: true, button: 0,
      buttons: 1, clientX: x, clientY: y }))
  }
  drag('pointerdown', 60, 50)
  drag('pointermove', 148, 80)
  drag('pointerup', 148, 80)
  assert.notDeepEqual(camera.update(1.2), initial)
  const landmark = [0.6, 0.1, 0.4, 1]
  function point(aspect, framing) {
    const clip = vec4.transformMat4(landmark, camera.update(aspect, framing))
    return [0.5 + 0.5 * clip[0] / clip[3], 0.5 - 0.5 * clip[1] / clip[3]]
  }
  for (const [params, expected] of layouts) {
    camera.frame(room)
    const area = roomViewport(...params)
    assert.deepEqual(area, expected)
    const aspect = area.width / area.height
    const preview = point(aspect, room)
    camera.frame({ target: [1,2,3], size: [8,9,10] })
    assert.deepEqual(point(aspect, room), preview)
    for (const edge of [1024, 2048, 4096]) {
      const { width, height } = getExportSize(aspect, edge)
      assert.equal(Math.max(width, height), edge)
      const image = point(width / height, room)
      assert.ok(Math.abs(image[0] - preview[0]) * width < 0.5)
      assert.ok(Math.abs(image[1] - preview[1]) * height < 0.5)
    }
  }
})

test('export pause preserves queued changes and training across cancel, completion and disposal', async t => {
  const frames = new Map()
  let sequence = 0
  globalThis.requestAnimationFrame = callback => { frames.set(++sequence, callback); return sequence }
  globalThis.cancelAnimationFrame = id => frames.delete(id)
  t.after(() => { delete globalThis.requestAnimationFrame; delete globalThis.cancelAnimationFrame })
  let gpuComplete
  const device = { queue: { submit() {}, onSubmittedWorkDone: () => new Promise(done => { gpuComplete = done }) } }
  const steps = []
  const loop = createFrameLoop(device, count => { steps.push(count); return {} }, {
    maxStepsPerFrame: 3, onError: assert.fail,
  })
  const tick = () => {
    assert.equal(frames.size, 1)
    const [[id, callback]] = frames
    frames.delete(id)
    callback()
  }
  loop.setRunning(true)
  loop.setPaused(true)
  assert.equal(frames.size, 0)
  loop.requestFrame()
  loop.requestFrame()
  assert.equal(frames.size, 0)
  loop.setPaused(false)
  tick()
  assert.deepEqual(steps, [3])

  // A submitted training batch can finish while an export is active.
  loop.setPaused(true)
  gpuComplete()
  await Promise.resolve()
  assert.equal(frames.size, 0)
  loop.setPaused(false)
  tick()
  assert.deepEqual(steps, [3, 3])

  // Cancel while another batch is in flight; its fence must wake exactly
  // one repainted frame, retaining any edit to the running flag.
  loop.setPaused(true)
  loop.setRunning(false)
  loop.requestFrame()
  loop.setPaused(false)
  assert.equal(frames.size, 0)
  gpuComplete()
  await Promise.resolve()
  tick()
  assert.deepEqual(steps, [3, 3, 0])
  loop.setPaused(true)
  loop.destroy()
  gpuComplete()
  await Promise.resolve()
  loop.setPaused(false)
  loop.requestFrame()
  assert.equal(frames.size, 0)
})
