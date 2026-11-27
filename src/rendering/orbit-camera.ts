import { mat4 } from 'wgpu-matrix'

export interface OrbitCameraOptions {
  target: [number, number, number]
  size: [number, number, number]
  azimuth?: number
  elevation?: number
  // Optional overhead lamp, centered above the subject. Its top and footprint
  // influence the initial shot while the subject remains the orbit pivot.
  lampTop?: { y: number; width: number; depth: number }
}

const standardElevation = Math.PI / 4

export function createOrbitCamera(
  canvas: HTMLCanvasElement,
  initial: OrbitCameraOptions,
  onChange: () => void,
) {
  const standardAzimuth = Math.atan2(1.3, 1.8)
  const verticalFov = Math.PI / 4
  const elevationLimit = Math.PI / 2 - 0.01
  let framing = initial
  let azimuth = initial.azimuth ?? standardAzimuth
  let elevation = initial.elevation ?? standardElevation
  let zoomFactor = 1
  const poses = new WeakMap<OrbitCameraOptions, { azimuth: number; elevation: number; zoomFactor: number }>()
  let activePointer: number | null = null
  let previousX = 0
  let previousY = 0
  const events = new AbortController()
  const eye = new Float32Array(3)
  const view = mat4.identity()
  const projection = mat4.identity()
  const viewProjection = mat4.identity()

  // Fit the subject and inlet separately. A single box as wide as the whole
  // model at lamp height would make the camera back away into empty space.
  function fitDistance(aspect: number, verticalShift: number) {
    const { size, target, lampTop, elevation: initialElevation = standardElevation } = framing
    const heading = framing.azimuth ?? standardAzimuth
    const sin = Math.sin(heading), cos = Math.cos(heading)
    const up = Math.sin(initialElevation), horizontal = Math.cos(initialElevation)
    const tanY = Math.tan(verticalFov / 2)
    const edge = 1 / 1.12
    let distance = 0
    function include(x: number, y: number, z: number) {
      const along = x * sin + z * cos
      const depth = along * horizontal + y * up
      const projectedY = y * horizontal - along * up
      const halfX = Math.abs(x * cos - z * sin) / (aspect * edge)
      const halfY = Math.abs(projectedY) / (edge - Math.sign(projectedY) * verticalShift)
      distance = Math.max(distance, depth + Math.max(halfX, halfY) / tanY)
    }
    for (const x of [-size[0] / 2, size[0] / 2])
      for (const y of [-size[1] / 2, size[1] / 2])
        for (const z of [-size[2] / 2, size[2] / 2]) include(x, y, z)
    if (lampTop) {
      for (const x of [-lampTop.width / 2, lampTop.width / 2])
        for (const z of [-lampTop.depth / 2, lampTop.depth / 2]) include(x, lampTop.y - target[1], z)
    }
    return distance
  }

  const releasePointer = () => {
    const pointer = activePointer
    activePointer = null
    canvas.classList.remove('is-orbiting')
    if (pointer !== null && canvas.hasPointerCapture(pointer)) {
      canvas.releasePointerCapture(pointer)
    }
  }

  const startOrbit = (event: PointerEvent) => {
    if (activePointer !== null || !event.isPrimary || event.button !== 0) return
    event.preventDefault()
    activePointer = event.pointerId
    previousX = event.clientX
    previousY = event.clientY
    canvas.setPointerCapture(event.pointerId)
    canvas.classList.add('is-orbiting')
  }

  const moveOrbit = (event: PointerEvent) => {
    if (event.pointerId !== activePointer) return
    if ((event.buttons & 1) === 0) {
      releasePointer()
      return
    }

    // Use CSS pixels so sensitivity does not depend on display density.
    const radiansPerPixel = Math.PI / Math.max(1, canvas.clientHeight)
    azimuth = (azimuth - (event.clientX - previousX) * radiansPerPixel) % (2 * Math.PI)
    elevation = Math.max(-elevationLimit, Math.min(
      elevationLimit,
      elevation + (event.clientY - previousY) * radiansPerPixel,
    ))
    previousX = event.clientX
    previousY = event.clientY
    onChange()
  }

  const finishOrbit = (event: PointerEvent) => {
    if (event.pointerId === activePointer) releasePointer()
  }

  const zoom = (event: WheelEvent) => {
    event.preventDefault()
    // Wheel deltas may be expressed in pixels, lines, or pages.
    const unit = event.deltaMode === WheelEvent.DOM_DELTA_LINE
      ? 16
      : event.deltaMode === WheelEvent.DOM_DELTA_PAGE
        ? Math.max(1, canvas.clientHeight)
        : 1
    const delta = Math.max(-1000, Math.min(1000, event.deltaY * unit))
    zoomFactor = Math.max(0.25, Math.min(5, zoomFactor * Math.exp(delta * 0.001)))
    onChange()
  }

  const reset = () => {
    releasePointer()
    azimuth = framing.azimuth ?? standardAzimuth
    elevation = framing.elevation ?? standardElevation
    zoomFactor = 1
    onChange()
  }

  const options = { signal: events.signal }
  canvas.addEventListener('pointerdown', startOrbit, options)
  canvas.addEventListener('pointermove', moveOrbit, options)
  canvas.addEventListener('pointerup', finishOrbit, options)
  canvas.addEventListener('pointercancel', finishOrbit, options)
  canvas.addEventListener('lostpointercapture', finishOrbit, options)
  canvas.addEventListener('wheel', zoom, { ...options, passive: false })
  canvas.addEventListener('dblclick', reset, options)
  window.addEventListener('blur', releasePointer, options)

  return {
    frame(next: OrbitCameraOptions) {
      if (framing === next) return
      releasePointer()
      poses.set(framing, { azimuth, elevation, zoomFactor })
      framing = next
      const saved = poses.get(next)
      azimuth = saved?.azimuth ?? next.azimuth ?? standardAzimuth
      elevation = saved?.elevation ?? next.elevation ?? standardElevation
      zoomFactor = saved?.zoomFactor ?? 1
      onChange()
    },

    update(aspect: number) {
      // Y is up; azimuth turns around Y and elevation rises above XZ.
      const { size, target, lampTop } = framing
      const span = Math.max(...size, lampTop ? lampTop.y - target[1] : 0)
      // Give the lamp headroom above the subject through an off-axis
      // projection. The optical setup never moves the center of rotation.
      const verticalShift = lampTop ? -0.14 : 0
      const distance = fitDistance(aspect, verticalShift) * zoomFactor
      const horizontalDistance = distance * Math.cos(elevation)
      eye[0] = target[0] + horizontalDistance * Math.sin(azimuth)
      eye[1] = target[1] + distance * Math.sin(elevation)
      eye[2] = target[2] + horizontalDistance * Math.cos(azimuth)
      mat4.lookAt(eye, target, [0, 1, 0], view)
      mat4.perspective(verticalFov, aspect, span * 0.005, distance + span * 2, projection)
      projection[9] = -verticalShift
      // Borrowed matrix, reused on the next update.
      return mat4.multiply(projection, view, viewProjection)
    },

    destroy() {
      events.abort()
      releasePointer()
    },
  }
}
