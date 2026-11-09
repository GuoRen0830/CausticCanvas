import { mat4 } from 'wgpu-matrix'

interface OrbitCameraOptions {
  target: [number, number, number]
  span: number
}

export function createOrbitCamera(
  canvas: HTMLCanvasElement,
  { target, span }: OrbitCameraOptions,
  onChange: () => void,
) {
  // Start at the same oblique angle as the original preview.
  const initialAzimuth = Math.atan2(1.3, 1.8)
  const initialElevation = Math.atan2(0.9, Math.hypot(1.3, 1.8))
  const initialDistance = span * Math.hypot(1.3, 0.9, 1.8)
  const elevationLimit = Math.PI / 2 - 0.01
  let azimuth = initialAzimuth
  let elevation = initialElevation
  let distance = initialDistance
  let activePointer: number | null = null
  let previousX = 0
  let previousY = 0
  const events = new AbortController()
  const eye = new Float32Array(3)
  const view = mat4.identity()
  const projection = mat4.identity()
  const viewProjection = mat4.identity()

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
    distance = Math.max(span * 0.6, Math.min(
      span * 10,
      distance * Math.exp(delta * 0.001),
    ))
    onChange()
  }

  const reset = () => {
    releasePointer()
    azimuth = initialAzimuth
    elevation = initialElevation
    distance = initialDistance
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
    update(aspect: number) {
      // Y is up; azimuth turns around Y and elevation rises above XZ.
      const horizontalDistance = distance * Math.cos(elevation)
      eye[0] = target[0] + horizontalDistance * Math.sin(azimuth)
      eye[1] = target[1] + distance * Math.sin(elevation)
      eye[2] = target[2] + horizontalDistance * Math.cos(azimuth)
      mat4.lookAt(eye, target, [0, 1, 0], view)
      mat4.perspective(Math.PI / 4, aspect, span * 0.01, span * 20, projection)
      // Borrowed matrix, reused on the next update.
      return mat4.multiply(projection, view, viewProjection)
    },

    destroy() {
      events.abort()
      releasePointer()
    },
  }
}
