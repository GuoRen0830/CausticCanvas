import type { TargetImage } from '../contracts/target'

export function createDrawingPad(
  canvas: HTMLCanvasElement,
  onChange: (target: TargetImage) => void,
) {
  const context = canvas.getContext('2d', { willReadFrequently: true })
  if (!context) throw new Error('Failed to create drawing context.')

  // Keep target resolution independent of CSS size and display density.
  canvas.width = 256
  canvas.height = 256
  context.strokeStyle = '#fff'
  context.fillStyle = '#fff'
  context.lineWidth = 12
  context.lineCap = 'round'
  context.lineJoin = 'round'

  let version = 0
  let activePointer: number | null = null
  let previous = { x: 0, y: 0 }
  const events = new AbortController()

  const publish = () => {
    const { width, height } = canvas
    const pixels = context.getImageData(0, 0, width, height).data
    const values = new Float32Array(width * height)

    // White strokes use alpha as their weight. Flip rows for optical coordinates.
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const source = 4 * (y * width + x) + 3
        const destination = (height - 1 - y) * width + x
        values[destination] = pixels[source] / 255
      }
    }

    // Publish a new snapshot after each completed stroke or clear.
    onChange({ width, height, version: ++version, values })
  }

  const pointFromEvent = (event: PointerEvent) => {
    const rect = canvas.getBoundingClientRect()
    return {
      x: ((event.clientX - rect.left) / rect.width) * canvas.width,
      y: ((event.clientY - rect.top) / rect.height) * canvas.height,
    }
  }

  const releasePointer = () => {
    const pointer = activePointer
    activePointer = null
    if (pointer !== null && canvas.hasPointerCapture(pointer)) {
      canvas.releasePointerCapture(pointer)
    }
  }

  const startStroke = (event: PointerEvent) => {
    if (activePointer !== null || !event.isPrimary || event.button !== 0) return
    activePointer = event.pointerId
    previous = pointFromEvent(event)

    // Keep receiving events when the pointer leaves the canvas.
    canvas.setPointerCapture(event.pointerId)
    context.beginPath()
    context.arc(previous.x, previous.y, context.lineWidth / 2, 0, Math.PI * 2)
    context.fill()
  }

  const moveStroke = (event: PointerEvent) => {
    if (event.pointerId !== activePointer) return
    const point = pointFromEvent(event)
    context.beginPath()
    context.moveTo(previous.x, previous.y)
    context.lineTo(point.x, point.y)
    context.stroke()
    previous = point
  }

  const finishStroke = (event: PointerEvent) => {
    if (event.pointerId !== activePointer) return
    if (event.type === 'pointerup') moveStroke(event)
    releasePointer()
    publish()
  }

  const clear = () => {
    releasePointer()
    context.clearRect(0, 0, canvas.width, canvas.height)
    publish()
  }

  const options = { signal: events.signal }
  canvas.addEventListener('pointerdown', startStroke, options)
  canvas.addEventListener('pointermove', moveStroke, options)
  canvas.addEventListener('pointerup', finishStroke, options)
  canvas.addEventListener('pointercancel', finishStroke, options)
  canvas.addEventListener('lostpointercapture', finishStroke, options)
  clear()

  return {
    clear,
    destroy() {
      events.abort()
      releasePointer()
    },
  }
}