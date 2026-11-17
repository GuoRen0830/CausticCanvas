import type { DesignView } from '../optics/design-view'
import type { TargetImage } from '../contracts'

export function createDrawingPad(
  canvas: HTMLCanvasElement,
  size: Pick<TargetImage, 'width' | 'height'>,
  onChange: (target: TargetImage) => void,
  view?: DesignView,
) {
  const { width, height } = size
  if (
    !Number.isInteger(width) || !Number.isInteger(height) ||
    width < 1 || height < 1
  ) {
    throw new Error('Drawing dimensions must be positive integers.')
  }

  const context = canvas.getContext('2d', { willReadFrequently: true })
  if (!context) throw new Error('Failed to create drawing context.')

  // Keep target resolution independent of CSS size and display density.
  canvas.width = width
  canvas.height = height
  if (view) {
    if (view.width !== width || view.height !== height) throw new Error('Drawing size must match the design view.')
    // CSS backing never enters the brush alpha channel used for the loss.
    const guide = document.createElement('canvas')
    guide.width = width
    guide.height = height
    const ctx = guide.getContext('2d')!
    const image = ctx.createImageData(width, height)
    const depths = new Float32Array(view.samples.buffer)
    let far = 0
    for (let i = 0; i < width * height; i++) far = Math.max(far, depths[2*i+1])
    for (let i = 0; i < width * height; i++) {
      const shade = view.surfaceAreas[i] > 0 ? 22 + 60 * (1 - depths[2*i+1] / far) : 2
      image.data.set([shade, shade, shade, 255], 4 * i)
    }
    ctx.putImageData(image, 0, 0)
    canvas.style.backgroundImage = `url(${guide.toDataURL()})`
    canvas.style.backgroundSize = '100% 100%'
  }
  context.strokeStyle = '#fff'
  context.fillStyle = '#fff'
  // Keep brush size proportional when the image resolution changes.
  context.lineWidth = Math.min(width, height) * 0.05
  context.lineCap = 'round'
  context.lineJoin = 'round'

  let version = 0
  let activePointer: number | null = null
  let previous = { x: 0, y: 0 }
  const events = new AbortController()

  const publish = () => {
    const { width, height } = canvas
    const strokes = context.getImageData(0, 0, width, height)
    const pixels = strokes.data
    let clipped = false
    const values = new Float32Array(width * height)

    // Looking along +Z with +X to the right puts optical +Y down the screen.
    // Guide, brush mask and solver target therefore share the same row order.
    for (let i = 0; i < values.length; i++) {
      const alpha = 4 * i + 3
      if (pixels[alpha] && view?.surfaceAreas[i] === 0) {
        pixels[alpha] = 0
        clipped = true
      }
      values[i] = pixels[alpha] / 255
    }

    if (clipped) context.putImageData(strokes, 0, 0)

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
