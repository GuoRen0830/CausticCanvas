import type { DesignView } from '../optics/design-view'
import type { TargetImage } from '../contracts'

const sRGBToLinear = Float32Array.from({ length: 256 }, (_, byte) => {
  const v = byte / 255
  return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4
})

export function createDrawingPad(
  canvas: HTMLCanvasElement,
  size: Pick<TargetImage, 'width' | 'height'>,
  onChange: (target: TargetImage) => void,
  view?: DesignView,
  onHistory?: (canUndo: boolean) => void,
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
    let near = Infinity, far = -Infinity
    for (let i = 0; i < width * height; i++) {
      if (view.surfaceAreas[i] === 0) continue
      near = Math.min(near, depths[2*i+1])
      far = Math.max(far, depths[2*i+1])
    }
    for (let i = 0; i < width * height; i++) {
      // Use the visible depth range so a low-profile OBJ is legible against the floor.
      const relief = far > near ? (far - depths[2*i+1]) / (far - near) : 0
      // A mid-grey guide keeps white light strokes visible over the floor.
      const shade = view.surfaceAreas[i] > 0 ? 153 - 45 * relief : 215
      image.data.set([shade, shade, shade, 255], 4 * i)
    }
    ctx.putImageData(image, 0, 0)
    canvas.style.backgroundImage = `url(${guide.toDataURL()})`
    canvas.style.backgroundSize = '100% 100%'
  }
  context.strokeStyle = context.fillStyle = '#ffffff'
  // Keep brush size proportional when the image resolution changes.
  context.lineWidth = Math.min(width, height) * 0.05
  context.lineCap = 'round'
  context.lineJoin = 'round'

  let version = 0, colorVersion = 0
  let previousPixels: ImageData | undefined
  let values = new Float32Array(width * height)
  // Committed RGBA snapshots preserve both the coverage and color across undo.
  const history: ImageData[] = []
  const historyLimit = 30
  let activePointer: number | null = null
  let previous = { x: 0, y: 0 }
  const events = new AbortController()

  const publish = (remember = true) => {
    const strokes = context.getImageData(0, 0, width, height)
    const pixels = strokes.data
    const old = previousPixels?.data
    let clipped = false, shapeChanged = false, colorChanged = false, usesColor = false

    for (let i = 0; i < values.length; i++) {
      const p = 4 * i
      if (pixels[p + 3] && view?.surfaceAreas[i] === 0) {
        pixels[p + 3] = 0
        clipped = true
      }
      const alpha = pixels[p + 3], oldAlpha = old?.[p + 3] ?? 0
      if (alpha !== oldAlpha) shapeChanged = true
      if (!alpha && !oldAlpha) continue
      for (let c = 0; c < 3; c++) {
        const channel = alpha ? pixels[p + c] : 255
        if (channel !== 255) usesColor = true
        if (channel !== (oldAlpha ? old![p + c] : 255)) colorChanged = true
      }
    }

    if (clipped) context.putImageData(strokes, 0, 0)
    if (previousPixels && !shapeChanged && !colorChanged) return
    if (remember && previousPixels) {
      history.push(previousPixels)
      if (history.length > historyLimit) history.shift()
    }
    previousPixels = strokes
    onHistory?.(history.length > 0)
    if (shapeChanged) {
      version++
      values = Float32Array.from({ length: width * height }, (_, i) => pixels[4 * i + 3] / 255)
    }
    if (colorChanged) colorVersion++
    // getImageData returns straight sRGB; the aperture filter consumes straight linear RGB.
    // Transparent pixels have neutral transmission and zero shape weight.
    let colors: Float32Array | undefined
    if (usesColor) {
      colors = new Float32Array(3 * width * height).fill(1)
      for (let i = 0; i < values.length; i++) {
        if (values[i] === 0) continue
        for (let c = 0; c < 3; c++) colors[3 * i + c] = sRGBToLinear[pixels[4 * i + c]]
      }
    }
    onChange({ width, height, version, colorVersion, values, colors })
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

  const undo = () => {
    if (activePointer !== null) return
    const snapshot = history.pop()
    if (!snapshot) return
    context.putImageData(snapshot, 0, 0)
    publish(false)
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
    setColor(color: string) {
      context.strokeStyle = context.fillStyle = color
    },
    setRecolor(enabled: boolean) {
      context.globalCompositeOperation = enabled ? 'source-atop' : 'source-over'
    },
    undo,
    // Brush diameter as a fraction of the shorter side.
    setBrush(fraction: number) {
      if (fraction > 0) context.lineWidth = Math.min(width, height) * fraction
    },
    destroy() {
      events.abort()
      releasePointer()
    },
  }
}
