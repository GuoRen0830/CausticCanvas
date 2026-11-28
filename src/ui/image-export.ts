interface ExportScene {
  name: string
  aspect: number
  materialMode?: 'pbr' | 'basecolor'
}

export interface ImageExportRequest {
  width: number
  height: number
  signal: AbortSignal
}

export function getExportSize(aspect: number, longEdge: number) {
  if (!(Number.isFinite(aspect) && aspect > 0 && Number.isSafeInteger(longEdge) && longEdge > 0)) {
    throw new Error('The image dimensions must be positive.')
  }
  return aspect >= 1
    ? { width: longEdge, height: Math.max(1, Math.round(longEdge / aspect)) }
    : { width: Math.max(1, Math.round(longEdge * aspect)), height: longEdge }
}

export async function encodeExportPng(image: ImageData, signal: AbortSignal): Promise<Blob> {
  signal.throwIfAborted()
  const canvas = document.createElement('canvas')
  canvas.width = image.width
  canvas.height = image.height
  try {
    const context = canvas.getContext('2d', { colorSpace: 'srgb' })
    if (!context) throw new Error('The browser could not prepare the PNG.')
    context.putImageData(image, 0, 0)
    const png = await new Promise<Blob>((resolve, reject) => {
      canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error('The PNG could not be encoded.')), 'image/png')
    })
    signal.throwIfAborted()
    return png
  } finally {
    canvas.width = 0
    canvas.height = 0
  }
}

function filename({ name, materialMode }: ExportScene, width: number, height: number) {
  const stem = name.replace(/\.(obj|gltf|glb)$/i, '').replace(/[<>:"/\\|?*\u0000-\u001f]/g, '-').trim().slice(0, 100)
  return [stem || 'Caustic Canvas', materialMode, `${width}x${height}.png`].filter(Boolean).join('-')
}

export function createImageExport(options: {
  signal: AbortSignal
  getScene: () => ExportScene | undefined
  render: (request: ImageExportRequest) => Promise<ImageData>
}) {
  const { signal, getScene, render } = options
  const open = document.querySelector<HTMLButtonElement>('#export-image')!
  const dialog = document.querySelector<HTMLDialogElement>('#export-dialog')!
  const form = document.querySelector<HTMLFormElement>('#export-form')!
  const size = document.querySelector<HTMLSelectElement>('#export-size')!
  const submit = document.querySelector<HTMLButtonElement>('#export-submit')!
  const cancelButton = document.querySelector<HTMLButtonElement>('#export-cancel')!
  const spinner = document.querySelector<HTMLSpanElement>('#export-spinner')!
  const errorMessage = document.querySelector<HTMLParagraphElement>('#export-error')!
  let job: AbortController | undefined
  let objectUrl: string | undefined
  let releaseTimer: number | undefined

  function clearError() {
    errorMessage.textContent = ''
  }

  function cancel() {
    job?.abort()
    if (dialog.open) dialog.close()
  }

  function refresh() {
    const current = getScene()
    open.disabled = signal.aborted || !current || !!job
    size.disabled = submit.disabled = !!job
    spinner.hidden = !job
    if (!current || signal.aborted) { cancel(); return }
    if (job) return
    for (const option of size.options) {
      const { width, height } = getExportSize(current.aspect, Number(option.value))
      option.textContent = `${width} × ${height} px`
    }
  }

  function releaseDownload() {
    if (releaseTimer !== undefined) window.clearTimeout(releaseTimer)
    releaseTimer = undefined
    if (objectUrl) URL.revokeObjectURL(objectUrl)
    objectUrl = undefined
  }

  async function start() {
    const scene = getScene()
    if (job || !scene || !dialog.open || signal.aborted) return
    const { width, height } = getExportSize(scene.aspect, Number(size.value))
    const request = new AbortController()
    job = request
    clearError()
    refresh()
    // Keep a keyboard target inside the modal once the submit button is disabled.
    cancelButton.focus()
    let downloaded = false
    try {
      const image = await render({ width, height, signal: request.signal })
      request.signal.throwIfAborted()
      if (image.width !== width || image.height !== height) throw new Error('The rendered image has unexpected dimensions.')
      const blob = await encodeExportPng(image, request.signal)
      request.signal.throwIfAborted()
      releaseDownload()
      objectUrl = URL.createObjectURL(blob)
      const download = document.createElement('a')
      download.href = objectUrl
      download.download = filename(scene, width, height)
      download.click()
      // The browser may consume the URL after this event has finished.
      releaseTimer = window.setTimeout(releaseDownload, 60_000)
      downloaded = true
    } catch (error) {
      if (!request.signal.aborted) {
        errorMessage.textContent = error instanceof Error ? error.message : 'Could not export the image.'
      }
    } finally {
      job = undefined
      refresh()
      if (downloaded && dialog.open) dialog.close()
    }
  }

  open.addEventListener('click', () => {
    if (open.disabled || dialog.open) return
    clearError()
    refresh()
    if (!open.disabled) dialog.showModal()
  }, { signal })
  size.addEventListener('change', clearError, { signal })
  cancelButton.addEventListener('click', cancel, { signal })
  dialog.addEventListener('cancel', event => { event.preventDefault(); cancel() }, { signal })
  dialog.addEventListener('click', event => {
    if (event.target !== dialog || job) return
    const { left, right, top, bottom } = dialog.getBoundingClientRect()
    if (event.clientX < left || event.clientX > right || event.clientY < top || event.clientY > bottom) cancel()
  }, { signal })
  form.addEventListener('submit', event => { event.preventDefault(); void start() }, { signal })
  window.addEventListener('resize', refresh, { signal })
  signal.addEventListener('abort', () => { cancel(); releaseDownload(); open.disabled = true }, { once: true })
  refresh()
  return { refresh, cancel }
}
