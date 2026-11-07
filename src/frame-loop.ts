interface FrameLoopOptions {
  maxStepsPerFrame: number
  onError(error: unknown): void
}

export function createFrameLoop(
  device: GPUDevice,
  encodeFrame: (steps: number) => GPUCommandBuffer,
  { maxStepsPerFrame, onError }: FrameLoopOptions,
) {
  if (!Number.isSafeInteger(maxStepsPerFrame) || maxStepsPerFrame < 1) {
    throw new Error('Frame step limit must be a positive safe integer.')
  }

  let frameId: number | null = null
  let gpuBusy = false
  let dirty = false
  let running = false
  let disposed = false

  function schedule() {
    if (disposed || gpuBusy || frameId !== null || (!dirty && !running)) return
    frameId = requestAnimationFrame(tick)
  }

  function destroy() {
    disposed = true
    running = false
    dirty = false
    if (frameId !== null) cancelAnimationFrame(frameId)
    frameId = null
  }

  function fail(error: unknown) {
    if (disposed) return
    destroy()
    onError(error)
  }

  function tick() {
    frameId = null
    if (disposed || (!dirty && !running)) return

    dirty = false
    gpuBusy = true

    try {
      // One synchronous batch per animation frame; paused frames get zero steps.
      const commands = encodeFrame(running ? maxStepsPerFrame : 0)
      device.queue.submit([commands])

      // Bound queued GPU work without blocking input events or reading results back.
      void device.queue.onSubmittedWorkDone().then(
        () => {
          gpuBusy = false
          schedule()
        },
        fail,
      )
    } catch (error) {
      fail(error)
    }
  }

  return {
    requestFrame() {
      if (disposed) return
      // Coalesce changes while a frame is scheduled or the GPU is busy.
      dirty = true
      schedule()
    },

    setRunning(value: boolean) {
      if (disposed) return
      running = value
      schedule()
    },

    destroy,
  }
}
