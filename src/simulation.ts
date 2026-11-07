import type {
  AppState,
  ForwardModel,
  ForwardResult,
  LensOptimizer,
} from './contracts'

// Modules own resources; the caller delivers targets and submits GPU commands.
export function createSimulation(
  state: AppState,
  forward: ForwardModel,
  optimizer?: LensOptimizer,
) {
  return {
    // Deliver targets before updates; consume and submit before the next batch.
    encode(
      encoder: GPUCommandEncoder,
      steps = 0,
    ): { result: ForwardResult; done: boolean } {
      if (!Number.isSafeInteger(steps) || steps < 0) {
        throw new Error('Optimization steps must be a non-negative safe integer.')
      }

      if (steps > 0 && !optimizer) {
        throw new Error('Optimization steps require an optimizer.')
      }

      // Snapshot small CPU metadata; height data stays on the GPU.
      let lens = { ...state.lens }
      const optics = { ...state.optics }
      const receiver = { ...state.receiver }
      let done = false

      if (optimizer) {
        for (let step = 0; step < steps; step++) {
          const update = optimizer.encodeStep(encoder, { lens, optics, receiver })
          lens = update.lens
          done = update.done
          if (done) break
        }
      }

      // Render the final heights, including the update that ends a run.
      const result = forward.encode(encoder, { lens, optics, receiver })

      // Publish the current buffer handle only after encoding succeeds.
      state.lens = lens
      return { result, done }
    },
  }
}
