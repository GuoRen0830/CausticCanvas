import type { ForwardInput, ForwardModel, LensOptimizer } from '../contracts'

// Prepare optimizer resources on the shared device without changing input heights.
// Capture the shared forward model for encodeStep; the caller retains ownership.
// Clean up partial allocations on failure.
export async function createLensOptimizer(
  _device: GPUDevice,
  _initial: ForwardInput,
  _forward: ForwardModel,
): Promise<LensOptimizer | null> {
  // TODO: Implement the agreed gradient and update method, then return an optimizer.
  // Null keeps the application in forward-preview mode.
  return null
}
