import type { ForwardInput, ForwardModel } from '../contracts'

// Create resources and await pipeline preparation before returning.
// Borrow the input heights; clean up partial allocations if creation fails.
export async function createForwardModel(
  _device: GPUDevice,
  _initial: ForwardInput,
): Promise<ForwardModel | null> {
  // TODO: Implement forward optics and return a ready ForwardModel.
  // Null marks this module as not implemented; it does not produce an image.
  return null
}
