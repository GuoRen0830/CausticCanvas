export interface TargetImage {
  width: number
  height: number
  version: number

  // Row-major weights in [0, 1], with row 0 at the bottom.
  values: Float32Array
}