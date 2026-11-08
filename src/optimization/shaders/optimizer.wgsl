// Loss gradient, height regularization, and Adam updates for the lens optimizer.

struct Params {
  // Lens vertex counts: columns, rows.
  gridSize: vec2<u32>,
  // Receiver pixel counts: width, height.
  imageSize: vec2<u32>,
  // Vertex spacing in meters: dx, dy.
  spacing: vec2<f32>,
  // Data loss: mean over pixels of ((E - T) / meanIrradiance)^2.
  invMeanIrradianceSq: f32,
  invPixelCount: f32,
  learningRate: f32,
  beta1: f32,
  beta2: f32,
  epsilon: f32,
  smoothnessWeight: f32,
  slopeWeight: f32,
  maxSlope: f32,
  minHeight: f32,
  invVertexCount: f32,
  invFaceCount: f32,
  maxIterations: u32,
  lossBlurSigma: f32,
}

struct StepState {
  // Completed Adam iterations in the current run.
  step: u32,
}

@group(0) @binding(0) var<uniform> params: Params;

// Loss bindings (group 0).
@group(0) @binding(1) var<storage, read> targetIrradiance: array<f32>;
@group(0) @binding(2) var<storage, read_write> pixelGradient: array<f32>;
@group(0) @binding(3) var<storage, read_write> lossPartials: array<f32>;

// Caustic image (group 1), one binding per output kind.
@group(1) @binding(0) var<storage, read> causticBuffer: array<f32>;
@group(1) @binding(1) var causticTexture: texture_2d<f32>;

// Update bindings (group 0 of the update layout).
@group(0) @binding(4) var<storage, read_write> heights: array<f32>;
@group(0) @binding(5) var<storage, read_write> heightGradient: array<f32>;
@group(0) @binding(6) var<storage, read_write> curvature: array<f32>;
@group(0) @binding(7) var<storage, read_write> moments: array<vec2<f32>>;
@group(0) @binding(8) var<storage, read_write> state: StepState;
@group(0) @binding(9) var<storage, read_write> filterA: array<f32>;
@group(0) @binding(10) var<storage, read_write> filterB: array<f32>;

const LOSS_WORKGROUP = 256u;
var<workgroup> partialSums: array<f32, LOSS_WORKGROUP>;

// Store the residual; log the unfiltered loss for comparable diagnostics.
fn pixelLoss(index: u32, irradiance: f32) -> f32 {
  let residual = irradiance - targetIrradiance[index];
  let scale = params.invMeanIrradianceSq * params.invPixelCount;
  pixelGradient[index] = residual;
  return residual * residual * scale;
}

fn reduceLoss(localIndex: u32, groupIndex: u32, value: f32) {
  partialSums[localIndex] = value;
  workgroupBarrier();
  for (var stride = LOSS_WORKGROUP / 2u; stride > 0u; stride = stride / 2u) {
    if (localIndex < stride) {
      partialSums[localIndex] += partialSums[localIndex + stride];
    }
    workgroupBarrier();
  }
  if (localIndex == 0u) {
    lossPartials[groupIndex] = partialSums[0];
  }
}

@compute @workgroup_size(LOSS_WORKGROUP)
fn lossFromBuffer(
  @builtin(global_invocation_id) id: vec3<u32>,
  @builtin(local_invocation_index) localIndex: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  var loss = 0.0;
  if (id.x < params.imageSize.x * params.imageSize.y) {
    loss = pixelLoss(id.x, causticBuffer[id.x]);
  }
  reduceLoss(localIndex, group.x, loss);
}

@compute @workgroup_size(LOSS_WORKGROUP)
fn lossFromTexture(
  @builtin(global_invocation_id) id: vec3<u32>,
  @builtin(local_invocation_index) localIndex: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  var loss = 0.0;
  if (id.x < params.imageSize.x * params.imageSize.y) {
    let pixel = vec2<u32>(id.x % params.imageSize.x, id.x / params.imageSize.x);
    loss = pixelLoss(id.x, textureLoad(causticTexture, pixel, 0).r);
  }
  reduceLoss(localIndex, group.x, loss);
}

// Symmetric, zero-padded Gaussian filtering: d||K(E-T)||²/dE = 2 K^T K(E-T).
fn filtered(index: u32, axis: u32, source: u32) -> f32 {
  let progress = f32(state.step) / max(1.0, 0.8 * f32(params.maxIterations));
  let sigma = params.lossBlurSigma * max(0.0, 1.0 - progress);
  let radius = i32(ceil(3.0 * sigma));
  let pixel = vec2<i32>(i32(index % params.imageSize.x), i32(index / params.imageSize.x));
  var sum = 0.0;
  var weightSum = 0.0;
  for (var offset = -radius; offset <= radius; offset++) {
    let weight = exp(-0.5 * f32(offset * offset) / max(sigma * sigma, 1e-8));
    weightSum += weight;
    var q = pixel;
    q[axis] += offset;
    if (all(q >= vec2<i32>(0)) && all(q < vec2<i32>(params.imageSize))) {
      let i = u32(q.y) * params.imageSize.x + u32(q.x);
      var value = 0.0;
      if (source == 0u) { value = pixelGradient[i]; }
      if (source == 1u) { value = filterA[i]; }
      if (source == 2u) { value = filterB[i]; }
      sum += weight * value;
    }
  }
  return sum / weightSum;
}

@compute @workgroup_size(256)
fn filterResidualX(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < params.imageSize.x * params.imageSize.y) {
    filterA[id.x] = filtered(id.x, 0u, 0u);
  }
}

@compute @workgroup_size(256)
fn filterResidualY(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < params.imageSize.x * params.imageSize.y) {
    filterB[id.x] = filtered(id.x, 1u, 1u);
  }
}

@compute @workgroup_size(256)
fn filterGradientY(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < params.imageSize.x * params.imageSize.y) {
    filterA[id.x] = filtered(id.x, 1u, 2u);
  }
}

@compute @workgroup_size(256)
fn filterGradientX(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < params.imageSize.x * params.imageSize.y) {
    pixelGradient[id.x] = 2.0 * params.invMeanIrradianceSq * params.invPixelCount * filtered(id.x, 0u, 1u);
  }
}

fn vertexIndex(column: u32, row: u32) -> u32 {
  return row * params.gridSize.x + column;
}

// Dimensionless curvature: discrete Laplacian times the mean spacing.
// Boundary vertices have no curvature term.
@compute @workgroup_size(64)
fn computeCurvature(@builtin(global_invocation_id) id: vec3<u32>) {
  let count = params.gridSize.x * params.gridSize.y;
  if (id.x >= count) {
    return;
  }
  let column = id.x % params.gridSize.x;
  let row = id.x / params.gridSize.x;
  if (
    column == 0u || row == 0u ||
    column + 1u == params.gridSize.x || row + 1u == params.gridSize.y
  ) {
    curvature[id.x] = 0.0;
    return;
  }
  let h = heights[id.x];
  let dx2 = params.spacing.x * params.spacing.x;
  let dy2 = params.spacing.y * params.spacing.y;
  let laplacian =
    (heights[id.x - 1u] + heights[id.x + 1u] - 2.0 * h) / dx2 +
    (heights[id.x - params.gridSize.x] + heights[id.x + params.gridSize.x] - 2.0 * h) / dy2;
  curvature[id.x] = laplacian * 0.5 * (params.spacing.x + params.spacing.y);
}

// dL/d(gx, gy) of the slope penalty for one face.
fn slopePenaltyGradient(gx: f32, gy: f32) -> vec2<f32> {
  let slope = sqrt(gx * gx + gy * gy);
  let excess = slope - params.maxSlope;
  if (excess <= 0.0 || slope <= 0.0) {
    return vec2<f32>(0.0);
  }
  let scale = 2.0 * params.slopeWeight * params.invFaceCount * excess / slope;
  return vec2<f32>(gx, gy) * scale;
}

// Slope-penalty gradient for the vertex at local corner (lx, ly) of a cell.
// Corners: a = (0, 0), b = (1, 0), c = (0, 1), d = (1, 1).
// Faces follow buildLensIndices: (a, b, c) and (b, d, c).
fn cellSlopeGradient(cellColumn: u32, cellRow: u32, lx: u32, ly: u32) -> f32 {
  let a = vertexIndex(cellColumn, cellRow);
  let ha = heights[a];
  let hb = heights[a + 1u];
  let hc = heights[a + params.gridSize.x];
  let hd = heights[a + params.gridSize.x + 1u];
  let invDx = 1.0 / params.spacing.x;
  let invDy = 1.0 / params.spacing.y;

  // Face (a, b, c): gx = (hb - ha) / dx, gy = (hc - ha) / dy.
  let first = slopePenaltyGradient((hb - ha) * invDx, (hc - ha) * invDy);
  // Face (b, d, c): gx = (hd - hc) / dx, gy = (hd - hb) / dy.
  let second = slopePenaltyGradient((hd - hc) * invDx, (hd - hb) * invDy);

  let corner = ly * 2u + lx;
  switch corner {
    case 0u: {
      return -first.x * invDx - first.y * invDy;
    }
    case 1u: {
      return first.x * invDx - second.y * invDy;
    }
    case 2u: {
      return first.y * invDy - second.x * invDx;
    }
    default: {
      return second.x * invDx + second.y * invDy;
    }
  }
}

// Add regularization gradients to the optical gradient written by the forward model.
@compute @workgroup_size(64)
fn addRegularization(@builtin(global_invocation_id) id: vec3<u32>) {
  let columns = params.gridSize.x;
  let rows = params.gridSize.y;
  if (id.x >= columns * rows) {
    return;
  }
  let column = id.x % columns;
  let row = id.x / columns;

  // Smoothness: L = w / N * sum(q^2), q = s * laplacian(h).
  let dx2 = params.spacing.x * params.spacing.x;
  let dy2 = params.spacing.y * params.spacing.y;
  var curvatureSum = -2.0 * curvature[id.x] * (1.0 / dx2 + 1.0 / dy2);
  if (column > 0u) { curvatureSum += curvature[id.x - 1u] / dx2; }
  if (column + 1u < columns) { curvatureSum += curvature[id.x + 1u] / dx2; }
  if (row > 0u) { curvatureSum += curvature[id.x - columns] / dy2; }
  if (row + 1u < rows) { curvatureSum += curvature[id.x + columns] / dy2; }
  let meanSpacing = 0.5 * (params.spacing.x + params.spacing.y);
  var gradient = 2.0 * params.smoothnessWeight * params.invVertexCount *
    meanSpacing * curvatureSum;

  // Slope limit: gather from the up to four cells sharing this vertex.
  for (var dy = 0u; dy < 2u; dy++) {
    for (var dx = 0u; dx < 2u; dx++) {
      if (column >= dx && row >= dy) {
        let cellColumn = column - dx;
        let cellRow = row - dy;
        if (cellColumn + 1u < columns && cellRow + 1u < rows) {
          gradient += cellSlopeGradient(cellColumn, cellRow, dx, dy);
        }
      }
    }
  }

  // Treat non-finite optical gradients as zero instead of corrupting the lens.
  let optical = heightGradient[id.x];
  let finiteOptical = select(0.0, optical, abs(optical) < 3.0e38);
  heightGradient[id.x] = finiteOptical + gradient;
}

@compute @workgroup_size(64)
fn adamUpdate(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= params.gridSize.x * params.gridSize.y) {
    return;
  }
  let t = f32(state.step + 1u);
  let g = heightGradient[id.x];
  let previous = moments[id.x];
  let m = params.beta1 * previous.x + (1.0 - params.beta1) * g;
  let v = params.beta2 * previous.y + (1.0 - params.beta2) * g * g;
  moments[id.x] = vec2<f32>(m, v);
  let mHat = m / (1.0 - pow(params.beta1, t));
  let vHat = v / (1.0 - pow(params.beta2, t));
  let remaining = max(0.0, 1.0 - t / f32(params.maxIterations));
  let rate = params.learningRate * (0.05 + 0.95 * remaining * remaining);
  let updated = heights[id.x] - rate * mHat / (sqrt(vHat) + params.epsilon);
  heights[id.x] = max(updated, params.minHeight);
}

// Runs after adamUpdate so every vertex sees the same step count.
@compute @workgroup_size(1)
fn advanceStep() {
  state.step += 1u;
}
