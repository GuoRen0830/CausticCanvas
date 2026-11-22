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
  // Fraction of maxIterations over which lossBlurSigma anneals to zero.
  lossBlurAnnealFraction: f32,
  // Final Adam step size as a fraction of learningRate.
  learningRateFloor: f32,
  // Length that makes the Laplacian dimensionless (defaults to mean spacing).
  curvatureScale: f32,
  // Gaussian width in meters for the optical gradient blur.
  gradientSigma: f32,
  // Final loss-filter width in pixels.
  lossBlurFinalSigma: f32,
  // Gaussian width in meters for the Adam update blur (0 = unsmoothed update).
  updateSigma: f32,
  // Slope difference that stops the gradient blur at a crease (0 = plain Gaussian).
  creaseSlope: f32,
  // Vertex motion: Adam step in vertex spacings (0 = fixed grid), weight of the
  // x/y Laplacian, smallest face area as a fraction of its grid area, and the
  // x/y gradient blur width in meters.
  xyLearningRate: f32,
  xySmoothnessWeight: f32,
  minAreaRatio: f32,
  xySigma: f32,
  // 1: a moving vertex stays on the current surface (its height follows the
  // local slope), so x/y motion only redistributes faces; 0: heights stay put.
  xyTangential: f32,
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
// Spill-penalty gradient per pixel (distance to the drawing), added unfiltered.
@group(0) @binding(11) var<storage, read> spillGradient: array<f32>;

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

// Vertex-motion bindings: x/y displacement from the grid in meters, its
// gradient, Adam moments (m.xy, v.xy), a proposed displacement (xy) with the
// surface slope at the vertex (zw), and scratch.
@group(0) @binding(12) var<storage, read_write> offsets: array<vec2<f32>>;
@group(0) @binding(13) var<storage, read_write> offsetGradient: array<vec2<f32>>;
@group(0) @binding(14) var<storage, read_write> offsetMoments: array<vec4<f32>>;
@group(0) @binding(15) var<storage, read_write> proposal: array<vec4<f32>>;
@group(0) @binding(16) var<storage, read_write> scratch: array<vec2<f32>>;

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
  if (localIndex == 0u && groupIndex < arrayLength(&lossPartials)) {
    lossPartials[groupIndex] = partialSums[0];
  }
}

@compute @workgroup_size(LOSS_WORKGROUP)
fn lossFromBuffer(
  @builtin(global_invocation_id) id: vec3<u32>,
  @builtin(local_invocation_index) localIndex: u32,
  @builtin(num_workgroups) groups: vec3u,
) {
  let index = id.x + id.y * groups.x * LOSS_WORKGROUP;
  var loss = 0.0;
  if (index < params.imageSize.x * params.imageSize.y) {
    loss = pixelLoss(index, causticBuffer[index]);
  }
  reduceLoss(localIndex, index / LOSS_WORKGROUP, loss);
}

@compute @workgroup_size(LOSS_WORKGROUP)
fn lossFromTexture(
  @builtin(global_invocation_id) id: vec3<u32>,
  @builtin(local_invocation_index) localIndex: u32,
  @builtin(num_workgroups) groups: vec3u,
) {
  let index = id.x + id.y * groups.x * LOSS_WORKGROUP;
  var loss = 0.0;
  if (index < params.imageSize.x * params.imageSize.y) {
    let pixel = vec2<u32>(index % params.imageSize.x, index / params.imageSize.x);
    loss = pixelLoss(index, textureLoad(causticTexture, pixel, 0).r);
  }
  reduceLoss(localIndex, index / LOSS_WORKGROUP, loss);
}

// Symmetric, zero-padded Gaussian filtering: d||K(E-T)||²/dE = 2 K^T K(E-T).
fn filtered(index: u32, axis: u32, source: u32) -> f32 {
  let progress = f32(state.step) / max(1.0, params.lossBlurAnnealFraction * f32(params.maxIterations));
  let sigma = params.lossBlurFinalSigma +
    (params.lossBlurSigma - params.lossBlurFinalSigma) * max(0.0, 1.0 - progress);
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

@compute @workgroup_size(LOSS_WORKGROUP)
fn filterResidualX(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let index = id.x + id.y * groups.x * LOSS_WORKGROUP;
  if (index < params.imageSize.x * params.imageSize.y) {
    filterA[index] = filtered(index, 0u, 0u);
  }
}

@compute @workgroup_size(LOSS_WORKGROUP)
fn filterResidualY(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let index = id.x + id.y * groups.x * LOSS_WORKGROUP;
  if (index < params.imageSize.x * params.imageSize.y) {
    filterB[index] = filtered(index, 1u, 1u);
  }
}

@compute @workgroup_size(LOSS_WORKGROUP)
fn filterGradientY(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let index = id.x + id.y * groups.x * LOSS_WORKGROUP;
  if (index < params.imageSize.x * params.imageSize.y) {
    filterA[index] = filtered(index, 1u, 2u);
  }
}

@compute @workgroup_size(LOSS_WORKGROUP)
fn filterGradientX(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let index = id.x + id.y * groups.x * LOSS_WORKGROUP;
  if (index < params.imageSize.x * params.imageSize.y) {
    pixelGradient[index] = 2.0 * params.invMeanIrradianceSq * params.invPixelCount * filtered(index, 0u, 1u) +
      spillGradient[index];
  }
}

fn vertexIndex(column: u32, row: u32) -> u32 {
  return row * params.gridSize.x + column;
}

fn finiteGradient(value: f32) -> f32 {
  return select(0.0, value, abs(value) < 3.0e38);
}

// Separable Gaussian blur of the optical gradient with width gradientSigma
// meters, renormalized at the lens boundary (no flux leaves the aperture).
// Runs before computeCurvature, so the curvature buffer is free scratch.
fn smoothedGradient(vertex: u32, axis: u32, source: u32) -> f32 {
  return gaussianAlong(vertex, axis, source, params.gradientSigma, params.creaseSlope);
}

// Surface slope at a grid point from central differences (one-sided at the border).
fn vertexSlope(q: vec2<i32>) -> vec2<f32> {
  let size = vec2<i32>(params.gridSize);
  let xm = max(q.x - 1, 0);
  let xp = min(q.x + 1, size.x - 1);
  let ym = max(q.y - 1, 0);
  let yp = min(q.y + 1, size.y - 1);
  let w = params.gridSize.x;
  let gx = (heights[u32(q.y) * w + u32(xp)] - heights[u32(q.y) * w + u32(xm)]) / (f32(xp - xm) * params.spacing.x);
  let gy = (heights[u32(yp) * w + u32(q.x)] - heights[u32(ym) * w + u32(q.x)]) / (f32(yp - ym) * params.spacing.y);
  return vec2<f32>(gx, gy);
}

// Gaussian along one axis. With creaseSlope > 0, each neighbor's weight also
// falls off with its slope difference from this vertex (a bilateral filter),
// so the blur does not cross creases.
fn gaussianAlong(vertex: u32, axis: u32, source: u32, width: f32, creaseSlope: f32) -> f32 {
  let coordinate = vec2<i32>(i32(vertex % params.gridSize.x), i32(vertex / params.gridSize.x));
  let sigma = width / params.spacing[axis];
  let radius = i32(ceil(3.0 * sigma));
  let size = vec2<i32>(params.gridSize);
  var center = vec2<f32>(0.0);
  if (creaseSlope > 0.0) { center = vertexSlope(coordinate); }
  var sum = 0.0;
  var weightSum = 0.0;
  for (var offset = -radius; offset <= radius; offset++) {
    var q = coordinate;
    q[axis] += offset;
    if (q[axis] < 0 || q[axis] >= size[axis]) { continue; }
    var weight = exp(-0.5 * f32(offset * offset) / max(sigma * sigma, 1e-8));
    if (creaseSlope > 0.0) {
      let difference = (vertexSlope(q) - center) / creaseSlope;
      weight *= exp(-dot(difference, difference));
    }
    let i = u32(q.y) * params.gridSize.x + u32(q.x);
    var value = 0.0;
    if (source == 0u) { value = heightGradient[i]; } else { value = curvature[i]; }
    sum += weight * finiteGradient(value);
    weightSum += weight;
  }
  return sum / weightSum;
}

@compute @workgroup_size(64)
fn smoothGradientX(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let vertex = id.x + id.y * groups.x * 64u;
  if (vertex >= params.gridSize.x * params.gridSize.y) {
    return;
  }
  curvature[vertex] = finiteGradient(smoothedGradient(vertex, 0u, 0u));
}

@compute @workgroup_size(64)
fn smoothGradientY(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let vertex = id.x + id.y * groups.x * 64u;
  if (vertex >= params.gridSize.x * params.gridSize.y) {
    return;
  }
  heightGradient[vertex] = finiteGradient(smoothedGradient(vertex, 1u, 1u));
}

// Dimensionless curvature: discrete Laplacian times curvatureScale.
// Boundary vertices have no curvature term. With vertex motion the grid is no
// longer uniform, so the part of the Laplacian that a plane through the vertex
// would produce (slope . Laplacian of the positions) is removed: a plane costs
// nothing however its vertices are spaced, while a crease still costs the same
// per vertex, so gathering vertices at a crease lets it be physically sharper.
// On the uniform grid the correction is zero. Its slope is treated as fixed in
// the gradient.
@compute @workgroup_size(64)
fn computeCurvature(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let vertex = id.x + id.y * groups.x * 64u;
  let count = params.gridSize.x * params.gridSize.y;
  if (vertex >= count) {
    return;
  }
  let column = vertex % params.gridSize.x;
  let row = vertex / params.gridSize.x;
  if (
    column == 0u || row == 0u ||
    column + 1u == params.gridSize.x || row + 1u == params.gridSize.y
  ) {
    curvature[vertex] = 0.0;
    return;
  }
  let h = heights[vertex];
  let dx2 = params.spacing.x * params.spacing.x;
  let dy2 = params.spacing.y * params.spacing.y;
  let w = params.gridSize.x;
  var laplacian =
    (heights[vertex - 1u] + heights[vertex + 1u] - 2.0 * h) / dx2 +
    (heights[vertex - w] + heights[vertex + w] - 2.0 * h) / dy2;
  let spread = (offsets[vertex - 1u] + offsets[vertex + 1u] - 2.0 * offsets[vertex]) / dx2 +
    (offsets[vertex - w] + offsets[vertex + w] - 2.0 * offsets[vertex]) / dy2;
  if (any(spread != vec2<f32>(0.0))) {
    laplacian -= dot(displacedSlope(vertex), spread);
  }
  curvature[vertex] = laplacian * params.curvatureScale;
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

// Vertex position in the lens plane, relative to the first vertex, including
// its x/y displacement.
fn position(vertex: u32) -> vec2<f32> {
  let grid = vec2<f32>(f32(vertex % params.gridSize.x), f32(vertex / params.gridSize.x));
  return grid * params.spacing + offsets[vertex];
}

// Slope-penalty gradient with respect to the height of corner j of a face.
// The face's height gradient g = sum_i h_i * grad(phi_i), with the linear
// basis functions phi_i of the displaced triangle.
fn faceSlopeGradient(ids: vec3<u32>, j: u32) -> f32 {
  let a = position(ids.x);
  let b = position(ids.y);
  let c = position(ids.z);
  let doubleArea = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
  if (!(doubleArea > 0.0)) {
    return 0.0;
  }
  // grad(phi_i) = perpendicular of the opposite edge / doubleArea.
  var basis = array<vec2<f32>, 3>(
    vec2<f32>(b.y - c.y, c.x - b.x),
    vec2<f32>(c.y - a.y, a.x - c.x),
    vec2<f32>(a.y - b.y, b.x - a.x),
  );
  let g = (heights[ids.x] * basis[0] + heights[ids.y] * basis[1] + heights[ids.z] * basis[2]) / doubleArea;
  return dot(slopePenaltyGradient(g.x, g.y), basis[j] / doubleArea);
}

// Faces follow buildLensIndices: (a, b, c) and (b, d, c) per cell.
fn cellFaces(cellColumn: u32, cellRow: u32) -> array<vec3<u32>, 2> {
  let a = vertexIndex(cellColumn, cellRow);
  let w = params.gridSize.x;
  return array<vec3<u32>, 2>(vec3<u32>(a, a + 1u, a + w), vec3<u32>(a + 1u, a + w + 1u, a + w));
}

// Add regularization gradients to the optical gradient written by the forward model.
@compute @workgroup_size(64)
fn addRegularization(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let vertex = id.x + id.y * groups.x * 64u;
  let columns = params.gridSize.x;
  let rows = params.gridSize.y;
  if (vertex >= columns * rows) {
    return;
  }
  let column = vertex % columns;
  let row = vertex / columns;

  // Smoothness: L = w / N * sum(q^2), q = s * laplacian(h).
  let dx2 = params.spacing.x * params.spacing.x;
  let dy2 = params.spacing.y * params.spacing.y;
  var curvatureSum = -2.0 * curvature[vertex] * (1.0 / dx2 + 1.0 / dy2);
  if (column > 0u) { curvatureSum += curvature[vertex - 1u] / dx2; }
  if (column + 1u < columns) { curvatureSum += curvature[vertex + 1u] / dx2; }
  if (row > 0u) { curvatureSum += curvature[vertex - columns] / dy2; }
  if (row + 1u < rows) { curvatureSum += curvature[vertex + columns] / dy2; }
  var gradient = 2.0 * params.smoothnessWeight * params.invVertexCount *
    params.curvatureScale * curvatureSum;

  // Slope limit: gather from the faces of the up to four cells sharing this vertex.
  for (var dy = 0u; dy < 2u; dy++) {
    for (var dx = 0u; dx < 2u; dx++) {
      if (column >= dx && row >= dy) {
        let cellColumn = column - dx;
        let cellRow = row - dy;
        if (cellColumn + 1u < columns && cellRow + 1u < rows) {
          var faces = cellFaces(cellColumn, cellRow);
          for (var f = 0u; f < 2u; f++) {
            for (var j = 0u; j < 3u; j++) {
              if (faces[f][j] == vertex) { gradient += faceSlopeGradient(faces[f], j); }
            }
          }
        }
      }
    }
  }

  // Treat non-finite optical gradients as zero instead of corrupting the lens.
  let optical = heightGradient[vertex];
  let finiteOptical = select(0.0, optical, abs(optical) < 3.0e38);
  heightGradient[vertex] = finiteOptical + gradient;
}

fn adamDelta(vertex: u32) -> f32 {
  let t = f32(state.step + 1u);
  let g = heightGradient[vertex];
  let previous = moments[vertex];
  let m = params.beta1 * previous.x + (1.0 - params.beta1) * g;
  let v = params.beta2 * previous.y + (1.0 - params.beta2) * g * g;
  moments[vertex] = vec2<f32>(m, v);
  let mHat = m / (1.0 - pow(params.beta1, t));
  let vHat = v / (1.0 - pow(params.beta2, t));
  let remaining = max(0.0, 1.0 - t / f32(params.maxIterations));
  let rateFloor = params.learningRateFloor;
  let rate = params.learningRate * (rateFloor + (1.0 - rateFloor) * remaining * remaining);
  return rate * mHat / (sqrt(vHat) + params.epsilon);
}

@compute @workgroup_size(64)
fn adamUpdate(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let vertex = id.x + id.y * groups.x * 64u;
  if (vertex >= params.gridSize.x * params.gridSize.y) {
    return;
  }
  heights[vertex] = max(heights[vertex] - adamDelta(vertex), params.minHeight);
}

// Smoothed-update variant: Adam's per-vertex normalization can turn a smooth
// gradient back into vertex-scale ripples, so blur the step itself instead.
// The curvature buffer is free after addRegularization and heightGradient is
// free after adamStep, so they hold the step between the three passes.
@compute @workgroup_size(64)
fn adamStep(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let vertex = id.x + id.y * groups.x * 64u;
  if (vertex >= params.gridSize.x * params.gridSize.y) {
    return;
  }
  curvature[vertex] = finiteGradient(adamDelta(vertex));
}

@compute @workgroup_size(64)
fn smoothUpdateX(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let vertex = id.x + id.y * groups.x * 64u;
  if (vertex >= params.gridSize.x * params.gridSize.y) {
    return;
  }
  heightGradient[vertex] = gaussianAlong(vertex, 0u, 1u, params.updateSigma, 0.0);
}

@compute @workgroup_size(64)
fn applySmoothUpdateY(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let vertex = id.x + id.y * groups.x * 64u;
  if (vertex >= params.gridSize.x * params.gridSize.y) {
    return;
  }
  let step = gaussianAlong(vertex, 1u, 0u, params.updateSigma, 0.0);
  heights[vertex] = max(heights[vertex] - step, params.minHeight);
}

// Runs after adamUpdate so every vertex sees the same step count.
@compute @workgroup_size(1)
fn advanceStep() {
  state.step += 1u;
}

// ---- Vertex motion -------------------------------------------------------

// Bilateral Gaussian along one axis of the x/y gradient (source 0) or of the
// scratch buffer (source 1), stopping at creases like smoothedGradient.
fn blurOffsets(vertex: u32, axis: u32, source: u32) -> vec2<f32> {
  let coordinate = vec2<i32>(i32(vertex % params.gridSize.x), i32(vertex / params.gridSize.x));
  let sigma = params.xySigma / params.spacing[axis];
  let radius = i32(ceil(3.0 * sigma));
  let size = vec2<i32>(params.gridSize);
  var center = vec2<f32>(0.0);
  if (params.creaseSlope > 0.0) { center = vertexSlope(coordinate); }
  var sum = vec2<f32>(0.0);
  var weightSum = 0.0;
  for (var offset = -radius; offset <= radius; offset++) {
    var q = coordinate;
    q[axis] += offset;
    if (q[axis] < 0 || q[axis] >= size[axis]) { continue; }
    var weight = exp(-0.5 * f32(offset * offset) / max(sigma * sigma, 1e-8));
    if (params.creaseSlope > 0.0) {
      let difference = (vertexSlope(q) - center) / params.creaseSlope;
      weight *= exp(-dot(difference, difference));
    }
    let i = u32(q.y) * params.gridSize.x + u32(q.x);
    var value = scratch[i];
    if (source == 0u) { value = offsetGradient[i]; }
    sum += weight * vec2<f32>(finiteGradient(value.x), finiteGradient(value.y));
    weightSum += weight;
  }
  return sum / weightSum;
}

fn motionVertex(id: vec3u, groups: vec3u) -> u32 {
  return id.x + id.y * groups.x * 64u;
}

// Surface slope at a vertex from central differences over the displaced
// positions (one-sided at the border): the slope s with s . e = dh along both
// grid directions, exact for a plane however the vertices are displaced.
fn displacedSlope(vertex: u32) -> vec2<f32> {
  let w = params.gridSize.x;
  let column = vertex % w;
  let row = vertex / w;
  let xm = select(vertex - 1u, vertex, column == 0u);
  let xp = select(vertex + 1u, vertex, column + 1u == w);
  let ym = select(vertex - w, vertex, row == 0u);
  let yp = select(vertex + w, vertex, row + 1u == params.gridSize.y);
  let e1 = position(xp) - position(xm);
  let e2 = position(yp) - position(ym);
  let h1 = heights[xp] - heights[xm];
  let h2 = heights[yp] - heights[ym];
  let determinant = e1.x * e2.y - e1.y * e2.x;
  if (!(abs(determinant) > 0.0)) { return vec2<f32>(0.0); }
  return vec2<f32>(h1 * e2.y - h2 * e1.y, e1.x * h2 - e2.x * h1) / determinant;
}

// With xyTangential, a step d in x/y also raises the vertex by slope . d, so
// dL/d(step) = dL/d(xy) + dL/dh * slope. Runs after the height update, while
// heightGradient still holds the full height gradient.
@compute @workgroup_size(64)
fn tangentialGradient(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let vertex = motionVertex(id, groups);
  if (vertex >= params.gridSize.x * params.gridSize.y) { return; }
  let optical = offsetGradient[vertex];
  var g = vec2<f32>(finiteGradient(optical.x), finiteGradient(optical.y));
  if (params.xyTangential > 0.0) {
    let slope = displacedSlope(vertex);
    g += finiteGradient(heightGradient[vertex]) * vec2<f32>(finiteGradient(slope.x), finiteGradient(slope.y));
  }
  offsetGradient[vertex] = g;
}

@compute @workgroup_size(64)
fn blurOffsetGradientX(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let vertex = motionVertex(id, groups);
  if (vertex >= params.gridSize.x * params.gridSize.y) { return; }
  scratch[vertex] = blurOffsets(vertex, 0u, 0u);
}

@compute @workgroup_size(64)
fn blurOffsetGradientY(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let vertex = motionVertex(id, groups);
  if (vertex >= params.gridSize.x * params.gridSize.y) { return; }
  offsetGradient[vertex] = blurOffsets(vertex, 1u, 1u);
}

// Discrete Laplacian of the displacement in vertex spacings; zero on the border.
@compute @workgroup_size(64)
fn offsetCurvature(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let vertex = motionVertex(id, groups);
  let columns = params.gridSize.x;
  let rows = params.gridSize.y;
  if (vertex >= columns * rows) { return; }
  let column = vertex % columns;
  let row = vertex / columns;
  if (column == 0u || row == 0u || column + 1u == columns || row + 1u == rows) {
    scratch[vertex] = vec2<f32>(0.0);
    return;
  }
  let sum = offsets[vertex - 1u] + offsets[vertex + 1u] + offsets[vertex - columns] + offsets[vertex + columns];
  scratch[vertex] = (sum - 4.0 * offsets[vertex]) / params.spacing;
}

// Adam on the displacement: proposal = offsets - step. Border vertices only
// slide along their edge of the aperture.
@compute @workgroup_size(64)
fn proposeOffsets(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let vertex = motionVertex(id, groups);
  let columns = params.gridSize.x;
  let rows = params.gridSize.y;
  if (vertex >= columns * rows) { return; }
  let column = vertex % columns;
  let row = vertex / columns;

  // Smoothness: L = w / N * sum(|q|^2), q = laplacian(offsets) / spacing.
  var laplacianSum = -4.0 * scratch[vertex];
  if (column > 0u) { laplacianSum += scratch[vertex - 1u]; }
  if (column + 1u < columns) { laplacianSum += scratch[vertex + 1u]; }
  if (row > 0u) { laplacianSum += scratch[vertex - columns]; }
  if (row + 1u < rows) { laplacianSum += scratch[vertex + columns]; }
  let optical = offsetGradient[vertex];
  var g = vec2<f32>(finiteGradient(optical.x), finiteGradient(optical.y)) +
    2.0 * params.xySmoothnessWeight * params.invVertexCount * laplacianSum / params.spacing;
  var movable = vec2<f32>(1.0);
  if (column == 0u || column + 1u == columns) { movable.x = 0.0; }
  if (row == 0u || row + 1u == rows) { movable.y = 0.0; }
  g *= movable;

  let t = f32(state.step + 1u);
  let previous = offsetMoments[vertex];
  let m = params.beta1 * previous.xy + (1.0 - params.beta1) * g;
  let v = params.beta2 * previous.zw + (1.0 - params.beta2) * g * g;
  offsetMoments[vertex] = vec4<f32>(m, v);
  let mHat = m / (1.0 - pow(params.beta1, t));
  let vHat = v / (1.0 - pow(params.beta2, t));
  let remaining = max(0.0, 1.0 - t / f32(params.maxIterations));
  let rateFloor = params.learningRateFloor;
  let rate = params.xyLearningRate * params.spacing * (rateFloor + (1.0 - rateFloor) * remaining * remaining);
  let slope = displacedSlope(vertex);
  proposal[vertex] = vec4<f32>((offsets[vertex] - rate * mHat / (sqrt(vHat) + params.epsilon)) * movable,
    finiteGradient(slope.x), finiteGradient(slope.y));
}

// Area of a face as a fraction of its grid area.
fn areaRatio(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> f32 {
  let doubleArea = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
  return doubleArea / (params.spacing.x * params.spacing.y);
}

fn grid(vertex: u32) -> vec2<f32> {
  return vec2<f32>(f32(vertex % params.gridSize.x), f32(vertex / params.gridSize.x)) * params.spacing;
}

// Accept a vertex's proposal only if every face around it keeps at least
// minAreaRatio of its grid area whether each neighbor moves or stays. Then
// any mix of accepted and rejected moves leaves every face valid.
@compute @workgroup_size(64)
fn acceptOffsets(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let vertex = motionVertex(id, groups);
  let columns = params.gridSize.x;
  let rows = params.gridSize.y;
  if (vertex >= columns * rows) { return; }
  let column = vertex % columns;
  let row = vertex / columns;
  var valid = true;
  for (var dy = 0u; dy < 2u; dy++) {
    for (var dx = 0u; dx < 2u; dx++) {
      if (column < dx || row < dy) { continue; }
      let cellColumn = column - dx;
      let cellRow = row - dy;
      if (cellColumn + 1u >= columns || cellRow + 1u >= rows) { continue; }
      var faces = cellFaces(cellColumn, cellRow);
      for (var f = 0u; f < 2u; f++) {
        let ids = faces[f];
        if (ids.x != vertex && ids.y != vertex && ids.z != vertex) { continue; }
        for (var combo = 0u; combo < 8u; combo++) {
          var p: array<vec2<f32>, 3>;
          for (var k = 0u; k < 3u; k++) {
            let moved = ids[k] == vertex || ((combo >> k) & 1u) == 1u;
            p[k] = grid(ids[k]) + select(offsets[ids[k]], proposal[ids[k]].xy, moved);
          }
          if (!(areaRatio(p[0], p[1], p[2]) >= params.minAreaRatio)) { valid = false; }
        }
      }
    }
  }
  scratch[vertex] = select(offsets[vertex], proposal[vertex].xy, valid);
}

// Each vertex writes only its own offset and height here, so no neighbor sees
// a half-updated mesh.
@compute @workgroup_size(64)
fn commitOffsets(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let vertex = motionVertex(id, groups);
  if (vertex >= params.gridSize.x * params.gridSize.y) { return; }
  if (params.xyTangential > 0.0) {
    let step = scratch[vertex] - offsets[vertex];
    heights[vertex] = max(heights[vertex] + dot(proposal[vertex].zw, step), params.minHeight);
  }
  offsets[vertex] = scratch[vertex];
}
