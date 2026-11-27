struct VolumeParams {
  grid: vec2u, strata: vec2u,
  aperture: vec2f, hasHits: u32, pad0: u32,
  lower: vec3f, pad1: f32,
  cellSize: vec3f, pad2: f32,
  resolution: vec3u, pad3: u32,
}
struct Hit { trianglePlusOne: u32, t: f32 }
struct Triangle {
  a: vec3f, objectId: u32,
  b: vec3f, originalId: u32,
  c: vec3f, pad: u32,
}
@group(0) @binding(0) var<uniform> volume: VolumeParams;
@group(0) @binding(1) var<storage, read> heights: array<f32>;
@group(0) @binding(2) var<storage, read> offsets: array<vec2f>;
@group(0) @binding(3) var<storage, read> beams: array<vec4f>;
@group(0) @binding(4) var<storage, read> hits: array<Hit>;
@group(0) @binding(5) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(6) var<storage, read_write> field: array<atomic<u32>>;
@group(0) @binding(7) var lightVolume: texture_storage_3d<rgba16float, write>;
@group(0) @binding(8) var<storage, read_write> filteredRows: array<u32>;

fn vertex(id: u32) -> vec3f {
  let xy = vec2f(vec2u(id % volume.grid.x, id / volume.grid.x)) / vec2f(volume.grid - 1u);
  return vec3f((xy - 0.5) * volume.aperture + offsets[id], heights[id]);
}

fn jitter(seed: u32) -> f32 {
  var n = seed ^ (seed >> 16u);
  n *= 0x7feb352du;
  n ^= n >> 15u;
  n *= 0x846ca68bu;
  return f32((n ^ (n >> 16u)) >> 8u) * (1.0 / 16777216.0);
}

fn flatIndex(cell: vec3u, size: vec3u) -> u32 {
  return (cell.z * size.y + cell.y) * size.x + cell.x;
}

fn deposit(cell: vec3i, amount: f32) {
  if (any(cell < vec3i(0)) || any(cell >= vec3i(volume.resolution)) || !(amount > 0.0)) { return; }
  let index = flatIndex(vec3u(cell), volume.resolution);
  // WGSL has no native atomicAdd on f32. Saturate at the representable
  // texture range even when many bright samples converge to the same cell.
  var old = atomicLoad(&field[index]);
  loop {
    let next = bitcast<u32>(min(65504.0, bitcast<f32>(old) + amount));
    if (old == next) { break; }
    let changed = atomicCompareExchangeWeak(&field[index], old, next);
    if (changed.exchanged) { break; }
    old = changed.old_value;
  }
}

@compute @workgroup_size(8, 8)
fn inject(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= volume.strata)) { return; }
  // Sampling in undeformed aperture UV gives each source triangle the same
  // probability. Its actual displaced area is already included in beam.w.
  let seed = id.x + id.y * volume.strata.x;
  let uv = (vec2f(id.xy) + vec2f(jitter(2u * seed), jitter(2u * seed + 1u))) / vec2f(volume.strata);
  let cells = volume.grid - 1u;
  let position = uv * vec2f(cells);
  let cell = min(vec2u(position), cells - 1u);
  let local = position - vec2f(cell);
  let second = local.x + local.y > 1.0;
  let face = 2u * (cell.y * cells.x + cell.x) + u32(second);
  let ray = beams[face];
  if (!(ray.w > 0.0)) { return; }
  let i = cell.y * volume.grid.x + cell.x;
  let a = vertex(select(i, i + 1u, second));
  let b = vertex(select(i + 1u, i + volume.grid.x + 1u, second));
  let c = vertex(i + volume.grid.x);
  let bary = select(vec3f(1.0 - local.x - local.y, local.x, local.y),
                    vec3f(1.0 - local.y, local.x + local.y - 1.0, 1.0 - local.x), second);
  let origin = a * bary.x + b * bary.y + c * bary.z;
  var end = 1e20;
  if (volume.hasHits != 0u) {
    let hit = hits[face];
    if (hit.trianglePlusOne != 0u) {
      // The forward pass traces from the centroid. Move that hit onto this
      // sample's parallel ray on the same first-hit plane.
      let tri = triangles[hit.trianglePlusOne - 1u];
      let n = cross(tri.b - tri.a, tri.c - tri.a);
      let den = dot(n, ray.xyz);
      end = hit.t;
      if (abs(den) > 1e-6 * length(n)) {
        end += dot(n, (a + b + c) / 3.0 - origin) / den;
      }
    }
  }
  let start = (origin - volume.lower) / volume.cellSize;
  let velocity = ray.xyz / volume.cellSize;
  var enter = 0.0;
  for (var k = 0u; k < 3u; k++) {
    if (abs(velocity[k]) < 1e-12) {
      if (start[k] < 0.0 || start[k] >= f32(volume.resolution[k])) { return; }
    } else {
      let near = -start[k] / velocity[k];
      let far = (f32(volume.resolution[k]) - start[k]) / velocity[k];
      enter = max(enter, min(near, far));
      end = min(end, max(near, far));
    }
  }
  if (!(end > enter)) { return; }

  // Slice perpendicular to the dominant voxel-space direction. This also
  // covers very oblique rays without skipping cells along their long axis.
  let speed = abs(velocity);
  var k = select(0u, 1u, speed.y > speed.x);
  k = select(k, 2u, speed.z >= speed[k]);
  let u = (k + 1u) % 3u;
  let v = (k + 2u) % 3u;
  let p0 = start[k] + enter * velocity[k];
  let p1 = start[k] + end * velocity[k];
  let first = max(0, i32(floor(min(p0, p1))));
  let last = min(i32(volume.resolution[k]) - 1, i32(floor(max(p0, p1))));
  let inverseProbability = 2.0 * f32(cells.x) * f32(cells.y) / f32(volume.strata.x * volume.strata.y);
  let voxelVolume = volume.cellSize.x * volume.cellSize.y * volume.cellSize.z;
  let fluxPerVolume = ray.w * inverseProbability / voxelVolume;
  let stepT = 1.0 / velocity[k];
  for (var layer = first; layer <= last; layer++) {
    let left = (f32(layer) - start[k]) * stepT;
    let right = left + stepT;
    let t0 = max(enter, min(left, right));
    let t1 = min(end, max(left, right));
    if (t1 <= t0) { continue; }
    let center = start + (0.5 * (t0 + t1)) * velocity - 0.5;
    let across = vec2f(center[u], center[v]);
    let lower = vec2i(floor(across));
    let f = fract(across);
    // Track length times power divided by voxel volume gives W/m².
    let fluenceRate = (t1 - t0) * fluxPerVolume;
    var index = vec3i(0);
    index[k] = layer;
    index[u] = lower.x;
    index[v] = lower.y;
    deposit(index, fluenceRate * (1.0 - f.x) * (1.0 - f.y));
    index[u] += 1;
    deposit(index, fluenceRate * f.x * (1.0 - f.y));
    index[v] += 1;
    deposit(index, fluenceRate * f.x * f.y);
    index[u] -= 1;
    deposit(index, fluenceRate * (1.0 - f.x) * f.y);
  }
}

fn rowSample(id: vec3u, dx: i32) -> f32 {
  let x = i32(id.x) + dx;
  if (x < 0 || x >= i32(volume.resolution.x)) { return 0.0; }
  let index = flatIndex(vec3u(u32(x), id.yz), volume.resolution);
  return bitcast<f32>(atomicLoad(&field[index]));
}

// Two separable transverse filters in one pair of passes: a one-voxel tent
// for fluence and a broader binomial kernel for its local contrast reference.
// Neither filter spans depth layers, so it cannot extend a terminated ray in Z.
@compute @workgroup_size(16, 16)
fn filterRows(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= volume.resolution)) { return; }
  let center = rowSample(id, 0);
  let narrow = (rowSample(id, -1) + 2.0 * center + rowSample(id, 1)) * 0.25;
  let broad = (rowSample(id, -6) + rowSample(id, 6)
    + 4.0 * (rowSample(id, -3) + rowSample(id, 3)) + 6.0 * center) / 16.0;
  filteredRows[flatIndex(id, volume.resolution)] = pack2x16float(vec2f(narrow, broad));
}

fn columnSample(id: vec3u, dy: i32, size: vec3u) -> vec2f {
  let y = i32(id.y) + dy;
  if (y < 0 || y >= i32(size.y)) { return vec2f(0); }
  return unpack2x16float(filteredRows[flatIndex(vec3u(id.x, u32(y), id.z), size)]);
}

@compute @workgroup_size(16, 16)
fn resolve(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(lightVolume);
  if (any(id >= size)) { return; }
  let center = columnSample(id, 0, size);
  let narrow = (columnSample(id, -1, size).x + 2.0 * center.x + columnSample(id, 1, size).x) * 0.25;
  let broad = (columnSample(id, -6, size).y + columnSample(id, 6, size).y
    + 4.0 * (columnSample(id, -3, size).y + columnSample(id, 3, size).y) + 6.0 * center.y) / 16.0;
  textureStore(lightVolume, id, vec4f(narrow, broad, 0.0, 1.0));
}
