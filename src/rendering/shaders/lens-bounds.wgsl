// Render-only bounds. Two small reductions keep exaggerated previews centered
// and framed without reading a million heights back to the CPU.
@group(0) @binding(0) var<storage, read> heights: array<f32>;
@group(0) @binding(1) var<storage, read_write> partials: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> bounds: vec2f;
var<workgroup> ranges: array<vec2f, 256>;

fn merge(a: vec2f, b: vec2f) -> vec2f {
  return vec2f(min(a.x, b.x), max(a.y, b.y));
}

fn reduce(lane: u32) {
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride /= 2u) {
    if (lane < stride) { ranges[lane] = merge(ranges[lane], ranges[lane + stride]); }
    workgroupBarrier();
  }
}

@compute @workgroup_size(256)
fn scan(@builtin(global_invocation_id) id: vec3u,
        @builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3u) {
  var range = vec2f(1e30, -1e30);
  for (var i = id.x; i < arrayLength(&heights); i += 65536u) {
    range = merge(range, vec2f(heights[i]));
  }
  ranges[lane] = range;
  reduce(lane);
  if (lane == 0u) { partials[group.x] = ranges[0]; }
}

@compute @workgroup_size(256)
fn finish(@builtin(local_invocation_index) lane: u32) {
  ranges[lane] = partials[lane];
  reduce(lane);
  if (lane == 0u) { bounds = ranges[0]; }
}
