struct Frame {
  localVP: mat4x4f,
  inverseLocalVP: mat4x4f,
  grid: vec2u,
  aperture: vec2f,
  environmentLight: f32,
  referenceIrradiance: f32,
  targetSize: vec2f,
  mode: u32,
  exposure: f32,
  hasColorFilter: u32,
  pad: u32,
}

// Fixed logarithmic scale relative to the incident lamp irradiance.
// Ticks at 0, 1, 3, 7 and 15 times the lamp intensity are evenly spaced.
fn irradianceHeat(value: f32) -> vec4f {
  let t = clamp(log2(1.0 + max(value, 0.0) / frame.referenceIrradiance) / 4.0, 0.0, 1.0);
  let dark = vec3f(0.035, 0.045, 0.10);
  let blue = vec3f(0.13, 0.45, 0.85);
  let cyan = vec3f(0.08, 0.80, 0.78);
  let yellow = vec3f(1.0, 0.82, 0.20);
  let white = vec3f(1.0, 0.97, 0.88);
  var color = mix(dark, blue, clamp(4.0 * t, 0.0, 1.0));
  color = mix(color, cyan, clamp(4.0 * t - 1.0, 0.0, 1.0));
  color = mix(color, yellow, clamp(4.0 * t - 2.0, 0.0, 1.0));
  color = mix(color, white, clamp(4.0 * t - 3.0, 0.0, 1.0));
  return vec4f(color, 1.0);
}

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var sceneDepth: texture_depth_2d;
@group(0) @binding(2) var normalAndFloor: texture_2d<f32>;
@group(0) @binding(3) var albedo: texture_2d<f32>;
@group(0) @binding(4) var lighting: texture_2d<f32>;
@group(1) @binding(0) var<storage, read> heights: array<f32>;
@group(1) @binding(1) var<storage, read> beams: array<vec4f>;
// Per-vertex XY displacement from the uniform grid, in meters.
@group(1) @binding(2) var<storage, read> offsets: array<vec2f>;
@group(1) @binding(5) var apertureFilter: texture_2d<f32>;

fn sourceVertex(id: u32) -> vec3f {
  let gridXY = vec2f(f32(id % frame.grid.x), f32(id / frame.grid.x));
  let xy = (gridXY / vec2f(frame.grid - 1u) - 0.5) * frame.aperture + offsets[id];
  return vec3f(xy, heights[id]);
}

fn sourceIds(face: u32) -> vec3u {
  let cell = face / 2u;
  let stride = frame.grid.x;
  let i = cell / (stride - 1u) * stride + cell % (stride - 1u);
  return select(vec3u(i, i + 1u, i + stride),
                   vec3u(i + 1u, i + stride + 1u, i + stride), face % 2u == 1u);
}
