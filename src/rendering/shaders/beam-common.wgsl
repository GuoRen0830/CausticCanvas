struct Frame {
  localVP: mat4x4f,
  inverseLocalVP: mat4x4f,
  grid: vec2u,
  aperture: vec2f,
  pad: vec2f,
  receiverSize: vec2f,
  image: vec2u,
  targetSize: vec2f,
  receiverZ: f32,
  mode: u32,
  exposure: f32,
  beamCount: u32,
}

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var sceneDepth: texture_depth_2d;
@group(0) @binding(2) var normalAndFloor: texture_2d<f32>;
@group(0) @binding(3) var albedo: texture_2d<f32>;
@group(0) @binding(4) var lighting: texture_2d<f32>;
@group(1) @binding(0) var<storage, read> heights: array<f32>;
@group(1) @binding(1) var<storage, read> beams: array<vec4f>;

fn sourceVertex(id: u32) -> vec3f {
  let gridXY = vec2f(f32(id % frame.grid.x), f32(id / frame.grid.x));
  let xy = (gridXY / vec2f(frame.grid - 1u) - 0.5) * frame.aperture;
  return vec3f(xy, heights[id]);
}

fn sourceIds(face: u32) -> vec3u {
  let cell = face / 2u;
  let stride = frame.grid.x;
  let i = cell / (stride - 1u) * stride + cell % (stride - 1u);
  return select(vec3u(i, i + 1u, i + stride),
                   vec3u(i + 1u, i + stride + 1u, i + stride), face % 2u == 1u);
}
