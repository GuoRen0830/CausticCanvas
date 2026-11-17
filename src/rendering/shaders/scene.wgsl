struct SceneUniforms {
  mvp: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> sceneUniforms: SceneUniforms;
struct Material { color: vec4f, uvTransform: vec4f }
@group(1) @binding(0) var<uniform> material: Material;
@group(1) @binding(1) var baseColorMap: texture_2d<f32>;
@group(1) @binding(2) var baseColorSampler: sampler;

struct VertexInput {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) color: vec3<f32>,
  @location(3) isFloor: f32,
  @location(4) objectId: f32,
  @location(5) uv: vec2f,
}

struct VertexOutput {
  @builtin(position) clipPosition: vec4<f32>,
  @location(0) localNormal: vec3<f32>,
  @location(1) color: vec3<f32>,
  @location(2) @interpolate(flat) isFloor: f32,
  @location(3) @interpolate(flat) objectId: u32,
  @location(4) localPosition: vec3f,
  @location(5) uv: vec2f,
}

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  output.clipPosition = sceneUniforms.mvp * vec4<f32>(input.position, 1.0);
  output.localNormal = input.normal;
  output.color = input.color;
  output.isFloor = input.isFloor;
  output.objectId = u32(input.objectId);
  output.localPosition = input.position;
  let uv = input.uv * material.uvTransform.xy + material.uvTransform.zw;
  output.uv = vec2f(uv.x, 1.0 - uv.y);
  return output;
}

struct GBuffer {
  @location(0) albedo: vec4<f32>,
  @location(1) normalAndFloor: vec4<f32>,
  @location(2) objectAndNormal: vec2u,
}

// Same bytes as pack4x8snorm(vec4f(n, 0)), which returns 0 on some Metal drivers.
fn packNormal(n: vec3f) -> u32 {
  let q = vec3i(round(clamp(n, vec3f(-1), vec3f(1)) * 127.0)) & vec3i(255);
  return u32(q.x) | (u32(q.y) << 8u) | (u32(q.z) << 16u);
}

@fragment
fn fragmentMain(input: VertexOutput) -> GBuffer {
  // Screen Y points down, so dy cross dx follows the front-face winding.
  // Keep this geometric normal independent of interpolated shading normals.
  let geometric = normalize(cross(dpdy(input.localPosition), dpdx(input.localPosition)));
  return GBuffer(
    vec4f(mix(input.color, material.color.rgb, material.color.a) *
      textureSample(baseColorMap, baseColorSampler, input.uv).rgb, 1.0),
    // Signed unit normals encoded in UNORM8; zero is exactly byte 128.
    vec4<f32>((normalize(input.localNormal) * 127.0 + 128.0) / 255.0, input.isFloor),
    vec2u(input.objectId, packNormal(geometric)),
  );
}
