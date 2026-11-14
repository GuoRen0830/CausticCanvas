struct SceneUniforms {
  mvp: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> sceneUniforms: SceneUniforms;

struct VertexInput {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) color: vec3<f32>,
  @location(3) isFloor: f32,
  @location(4) objectId: f32,
}

struct VertexOutput {
  @builtin(position) clipPosition: vec4<f32>,
  @location(0) localNormal: vec3<f32>,
  @location(1) color: vec3<f32>,
  @location(2) @interpolate(flat) isFloor: f32,
  @location(3) @interpolate(flat) objectId: u32,
  @location(4) localPosition: vec3f,
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
  return output;
}

struct GBuffer {
  @location(0) albedo: vec4<f32>,
  @location(1) normalAndFloor: vec4<f32>,
  @location(2) objectAndNormal: vec2u,
}

@fragment
fn fragmentMain(input: VertexOutput) -> GBuffer {
  // Screen Y points down, so dy cross dx follows the front-face winding.
  // Keep this geometric normal independent of interpolated shading normals.
  let geometric = normalize(cross(dpdy(input.localPosition), dpdx(input.localPosition)));
  return GBuffer(
    vec4<f32>(input.color, 1.0),
    // Signed unit normals encoded in UNORM8; zero is exactly byte 128.
    vec4<f32>((normalize(input.localNormal) * 127.0 + 128.0) / 255.0, input.isFloor),
    vec2u(input.objectId, pack4x8snorm(vec4f(geometric, 0))),
  );
}
