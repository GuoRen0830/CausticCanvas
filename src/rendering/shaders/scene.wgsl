struct SceneUniforms {
  mvp: mat4x4<f32>,
  model: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> sceneUniforms: SceneUniforms;

struct VertexInput {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) color: vec3<f32>,
  @location(3) isFloor: f32,
}

struct VertexOutput {
  @builtin(position) clipPosition: vec4<f32>,
  @location(0) worldNormal: vec3<f32>,
  @location(1) color: vec3<f32>,
  @location(2) @interpolate(flat) isFloor: f32,
}

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  output.clipPosition = sceneUniforms.mvp * vec4<f32>(input.position, 1.0);
  // The shared model is rigid, so its rotation also transforms normals.
  output.worldNormal = (sceneUniforms.model * vec4<f32>(input.normal, 0.0)).xyz;
  output.color = input.color;
  output.isFloor = input.isFloor;
  return output;
}

struct GBuffer {
  @location(0) albedo: vec4<f32>,
  @location(1) normalAndFloor: vec4<f32>,
}

@fragment
fn fragmentMain(input: VertexOutput) -> GBuffer {
  return GBuffer(
    vec4<f32>(input.color, 1.0),
    // Signed unit normals encoded in UNORM8; zero is exactly byte 128.
    vec4<f32>((normalize(input.worldNormal) * 127.0 + 128.0) / 255.0, input.isFloor),
  );
}
