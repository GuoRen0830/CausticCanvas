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
}

struct VertexOutput {
  @builtin(position) clipPosition: vec4<f32>,
  @location(0) worldNormal: vec3<f32>,
  @location(1) color: vec3<f32>,
}

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  output.clipPosition = sceneUniforms.mvp * vec4<f32>(input.position, 1.0);
  // The shared model is rigid, so its rotation also transforms normals.
  output.worldNormal = (sceneUniforms.model * vec4<f32>(input.normal, 0.0)).xyz;
  output.color = input.color;
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
  // Inspection shading only; beam irradiance and light visibility come later.
  let normal = normalize(input.worldNormal);
  let toLight = normalize(vec3<f32>(-0.4, 0.8, 0.6));
  let diffuse = max(dot(normal, toLight), 0.0);
  return vec4<f32>(input.color * (0.25 + 0.75 * diffuse), 1.0);
}
