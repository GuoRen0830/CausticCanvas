struct TransformUniforms {
  mvp: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> transforms: TransformUniforms;

@vertex
fn vertexMain(
  @location(0) position: vec3<f32>,
) -> @builtin(position) vec4<f32> {
  // Transform lens-local coordinates into clip space.
  return transforms.mvp * vec4<f32>(position, 1.0);
}

@fragment
fn fragmentMain() -> @location(0) vec4<f32> {
  // Use a solid color to inspect the lens geometry.
  return vec4<f32>(0.2, 0.75, 0.95, 1.0);
}