struct LensUniforms {
  mvp: mat4x4<f32>,
  gridSize: vec2<u32>,
  apertureSize: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> lensUniforms: LensUniforms;

@group(0) @binding(1)
var<storage, read> heights: array<f32>;

struct VertexOutput {
  @builtin(position) clipPosition: vec4<f32>,
  @location(0) localPosition: vec3<f32>,
}

@vertex
fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
) -> VertexOutput {
  // Recover the grid coordinates from the indexed vertex.
  let column = vertexIndex % lensUniforms.gridSize.x;
  let row = vertexIndex / lensUniforms.gridSize.x;

  let gridPosition = vec2<f32>(f32(column), f32(row));
  let gridExtent = vec2<f32>(
    lensUniforms.gridSize - vec2<u32>(1u),
  );

  // Map the grid onto the centered physical aperture.
  let xy = (
    gridPosition / gridExtent - vec2<f32>(0.5)
  ) * lensUniforms.apertureSize;

  let position = vec3<f32>(xy, heights[vertexIndex]);

  var output: VertexOutput;
  output.clipPosition = lensUniforms.mvp * vec4<f32>(position, 1.0);
  output.localPosition = position;

  return output;
}

@fragment
fn fragmentMain(
  @location(0) localPosition: vec3<f32>,
) -> @location(0) vec4<f32> {
  // Estimate surface tangents from neighboring fragment positions.
  let tangentX = dpdx(localPosition);
  let tangentY = dpdy(localPosition);

  // Framebuffer Y points down, so reverse the cross-product order.
  let normal = normalize(cross(tangentY, tangentX));

  // Use a fixed inspection light in lens-local space.
  let toLight = normalize(vec3<f32>(-0.4, 0.6, 1.0));
  let diffuse = max(dot(normal, toLight), 0.0);

  let baseColor = vec3<f32>(0.2, 0.75, 0.95);
  let brightness = 0.2 + 0.8 * diffuse;

  return vec4<f32>(baseColor * brightness, 1.0);
}