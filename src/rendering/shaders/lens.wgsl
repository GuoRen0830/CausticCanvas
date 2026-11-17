struct LensUniforms {
  mvp: mat4x4<f32>,
  model: mat4x4<f32>,
  gridSize: vec2<u32>,
  apertureSize: vec2<f32>,
  relief: f32,
  referenceHeight: f32,
  padding: vec2f,
}

@group(0) @binding(0)
var<uniform> lensUniforms: LensUniforms;

@group(0) @binding(1)
var<storage, read> heights: array<f32>;

@group(0) @binding(2) var<storage, read> heightBounds: vec2f;

struct VertexOutput {
  @builtin(position) clipPosition: vec4<f32>,
  @location(0) worldPosition: vec3<f32>,
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

  // Exaggerate displacement around the midpoint in this preview only.
  // The shared optical height buffer is read-only and remains in meters.
  let midpoint = 0.5 * (heightBounds.x + heightBounds.y);
  let span = (heightBounds.y - heightBounds.x) * lensUniforms.relief;
  let aperture = max(lensUniforms.apertureSize.x, lensUniforms.apertureSize.y);
  // Uniform fitting preserves the exaggerated height-to-width ratio.
  let fit = min(1.0, 0.6 * aperture / max(span, 1e-8));
  let height = (heights[vertexIndex] - midpoint) * lensUniforms.relief;
  let position = vec3<f32>(xy, height) * fit + vec3f(0, 0, lensUniforms.referenceHeight);

  var output: VertexOutput;
  output.clipPosition = lensUniforms.mvp * vec4<f32>(position, 1.0);
  output.worldPosition = (lensUniforms.model * vec4<f32>(position, 1.0)).xyz;

  return output;
}

@fragment
fn fragmentMain(
  @location(0) worldPosition: vec3<f32>,
) -> @location(0) vec4<f32> {
  // Estimate surface tangents from neighboring fragment positions.
  let tangentX = dpdx(worldPosition);
  let tangentY = dpdy(worldPosition);

  // Framebuffer Y points down, so reverse the cross-product order.
  let normal = normalize(cross(tangentY, tangentX));

  // Glass-like shading for the inspection view: a cool tint that darkens on
  // steep slopes, plus a highlight from the lamp above. The orbit camera
  // usually looks down from the front, so a fixed viewing direction is enough.
  let toLight = normalize(vec3<f32>(-0.4, 0.8, 0.6));
  let toViewer = normalize(vec3<f32>(0.5, 0.7, 0.6));
  let n = select(normal, -normal, dot(normal, toViewer) < 0.0);
  let diffuse = max(dot(n, toLight), 0.0);
  let halfway = normalize(toLight + toViewer);
  let highlight = pow(max(dot(n, halfway), 0.0), 80.0);
  let rim = pow(1.0 - max(dot(n, toViewer), 0.0), 3.0);

  let deep = vec3<f32>(0.10, 0.13, 0.14);
  let pale = vec3<f32>(0.62, 0.72, 0.74);
  let color = mix(deep, pale, diffuse * diffuse) + 0.25 * rim * pale
    + vec3<f32>(1.0) * highlight;
  return vec4<f32>(min(color, vec3<f32>(1.0)), 1.0);
}
