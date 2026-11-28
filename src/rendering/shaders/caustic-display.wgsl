struct DisplayUniforms {
  imageSize: vec2<u32>,
  exposure: f32,
  padding: f32,
}

@group(0) @binding(0)
var<uniform> display: DisplayUniforms;

@group(0) @binding(1)
var<storage, read> imageBuffer: array<f32>;

@group(0) @binding(2)
var imageTexture: texture_2d<f32>;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

@vertex
fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
) -> VertexOutput {
  // One oversized triangle covers the entire viewport.
  let positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0),
  );

  let position = positions[vertexIndex];

  var output: VertexOutput;
  output.position = vec4<f32>(position, 0.0, 1.0);

  // The top of the drawing pad and preview both show optical row 0 (-Y).
  output.uv = position * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);

  return output;
}

fn pixelFromUV(uv: vec2<f32>) -> vec2<u32> {
  let boundedUV = clamp(
    uv,
    vec2<f32>(0.0),
    vec2<f32>(1.0),
  );

  let pixel = vec2<u32>(
    boundedUV * vec2<f32>(display.imageSize),
  );

  // Keep edge coordinates inside the source image.
  return min(pixel, display.imageSize - vec2<u32>(1u));
}

@fragment
fn fragmentBuffer(
  @location(0) uv: vec2<f32>,
) -> @location(0) vec4<f32> {
  let pixel = pixelFromUV(uv);
  let index = pixel.y * display.imageSize.x + pixel.x;

  return displayColor(imageBuffer[index], display.exposure);
}

@fragment
fn fragmentTexture(
  @location(0) uv: vec2<f32>,
) -> @location(0) vec4<f32> {
  let pixel = pixelFromUV(uv);
  let irradiance = textureLoad(imageTexture, pixel, 0).r;

  return displayColor(irradiance, display.exposure);
}

@fragment
fn fragmentRGB(@location(0) uv: vec2f) -> @location(0) vec4f {
  return displayColorRGB(textureLoad(imageTexture, pixelFromUV(uv), 0).rgb, display.exposure);
}
