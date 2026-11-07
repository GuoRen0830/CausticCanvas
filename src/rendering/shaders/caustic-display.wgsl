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

  // Match the image convention: row 0 is at the bottom.
  output.uv = position * 0.5 + vec2<f32>(0.5);

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

fn displayColor(irradiance: f32) -> vec4<f32> {
  // Apply exposure and compress the display range.
  let scaled = max(irradiance, 0.0) * display.exposure;
  let mapped = scaled / (1.0 + scaled);

  // Encode for the current non-sRGB UNORM canvas view.
  let encoded = select(
    12.92 * mapped,
    1.055 * pow(mapped, 1.0 / 2.4) - 0.055,
    mapped > 0.0031308,
  );

  return vec4<f32>(vec3<f32>(encoded), 1.0);
}

@fragment
fn fragmentBuffer(
  @location(0) uv: vec2<f32>,
) -> @location(0) vec4<f32> {
  let pixel = pixelFromUV(uv);
  let index = pixel.y * display.imageSize.x + pixel.x;

  return displayColor(imageBuffer[index]);
}

@fragment
fn fragmentTexture(
  @location(0) uv: vec2<f32>,
) -> @location(0) vec4<f32> {
  let pixel = pixelFromUV(uv);
  let irradiance = textureLoad(imageTexture, pixel, 0).r;

  return displayColor(irradiance);
}