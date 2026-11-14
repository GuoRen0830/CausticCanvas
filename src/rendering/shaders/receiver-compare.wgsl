@fragment
fn integrateFS(input: Bounds) -> @location(0) vec4f {
  // Framebuffer row 0 is at the top. Optical row 0 is at the bottom.
  let pixel = vec2f(floor(input.position.x), f32(frame.image.y) - 1.0 - floor(input.position.y));
  let coverage = pixelCoverage(input.a - pixel, input.b - pixel, input.c - pixel);
  return vec4f(coverage * input.irradiance, 0, 0, 0);
}

@group(1) @binding(2) var<storage, read> solverBuffer: array<f32>;
@group(1) @binding(3) var solverTexture: texture_2d<f32>;

struct PlaneSample { texel: vec2i, opticalPixel: vec2u, inside: bool }

fn planeSample(frag: vec2f) -> PlaneSample {
  let image = vec2f(frame.image);
  // Preserve receiver aspect ratio and center it in the viewport.
  let scale = min(frame.targetSize.x / image.x, frame.targetSize.y / image.y);
  let p = (frag - 0.5 * (frame.targetSize - image * scale)) / scale;
  let texel = vec2u(clamp(floor(p), vec2f(0), image - 1.0));
  return PlaneSample(vec2i(texel), vec2u(texel.x, frame.image.y - 1u - texel.y),
                     all(p >= vec2f(0)) && all(p < image));
}

fn planeColor(p: PlaneSample, solver: f32) -> vec4f {
  if (!p.inside) { return vec4f(0.03, 0.06, 0.08, 1); }
  let raw = textureLoad(lighting, p.texel, 0).r;
  if (frame.mode == 4u) { return displayColor(abs(raw - solver), frame.exposure); }
  return displayColor(raw, frame.exposure);
}

@fragment
fn compareBufferFS(@builtin(position) frag: vec4f) -> @location(0) vec4f {
  let p = planeSample(frag.xy);
  return planeColor(p, solverBuffer[p.opticalPixel.y * frame.image.x + p.opticalPixel.x]);
}

@fragment
fn compareTextureFS(@builtin(position) frag: vec4f) -> @location(0) vec4f {
  let p = planeSample(frag.xy);
  return planeColor(p, textureLoad(solverTexture, vec2i(p.opticalPixel), 0).r);
}
