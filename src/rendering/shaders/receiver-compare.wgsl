// Integrate the constant irradiance of a triangular footprint over a unit
// receiver pixel. Clipping in pixel-local coordinates improves precision.
struct Polygon { points: array<vec2f, 8>, count: u32 }

fn clipEdge(poly: Polygon, axis: u32, bound: f32, keepGreater: bool) -> Polygon {
  var result: Polygon;
  if (poly.count == 0u) { return result; }
  var prev = poly.points[poly.count - 1u];
  for (var i = 0u; i < poly.count; i++) {
    let curr = poly.points[i];
    let prevIn = select(prev[axis] <= bound, prev[axis] >= bound, keepGreater);
    let currIn = select(curr[axis] <= bound, curr[axis] >= bound, keepGreater);
    if (prevIn != currIn) {
      let t = (bound - prev[axis]) / (curr[axis] - prev[axis]);
      result.points[result.count] = mix(prev, curr, t);
      result.count++;
    }
    if (currIn) {
      result.points[result.count] = curr;
      result.count++;
    }
    prev = curr;
  }
  return result;
}

@fragment
fn integrateFS(input: Bounds) -> @location(0) vec4f {
  // Framebuffer row 0 is at the top. Optical row 0 is at the bottom.
  let pixel = vec2f(floor(input.position.x), f32(frame.image.y) - 1.0 - floor(input.position.y));
  var poly: Polygon;
  poly.count = 3u;
  poly.points[0] = input.a - pixel;
  poly.points[1] = input.b - pixel;
  poly.points[2] = input.c - pixel;
  poly = clipEdge(poly, 0u, 0.0, true);
  poly = clipEdge(poly, 0u, 1.0, false);
  poly = clipEdge(poly, 1u, 0.0, true);
  poly = clipEdge(poly, 1u, 1.0, false);
  var twiceArea = 0.0;
  for (var i = 1u; i + 1u < poly.count; i++) {
    twiceArea += cross2(poly.points[i] - poly.points[0], poly.points[i+1u] - poly.points[0]);
  }
  let coverage = clamp(0.5 * abs(twiceArea), 0.0, 1.0);
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
