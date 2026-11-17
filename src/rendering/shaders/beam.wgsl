struct Bounds {
  @builtin(position) position: vec4f,
  // Receiver pixel units, with positive Y up. Constant across a beam rectangle.
  @location(0) @interpolate(flat) a: vec2f,
  @location(1) @interpolate(flat) b: vec2f,
  @location(2) @interpolate(flat) c: vec2f,
  @location(3) @interpolate(flat) irradiance: f32,
}

fn receiverXY(s: vec3f, d: vec3f) -> vec2f {
  return s.xy + ((frame.receiverZ - s.z) / d.z) * d.xy;
}

fn receiverPixel(xy: vec2f) -> vec2f {
  return (xy / frame.receiverSize + 0.5) * vec2f(frame.image);
}

@vertex
fn boundsVS(@builtin(vertex_index) vertex: u32, @builtin(instance_index) face: u32) -> Bounds {
  var out: Bounds;
  out.position = vec4f(2, 2, 0, 1); // Collapse all four vertices of a skipped beam.
  let ray = beams[face];
  if (!(ray.w > 0.0 && ray.z > 1e-6)) { return out; }

  let ids = sourceIds(face);
  var lo = vec2f(1e30);
  var hi = vec2f(-1e30);
  var corners: array<vec2f, 3>;
  for (var j = 0u; j < 3u; j++) {
    let s = sourceVertex(ids[j]);
    if (!(frame.receiverZ > s.z)) { return out; }
    let foot = receiverXY(s, ray.xyz);
    corners[j] = receiverPixel(foot);
    let ndc = 2.0 * foot / frame.receiverSize;
    lo = min(lo, ndc);
    hi = max(hi, ndc);
  }
  let doubleArea = cross2(corners[1] - corners[0], corners[2] - corners[0]);
  if (!(0.5 * abs(doubleArea) > 1e-8)) { return out; }
  // CCW winding lets adjacent faces use a complementary half-open edge rule.
  out.a = corners[0];
  out.b = select(corners[2], corners[1], doubleArea > 0.0);
  out.c = select(corners[1], corners[2], doubleArea > 0.0);
  let pixelSize = frame.receiverSize / vec2f(frame.image);
  out.irradiance = ray.w / (0.5 * abs(doubleArea) * pixelSize.x * pixelSize.y);
  let pad = 2.0 / vec2f(frame.image);
  lo = clamp(lo - pad, vec2f(-1), vec2f(1));
  hi = clamp(hi + pad, vec2f(-1), vec2f(1));
  let corner = vec2f(f32(vertex & 1u), f32(vertex >> 1u));
  out.position = vec4f(mix(lo, hi, corner), 0, 1);
  return out;
}

@vertex
fn fullscreenVS(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  let corners = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
  return vec4f(corners[index], 0, 1);
}

override materialLighting: bool = false;
override encodeCanvas: bool = false;

// Near-white lamp in a dark room; a faint neutral fill keeps the model readable.
const lampColor = vec3f(1.0, 0.97, 0.92);
const fillSky = vec3f(0.08);
const fillGround = vec3f(0.025);

fn decodeNormal(pixel: vec2i) -> vec3f {
  return textureLoad(normalAndFloor, pixel, 0).xyz * (255.0 / 127.0) - 128.0 / 127.0;
}

fn insideTarget(pixel: vec2i) -> bool {
  return all(pixel >= vec2i(0)) && all(pixel < vec2i(frame.targetSize));
}

// Footprints smaller than a screen pixel leave sparkle on surfaces. Average a
// 5 x 5 neighbourhood, but only across texels that face the same way.
fn smoothedIrradiance(pixel: vec2i, n: vec3f) -> f32 {
  var sum = 0.0;
  var weights = 0.0;
  for (var y = -2; y <= 2; y++) {
    for (var x = -2; x <= 2; x++) {
      let q = pixel + vec2i(x, y);
      if (!insideTarget(q) || textureLoad(albedo, q, 0).a < 0.5) { continue; }
      let facing = max(dot(decodeNormal(q), n), 0.0);
      let w = exp(-0.25 * f32(x * x + y * y)) * facing * facing * facing * facing;
      sum += w * textureLoad(lighting, q, 0).r;
      weights += w;
    }
  }
  return select(textureLoad(lighting, pixel, 0).r, sum / weights, weights > 0.0);
}

// Filmic curve (Narkowicz ACES fit): bright light rolls off instead of clipping.
fn filmic(x: vec3f) -> vec3f {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0), vec3f(1));
}

fn present(linear: vec3f) -> vec4f {
  if (!encodeCanvas) { return vec4f(linear, 1); }
  let srgb = select(12.92 * linear, 1.055 * pow(linear, vec3f(1.0 / 2.4)) - 0.055, linear > vec3f(0.0031308));
  return vec4f(srgb, 1);
}

@fragment
fn compositeFS(@builtin(position) frag: vec4f) -> @location(0) vec4f {
  let pixel = vec2i(frag.xy);
  let light = textureLoad(lighting, pixel, 0).rg;
  if (frame.mode == 1u) {
    if (light.g == 0.0) { return vec4f(0, 0, 0, 1); }
    let blue = vec3f(0.13, 0.6, 0.93);
    let cyan = vec3f(0.1, 0.95, 0.95);
    let green = vec3f(0.25, 0.85, 0.35);
    let yellow = vec3f(1.0, 0.82, 0.1);
    let red = vec3f(1.0, 0.2, 0.15);
    let level = log2(light.g);
    var color = mix(blue, cyan, clamp(level, 0.0, 1.0));
    color = mix(color, green, clamp(level - 1.0, 0.0, 1.0));
    color = mix(color, yellow, clamp(0.5 * (level - 2.0), 0.0, 1.0));
    color = mix(color, red, clamp(0.5 * (level - 4.0), 0.0, 1.0));
    return vec4f(color, 1);
  }
  if (frame.mode == 2u) { return displayColor(light.r, frame.exposure); }

  let uv = frag.xy / frame.targetSize;

  let surface = textureLoad(albedo, pixel, 0);
  if (surface.a < 0.5) {
    // Dark neutral backdrop, slightly lighter at the top, so the light stands out.
    return present(mix(vec3f(0.018), vec3f(0.004), uv.y));
  }

  var base = surface.rgb;
  if (!materialLighting) { base *= vec3f(1.0, 0.78, 0.38); }
  let n = decodeNormal(pixel);
  // The lamp shines along +Z, so surfaces facing -Z look up towards it.
  let fill = mix(fillGround, fillSky, 0.5 - 0.5 * n.z);
  let irradiance = smoothedIrradiance(pixel, n);
  let radiance = base * (fill + lampColor * (frame.exposure * irradiance / 3.14159265));
  return present(filmic(0.9 * radiance));
}
