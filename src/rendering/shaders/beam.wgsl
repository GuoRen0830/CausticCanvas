@vertex
fn fullscreenVS(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  let corners = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
  return vec4f(corners[index], 0, 1);
}

override materialLighting: bool = false;
override encodeCanvas: bool = false;

// Near-white lamp with a neutral hemispheric fill configured independently of the lamp.
const lampColor = vec3f(1.0, 0.97, 0.92);

fn decodeNormal(pixel: vec2i) -> vec3f {
  return textureLoad(normalAndFloor, pixel, 0).xyz * (255.0 / 127.0) - 128.0 / 127.0;
}

fn insideTarget(pixel: vec2i) -> bool {
  return all(pixel >= vec2i(0)) && all(pixel < vec2i(frame.targetSize));
}

// Footprints smaller than a screen pixel leave sparkle on surfaces. Average a
// 5 x 5 neighbourhood, but only across texels that face the same way.
fn smoothedIrradiance(pixel: vec2i, n: vec3f) -> vec3f {
  var sum = vec3f(0);
  var weights = 0.0;
  for (var y = -2; y <= 2; y++) {
    for (var x = -2; x <= 2; x++) {
      let q = pixel + vec2i(x, y);
      if (!insideTarget(q) || textureLoad(albedo, q, 0).a < 0.5) { continue; }
      let facing = max(dot(decodeNormal(q), n), 0.0);
      let w = exp(-0.25 * f32(x * x + y * y)) * facing * facing * facing * facing;
      sum += w * textureLoad(lighting, q, 0).rgb;
      weights += w;
    }
  }
  return select(textureLoad(lighting, pixel, 0).rgb, sum / max(weights, 1e-6), weights > 0.0);
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

fn backdrop(uv: vec2f) -> f32 {
  return mix(0.018, 0.004, uv.y);
}

fn surfaceRadiance(pixel: vec2i, surface: vec4f) -> vec3f {
  var base = surface.rgb;
  if (!materialLighting) { base *= vec3f(1.0, 0.78, 0.38); }
  let n = decodeNormal(pixel);
  let fill = frame.environmentLight * mix(0.35, 1.0, clamp(0.5 - 0.5 * n.z, 0.0, 1.0));
  let irradiance = smoothedIrradiance(pixel, n);
  return base * (fill + lampColor * (frame.exposure * irradiance / 3.14159265));
}

@fragment
fn compositeFS(@builtin(position) frag: vec4f) -> @location(0) vec4f {
  let pixel = vec2i(frag.xy);
  let light = textureLoad(lighting, pixel, 0);
  if (frame.mode == 1u) {
    if (light.a == 0.0) { return vec4f(0, 0, 0, 1); }
    let blue = vec3f(0.13, 0.6, 0.93);
    let cyan = vec3f(0.1, 0.95, 0.95);
    let green = vec3f(0.25, 0.85, 0.35);
    let yellow = vec3f(1.0, 0.82, 0.1);
    let red = vec3f(1.0, 0.2, 0.15);
    let level = log2(light.a);
    var color = mix(blue, cyan, clamp(level, 0.0, 1.0));
    color = mix(color, green, clamp(level - 1.0, 0.0, 1.0));
    color = mix(color, yellow, clamp(0.5 * (level - 2.0), 0.0, 1.0));
    color = mix(color, red, clamp(0.5 * (level - 4.0), 0.0, 1.0));
    return vec4f(color, 1);
  }
  let uv = frag.xy / frame.targetSize;

  let surface = textureLoad(albedo, pixel, 0);
  if (surface.a < 0.5) {
    // Dark neutral backdrop, slightly lighter at the top, so the light stands out.
    return present(vec3f(backdrop(uv)));
  }

  // Read the unfiltered irradiance so this view exposes sampling noise.
  if (frame.mode == 2u) { return irradianceHeat(light.r); }
  if (frame.mode == 3u) {
    let local = normalize(decodeNormal(pixel));
    let world = vec3f(local.x, -local.z, local.y);
    return vec4f(world * 0.5 + 0.5, 1);
  }
  if (frame.mode == 4u) { return present(surface.rgb); }

  return present(filmic(0.9 * surfaceRadiance(pixel, surface)));
}
