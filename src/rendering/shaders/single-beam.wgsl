struct Beam {
  vp: mat4x4<f32>,
  inverseVP: mat4x4<f32>,
  a: vec3<f32>, flux: f32,
  b: vec3<f32>, lengthMeters: f32,
  c: vec3<f32>, mode: f32,
  direction: vec3<f32>,
}
@group(0) @binding(0) var<uniform> beam: Beam;
@group(0) @binding(1) var sceneDepth: texture_depth_2d;
@group(0) @binding(2) var normalAndFloor: texture_2d<f32>;
@group(0) @binding(3) var albedo: texture_2d<f32>;
@group(0) @binding(4) var lighting: texture_2d<f32>;

@vertex
fn boundsVS(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
  let triangle = array(beam.a, beam.b, beam.c);
  var lo = vec2f(1e30);
  var hi = vec2f(-1e30);
  for (var i = 0u; i < 6u; i++) {
    let end = f32(i / 3u) * beam.lengthMeters;
    let clip = beam.vp * vec4f(triangle[i % 3u] + end * beam.direction, 1.0);
    if (clip.z <= 0.0 || clip.w <= 0.0) {
      // Clipping an edge can create new extrema. A full-screen fallback is conservative.
      lo = vec2f(-1.0);
      hi = vec2f(1.0);
      break;
    }
    let ndc = clip.xy / clip.w;
    lo = min(lo, ndc);
    hi = max(hi, ndc);
  }
  let onePixel = 2.0 / vec2f(textureDimensions(sceneDepth));
  lo = clamp(lo - onePixel, vec2f(-1.0), vec2f(1.0));
  hi = clamp(hi + onePixel, vec2f(-1.0), vec2f(1.0));
  let corner = vec2f(f32(index & 1u), f32(index >> 1u));
  return vec4f(mix(lo, hi, corner), 0.0, 1.0);
}

fn decodeNormal(encoded: vec3f) -> vec3f {
  return normalize((encoded * 255.0 - 128.0) / 127.0);
}

// Return irradiance in W/m². All vectors and distances here are in world space.
fn floorIrradiance(p: vec3f, n: vec3f) -> f32 {
  let ab = beam.b - beam.a;
  let ac = beam.c - beam.a;
  let areaVector = cross(ab, ac);
  let projectedDoubleArea = dot(areaVector, beam.direction);
  if (abs(projectedDoubleArea) < 1e-10) { return 0.0; }

  // Intersect P - t*d with the source plane; reject before/after the finite prism.
  let t = dot(areaVector, p - beam.a) / projectedDoubleArea;
  if (t < 0.0 || t > beam.lengthMeters) { return 0.0; }
  let aq = p - t * beam.direction - beam.a;
  let areaSquared = dot(areaVector, areaVector);
  let u = dot(cross(aq, ac), areaVector) / areaSquared;
  let v = dot(cross(ab, aq), areaVector) / areaSquared;
  if (u < 0.0 || v < 0.0 || u + v > 1.0) { return 0.0; }

  let perpendicularArea = 0.5 * abs(projectedDoubleArea);
  return beam.flux / perpendicularArea * max(dot(n, -beam.direction), 0.0);
}

@fragment
fn accumulateFS(@builtin(position) frag: vec4f) -> @location(0) vec4f {
  let pixel = vec2i(frag.xy);
  let surface = textureLoad(normalAndFloor, pixel, 0);
  var irradiance = 0.0;
  if (surface.a > 0.5) {
    let uv = frag.xy / vec2f(textureDimensions(sceneDepth));
    let ndcXY = uv * vec2f(2.0, -2.0) + vec2f(-1.0, 1.0);
    let depth = textureLoad(sceneDepth, pixel, 0);
    let h = beam.inverseVP * vec4f(ndcXY, depth, 1.0);
    irradiance = floorIrradiance(h.xyz / h.w, decodeNormal(surface.xyz));
  }
  // R carries physical illumination. G counts rasterized candidates, even on a miss.
  return vec4f(irradiance, 1.0, 0.0, 0.0);
}

@vertex
fn fullscreenVS(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  let corners = array(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(corners[index], 0.0, 1.0);
}

@fragment
fn compositeFS(@builtin(position) frag: vec4f) -> @location(0) vec4f {
  let pixel = vec2i(frag.xy);
  let light = textureLoad(lighting, pixel, 0).rg;
  if (beam.mode == 1.0) {
    return vec4f(vec3f(0.13, 0.6, 0.93) * light.g, 1.0);
  }
  if (beam.mode == 2.0) {
    return vec4f(select(vec3f(0.0), vec3f(1.0, 0.8, 0.25), light.r > 0.0), 1.0);
  }
  let base = textureLoad(albedo, pixel, 0);
  if (base.a == 0.0) { return vec4f(0.03, 0.18, 0.22, 1.0); }
  let n = decodeNormal(textureLoad(normalAndFloor, pixel, 0).xyz);
  let inspection = 0.25 + 0.75 * max(dot(n, normalize(vec3f(-0.4, 0.8, 0.6))), 0.0);
  // Use a warm display tint for the test beam. The accumulation texture stays scalar.
  let beamDiffuse = vec3f(1.0, 0.78, 0.38) * (light.r / 3.14159265);
  return vec4f(base.rgb * (vec3f(inspection) + beamDiffuse), 1.0);
}
