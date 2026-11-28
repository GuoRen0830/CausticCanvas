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

// Filmic curve (Narkowicz ACES fit): bright light rolls off instead of clipping.
fn filmic(radiance: vec3f) -> vec3f {
  // Even a saturated half-float caustic must roll off to white, including on
  // devices without float32 blending. The fitted curve already clips above 8.
  let x = min(radiance, vec3f(16.0));
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0), vec3f(1));
}

fn present(linear: vec3f) -> vec4f {
  if (!encodeCanvas) { return vec4f(linear, 1); }
  let srgb = select(12.92 * linear, 1.055 * pow(linear, vec3f(1.0 / 2.4)) - 0.055, linear > vec3f(0.0031308));
  return vec4f(srgb, 1);
}

fn unpackEmission(word: u32) -> vec3f {
  return unpack4x8unorm(word).rgb * exp2(f32(word >> 24u) - 128.0);
}

fn backdrop(uv: vec2f) -> f32 {
  return mix(0.018, 0.004, uv.y);
}

// The existing backdrop is authored in display-linear space. Invert its
// filmic curve before adding scene-linear glass and scattered light.
fn backdropRadiance(tone: f32) -> f32 {
  let a = 2.51 - 2.43 * tone;
  let b = 0.03 - 0.59 * tone;
  return (sqrt(b * b + 0.56 * a * tone) - b) / (1.8 * a);
}

fn environmentPbr(base: vec3f, n: vec3f, v: vec3f, roughness: f32, metallic: f32, occlusion: f32) -> vec3f {
  let f0 = mix(vec3f(0.04), base, metallic);
  let nv = clamp(dot(n,v), 0.0, 1.0);
  // Reflect the existing adjustable hemispheric fill into the material.
  // Broader lobes approach the hemisphere average, including on metal.
  let sky = clamp(0.5 - 0.5*n.z, 0.0, 1.0);
  let reflectedSky = clamp(0.5 - 0.5*reflect(-v,n).z, 0.0, 1.0);
  let envF = f0 + (max(vec3f(1.0-roughness),f0)-f0) * pow(1.0-nv, 5.0);
  return frame.environmentLight * occlusion * ((1.0-metallic)*(vec3f(1)-envF)*base*mix(0.35,1.0,sky)
    + envF * mix(0.35,1.0,mix(reflectedSky,0.5,roughness*roughness)));
}

fn withLightPaths(base: vec3f, fog: vec3f, overlay: vec4f) -> vec3f {
  // Roll off the concentrated airborne light independently of the surface.
  let peak = max(fog.r, max(fog.g, fog.b));
  let air = fog / (1.0 + peak / 0.34);
  return (base + lampColor * air) * (1.0 - overlay.a) + overlay.rgb;
}

// Spatial dither within one sRGB code value, stable in final-image coordinates.
fn airDither(pixel: vec2f, peak: f32) -> f32 {
  let noise = fract(52.9829189 * fract(dot(pixel, vec2f(0.06711056, 0.00583715)))) - 0.5;
  return noise * smoothstep(0.0, 0.002, peak) / 255.0;
}
