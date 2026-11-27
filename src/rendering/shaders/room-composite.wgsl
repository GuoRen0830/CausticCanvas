// R is the half-resolution view integral; G is ray distance in optical metres.
@group(1) @binding(3) var fogImage: texture_2d<f32>;
@group(1) @binding(4) var pathOverlay: texture_2d<f32>;

fn roomFog(uv: vec2f, depth: f32) -> f32 {
  var distance = 60000.0; // same background marker as the half-resolution pass
  if (depth < 1.0) {
    let ndc = uv * vec2f(2, -2) + vec2f(-1, 1);
    let near = frame.inverseLocalVP * vec4f(ndc, 0, 1);
    let hit = near + frame.inverseLocalVP[2] * depth;
    distance = length(hit.xyz / hit.w - near.xyz / near.w);
  }
  let size = vec2i(textureDimensions(fogImage));
  let p = uv * vec2f(size) - 0.5;
  let cell = vec2i(floor(p));
  let f = fract(p);
  let tolerance = max(0.001, distance * 0.015);
  var sum = 0.0;
  var weights = 0.0;
  for (var y = 0; y <= 1; y++) {
    for (var x = 0; x <= 1; x++) {
      let q = clamp(cell + vec2i(x, y), vec2i(0), size - 1);
      let fog = textureLoad(fogImage, q, 0).rg;
      let w = mix(1.0 - f.x, f.x, f32(x)) * mix(1.0 - f.y, f.y, f32(y));
      // Reject discontinuous depths so light behind a thin foreground edge
      // does not bleed onto it, even if all four coarse samples miss the edge.
      let depthWeight = 1.0 - smoothstep(tolerance, 4.0 * tolerance, abs(distance - fog.g));
      sum += w * depthWeight * fog.r;
      weights += w * depthWeight;
    }
  }
  return sum / max(weights, 1e-5);
}

// The existing backdrop is authored in display-linear space. Invert its
// filmic curve before adding scene-linear glass and scattered light.
fn backdropRadiance(tone: f32) -> f32 {
  let a = 2.51 - 2.43 * tone;
  let b = 0.03 - 0.59 * tone;
  return (sqrt(b * b + 0.56 * a * tone) - b) / (1.8 * a);
}

@fragment
fn roomPathsFS(@builtin(position) frag: vec4f) -> @location(0) vec4f {
  let pixel = vec2i(frag.xy);
  let uv = frag.xy / frame.targetSize;
  let fog = roomFog(uv, textureLoad(sceneDepth, pixel, 0));
  let overlay = textureLoad(pathOverlay, pixel, 0);
  let surface = textureLoad(albedo, pixel, 0);
  var base: vec3f;
  if (surface.a >= 0.5) {
    base = surfaceRadiance(pixel, surface);
  } else {
    let sky = backdrop(uv);
    if (fog == 0.0 && all(overlay == vec4f(0))) { return present(vec3f(sky)); }
    base = vec3f(backdropRadiance(sky));
  }
  // The tiny exit concentrates a lot of flux. Roll off only the airborne
  // highlights so the glass stays translucent and the painted surface can peak white.
  let air = fog / (1.0 + fog / 0.34);
  let radiance = (base + lampColor * air) * (1.0 - overlay.a) + overlay.rgb;
  let color = present(filmic(0.9 * radiance));
  if (encodeCanvas && air > 0.0) {
    // Static, zero-centred dither below one sRGB code value removes display
    // contours in the faint air gradients without flicker or a history buffer.
    let noise = fract(52.9829189 * fract(dot(frag.xy, vec2f(0.06711056, 0.00583715)))) - 0.5;
    return vec4f(color.rgb + noise * smoothstep(0.0, 0.002, air) / 255.0, color.a);
  }
  return color;
}
