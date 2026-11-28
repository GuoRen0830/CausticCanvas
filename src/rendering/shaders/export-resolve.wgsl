// The G-buffer and direct radiance are sampled at twice the output resolution.
// Tile coordinates are in final-image pixels, with the origin at the top left.
@group(0) @binding(7) var<uniform> tile: vec4f;
@group(1) @binding(3) var exportFog: texture_2d<f32>;
@group(1) @binding(4) var exportGlass: texture_2d<f32>;

fn exportSample(pixel: vec2i) -> vec3f {
  let surface = textureLoad(albedo, pixel, 0);
  if (surface.a < 0.5) {
    let uv = (tile.xy + (vec2f(pixel) + 0.5) * 0.5) / tile.zw;
    return vec3f(backdropRadiance(backdrop(uv)));
  }
  let n = decodeNormal(pixel);
  let base = litBaseColor(surface.rgb, materialLighting);
  let packed = textureLoad(sceneMaterial, pixel, 0).rg;
  let material = unpack4x8unorm(packed.x);
  var ambient = base * frame.environmentLight * mix(0.35, 1.0, clamp(0.5 - 0.5 * n.z, 0.0, 1.0));
  if (material.b > 0.5) {
    let ndc = (vec2f(pixel) + 0.5) / frame.targetSize * vec2f(2,-2) + vec2f(-1,1);
    let near = frame.inverseLocalVP * vec4f(ndc, 0, 1);
    let dz = frame.inverseLocalVP[2];
    let v = normalize(near.xyz * dz.w - dz.xyz * near.w);
    ambient = environmentPbr(base, normalize(n), v, material.r, material.g, material.a);
  }
  var radiance = ambient + unpackEmission(packed.y);
  if (frame.exposure > 0.0) { radiance += frame.exposure * textureLoad(lighting, pixel, 0).rgb; }
  return radiance;
}

override showPaths: bool = false;

@fragment
fn resolveExport(@builtin(position) frag: vec4f) -> @location(0) vec4f {
  let corner = vec2i(frag.xy) * 2;
  var radiance = vec3f(0);
  var peak = 0.0;
  for (var y = 0; y < 2; y++) {
    for (var x = 0; x < 2; x++) {
      let pixel = corner + vec2i(x, y);
      var sample = exportSample(pixel);
      if (showPaths) {
        let fog = textureLoad(exportFog, pixel, 0).rgb;
        peak = max(peak, max(fog.r, max(fog.g, fog.b)));
        sample = withLightPaths(sample, fog, textureLoad(exportGlass, pixel, 0));
      }
      radiance += sample;
    }
  }
  let color = present(filmic((0.9 * 0.25) * radiance));
  return vec4f(color.rgb + airDither(tile.xy + frag.xy, peak), color.a);
}
