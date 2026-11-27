fn displayColorRGB(irradiance: vec3f, exposure: f32) -> vec4f {
  let scaled = max(irradiance, vec3f(0)) * exposure;
  // One exposure curve preserves the hue of bright mixed-color strokes.
  let linear = scaled / (1.0 + max(scaled.r, max(scaled.g, scaled.b)));
  let encoded = select(12.92 * linear, 1.055 * pow(linear, vec3f(1.0 / 2.4)) - 0.055, linear > vec3f(0.0031308));
  return vec4f(encoded, 1.0);
}

fn displayColor(irradiance: f32, exposure: f32) -> vec4f {
  return displayColorRGB(vec3f(irradiance), exposure);
}
