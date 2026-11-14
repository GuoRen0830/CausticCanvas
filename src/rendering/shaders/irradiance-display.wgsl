fn displayColor(irradiance: f32, exposure: f32) -> vec4f {
  let scaled = max(irradiance, 0.0) * exposure;
  let mapped = scaled / (1.0 + scaled);
  let encoded = select(12.92 * mapped, 1.055 * pow(mapped, 1.0 / 2.4) - 0.055, mapped > 0.0031308);
  return vec4f(vec3f(encoded), 1.0);
}
