fn displayColor(irradiance: f32, exposure: f32) -> vec4f {
  let scaled = max(irradiance, 0.0) * exposure;
  let mapped = scaled / (1.0 + scaled);
  let linear = vec3f(mapped);
  let encoded = select(12.92 * linear, 1.055 * pow(linear, vec3f(1.0 / 2.4)) - 0.055, linear > vec3f(0.0031308));
  return vec4f(encoded, 1.0);
}
