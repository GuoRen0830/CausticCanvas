// The entrance filter is stationary in physical XY, including when lens vertices slide.
// Its first row lies on the optical -Y edge, matching the design raster.
fn apertureTexel(xy: vec2f, span: vec2f, size: vec2u) -> vec2i {
  let uv = clamp(xy / span + 0.5, vec2f(0), vec2f(1));
  return vec2i(min(vec2u(uv * vec2f(size)), size - 1u));
}
