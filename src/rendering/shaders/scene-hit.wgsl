struct Triangle {
  a: vec3f, objectId: u32,
  b: vec3f, originalId: u32,
  c: vec3f, doubleSided: u32,
}
fn receivingNormal(tri: Triangle, direction: vec3f) -> vec3f {
  let n = normalize(cross(tri.b - tri.a, tri.c - tri.a));
  return select(n, -n, tri.doubleSided != 0u && dot(n, direction) > 0.0);
}
// 8 bytes per source face. Zero trianglePlusOne denotes a miss.
struct Hit { trianglePlusOne: u32, t: f32 }
