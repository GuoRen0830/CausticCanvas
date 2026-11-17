struct Triangle {
  a: vec3f, objectId: u32,
  b: vec3f, originalId: u32,
  c: vec3f, pad: u32,
}
// 8 bytes per source face. Zero trianglePlusOne denotes a miss.
struct Hit { trianglePlusOne: u32, t: f32 }
