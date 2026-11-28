// Each child is a parallel sub-beam carrying one quarter of the face flux.
// Index 4 is the original triangle, used when all children have the same receiver.
fn sourcePatch(face: u32, child: u32) -> array<vec3f, 3> {
  let ids = sourceIds(face);
  let a = sourceVertex(ids.x);
  let b = sourceVertex(ids.y);
  let c = sourceVertex(ids.z);
  let ab = 0.5 * (a + b);
  let ac = 0.5 * (a + c);
  let bc = 0.5 * (b + c);
  switch child {
    case 0u: { return array(a, ab, ac); }
    case 1u: { return array(ab, b, bc); }
    case 2u: { return array(ac, bc, c); }
    case 3u: { return array(ab, bc, ac); }
    default: { return array(a, b, c); }
  }
}

fn perpendicularArea(vertices: array<vec3f, 3>, direction: vec3f) -> f32 {
  return 0.5 * abs(dot(cross(vertices[1] - vertices[0], vertices[2] - vertices[0]), direction));
}
