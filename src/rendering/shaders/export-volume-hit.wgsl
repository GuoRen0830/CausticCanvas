@group(0) @binding(4) var<storage, read> nodes: array<Node>;

fn volumeHitDistance(origin: vec3f, direction: vec3f, centroid: vec3f, face: u32) -> f32 {
  // A sample can reach a narrow foreground object missed by its face centroid.
  let hit = closestHit(origin, direction);
  return select(1e20, hit.t, hit.trianglePlusOne != 0u);
}
