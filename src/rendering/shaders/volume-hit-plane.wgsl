@group(0) @binding(4) var<storage, read> hits: array<Hit>;

fn volumeHitDistance(origin: vec3f, direction: vec3f, centroid: vec3f, face: u32) -> f32 {
  let hit = hits[face];
  if (hit.trianglePlusOne == 0u) { return 1e20; }
  // Move the preview's cached centroid hit onto the sample's parallel ray.
  let tri = triangles[hit.trianglePlusOne - 1u];
  let n = cross(tri.b - tri.a, tri.c - tri.a);
  let den = dot(n, direction);
  if (abs(den) > 1e-6 * length(n)) { return hit.t + dot(n, centroid - origin) / den; }
  return hit.t;
}
