@group(2) @binding(0) var<storage, read> triangles: array<Triangle>;
@group(2) @binding(1) var<storage, read> nodes: array<Node>;
@group(2) @binding(2) var<storage, read_write> traceOutput: array<Hit>;

@compute @workgroup_size(64)
fn traceFirstHit(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let face = id.y * (groups.x * 64u) + id.x;
  if (face >= frame.beamCount) { return; }
  let ray = beams[face];
  var hit = Hit(0u, 0.0);
  if (ray.w > 0.0) {
    let ids = sourceIds(face);
    let o = (sourceVertex(ids.x) + sourceVertex(ids.y) + sourceVertex(ids.z)) / 3.0;
    hit = closestHit(o, ray.xyz);
  }
  traceOutput[face] = hit;
}
