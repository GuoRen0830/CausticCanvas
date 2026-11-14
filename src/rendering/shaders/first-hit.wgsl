struct Node {
  lo: vec3f, first: u32,
  hi: vec3f, count: u32,
  escape: u32,
}
struct Triangle {
  a: vec3f, objectId: u32,
  ab: vec3f, originalId: u32,
  ac: vec3f, pad: u32,
}
// 8 bytes per source face. Zero trianglePlusOne denotes a miss.
struct Hit { trianglePlusOne: u32, t: f32 }
@group(2) @binding(0) var<storage, read> triangles: array<Triangle>;
@group(2) @binding(1) var<storage, read> nodes: array<Node>;
@group(2) @binding(2) var<storage, read_write> traceOutput: array<Hit>;

fn intersectsNode(node: Node, o: vec3f, d: vec3f, closest: f32) -> bool {
  var enter = 0.0;
  var leave = closest;
  for (var axis = 0u; axis < 3u; axis++) {
    if (d[axis] == 0.0) {
      if (o[axis] < node.lo[axis] || o[axis] > node.hi[axis]) { return false; }
    } else {
      let t0 = (node.lo[axis] - o[axis]) / d[axis];
      let t1 = (node.hi[axis] - o[axis]) / d[axis];
      enter = max(enter, min(t0, t1));
      leave = min(leave, max(t0, t1));
    }
  }
  return enter <= leave;
}

// Two-sided intersection: an opaque back face still stops the light.
// An actual positive distance is required, with a tiny bias in optical meters.
fn triangleDistance(tri: Triangle, o: vec3f, d: vec3f) -> f32 {
  let p = cross(d, tri.ac);
  let det = dot(tri.ab, p);
  if (det == 0.0) { return -1.0; }
  let delta = o - tri.a;
  let u = dot(delta, p) / det;
  let q = cross(delta, tri.ab);
  let v = dot(d, q) / det;
  if (u < 0.0 || v < 0.0 || u + v > 1.0) { return -1.0; }
  return dot(tri.ac, q) / det;
}

@compute @workgroup_size(64)
fn traceFirstHit(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let face = id.y * (groups.x * 64u) + id.x;
  if (face >= frame.beamCount) { return; }
  let ray = beams[face];
  var hit = Hit(0u, 0.0);
  if (ray.w > 0.0) {
    let ids = sourceIds(face);
    let o = (sourceVertex(ids.x) + sourceVertex(ids.y) + sourceVertex(ids.z)) / 3.0;
    var closest = 3.4e38;
    var nodeId = 0u;
    loop {
      if (nodeId >= arrayLength(&nodes)) { break; }
      let node = nodes[nodeId];
      if (!intersectsNode(node, o, ray.xyz, closest)) { nodeId = node.escape; continue; }
      for (var j = node.first; j < node.first + node.count; j++) {
        let t = triangleDistance(triangles[j], o, ray.xyz);
        if (t > 1e-7 && t < closest) {
          closest = t;
          hit = Hit(j + 1u, t);
        }
      }
      nodeId++;
    }
  }
  traceOutput[face] = hit;
}
