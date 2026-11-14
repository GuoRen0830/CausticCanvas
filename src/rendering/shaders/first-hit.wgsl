struct Node {
  lo: vec3f, first: u32,
  hi: vec3f, count: u32,
  escape: u32,
}
struct Triangle {
  a: vec3f, objectId: u32,
  b: vec3f, originalId: u32,
  c: vec3f, pad: u32,
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

// Project along the ray's dominant axis. The ray becomes the 2D origin.
struct RayAxes { x: u32, y: u32, z: u32, shear: vec3f }

fn rayAxes(d: vec3f) -> RayAxes {
  let v = abs(d);
  var z = select(0u, 1u, v.y > v.x);
  z = select(z, 2u, v.z > v[z]);
  let x = (z + 1u) % 3u;
  let y = (z + 2u) % 3u;
  return RayAxes(x, y, z, vec3f(d[x]/d[z], d[y]/d[z], 1.0/d[z]));
}

fn raySpace(p: vec3f, axes: RayAxes) -> vec3f {
  return vec3f(vec2f(p[axes.x], p[axes.y]) - axes.shear.xy * p[axes.z], p[axes.z] * axes.shear.z);
}

// Use identical arithmetic on both sides of a shared edge, even when the
// compiler contracts multiply/subtract. Reverse only the final sign.
fn orientedEdge(p: vec2f, q: vec2f) -> f32 {
  let ordered = p.x < q.x || (p.x == q.x && p.y <= q.y);
  let a = select(q, p, ordered);
  let b = select(p, q, ordered);
  let value = a.x * b.y - a.y * b.x;
  return select(-value, value, ordered);
}

fn triangleDistance(tri: Triangle, o: vec3f, axes: RayAxes) -> f32 {
  let a = raySpace(tri.a - o, axes);
  let b = raySpace(tri.b - o, axes);
  let c = raySpace(tri.c - o, axes);
  let e = vec3f(orientedEdge(b.xy,c.xy), orientedEdge(c.xy,a.xy), orientedEdge(a.xy,b.xy));
  if (any(e < vec3f(0)) && any(e > vec3f(0))) { return -1.0; }
  let det = e.x + e.y + e.z;
  if (det == 0.0) { return -1.0; }
  return dot(e, vec3f(a.z,b.z,c.z)) / det;
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
    let axes = rayAxes(ray.xyz);
    var closest = 3.4e38;
    var nodeId = 0u;
    loop {
      if (nodeId >= arrayLength(&nodes)) { break; }
      let node = nodes[nodeId];
      if (!intersectsNode(node, o, ray.xyz, closest)) { nodeId = node.escape; continue; }
      for (var j = node.first; j < node.first + node.count; j++) {
        let t = triangleDistance(triangles[j], o, axes);
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
