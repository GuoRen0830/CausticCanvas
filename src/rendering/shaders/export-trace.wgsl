struct Bounds { lo: vec4f, hi: vec4f }
struct Batch { firstFace: u32 }
@group(2) @binding(2) var<storage, read_write> hitIds: array<vec4u>;
@group(2) @binding(3) var<storage, read_write> blockBounds: array<Bounds>;
@group(2) @binding(4) var<uniform> batch: Batch;
var<workgroup> lower: array<vec3f, 128>;
var<workgroup> upper: array<vec3f, 128>;

@compute @workgroup_size(128)
fn traceChildren(@builtin(local_invocation_index) lane: u32, @builtin(global_invocation_id) gid: vec3u) {
  let face = batch.firstFace + gid.x;
  var lo = vec3f(3.4e38);
  var hi = -lo;
  if (face < arrayLength(&hitIds)) {
    let ray = beams[face];
    var hits = vec4u(0u);
    if (ray.w > 0.0) {
      for (var child = 0u; child < 4u; child++) {
        let vertices = sourcePatch(face, child);
        let center = (vertices[0] + vertices[1] + vertices[2]) / 3.0;
        let hit = closestHit(center, ray.xyz);
        if (hit.trianglePlusOne == 0u) { continue; }
        let tri = triangles[hit.trianglePlusOne - 1u];
        let n = receivingNormal(tri, ray.xyz);
        let cosine = -dot(n, ray.xyz);
        if (cosine <= 0.0) { continue; }
        hits[child] = hit.trianglePlusOne;
        for (var corner = 0u; corner < 3u; corner++) {
          let p = vertices[corner];
          let destination = p + ray.xyz * (dot(n, p - tri.a) / cosine);
          lo = min(lo, destination);
          hi = max(hi, destination);
        }
      }
      if (hits.x != 0u && all(hits == vec4u(hits.x))) {
        var uniformFilter = true;
        if (hasApertureColor()) {
          let p = sourcePatch(face, 4u);
          let size = textureDimensions(apertureFilter);
          let t = apertureTexel(p[0].xy, frame.aperture, size);
          uniformFilter = all(t == apertureTexel(p[1].xy, frame.aperture, size)) &&
                          all(t == apertureTexel(p[2].xy, frame.aperture, size));
        }
        if (uniformFilter) { hits.x |= 0x80000000u; }
      }
    }
    hitIds[face] = hits;
  }
  // Bounds across a consecutive group of source faces make tile culling conservative.
  lower[lane] = lo;
  upper[lane] = hi;
  workgroupBarrier();
  for (var stride = 64u; stride > 0u; stride >>= 1u) {
    if (lane < stride) {
      lower[lane] = min(lower[lane], lower[lane + stride]);
      upper[lane] = max(upper[lane], upper[lane + stride]);
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    blockBounds[face / 128u] = Bounds(vec4f(lower[0], 0.0), vec4f(upper[0], 0.0));
  }
}
