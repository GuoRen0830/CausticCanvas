override materialLighting: bool = true;

struct ExportSplat {
  @builtin(position) position: vec4f,
  @location(0) @interpolate(flat) a: vec2f,
  @location(1) @interpolate(flat) b: vec2f,
  @location(2) @interpolate(flat) c: vec2f,
  @location(3) @interpolate(flat) d: vec2f,
  @location(4) @interpolate(flat) hitPlane: vec4f,
  @location(5) @interpolate(flat) sourcePlane: vec4f,
  @location(6) @interpolate(flat) directionAndTinyArea: vec4f,
  @location(7) @interpolate(flat) fluxDensity: vec3f,
  @location(8) @interpolate(flat) receiver: vec2u,
}

@vertex
fn exportVertex(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> ExportSplat {
  var out: ExportSplat;
  out.position = vec4f(2, 2, 0, 1);
  let face = instance / 4u;
  let child = instance % 4u;
  let hits = hitIds[face];
  let merged = (hits.x & 0x80000000u) != 0u;
  if (merged && child != 0u) { return out; }
  let hitId = hits[child] & 0x7fffffffu;
  if (hitId == 0u) { return out; }

  let ray = beams[face];
  let parent = sourcePatch(face, 4u);
  let area = perpendicularArea(parent, ray.xyz);
  if (!(area > 0.0)) { return out; }
  var source = parent;
  if (!merged) { source = sourcePatch(face, child); }
  var transmission = vec3f(1);
  if (hasApertureColor()) {
    let center = (source[0] + source[1] + source[2]) / 3.0;
    let texel = apertureTexel(center.xy, frame.aperture, textureDimensions(apertureFilter));
    transmission = textureLoad(apertureFilter, texel, 0).rgb;
    if (all(transmission <= vec3f(0))) { return out; }
  }
  // Equal-area children each carry one quarter of the parent flux. Dividing
  // parent flux by parent area already gives their common perpendicular density.
  out.fluxDensity = transmission * (ray.w / area);
  let tri = triangles[hitId - 1u];
  let n = receivingNormal(tri, ray.xyz);
  let incidence = -dot(n, ray.xyz);
  if (!(incidence > 1e-7)) { return out; }
  out.receiver = vec2u(tri.objectId, hitId - 1u);
  out.hitPlane = vec4f(n, -dot(n, tri.a));
  let sourceNormal = normalize(cross(parent[1]-parent[0], parent[2]-parent[0]));
  out.sourcePlane = vec4f(sourceNormal, -dot(sourceNormal, parent[0]));
  out.directionAndTinyArea = vec4f(ray.xyz, 0.0);
  var projected: array<vec4f, 3>;
  for (var i = 0u; i < 3u; i++) {
    let t = dot(n, source[i] - tri.a) / incidence;
    projected[i] = frame.localVP * vec4f(source[i] + t * ray.xyz, 1.0);
  }

  // Clip before dividing by w, including beams seen from inside their volume.
  var clipped: array<vec4f, 4>;
  var count = 0u;
  var prev = projected[2];
  for (var i = 0u; i < 3u; i++) {
    let p = projected[i];
    if ((prev.z < 0.0) != (p.z < 0.0)) {
      clipped[count] = mix(prev, p, prev.z / (prev.z - p.z));
      count++;
    }
    if (p.z >= 0.0) { clipped[count] = p; count++; }
    prev = p;
  }
  if (count < 3u) { return out; }
  var screen: array<vec2f, 4>;
  var lo = vec2f(1e30);
  var hi = -lo;
  for (var i = 0u; i < count; i++) {
    let p = clipped[i];
    screen[i] = (p.xy / p.w * vec2f(0.5,-0.5) + 0.5) * frame.targetSize;
    lo = min(lo, screen[i]);
    hi = max(hi, screen[i]);
  }
  out.a = screen[0];
  out.b = screen[1];
  out.c = screen[2];
  out.d = select(screen[2], screen[3], count == 4u);
  if (count == 3u && all(floor(lo) == floor(hi))) {
    out.directionAndTinyArea.w = 0.5 * abs(cross2(out.b - out.a, out.c - out.a));
  }
  let corner = vec2f(f32(vertex & 1u), f32(vertex >> 1u));
  let uv = mix(floor(lo), ceil(hi), corner) / frame.targetSize;
  out.position = vec4f(uv * vec2f(2,-2) + vec2f(-1,1), 0, 1);
  return out;
}

// Integrate both coverage and its first moment. The shadow segment should
// start on the illuminated part of a pixel, even for a sliver at its edge.
fn coveredSample(input: ExportSplat) -> vec3f {
  let origin = floor(input.position.xy);
  if (input.directionAndTinyArea.w > 0.0) {
    return vec3f(((input.a - origin) + (input.b - origin) + (input.c - origin)) / 3.0,
                 input.directionAndTinyArea.w);
  }
  var poly: Polygon;
  poly.points[0] = input.a - origin;
  poly.points[1] = input.b - origin;
  poly.points[2] = input.c - origin;
  poly.points[3] = input.d - origin;
  poly.count = select(3u, 4u, any(input.c != input.d));
  poly = clipEdge(poly, 0u, 0.0, true);
  poly = clipEdge(poly, 0u, 1.0, false);
  poly = clipEdge(poly, 1u, 0.0, true);
  poly = clipEdge(poly, 1u, 1.0, false);
  var twiceArea = 0.0;
  var moment = vec2f(0);
  let a = poly.points[0];
  for (var i = 1u; i + 1u < poly.count; i++) {
    let b = poly.points[i];
    let c = poly.points[i+1u];
    let weight = abs(cross2(b-a, c-a));
    twiceArea += weight;
    moment += weight * (a + b + c);
  }
  if (!(twiceArea > 0.0)) { return vec3f(0); }
  return vec3f(moment / (3.0 * twiceArea), min(1.0, 0.5 * twiceArea));
}

fn shadowed(p: vec3f, l: vec3f, distanceToLens: f32, receiver: u32) -> bool {
  // Roundoff is proportional to coordinate magnitude. Skip the receiver
  // itself and use a finite upper bound so objects behind the lens never mask it.
  let magnitude = max(max(abs(p.x), abs(p.y)), max(abs(p.z), distanceToLens));
  let bias = max(1e-7, 8.0 * 1.192093e-7 * magnitude);
  let limit = distanceToLens - bias;
  let axes = rayAxes(l);
  var id = 0u;
  loop {
    if (id >= arrayLength(&nodes)) { break; }
    let node = nodes[id];
    if (!intersectsNode(node, p, l, limit)) { id = node.escape; continue; }
    for (var j = node.first; j < node.first + node.count; j++) {
      if (j == receiver) { continue; }
      let t = triangleDistance(triangles[j], p, axes);
      if (t > bias && t < limit) { return true; }
    }
    id++;
  }
  return false;
}

fn directBrdf(base: vec3f, n: vec3f, l: vec3f, v: vec3f, roughness: f32, metallic: f32) -> vec3f {
  let nv = clamp(dot(n, v), 0.0, 1.0);
  let nl = clamp(dot(n, l), 0.0, 1.0);
  if (nv <= 0.0 || nl <= 0.0) { return vec3f(0); }
  let h = normalize(l + v);
  let nh = clamp(dot(n, h), 0.0, 1.0);
  let vh = clamp(dot(v, h), 0.0, 1.0);
  let f0 = mix(vec3f(0.04), base, metallic);
  let fresnel = f0 + (vec3f(1)-f0) * pow(1.0-vh, 5.0);
  // Integrate the actual angular distribution one beam at a time. The
  // numerical floor keeps a perfectly polished microfacet peak representable.
  let alpha = max(roughness*roughness, 0.002);
  let a2 = alpha*alpha;
  let denom = (1.0 - nh*nh) + nh*nh*a2;
  let distribution = a2 / (3.14159265 * denom * denom);
  let lambdaV = nl * sqrt(nv*nv*(1.0-a2) + a2);
  let lambdaL = nv * sqrt(nl*nl*(1.0-a2) + a2);
  let visibility = 0.5 / max(lambdaV + lambdaL, 1e-10);
  let diffuse = (1.0-metallic) * (vec3f(1)-fresnel) * base / 3.14159265;
  return nl * (diffuse + fresnel * distribution * visibility);
}

fn exportRadiance(input: ExportSplat) -> vec4f {
  let pixel = vec2i(input.position.xy);
  let visible = textureLoad(sceneObject, pixel, 0);
  if (visible.r != input.receiver.x) { discard; }
  let ng = normalize(unpack4x8snorm(visible.g).xyz);
  let l = -input.directionAndTinyArea.xyz;
  let ngl = dot(ng,l);
  if (!(dot(ng,input.hitPlane.xyz) >= 0.85 && ngl > 0.0)) { discard; }
  let covered = coveredSample(input);
  if (!(covered.z > 0.0)) { discard; }

  let inverse = frame.inverseLocalVP;
  let pixelToNdc = vec2f(2,-2) / frame.targetSize;
  let ndc = input.position.xy * pixelToNdc + vec2f(-1,1);
  let h = inverse * vec4f(ndc, textureLoad(sceneDepth, pixel, 0), 1.0);
  let p = h.xyz/h.w;
  let hx = h + inverse[0] * pixelToNdc.x;
  let hy = h + inverse[1] * pixelToNdc.y;
  let pixelMeters = max(length(hx.xyz/hx.w-p), length(hy.xyz/hy.w-p));
  if (abs(dot(input.hitPlane, vec4f(p,1))) > 2e-5 + 0.25 * pixelMeters) { discard; }

  // Intersect the ray through the coverage centroid with the G-buffer
  // surface's tangent plane. This works for off-axis and orthographic tiles.
  let sampleNdc = ndc + (covered.xy - 0.5) * pixelToNdc;
  let near = inverse * vec4f(sampleNdc, 0, 1);
  let depthColumn = inverse[2];
  let v = normalize(near.xyz * depthColumn.w - depthColumn.xyz * near.w);
  let delta = (covered.xy - 0.5) * pixelToNdc;
  let hc = h + delta.x * inverse[0] + delta.y * inverse[1];
  let q = hc.xyz / hc.w;
  let nv = dot(ng, v);
  var samplePoint = p;
  if (abs(nv) > 1e-5) { samplePoint = q + v * (dot(ng,p-q) / nv); }
  let distanceToLens = -dot(input.sourcePlane, vec4f(samplePoint,1)) / dot(input.sourcePlane.xyz,l);
  if (!(distanceToLens > 0.0)) { discard; }
  if (shadowed(samplePoint, l, distanceToLens, input.receiver.y)) { discard; }

  let base = litBaseColor(textureLoad(albedo, pixel, 0).rgb, materialLighting);
  let physical = unpack4x8unorm(textureLoad(sceneMaterial, pixel, 0).r);
  var response = base * (ngl / 3.14159265);
  if (physical.b > 0.5) {
    let n = normalize((textureLoad(normalAndFloor, pixel, 0).xyz * 255.0 - 128.0) / 127.0);
    response = directBrdf(base, n, l, v, physical.r, physical.g);
  }
  return vec4f(covered.z * input.fluxDensity * vec3f(1.0,0.97,0.92) * response, 0.0);
}
