@group(0) @binding(5) var sceneObject: texture_2d<u32>;
@group(2) @binding(3) var<storage, read> cachedHits: array<Hit>;

struct Splat {
  @builtin(position) position: vec4f,
  @location(0) @interpolate(flat) a: vec2f,
  @location(1) @interpolate(flat) b: vec2f,
  @location(2) @interpolate(flat) c: vec2f,
  @location(3) @interpolate(flat) plane: vec4f,
  @location(4) @interpolate(flat) irradiance: f32,
  @location(5) @interpolate(flat) objectId: u32,
  @location(6) @interpolate(flat) tinyArea: f32,
  @location(7) @interpolate(flat) d: vec2f,
}

@vertex
fn splatVS(@builtin(vertex_index) vertex: u32, @builtin(instance_index) face: u32) -> Splat {
  var out: Splat;
  out.position = vec4f(2, 2, 0, 1);
  let hit = cachedHits[face];
  if (hit.trianglePlusOne == 0u) { return out; }
  let tri = triangles[hit.trianglePlusOne - 1u];
  let n = normalize(cross(tri.ab, tri.ac));
  let ray = beams[face];
  let incidence = -dot(n, ray.xyz);
  if (!(incidence > 1e-7)) { return out; }
  let ids = sourceIds(face);
  let source = array(sourceVertex(ids.x), sourceVertex(ids.y), sourceVertex(ids.z));
  let centroid = (source[0] + source[1] + source[2]) / 3.0;
  let center = centroid + hit.t * ray.xyz;
  let perpArea = 0.5 * abs(dot(cross(source[1]-source[0], source[2]-source[0]), ray.xyz));
  if (!(perpArea > 0.0)) { return out; }
  out.irradiance = ray.w * incidence / perpArea;
  out.objectId = tri.objectId;
  out.plane = vec4f(n, -dot(n, tri.a));

  var projected: array<vec4f, 3>;
  for (var j = 0u; j < 3u; j++) {
    let t = dot(n, source[j] - center) / incidence;
    projected[j] = frame.localVP * vec4f(source[j] + t * ray.xyz, 1.0);
  }
  // Clip against the WebGPU near plane before dividing by w. A straddling
  // triangle produces at most four vertices, integrated as a pair of triangles.
  var clipped: array<vec4f, 4>;
  var count = 0u;
  var prev = projected[2];
  for (var j = 0u; j < 3u; j++) {
    let curr = projected[j];
    if ((prev.z < 0.0) != (curr.z < 0.0)) {
      clipped[count] = mix(prev, curr, prev.z / (prev.z - curr.z));
      count++;
    }
    if (curr.z >= 0.0) { clipped[count] = curr; count++; }
    prev = curr;
  }
  if (count < 3u) { return out; }
  // A single rectangle bounds the whole clipped polygon. Store its fourth
  // corner separately only on the rare clipped-quad path (see Splat.d).
  var screen: array<vec2f, 4>;
  var lo = vec2f(1e30);
  var hi = -lo;
  for (var j = 0u; j < count; j++) {
    let clip = clipped[j];
    screen[j] = (clip.xy / clip.w * vec2f(0.5, -0.5) + 0.5) * frame.targetSize;
    lo = min(lo, screen[j]); hi = max(hi, screen[j]);
  }
  out.a = screen[0]; out.b = screen[1]; out.c = screen[2];
  out.d = select(screen[2], screen[3], count == 4u);
  // Subpixel triangles inside one pixel have analytically known coverage.
  if (count == 3u && all(floor(lo) == floor(hi))) {
    out.tinyArea = 0.5 * abs(cross2(out.b-out.a, out.c-out.a));
  }
  let uv = mix(floor(lo), ceil(hi), vec2f(f32(vertex & 1u), f32(vertex >> 1u))) / frame.targetSize;
  out.position = vec4f(uv * vec2f(2,-2) + vec2f(-1,1), 0, 1);
  return out;
}

@fragment
fn splatFS(input: Splat) -> @location(0) vec4f {
  if (frame.mode == 1u) { return vec4f(0,1,0,0); }
  let pixel = vec2i(input.position.xy);
  let visible = textureLoad(sceneObject, pixel, 0);
  if (visible.r != input.objectId) { discard; }
  if (frame.mode == 2u && textureLoad(normalAndFloor, pixel, 0).a < 0.5) { discard; }
  let geometric = unpack4x8snorm(visible.g).xyz;
  if (dot(geometric, input.plane.xyz) < 0.85) { discard; }
  let ndc = input.position.xy / frame.targetSize * vec2f(2,-2) + vec2f(-1,1);
  let h = frame.inverseLocalVP * vec4f(ndc, textureLoad(sceneDepth, pixel, 0), 1);
  let p = h.xyz / h.w;
  // Allow depth quantization and gentle curvature within a screen pixel.
  let pixelStep = 2.0 / frame.targetSize;
  let hx = h + frame.inverseLocalVP[0] * pixelStep.x;
  let hy = h + frame.inverseLocalVP[1] * pixelStep.y;
  let pixelMeters = max(length(hx.xyz/hx.w - p), length(hy.xyz/hy.w - p));
  if (abs(dot(input.plane, vec4f(p,1))) > 2e-5 + 0.25 * pixelMeters) { discard; }
  var coverage = input.tinyArea;
  if (coverage == 0.0) {
    let origin = floor(input.position.xy);
    coverage = pixelCoverage(input.a-origin, input.b-origin, input.c-origin);
    if (any(input.d != input.c)) {
      coverage += pixelCoverage(input.a-origin, input.c-origin, input.d-origin);
    }
  }
  return vec4f(input.irradiance * coverage, 1, 0, 0);
}
