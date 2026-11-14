struct Frame {
  localVP: mat4x4f,
  inverseLocalVP: mat4x4f,
  grid: vec2u,
  aperture: vec2f,
  offset: vec2f,
  receiverSize: vec2f,
  image: vec2u,
  targetSize: vec2f,
  receiverZ: f32,
  mode: u32,
  exposure: f32,
  boxMin: vec3f,
  boxMax: vec3f,
  inspectionDirection: vec3f,
}

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var sceneDepth: texture_depth_2d;
@group(0) @binding(2) var normalAndFloor: texture_2d<f32>;
@group(0) @binding(3) var albedo: texture_2d<f32>;
@group(0) @binding(4) var lighting: texture_2d<f32>;
@group(1) @binding(0) var<storage, read> heights: array<f32>;
@group(1) @binding(1) var<storage, read> beams: array<vec4f>;

fn sourceVertex(id: u32) -> vec3f {
  let gridXY = vec2f(f32(id % frame.grid.x), f32(id / frame.grid.x));
  let xy = (gridXY / vec2f(frame.grid - 1u) - 0.5) * frame.aperture + frame.offset;
  return vec3f(xy, heights[id]);
}

struct Bounds {
  @builtin(position) position: vec4f,
  // Receiver pixel units, with positive Y up. Constant across a beam rectangle.
  @location(0) @interpolate(flat) a: vec2f,
  @location(1) @interpolate(flat) b: vec2f,
  @location(2) @interpolate(flat) c: vec2f,
  @location(3) @interpolate(flat) irradiance: f32,
  @location(4) @interpolate(flat) direction: vec3f,
  @location(5) @interpolate(flat) sourcePoint: vec3f,
  // Source normal divided by dot(normal, direction), for the distance back to Q.
  @location(6) @interpolate(flat) sourceDepthGradient: vec3f,
}

fn cross2(a: vec2f, b: vec2f) -> f32 { return a.x * b.y - a.y * b.x; }

@vertex
fn boundsVS(@builtin(vertex_index) vertex: u32, @builtin(instance_index) face: u32) -> Bounds {
  var out: Bounds;
  out.position = vec4f(2, 2, 0, 1); // Collapse all four vertices of a skipped beam.
  let ray = beams[face];
  if (!(ray.w > 0.0 && ray.z > 1e-6)) { return out; }

  let cell = face / 2u;
  let stride = frame.grid.x;
  let i = cell / (stride - 1u) * stride + cell % (stride - 1u);
  let ids = select(vec3u(i, i + 1u, i + stride),
                   vec3u(i + 1u, i + stride + 1u, i + stride), face % 2u == 1u);
  var lo = vec2f(1e30);
  var hi = vec2f(-1e30);
  var nearClipped = false;
  var corners: array<vec2f, 3>;
  var sources: array<vec3f, 3>;
  let planeView = frame.mode == 3u || frame.mode == 4u;
  for (var j = 0u; j < 3u; j++) {
    let s = sourceVertex(ids[j]);
    sources[j] = s;
    let t = (frame.receiverZ - s.z) / ray.z;
    if (!(t > 0.0)) { return out; }
    let foot = s.xy + t * ray.xy;
    corners[j] = (foot / frame.receiverSize + 0.5) * vec2f(frame.image);
    if (planeView) {
      let ndc = 2.0 * foot / frame.receiverSize;
      lo = min(lo, ndc);
      hi = max(hi, ndc);
    } else {
      // Conservative screen bounds of the finite prism with its sloped source cap.
      let ends = array(s, vec3f(foot, frame.receiverZ));
      for (var k = 0u; k < 2u; k++) {
        let clip = frame.localVP * vec4f(ends[k], 1.0);
        if (clip.z <= 0.0 || clip.w <= 0.0) { nearClipped = true; }
        else {
          lo = min(lo, clip.xy / clip.w);
          hi = max(hi, clip.xy / clip.w);
        }
      }
    }
  }
  let doubleArea = cross2(corners[1] - corners[0], corners[2] - corners[0]);
  if (!(0.5 * abs(doubleArea) > 1e-8)) { return out; }
  // CCW winding lets adjacent faces use a complementary half-open edge rule.
  out.a = corners[0];
  out.b = select(corners[2], corners[1], doubleArea > 0.0);
  out.c = select(corners[1], corners[2], doubleArea > 0.0);
  let pixelSize = frame.receiverSize / vec2f(frame.image);
  out.irradiance = ray.w / (0.5 * abs(doubleArea) * pixelSize.x * pixelSize.y);
  if (!planeView) {
    let normal = cross(sources[1] - sources[0], sources[2] - sources[0]);
    let projection = dot(normal, ray.xyz);
    if (projection == 0.0) { return out; }
    out.direction = ray.xyz;
    out.sourcePoint = sources[0];
    out.sourceDepthGradient = normal / projection;
  }

  let pad = 2.0 / select(frame.targetSize, vec2f(frame.image), planeView);
  lo = clamp(lo - pad, vec2f(-1), vec2f(1));
  hi = clamp(hi + pad, vec2f(-1), vec2f(1));
  if (nearClipped) { lo = vec2f(-1); hi = vec2f(1); }
  let corner = vec2f(f32(vertex & 1u), f32(vertex >> 1u));
  out.position = vec4f(mix(lo, hi, corner), 0, 1);
  return out;
}

fn insideEdge(a: vec2f, b: vec2f, p: vec2f) -> bool {
  // Evaluate a shared edge in canonical order so both faces see the same zero.
  let forward = a.y < b.y || (a.y == b.y && a.x < b.x);
  let start = select(b, a, forward);
  let end = select(a, b, forward);
  let edge = cross2(end - start, p - start);
  if (forward) { return edge >= 0.0; }
  return edge < 0.0;
}

// Intersect the open light-path segment P - t*d, 0 < t < distance(P,Q),
// with the interior of the box. A tangent touching one edge blocks no energy.
fn boxBlocks(p: vec3f, d: vec3f, distanceToSource: f32) -> bool {
  var enter = 0.0;
  var leave = distanceToSource;
  for (var axis = 0u; axis < 3u; axis++) {
    let lo = frame.boxMin[axis];
    let hi = frame.boxMax[axis];
    if (d[axis] == 0.0) {
      if (p[axis] <= lo || p[axis] >= hi) { return false; }
    } else {
      let t0 = (p[axis] - lo) / d[axis];
      let t1 = (p[axis] - hi) / d[axis];
      enter = max(enter, min(t0, t1));
      leave = min(leave, max(t0, t1));
    }
  }
  return enter < leave;
}

fn sceneIrradiance(input: Bounds) -> f32 {
  let pixel = vec2i(input.position.xy);
  let depth = textureLoad(sceneDepth, pixel, 0);
  if (depth >= 1.0) { return 0.0; }
  let surface = textureLoad(normalAndFloor, pixel, 0);
  let floor = surface.a > 0.5;
  if (!floor && (frame.mode == 2u || frame.mode == 5u)) { return 0.0; }
  let n = normalize((surface.xyz * 255.0 - 128.0) / 127.0);
  let d = input.direction;
  let cosine = dot(n, -d);
  if (cosine <= 0.0) { return 0.0; }

  let uv = input.position.xy / frame.targetSize;
  let ndc = uv * vec2f(2, -2) + vec2f(-1, 1);
  let h = frame.inverseLocalVP * vec4f(ndc, depth, 1);
  var p = h.xyz / h.w;
  if (floor) { p.z = frame.receiverZ; } // The planar receiver depth is known exactly.
  if (p.z > frame.receiverZ) { return 0.0; }
  // All points on one beam ray project onto the same receiver coordinate.
  // This reuses the half-open coverage test, including for vertical box faces.
  let foot = p.xy + (frame.receiverZ - p.z) * (d.xy / d.z);
  let f = (foot / frame.receiverSize + 0.5) * vec2f(frame.image);
  if (!(insideEdge(input.a, input.b, f) && insideEdge(input.b, input.c, f) && insideEdge(input.c, input.a, f))) {
    return 0.0;
  }
  let distanceToSource = dot(p - input.sourcePoint, input.sourceDepthGradient);
  if (distanceToSource <= 0.0) { return 0.0; }
  // A convex box cannot block its own incoming ray on a front-facing face.
  // For the floor, the same box can lie anywhere between the lens and P.
  let blocked = floor && boxBlocks(p, d, distanceToSource);
  if (blocked != (frame.mode == 5u)) { return 0.0; }
  // E_floor = flux / A_floor and A_perp = A_floor * d.z.
  return input.irradiance * (cosine / d.z);
}

@fragment
fn accumulateFS(input: Bounds) -> @location(0) vec4f {
  var e = 0.0;
  if (frame.mode != 1u) { e = sceneIrradiance(input); }
  return vec4f(e, 1, 0, 0);
}

@vertex
fn fullscreenVS(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  let corners = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
  return vec4f(corners[index], 0, 1);
}

@fragment
fn compositeFS(@builtin(position) frag: vec4f) -> @location(0) vec4f {
  let pixel = vec2i(frag.xy);
  let light = textureLoad(lighting, pixel, 0).rg;
  if (frame.mode == 1u) {
    if (light.g == 0.0) { return vec4f(0, 0, 0, 1); }
    let blue = vec3f(0.13, 0.6, 0.93);
    let cyan = vec3f(0.1, 0.95, 0.95);
    let green = vec3f(0.25, 0.85, 0.35);
    let yellow = vec3f(1.0, 0.82, 0.1);
    let red = vec3f(1.0, 0.2, 0.15);
    let level = log2(light.g);
    var color = mix(blue, cyan, clamp(level, 0.0, 1.0));
    color = mix(color, green, clamp(level - 1.0, 0.0, 1.0));
    color = mix(color, yellow, clamp(0.5 * (level - 2.0), 0.0, 1.0));
    color = mix(color, red, clamp(0.5 * (level - 4.0), 0.0, 1.0));
    return vec4f(color, 1);
  }
  if (frame.mode == 2u || frame.mode == 5u) { return displayColor(light.r, frame.exposure); }
  let base = textureLoad(albedo, pixel, 0);
  if (base.a == 0.0) { return vec4f(0.03, 0.18, 0.22, 1); }
  let n = normalize((textureLoad(normalAndFloor, pixel, 0).xyz * 255.0 - 128.0) / 127.0);
  let inspection = 0.25 + 0.75 * max(dot(n, frame.inspectionDirection), 0.0);
  let beamDiffuse = vec3f(1.0, 0.78, 0.38) * (light.r / 3.14159265);
  return vec4f(base.rgb * (vec3f(inspection) + beamDiffuse), 1);
}
