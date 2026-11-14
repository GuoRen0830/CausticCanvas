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
  for (var j = 0u; j < 3u; j++) {
    let s = sourceVertex(ids[j]);
    let t = (frame.receiverZ - s.z) / ray.z;
    if (!(t > 0.0)) { return out; }
    let foot = s.xy + t * ray.xy;
    corners[j] = (foot / frame.receiverSize + 0.5) * vec2f(frame.image);
    if (frame.mode >= 3u) {
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

  let pad = 2.0 / select(frame.targetSize, vec2f(frame.image), frame.mode >= 3u);
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

@fragment
fn accumulateFS(input: Bounds) -> @location(0) vec4f {
  let pixel = vec2i(input.position.xy);
  var e = 0.0;
  if (textureLoad(normalAndFloor, pixel, 0).a > 0.5) {
    let uv = input.position.xy / frame.targetSize;
    let ndc = uv * vec2f(2, -2) + vec2f(-1, 1);
    let h = frame.inverseLocalVP * vec4f(ndc, textureLoad(sceneDepth, pixel, 0), 1);
    let p = (h.xy / h.w / frame.receiverSize + 0.5) * vec2f(frame.image);
    if (insideEdge(input.a, input.b, p) && insideEdge(input.b, input.c, p) && insideEdge(input.c, input.a, p)) {
      e = input.irradiance;
    }
  }
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
  if (frame.mode == 2u) { return displayColor(light.r, frame.exposure); }
  let base = textureLoad(albedo, pixel, 0);
  if (base.a == 0.0) { return vec4f(0.03, 0.18, 0.22, 1); }
  let n = normalize((textureLoad(normalAndFloor, pixel, 0).xyz * 255.0 - 128.0) / 127.0);
  let inspection = 0.25 + 0.75 * max(dot(n, normalize(vec3f(-0.4, 0.8, 0.6))), 0.0);
  let beamDiffuse = vec3f(1.0, 0.78, 0.38) * (light.r / 3.14159265);
  return vec4f(base.rgb * (vec3f(inspection) + beamDiffuse), 1);
}
