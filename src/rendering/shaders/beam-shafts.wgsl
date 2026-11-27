struct Frame {
  vp: mat4x4f,
  viewport: vec2f, grid: vec2u,
  aperture: vec2f, receiver: vec2f,
  imageSize: vec2u, hasImage: u32, painted: u32,
  meanIrradiance: f32, padding: f32, strata: vec2u,
}
struct Hit { trianglePlusOne: u32, t: f32 }
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> heights: array<f32>;
@group(0) @binding(2) var<storage, read> offsets: array<vec2f>;
@group(0) @binding(3) var<storage, read> rays: array<vec4f>;
@group(0) @binding(4) var<storage, read> hits: array<Hit>;
@group(0) @binding(5) var<storage, read> irradiance: array<f32>;

fn jitter(seed: u32) -> f32 {
  var n = seed ^ (seed >> 16u);
  n *= 0x7feb352du;
  n ^= n >> 15u;
  n *= 0x846ca68bu;
  return f32((n ^ (n >> 16u)) >> 8u) / 16777216.0;
}

fn faceAt(uv: vec2f) -> u32 {
  let cells = frame.grid - 1u;
  let p = uv * vec2f(cells);
  let cell = min(vec2u(p), cells - 1u);
  let f = p - vec2f(cell);
  return 2u * (cell.y * cells.x + cell.x) + u32(f.x + f.y > 1.0);
}

fn exitVertex(id: u32) -> vec3f {
  let uv = vec2f(vec2u(id % frame.grid.x, id / frame.grid.x)) / vec2f(frame.grid - 1u);
  return vec3f((uv - 0.5) * frame.aperture + offsets[id], heights[id]);
}

fn imageAt(pixel: vec2i) -> f32 {
  let p = vec2u(clamp(pixel, vec2i(0), vec2i(frame.imageSize) - 1));
  return irradiance[p.y * frame.imageSize.x + p.x];
}

fn endpointIrradiance(xy: vec2f) -> f32 {
  let p = (xy / frame.receiver + 0.5) * vec2f(frame.imageSize) - 0.5;
  let i = vec2i(floor(p));
  let f = fract(p);
  return mix(mix(imageAt(i), imageAt(i + vec2i(1, 0)), f.x),
             mix(imageAt(i + vec2i(0, 1)), imageAt(i + 1), f.x), f.y);
}

struct Ribbon {
  @builtin(position) clip: vec4f,
  @location(0) @interpolate(linear) across: f32,
  @location(1) along: f32,
  @location(2) @interpolate(flat) profile: vec2f, // Gaussian width in pixels, peak radiance
}

@vertex
fn shaftVS(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> Ribbon {
  var out: Ribbon;
  out.clip = vec4f(2, 2, 2, 1);
  let count = frame.strata.x * frame.strata.y;
  let root = instance % count;
  let companion = instance >= count;
  // Anchor each pair to the lens. Camera motion never changes which optical
  // paths we see, and a change in target continuously moves the same faces.
  let seed = root * 5u;
  let uv = (vec2f(vec2u(root % frame.strata.x, root / frame.strata.x)) + 0.22
    + 0.56 * vec2f(jitter(seed + 1u), jitter(seed + 2u))) / vec2f(frame.strata);
  var face = faceAt(uv);
  if (companion) {
    let angle = 6.2831853 * jitter(seed + 3u);
    let gap = 0.07 + 0.07 * jitter(seed + 4u);
    let neighbour = faceAt(uv + gap * vec2f(cos(angle), sin(angle)) / vec2f(frame.strata));
    if (neighbour == face) { return out; }
    face = neighbour;
  }

  let ray = rays[face];
  let hit = hits[face];
  if (!(ray.w > 0.0) || hit.trianglePlusOne == 0u || !(hit.t > 0.0)) { return out; }
  let cell = face / 2u;
  let columns = frame.grid.x;
  let cellX = columns - 1u;
  let i = (cell / cellX) * columns + cell % cellX;
  let second = (face & 1u) != 0u;
  let source = (exitVertex(select(i, i + 1u, second))
    + exitVertex(select(i + 1u, i + columns + 1u, second)) + exitVertex(i + columns)) / 3.0;
  let end = source + ray.xyz * hit.t;
  let a = frame.vp * vec4f(source, 1);
  let b = frame.vp * vec4f(end, 1);
  if (a.z <= 0.0 && b.z <= 0.0) { return out; }

  // Clip the centerline in homogeneous space before expanding in screen pixels.
  var t0 = 0.0;
  var t1 = 1.0;
  if (a.z < 0.0) { t0 = a.z / (a.z - b.z); }
  if (b.z < 0.0) { t1 = a.z / (a.z - b.z); }
  let p0 = mix(a, b, t0);
  let p1 = mix(a, b, t1);
  if (min(p0.w, p1.w) < 1e-7) { return out; }
  let delta = (p1.xy / p1.w - p0.xy / p0.w) * 0.5 * frame.viewport;
  let pixels = length(delta);
  if (pixels < 0.5) { return out; }
  let perpendicular = vec2f(-delta.y, delta.x) / pixels;
  let variation = jitter(seed + 5u);
  let width = select(0.79, 0.51, companion) * mix(0.85, 1.16, variation)
    * clamp(pixels * 0.0036, 0.75, 2.2);

  let amount = 2.0 * f32(cellX) * f32(frame.grid.y - 1u) * ray.w / (frame.receiver.x * frame.receiver.y);
  // Leave dark space between groups even when several paths overlap in projection.
  var gain = min(1.6, sqrt(amount / 3.0)) * mix(0.18, 0.62, variation * variation);
  if (frame.painted != 0u) {
    // Favor paths that arrive on a bright part of the real caustic. Diffuse
    // spill still has its volumetric representation in the continuous field.
    if (frame.hasImage != 0u) {
      let relative = endpointIrradiance(end.xy) / max(frame.meanIrradiance, 1e-12);
      gain *= mix(0.14, 1.0, smoothstep(0.10, 1.8, relative));
    }
    if (companion) { gain *= 0.32; }
  } else {
    gain *= 0.15;
  }
  gain *= smoothstep(0.5, 6.0, pixels);

  let tip = vertex >= 2u;
  out.along = select(t0, t1, tip);
  // The full quad covers the short halo. Native depth testing occludes it
  // independently at every pixel, including across foreground silhouettes.
  let radius = 8.5 * width + 1.0;
  out.across = select(-radius, radius, (vertex & 1u) != 0u);
  let p = select(p0, p1, tip);
  out.clip = vec4f(p.xy + out.across * perpendicular * (2.0 / frame.viewport) * p.w, p.zw);
  out.profile = vec2f(width, gain);
  return out;
}

@fragment
fn shaftFS(v: Ribbon) -> @location(0) vec4f {
  let t = clamp(v.along, 0.0, 1.0);
  let width = v.profile.x * mix(0.50, 1.22, t);
  // Convolve the narrow core with a one-pixel box (variance 1/12), so a
  // subpixel path can cross a pixel center without acquiring excess energy.
  let sigma = sqrt(width * width + 1.0 / 12.0);
  let x = v.across / sigma;
  let h = v.across / (3.1 * width + 0.25);
  let core = (width / sigma) * exp(-0.5 * x * x);
  let halo = 0.17 * exp(-0.5 * h * h);
  let envelope = mix(1.0, 0.58, t) * (1.0 - 0.38 * smoothstep(0.92, 1.0, t));
  return vec4f(vec3f(1.0, 0.97, 0.92) * (v.profile.y * envelope * (core + halo)), 0);
}
