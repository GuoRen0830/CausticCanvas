struct GlassFrame {
  localVP: mat4x4f,
  eye: vec4f,
  grid: vec2u,
  cells: vec2u,
  aperture: vec2f,
  reflectance: f32,
}
@group(0) @binding(0) var<uniform> glass: GlassFrame;
@group(0) @binding(1) var<storage, read> physicalHeight: array<f32>;
@group(0) @binding(2) var<storage, read> vertexOffset: array<vec2f>;

struct ShellVertex {
  @builtin(position) clip: vec4f,
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
}

fn exitPosition(node: vec2u) -> vec3f {
  // Select actual solver vertices, including both aperture boundaries.
  let last = glass.grid - vec2u(1);
  let ij = (node * last + glass.cells / vec2u(2)) / glass.cells;
  let id = ij.y * glass.grid.x + ij.x;
  let xy = (vec2f(ij) / vec2f(last) - 0.5) * glass.aperture + vertexOffset[id];
  return vec3f(xy, physicalHeight[id]);
}

@vertex
fn shellVS(@builtin(vertex_index) vertex: u32) -> ShellVertex {
  let corner = array<vec2u, 6>(
    vec2u(0, 0), vec2u(1, 0), vec2u(0, 1),
    vec2u(0, 1), vec2u(1, 0), vec2u(1, 1))[vertex % 6u];
  let quad = vertex / 6u;
  let cells = glass.cells;
  let exitCount = cells.x * cells.y;
  var v: ShellVertex;
  if quad < exitCount {
    let node = vec2u(quad % cells.x, quad / cells.x) + corner;
    v.position = exitPosition(node);
    // Neighboring sampled surface vertices give smooth normals in physical units.
    let lo = max(node, vec2u(1)) - vec2u(1);
    let hi = min(node + vec2u(1), cells);
    let dx = exitPosition(vec2u(hi.x, node.y)) - exitPosition(vec2u(lo.x, node.y));
    let dy = exitPosition(vec2u(node.x, hi.y)) - exitPosition(vec2u(node.x, lo.y));
    v.normal = normalize(cross(dx, dy));
    v.uv = vec2f(node) / vec2f(cells);
  } else if quad == exitCount {
    // Reverse the winding on the planar air-to-glass entrance.
    v.uv = vec2f(corner.yx);
    v.position = vec3f((v.uv - 0.5) * glass.aperture, 0);
    v.normal = vec3f(0, 0, -1);
  } else {
    let segment = quad - exitCount - 1u;
    var node: vec2u;
    if segment < cells.x {
      node = vec2u(segment + corner.x, 0);
      v.normal = vec3f(0, -1, 0);
    } else if segment < cells.x + cells.y {
      node = vec2u(cells.x, segment - cells.x + corner.x);
      v.normal = vec3f(1, 0, 0);
    } else if segment < 2u * cells.x + cells.y {
      node = vec2u(2u * cells.x + cells.y - segment - corner.x, cells.y);
      v.normal = vec3f(0, 1, 0);
    } else {
      node = vec2u(0, 2u * (cells.x + cells.y) - segment - corner.x);
      v.normal = vec3f(-1, 0, 0);
    }
    v.position = exitPosition(node);
    v.position.z *= f32(corner.y);
    // Local coordinates on the side face limit the bevel glint to its perimeter.
    let along = select(f32(node.y) / f32(cells.y), f32(node.x) / f32(cells.x), v.normal.y != 0.0);
    v.uv = vec2f(along, f32(corner.y));
  }
  v.clip = glass.localVP * vec4f(v.position, 1);
  return v;
}

// Reflections of finite studio strip lights. Integrate their angular width over
// the pixel so a highlight on a steep solved surface does not flicker on orbit.
fn studioStrip(ray: vec3f, rayWidth: vec3f, across: vec3f, center: vec3f, width: f32) -> f32 {
  let axis = normalize(across);
  let d = dot(ray, axis);
  let footprint = dot(rayWidth, abs(axis));
  let variance = width * width + footprint * footprint / 12.0;
  return (width * inverseSqrt(variance)) * exp(-0.5 * d * d / variance)
    * smoothstep(0.45, 0.88, dot(ray, normalize(center)));
}

@fragment
fn shellFS(v: ShellVertex) -> @location(0) vec4f {
  let view = normalize(glass.eye.xyz - glass.eye.w * v.position);
  let n = normalize(v.normal);

  // Round the shading normal over the last pixel of each polished edge.
  // Tangents follow the actual surface, including mirrored side-face UVs.
  let duvX = dpdx(v.uv);
  let duvY = dpdy(v.uv);
  let dpX = dpdx(v.position);
  let dpY = dpdy(v.position);
  let handed = sign(duvX.x * duvY.y - duvX.y * duvY.x);
  let alongU = dpX * duvY.y - dpY * duvX.y;
  let alongV = dpY * duvX.x - dpX * duvY.x;
  let tangentU = alongU * (handed * inverseSqrt(max(dot(alongU, alongU), 1e-30)));
  let tangentV = alongV * (handed * inverseSqrt(max(dot(alongV, alongV), 1e-30)));
  let bevelWidth = clamp(1.25 * fwidth(v.uv), vec2f(1e-6), vec2f(0.16));
  let bend = smoothstep(1.0 - bevelWidth, vec2f(1), v.uv)
    + smoothstep(vec2f(0), bevelWidth, v.uv) - 1.0;
  let polished = normalize(n + bend.x * tangentU + bend.y * tangentV);
  let facing = clamp(abs(dot(polished, view)), 0.0, 1.0);
  let fresnel = glass.reflectance + (1.0 - glass.reflectance) * pow(1.0 - facing, 5.0);

  let r = reflect(-view, polished);
  let rayWidth = fwidth(reflect(-view, n));
  let key = studioStrip(r, rayWidth, vec3f(0.82, -0.58, 0), vec3f(-0.48, -0.68, -0.56), 0.085);
  let rim = studioStrip(r, rayWidth, vec3f(0.35, 0.74, -0.58), vec3f(0.85, -0.46, -0.10), 0.055);
  let sky = smoothstep(-0.45, 0.75, -r.z);
  let environment = mix(vec3f(0.025, 0.029, 0.032), vec3f(0.11, 0.125, 0.14), sky)
    + key * vec3f(2.0, 1.96, 1.88) + rim * vec3f(1.2, 1.28, 1.36);
  return vec4f(environment * fresnel, fresnel);
}
