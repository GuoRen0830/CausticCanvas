struct SceneUniforms {
  mvp: mat4x4<f32>,
  inspectMaps: u32,
  baseColorOnly: u32,
}

@group(0) @binding(0)
var<uniform> sceneUniforms: SceneUniforms;
struct Material {
  color: vec4f,
  physical: vec4f, // Roughness, metallic, tangent-space normal scale and Y convention.
  flags: vec4f, // Texture presence mask, PBR enabled, vertex color multiplier, double-sided.
  emissionAndOcclusion: vec4f,
  // Two affine rows per texture, with the source UV set in the first row's W.
  uvTransform: array<vec4f, 14>,
  vertexWetness: vec4f, // Inverse dry-to-wet red range (zero disables), saturated roughness.
}
@group(1) @binding(0) var<uniform> material: Material;
@group(1) @binding(1) var baseColorMap: texture_2d<f32>;
@group(1) @binding(2) var baseColorSampler: sampler;
@group(1) @binding(3) var metallicRoughnessMap: texture_2d<f32>;
@group(1) @binding(4) var metallicRoughnessSampler: sampler;
@group(1) @binding(5) var normalMap: texture_2d<f32>;
@group(1) @binding(6) var normalSampler: sampler;
@group(1) @binding(7) var roughnessMap: texture_2d<f32>;
@group(1) @binding(8) var roughnessSampler: sampler;
@group(1) @binding(9) var metallicMap: texture_2d<f32>;
@group(1) @binding(10) var metallicSampler: sampler;
@group(1) @binding(11) var occlusionMap: texture_2d<f32>;
@group(1) @binding(12) var occlusionSampler: sampler;
@group(1) @binding(13) var emissiveMap: texture_2d<f32>;
@group(1) @binding(14) var emissiveSampler: sampler;

struct VertexInput {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) color: vec3<f32>,
  @location(3) isFloor: f32,
  @location(4) objectId: f32,
  @location(5) uv: vec2f,
  @location(6) uv1: vec2f,
  @location(7) tangent: vec4f,
}

struct VertexOutput {
  @builtin(position) clipPosition: vec4<f32>,
  @location(0) localNormal: vec3<f32>,
  @location(1) color: vec3<f32>,
  @location(2) @interpolate(flat) isFloor: f32,
  @location(3) @interpolate(flat) objectId: u32,
  @location(4) localPosition: vec3f,
  @location(5) uvs: vec4f,
  @location(6) tangent: vec4f,
}

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  output.clipPosition = sceneUniforms.mvp * vec4<f32>(input.position, 1.0);
  output.localNormal = input.normal;
  output.color = input.color;
  output.isFloor = input.isFloor;
  output.objectId = u32(input.objectId);
  output.localPosition = input.position;
  output.uvs = vec4f(input.uv, input.uv1);
  output.tangent = input.tangent;
  return output;
}

struct GBuffer {
  @location(0) albedo: vec4<f32>,
  @location(1) normalAndFloor: vec4<f32>,
  @location(2) objectAndNormal: vec2u,
  @location(3) physical: vec2u,
}

// Same bytes as pack4x8snorm(vec4f(n, 0)), which returns 0 on some Metal drivers.
fn packNormal(n: vec3f) -> u32 {
  let q = vec3i(round(clamp(n, vec3f(-1), vec3f(1)) * 127.0)) & vec3i(255);
  return u32(q.x) | (u32(q.y) << 8u) | (u32(q.z) << 16u);
}

fn mapUV(uvs: vec4f, slot: u32) -> vec2f {
  let a = material.uvTransform[2u*slot];
  let b = material.uvTransform[2u*slot+1u];
  let uv = select(uvs.xy, uvs.zw, a.w > 0.5);
  let p = vec3f(uv.x, 1.0-uv.y, 1.0);
  return vec2f(dot(a.xyz,p), dot(b.xyz,p));
}

fn mappedNormal(n: vec3f, px: vec3f, py: vec3f, sampleUV: vec2f, authored: vec4f, raw: vec3f) -> vec3f {
  // Tangent and handedness come from the derivatives of the actual normal-map
  // UVs. Undo the image-coordinate V flip to express the basis in OBJ space.
  let dx = dpdx(sampleUV) * vec2f(1.0, -1.0);
  let dy = dpdy(sampleUV) * vec2f(1.0, -1.0);
  let det = dx.x * dy.y - dx.y * dy.x;
  var tangent = authored.xyz;
  var handedness = sign(authored.w);
  if (abs(authored.w) < 0.5) {
    if (abs(det) <= 1e-6 * length(dx) * length(dy)) { return n; }
    tangent = (px * dy.y - py * dx.y) * sign(det);
    let bitangent = (py * dx.x - px * dy.x) * sign(det);
    handedness = select(-1.0, 1.0, dot(cross(n,tangent),bitangent) >= 0.0);
  }
  let ortho = tangent - n * dot(n, tangent);
  if (dot(ortho, ortho) < 1e-28) { return n; }
  let t = normalize(ortho);
  let b = cross(n, t) * handedness;
  let local = (t * raw.x + b * (raw.y * material.physical.w)) * material.physical.z + n * raw.z;
  let length2 = dot(local, local);
  if (length2 < 1e-12) { return n; }
  return local * inverseSqrt(length2);
}

// RGBE shares the exponent between three HDR emission channels. A zero texel
// represents a surface with no emission.
fn packEmission(rgb: vec3f) -> u32 {
  let peak = max(rgb.r, max(rgb.g,rgb.b));
  if (peak < 1e-30) { return 0u; }
  let e = clamp(ceil(log2(peak)), -100.0, 126.0);
  let mantissa = vec3u(round(clamp(rgb * exp2(-e), vec3f(0), vec3f(1)) * 255.0));
  return mantissa.r | (mantissa.g << 8u) | (mantissa.b << 16u) | (u32(e + 128.0) << 24u);
}

@fragment
fn fragmentMain(input: VertexOutput, @builtin(front_facing) front: bool) -> GBuffer {
  // Screen Y points down, so dy cross dx faces the camera on both sides.
  // Keep this geometric normal independent of interpolated shading normals.
  let px = dpdx(input.localPosition);
  let py = dpdy(input.localPosition);
  let geometric = normalize(cross(py, px));
  // Base-color mode retains the authored color and UV transform, then uses
  // the existing Lambert lighting path with the mesh's own normals.
  let baseColorOnly = sceneUniforms.baseColorOnly != 0u;
  let wetness = select(clamp((1.0 - input.color.r) * material.vertexWetness.x, 0.0, 1.0), 0.0, baseColorOnly);
  let flags = u32(material.flags.x) & select(0xffffffffu, 1u, baseColorOnly);
  var shading = normalize(input.localNormal);
  var surface = select(material.physical.xy, vec2f(1.0, 0.0), baseColorOnly);
  if ((flags & 2u) != 0u) {
    surface *= textureSample(metallicRoughnessMap, metallicRoughnessSampler, mapUV(input.uvs, 1u)).gb;
  }
  var normalTexel = vec3f(0);
  if ((flags & 4u) != 0u) {
    let uv = mapUV(input.uvs, 2u);
    normalTexel = textureSample(normalMap, normalSampler, uv).xyz;
    // A continuous water film gradually covers the grain's microscopic slopes.
    let raw = 2.0 * normalTexel - 1.0;
    let coated = mix(raw, vec3f(0.0, 0.0, 1.0), wetness);
    shading = mappedNormal(shading, px, py, uv, input.tangent, coated);
  }
  if ((flags & 8u) != 0u) {
    surface.x *= textureSample(roughnessMap, roughnessSampler, mapUV(input.uvs, 3u)).r;
  }
  if ((flags & 16u) != 0u) {
    surface.y *= textureSample(metallicMap, metallicSampler, mapUV(input.uvs, 4u)).r;
  }
  surface.x = mix(surface.x, material.vertexWetness.y, wetness);
  let tint = select(material.color.rgb, material.color.rgb * input.color, material.flags.z > 0.5);
  var baseColor = mix(input.color, tint, material.color.a);
  if ((flags & 1u) != 0u) {
    baseColor *= textureSample(baseColorMap, baseColorSampler, mapUV(input.uvs, 0u)).rgb;
  }
  var ao = 1.0;
  if ((flags & 32u) != 0u) {
    let sampleAO = textureSample(occlusionMap, occlusionSampler, mapUV(input.uvs, 5u)).r;
    ao = mix(1.0, sampleAO, material.emissionAndOcclusion.w * (1.0-wetness));
  }
  var emission = select(material.emissionAndOcclusion.rgb, vec3f(0), baseColorOnly);
  if ((flags & 64u) != 0u) {
    emission *= textureSample(emissiveMap, emissiveSampler, mapUV(input.uvs, 6u)).rgb;
  }
  if (!front) {
    if (material.flags.w < 0.5) { discard; }
    shading = -shading;
  }
  var detail: u32;
  if (sceneUniforms.inspectMaps != 0u) {
    detail = pack4x8unorm(vec4f(normalTexel, 0)) | (flags << 24u);
  } else {
    detail = packEmission(emission);
  }
  return GBuffer(
    vec4f(baseColor, 1.0),
    // Signed unit normals encoded in UNORM8; zero is exactly byte 128.
    vec4<f32>((shading * 127.0 + 128.0) / 255.0, input.isFloor),
    vec2u(input.objectId, packNormal(geometric)),
    vec2u(pack4x8unorm(vec4f(surface, select(material.flags.y, 0.0, baseColorOnly), ao)), detail),
  );
}
