struct MarchFrame {
  inverseLocalVP: mat4x4f,
  volumeMin: vec3f,
  incidentLength: f32,
  volumeMax: vec3f,
  incidentIrradiance: f32,
  aperture: vec2f,
  halfSize: vec2f,
  maxRaySlope: f32,
  receiverMagnification: vec2f,
}

@group(0) @binding(0) var<uniform> frame: MarchFrame;
@group(0) @binding(1) var opaqueDepth: texture_depth_2d;
@group(0) @binding(2) var lightField: texture_3d<f32>;
@group(0) @binding(3) var fieldSampler: sampler;

const backgroundDistance = 60000.0;
override scatteringPerMeter: f32;

const diffuseWeight = 0.075;

@vertex
fn fullscreen(@builtin(vertex_index) id: u32) -> @builtin(position) vec4f {
  let corners = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
  return vec4f(corners[id], 0, 1);
}

fn boxInterval(origin: vec3f, direction: vec3f, lo: vec3f, hi: vec3f, limit: f32) -> vec2f {
  let parallel = abs(direction) < vec3f(1e-8);
  if (any(parallel & ((origin < lo) | (origin > hi)))) { return vec2f(0); }
  let divisor = select(direction, vec3f(1), parallel);
  let a = (lo - origin) / divisor;
  let b = (hi - origin) / divisor;
  let entry = select(min(a, b), vec3f(0), parallel);
  let exit = select(max(a, b), vec3f(limit), parallel);
  let start = max(0.0, max(entry.x, max(entry.y, entry.z)));
  let end = min(limit, min(exit.x, min(exit.y, exit.z)));
  return vec2f(start, max(start, end));
}

// Intersect a camera ray with one side of the diverging beam envelope.
// Inside is atOrigin + t * slope >= 0.
fn clipSide(interval: vec2f, atOrigin: f32, slope: f32) -> vec2f {
  if (abs(slope) < 1e-8) { return select(vec2f(1, 0), interval, atOrigin >= 0.0); }
  let t = -atOrigin / slope;
  if (slope > 0.0) { return vec2f(max(interval.x, t), interval.y); }
  return vec2f(interval.x, min(interval.y, t));
}

@fragment
fn integrate(@builtin(position) pixel: vec4f) -> @location(0) vec4f {
  // Sample the ray through the chosen full-resolution depth texel's center,
  // including the unpaired final column/row of an odd-size viewport.
  let fullSize = textureDimensions(opaqueDepth);
  let tap = min(vec2u(pixel.xy * vec2f(fullSize) / frame.halfSize), fullSize - 1u);
  let ndc = ((vec2f(tap) + 0.5) / vec2f(fullSize)) * vec2f(2, -2) + vec2f(-1, 1);
  let depth = textureLoad(opaqueDepth, vec2i(tap), 0);
  let nearH = frame.inverseLocalVP * vec4f(ndc, 0, 1);
  let farH = nearH + frame.inverseLocalVP[2];
  let origin = nearH.xyz / nearH.w;
  // Homogeneous difference works for perspective, orthographic, and infinite far planes.
  let direction = normalize(farH.xyz * nearH.w - nearH.xyz * farH.w);
  let endH = nearH + depth * frame.inverseLocalVP[2];
  var limit = backgroundDistance;
  if (abs(endH.w) > 1e-8) {
    limit = min(length(endH.xyz / endH.w - origin), backgroundDistance);
  }
  let surfaceDistance = select(limit, backgroundDistance, depth >= 1.0);

  // The finite, uniform, collimated column above the planar entrance face is
  // integrated analytically, so its brightness tracks path length at every angle.
  let halfAperture = 0.5 * frame.aperture;
  let incident = boxInterval(origin, direction, vec3f(-halfAperture, -frame.incidentLength),
    vec3f(halfAperture, 0), limit);
  var integrated = vec3f((incident.y - incident.x) * frame.incidentIrradiance * diffuseWeight);

  var refracted = boxInterval(origin, direction, frame.volumeMin, frame.volumeMax, limit);
  let extent = frame.volumeMax - frame.volumeMin;
  let cells = vec3f(textureDimensions(lightField));
  if (frame.maxRaySlope >= 0.0 && refracted.y > refracted.x) {
    // A ray can start anywhere on the aperture. Snell's maximum exit angle
    // bounds its lateral travel; pad for splatting, XY blur and interpolation.
    let cell = extent / cells;
    let radius = halfAperture + 3.0 * cell.xy + 2.0 * cell.z * frame.maxRaySlope;
    let atOrigin = radius + frame.maxRaySlope * origin.z;
    let slope = frame.maxRaySlope * direction.z;
    refracted = clipSide(refracted, atOrigin.x - origin.x, slope - direction.x);
    refracted = clipSide(refracted, atOrigin.x + origin.x, slope + direction.x);
    refracted = clipSide(refracted, atOrigin.y - origin.y, slope - direction.y);
    refracted = clipSide(refracted, atOrigin.y + origin.y, slope + direction.y);
  }
  let pathLength = refracted.y - refracted.x;
  if (pathLength > 0.0) {
    let uvEntry = (origin + direction * refracted.x - frame.volumeMin) / extent;
    let uvPerMeter = direction / extent;
    let speed = abs(uvPerMeter) * cells;
    var axis = select(0u, 1u, speed.y > speed.x);
    axis = select(axis, 2u, speed.z > speed[axis]);
    // Integrate between planes anchored in the optical grid. A neighbouring
    // pixel or a moving depth edge trims only the end intervals of its ray.
    let spacing = max(0.75, cells[axis] / 192.0);
    let stepLength = spacing / speed[axis];
    let gridEntry = (uvEntry[axis] * cells[axis] - 0.5) * sign(direction[axis]);
    let firstLength = (1.0 - fract(gridEntry / spacing)) * stepLength;
    let steps = 1u + u32(ceil(max(pathLength - firstLength, 0.0) / stepLength));
    let invIncident = select(0.0, 1.0 / max(frame.incidentIrradiance, 1e-12),
      frame.incidentIrradiance > 0.0);
    var left = 0.0;
    for (var i = 0u; i < steps; i++) {
      let right = min(firstLength + f32(i) * stepLength, pathLength);
      let uvw = uvEntry + uvPerMeter * (0.5 * (left + right));
      let field = textureSampleLevel(lightField, fieldSampler, uvw, 0);
      // RGB carries the filtered rays. Alpha is their broader contrast reference.
      // Scale both by the same unfocused fan so the tiny bright exit and a
      // broad stroke retain a continuous body; folded sheets have a brighter core.
      let span = mix(vec2f(1.0), frame.receiverMagnification, uvw.z);
      let relative = vec2f(max(field.r, max(field.g, field.b)), field.a) * (span.x * span.y * invIncident);
      let prominence = (relative.x - relative.y) / (relative.x + relative.y + 0.5);
      let ridge = smoothstep(0.04, 0.38, prominence);
      let focus = smoothstep(1.45, 5.75, relative.x);
      let weight = diffuseWeight + focus * (0.5 + 1.7 * ridge);
      integrated += field.rgb * weight * (right - left);
      left = right;
    }
  }
  return vec4f(min(scatteringPerMeter * integrated / 12.56637, vec3f(60000.0)), surfaceDistance);
}
