// Shares sceneSpot, sameSurface and the area integrator with the scalar solve.
// RGB is calibrated after that solve; it cannot change a triangle's direction.
@group(0) @binding(13) var<storage, read> paint: array<vec4f>;
@group(0) @binding(14) var entranceFilter: texture_2d<f32>;
@group(0) @binding(15) var colorResult: texture_storage_2d<rgba32float, write>;

struct SourceFace {
  @builtin(position) clip: vec4f,
  @location(0) @interpolate(flat) face: u32,
}

@vertex
fn apertureVS(@builtin(vertex_index) corner: u32, @builtin(instance_index) face: u32) -> SourceFace {
  let i = indices(face)[corner];
  let uv = vec2f(vec2u(i % p.grid.x, i / p.grid.x)) / vec2f(p.grid - 1u) + offsets[i] / p.aperture;
  return SourceFace(vec4f((2.0 * uv - 1.0) * vec2f(1, -1), 0, 1), face);
}

// At fine lens resolution most footprints lie entirely within one pixel.
// Their analytic triangle area is the same integral as four clipping steps.
fn colorCoverage(f: Footprint, xy: vec2u) -> f32 {
  let a = vec2f(f.corners[0].x.x, f.corners[0].y.x);
  let b = vec2f(f.corners[1].x.x, f.corners[1].y.x);
  let c = vec2f(f.corners[2].x.x, f.corners[2].y.x);
  let lo = vec2f(xy);
  if (all(min(a, min(b, c)) >= lo) && all(max(a, max(b, c)) <= lo + 1.0)) {
    let ab = b - a;
    let ac = c - a;
    return 0.5 * abs(ab.x * ac.y - ab.y * ac.x);
  }
  return overlap(f, xy).x;
}

@fragment
fn calibrateAperture(in: SourceFace) -> @location(0) vec4f {
  // The supplement is the mean of all painted colors. Misfocused or occluded
  // rays remain visible instead of artificially blacking out solver errors.
  let fallback = paint[p.image.x * p.image.y];
  let hit = sceneHits[in.face];
  if (hit.trianglePlusOne == 0u) { return fallback; }
  let tri = triangles[hit.trianglePlusOne - 1u];
  let spot = sceneSpot(in.face, tri, false);
  if (!(spot.density.x > 0)) { return fallback; }
  let box = bounds(spot.polygon);
  var sum = vec4f(0);
  for (var y = box.y; y < box.w; y++) {
    for (var x = box.x; x < box.z; x++) {
      let xy = vec2u(x,y);
      let brush = paint[y * p.image.x + x];
      if (brush.a > 0 && sameSurface(xy, hit.trianglePlusOne, tri, spot.normal)) {
        let weight = colorCoverage(spot.polygon, xy) * brush.a;
        sum += vec4f(brush.rgb, 1) * weight;
      }
    }
  }
  if (sum.a > 0) { return vec4f(clamp(sum.rgb / sum.a, vec3f(0), vec3f(1)), 1); }
  return fallback;
}

@compute @workgroup_size(64)
fn colorForward(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let face = opticalIndex(id, groups);
  if (face >= arrayLength(&sceneHits)) { return; }
  let hit = sceneHits[face];
  if (hit.trianglePlusOne == 0u) { return; }
  let tri = triangles[hit.trianglePlusOne - 1u];
  let spot = sceneSpot(face, tri, false);
  if (!(spot.density.x > 0)) { return; }
  let ids = indices(face);
  let center = (values(vertex(ids.x, vec3f(0))) + values(vertex(ids.y, vec3f(0))) + values(vertex(ids.z, vec3f(0)))) / 3;
  let tint = textureLoad(entranceFilter, apertureTexel(center.xy, p.aperture, textureDimensions(entranceFilter)), 0).rgb;
  let box = bounds(spot.polygon);
  for (var y = box.y; y < box.w; y++) {
    for (var x = box.x; x < box.z; x++) {
      let xy = vec2u(x,y);
      if (sameSurface(xy, hit.trianglePlusOne, tri, spot.normal)) {
        let irradiance = tint * (spot.density.x * colorCoverage(spot.polygon, xy));
        let i = 3u * (y * p.image.x + x);
        addEnergy(i, irradiance.r);
        addEnergy(i + 1u, irradiance.g);
        addEnergy(i + 2u, irradiance.b);
      }
    }
  }
}

@compute @workgroup_size(8,8)
fn resolveColor(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(colorResult);
  if (any(id.xy >= size)) { return; }
  let i = 3u * (id.y * size.x + id.x);
  let rgb = vec3u(atomicLoad(&energy[i]), atomicLoad(&energy[i + 1u]), atomicLoad(&energy[i + 2u]));
  textureStore(colorResult, id.xy, vec4f(bitcast<vec3f>(rgb), 1));
}

