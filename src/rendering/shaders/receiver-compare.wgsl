@fragment
fn integrateFS(input: Bounds) -> @location(0) vec4f {
  // Framebuffer row 0 is at the top. Optical row 0 is at the bottom.
  let pixel = vec2f(floor(input.position.x), f32(frame.image.y) - 1.0 - floor(input.position.y));
  let coverage = pixelCoverage(input.a - pixel, input.b - pixel, input.c - pixel);
  return vec4f(coverage * input.irradiance, 0, 0, 0);
}

@group(1) @binding(2) var<storage, read> solverBuffer: array<f32>;
@group(1) @binding(3) var solverTexture: texture_2d<f32>;

// Portable f32 accumulation for adapters without float32-blendable.
@group(0) @binding(6) var<storage, read_write> planeSums: array<atomic<u32>>;

@compute @workgroup_size(64)
fn integrateCS(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let face = id.x + id.y * groups.x * 64u;
  if (face >= frame.beamCount) { return; }
  let ray = beams[face];
  if (!(ray.w > 0.0 && ray.z > 1e-6)) { return; }
  let ids = sourceIds(face);
  var corners: array<vec2f, 3>;
  for (var i = 0u; i < 3u; i++) {
    let s = sourceVertex(ids[i]);
    if (!(frame.receiverZ > s.z)) { return; }
    corners[i] = receiverPixel(receiverXY(s, ray.xyz));
  }
  let a = corners[0]; let b = corners[1]; let c = corners[2];
  let area = 0.5 * abs(cross2(b-a, c-a));
  if (!(area > 1e-8)) { return; }
  let pixelSize = frame.receiverSize / vec2f(frame.image);
  let irradiance = ray.w / (area * pixelSize.x * pixelSize.y);
  let lo = vec2u(clamp(floor(min(a,min(b,c))), vec2f(0), vec2f(frame.image)));
  let hi = vec2u(clamp(ceil(max(a,max(b,c))), vec2f(0), vec2f(frame.image)));
  let rowWords = ((frame.image.x + 63u) / 64u) * 64u;
  for (var y = lo.y; y < hi.y; y++) {
    for (var x = lo.x; x < hi.x; x++) {
      let p = vec2f(f32(x), f32(y));
      let delta = irradiance * pixelCoverage(a-p, b-p, c-p);
      if (!(delta > 0.0)) { continue; }
      let index = (frame.image.y - 1u - y) * rowWords + x;
      var old = atomicLoad(&planeSums[index]);
      loop {
        let next = bitcast<u32>(bitcast<f32>(old) + delta);
        let attempt = atomicCompareExchangeWeak(&planeSums[index], old, next);
        if (attempt.exchanged) { break; }
        old = attempt.old_value;
      }
    }
  }
}

struct PlaneSample { texel: vec2i, opticalPixel: vec2u, inside: bool }

fn planeSample(frag: vec2f) -> PlaneSample {
  let image = vec2f(frame.image);
  // Preserve receiver aspect ratio and center it in the viewport.
  let scale = min(frame.targetSize.x / image.x, frame.targetSize.y / image.y);
  let p = (frag - 0.5 * (frame.targetSize - image * scale)) / scale;
  let texel = vec2u(clamp(floor(p), vec2f(0), image - 1.0));
  return PlaneSample(vec2i(texel), vec2u(texel.x, frame.image.y - 1u - texel.y),
                     all(p >= vec2f(0)) && all(p < image));
}

fn planeColor(p: PlaneSample, solver: f32) -> vec4f {
  if (!p.inside) { return vec4f(0.03, 0.06, 0.08, 1); }
  let raw = textureLoad(lighting, p.texel, 0).r;
  if (frame.mode == 4u) { return displayColor(abs(raw - solver), frame.exposure); }
  return displayColor(raw, frame.exposure);
}

@fragment
fn compareBufferFS(@builtin(position) frag: vec4f) -> @location(0) vec4f {
  let p = planeSample(frag.xy);
  return planeColor(p, solverBuffer[p.opticalPixel.y * frame.image.x + p.opticalPixel.x]);
}

@fragment
fn compareTextureFS(@builtin(position) frag: vec4f) -> @location(0) vec4f {
  let p = planeSample(frag.xy);
  return planeColor(p, textureLoad(solverTexture, vec2i(p.opticalPixel), 0).r);
}
