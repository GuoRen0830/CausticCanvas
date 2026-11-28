override srgbDebugTarget: bool = false;

fn insideTarget(pixel: vec2i) -> bool {
  return all(pixel >= vec2i(0)) && all(pixel < vec2i(frame.targetSize));
}

// Footprints smaller than a screen pixel leave sparkle on surfaces. Average a
// 5 x 5 neighbourhood, but only across texels that face the same way.
fn smoothedIrradiance(pixel: vec2i, n: vec3f, objectId: u32) -> vec3f {
  var sum = vec3f(0);
  var weights = 0.0;
  for (var y = -2; y <= 2; y++) {
    for (var x = -2; x <= 2; x++) {
      let q = pixel + vec2i(x, y);
      if (!insideTarget(q)) { continue; }
      var neighborNormal: vec3f;
      if (objectId != 0u) {
        let visible = textureLoad(sceneObject, q, 0);
        if (visible.r != objectId) { continue; }
        neighborNormal = unpack4x8snorm(visible.g).xyz;
      } else {
        if (textureLoad(albedo, q, 0).a < 0.5) { continue; }
        neighborNormal = decodeNormal(q);
      }
      let facing = max(dot(neighborNormal, n), 0.0);
      let w = exp(-0.25 * f32(x * x + y * y)) * facing * facing * facing * facing;
      sum += w * textureLoad(lighting, q, 0).rgb;
      weights += w;
    }
  }
  return select(textureLoad(lighting, pixel, 0).rgb, sum / max(weights, 1e-6), weights > 0.0);
}

// Parameter views display numeric channel values directly (0.5 is byte 128).
// Compensate only if the render target itself applies an sRGB transfer.
fn displayData(value: vec3f) -> vec4f {
  if (!srgbDebugTarget) { return vec4f(value, 1); }
  let linear = select(value / 12.92, pow((value + 0.055) / 1.055, vec3f(2.4)), value > vec3f(0.04045));
  return vec4f(linear, 1);
}

fn displayWorldNormal(localNormal: vec3f) -> vec4f {
  let n = normalize(localNormal);
  return displayData(vec3f(n.x, -n.z, n.y) * 0.5 + 0.5);
}

fn schlick(f0: vec3f, cosine: f32) -> vec3f {
  return f0 + (vec3f(1)-f0) * pow(1.0-clamp(cosine, 0.0, 1.0), 5.0);
}

fn pbrRadiance(pixel: vec2i, base: vec3f, n: vec3f, roughness: f32, metallic: f32, occlusion: f32) -> vec3f {
  let ndc = (vec2f(pixel) + 0.5) / frame.targetSize * vec2f(2,-2) + vec2f(-1,1);
  let near = frame.inverseLocalVP * vec4f(ndc, 0, 1);
  let depthColumn = frame.inverseLocalVP[2];
  let hit = near + depthColumn * textureLoad(sceneDepth, pixel, 0);
  let p = hit.xyz / hit.w;
  // The homogeneous depth derivative gives the pixel's view ray for perspective
  // and orthographic cameras, including the off-center canvas viewport.
  let v = normalize(near.xyz * depthColumn.w - depthColumn.xyz * near.w);
  let toLens = vec3f(0, 0, frame.lensCenterZ) - p;
  let distance = max(length(toLens), 1e-6);
  let l = toLens / distance;
  let nv = clamp(dot(n,v), 0.0, 1.0);
  let nl = clamp(dot(n,l), 0.0, 1.0);
  let f0 = mix(vec3f(0.04), base, metallic);

  let fill = environmentPbr(base, n, v, roughness, metallic, occlusion);

  let visible = textureLoad(sceneObject, pixel, 0);
  let geometric = normalize(unpack4x8snorm(visible.g).xyz);
  let ngl = dot(geometric,l);
  if (nv <= 0.0 || nl <= 0.0 || ngl <= 0.0) { return fill; }
  let irradiance = smoothedIrradiance(pixel, geometric, visible.r);
  if (all(irradiance <= vec3f(0))) { return fill; }
  let h = normalize(l + v);
  let nh = clamp(dot(n,h), 0.0, 1.0);
  let f = schlick(f0, dot(v,h));
  // The compact lens subtends a finite angle. Widen the GGX peak for very
  // smooth surfaces to avoid point-light spikes as a highlight crosses a pixel.
  let angularRadius = 0.5 * min(frame.aperture.x, frame.aperture.y) / distance;
  let alpha = max(roughness*roughness, clamp(0.5*angularRadius, 0.02, 0.35));
  let a2 = alpha*alpha;
  let denominator = (1.0-nh*nh) + nh*nh*a2;
  let distribution = a2 / (3.14159265 * denominator*denominator);
  // Height-correlated Smith GGX visibility, including 1 / (4 NoV NoL).
  let lambdaV = nl * sqrt(nv*nv*(1.0-a2) + a2);
  let lambdaL = nv * sqrt(nl*nl*(1.0-a2) + a2);
  let visibility = 0.5 / max(lambdaV + lambdaL, 1e-5);
  let diffuse = (1.0-metallic)*(vec3f(1)-f)*base / 3.14159265;
  let brdf = diffuse + f*distribution*visibility;
  // Splat irradiance already contains the geometric incidence cosine. The
  // bounded ratio only replaces it with shading-normal incidence.
  let correction = min(nl/max(ngl, 1e-4), 4.0);
  return fill + lampColor * frame.exposure * irradiance * correction * brdf;
}

fn surfaceRadiance(pixel: vec2i, surface: vec4f) -> vec3f {
  let base = litBaseColor(surface.rgb, materialLighting);
  let n = decodeNormal(pixel);
  let packed = textureLoad(sceneMaterial, pixel, 0).rg;
  let material = unpack4x8unorm(packed.x);
  let emission = unpackEmission(packed.y);
  if (material.b > 0.5) {
    return emission + pbrRadiance(pixel, base, normalize(n), material.r, material.g, material.a);
  }
  let fill = frame.environmentLight * mix(0.35, 1.0, clamp(0.5 - 0.5 * n.z, 0.0, 1.0));
  var filterNormal = n;
  var objectId = 0u;
  if (isBaseColorView()) {
    let visible = textureLoad(sceneObject, pixel, 0);
    filterNormal = normalize(unpack4x8snorm(visible.g).xyz);
    objectId = visible.r;
  }
  let irradiance = smoothedIrradiance(pixel, filterNormal, objectId);
  return base * (fill + lampColor * (frame.exposure * irradiance / 3.14159265));
}

@fragment
fn compositeFS(@builtin(position) frag: vec4f) -> @location(0) vec4f {
  let pixel = vec2i(frag.xy);
  if (frame.mode == VIEW_CANDIDATES) {
    let light = textureLoad(lighting, pixel, 0);
    if (light.a == 0.0) { return vec4f(0, 0, 0, 1); }
    let blue = vec3f(0.13, 0.6, 0.93);
    let cyan = vec3f(0.1, 0.95, 0.95);
    let green = vec3f(0.25, 0.85, 0.35);
    let yellow = vec3f(1.0, 0.82, 0.1);
    let red = vec3f(1.0, 0.2, 0.15);
    let level = log2(light.a);
    var color = mix(blue, cyan, clamp(level, 0.0, 1.0));
    color = mix(color, green, clamp(level - 1.0, 0.0, 1.0));
    color = mix(color, yellow, clamp(0.5 * (level - 2.0), 0.0, 1.0));
    color = mix(color, red, clamp(0.5 * (level - 4.0), 0.0, 1.0));
    return vec4f(color, 1);
  }
  let uv = frag.xy / frame.targetSize;

  let surface = textureLoad(albedo, pixel, 0);
  if (surface.a < 0.5) {
    // Dark neutral backdrop, slightly lighter at the top, so the light stands out.
    return present(vec3f(backdrop(uv)));
  }

  // Read the unfiltered irradiance so this view exposes sampling noise.
  switch frame.mode {
    case VIEW_IRRADIANCE: { return irradianceHeat(textureLoad(lighting, pixel, 0).r); }
    case VIEW_NORMALS: { return displayWorldNormal(decodeNormal(pixel)); }
    case VIEW_GEOMETRIC_NORMALS: {
      return displayWorldNormal(unpack4x8snorm(textureLoad(sceneObject, pixel, 0).g).xyz);
    }
    case VIEW_ALBEDO: { return present(surface.rgb); }
    case VIEW_LIT: { return present(filmic(0.9 * surfaceRadiance(pixel, surface))); }
    default: {}
  }
  let packed = textureLoad(sceneMaterial, pixel, 0).rg;
  let physical = unpack4x8unorm(packed.x);
  switch frame.mode {
    case VIEW_ROUGHNESS: { return displayData(vec3f(physical.r)); }
    case VIEW_METALLIC: { return displayData(vec3f(physical.g)); }
    case VIEW_OCCLUSION: { return displayData(vec3f(physical.a)); }
    case VIEW_EMISSIVE: { return present(filmic(unpackEmission(packed.y))); }
    case VIEW_NORMAL_MAP: { return displayData(unpack4x8unorm(packed.y).rgb); }
    case VIEW_MAP_COVERAGE: {
      let flags = packed.y >> 24u;
      // RGB indicates actual normal, roughness and metallic texture reads.
      let mapBits = vec3u(flags & 4u, flags & (2u | 8u), flags & (2u | 16u));
      return displayData(select(vec3f(0), vec3f(1), mapBits != vec3u(0)));
    }
    default: { return vec4f(0, 0, 0, 1); }
  }
}
