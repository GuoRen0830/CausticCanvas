override cacheFootprints: bool = false;

struct Params {
  grid: vec2u,
  image: vec2u,
  aperture: vec2f,
  receiverSize: vec2f,
  receiverZ: f32,
  eta: f32,
  irradiance: f32,
  padding: f32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> heights: array<f32>;
@group(0) @binding(2) var<storage, read_write> beams: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> energy: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> pixelGradient: array<f32>;
@group(0) @binding(5) var<storage, read_write> triangleGradient: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> heightGradient: array<f32>;

// A dual scalar stores its value and derivatives with respect to three heights.
fn constant(v: f32) -> vec4f { return vec4f(v, 0, 0, 0); }
fn mul(a: vec4f, b: vec4f) -> vec4f {
  return vec4f(a.x * b.x, a.yzw * b.x + b.yzw * a.x);
}
fn divide(a: vec4f, b: vec4f) -> vec4f {
  let value = a.x / b.x;
  return vec4f(value, (a.yzw - value * b.yzw) / b.x);
}
fn root(a: vec4f) -> vec4f {
  let value = sqrt(a.x);
  return vec4f(value, a.yzw / (2 * value));
}
struct Dual3 { x: vec4f, y: vec4f, z: vec4f }
struct Dual2 { x: vec4f, y: vec4f }
fn subtract(a: Dual3, b: Dual3) -> Dual3 {
  return Dual3(a.x-b.x, a.y-b.y, a.z-b.z);
}
fn crossDual(a: Dual3, b: Dual3) -> Dual3 {
  return Dual3(mul(a.y,b.z)-mul(a.z,b.y), mul(a.z,b.x)-mul(a.x,b.z), mul(a.x,b.y)-mul(a.y,b.x));
}
fn indices(face: u32) -> vec3u {
  let cell = face / 2u;
  let a = (cell / (p.grid.x-1u)) * p.grid.x + cell % (p.grid.x-1u);
  if face % 2u == 0u { return vec3u(a, a+1u, a+p.grid.x); }
  return vec3u(a+1u, a+p.grid.x+1u, a+p.grid.x);
}
fn vertex(index: u32, derivative: vec3f) -> Dual3 {
  let xy = (vec2f(f32(index % p.grid.x), f32(index / p.grid.x)) / vec2f(p.grid-vec2u(1)) - 0.5) * p.aperture;
  return Dual3(constant(xy.x), constant(xy.y), vec4f(heights[index], derivative));
}
fn project(v: Dual3, d: Dual3) -> Dual2 {
  let t = divide(constant(p.receiverZ)-v.z, d.z);
  return Dual2((v.x+mul(t,d.x)+constant(p.receiverSize.x*0.5)) * (f32(p.image.x)/p.receiverSize.x),
               (v.y+mul(t,d.y)+constant(p.receiverSize.y*0.5)) * (f32(p.image.y)/p.receiverSize.y));
}
fn cross2(a: Dual2, b: Dual2) -> vec4f { return mul(a.x,b.y)-mul(a.y,b.x); }
fn difference(a: Dual2, b: Dual2) -> Dual2 { return Dual2(a.x-b.x,a.y-b.y); }
struct Footprint {
  corners: array<Dual2,3>,
  area: vec4f,
  beam: vec4f,
}
@group(0) @binding(7) var<storage, read_write> footprints: array<Footprint>;

fn footprint(face: u32) -> Footprint {
  var result: Footprint;
  result.beam = vec4f(0,0,1,0);
  let ids = indices(face);
  let a = vertex(ids.x, vec3f(1,0,0));
  let b = vertex(ids.y, vec3f(0,1,0));
  let c = vertex(ids.z, vec3f(0,0,1));
  let normal = crossDual(subtract(b,a), subtract(c,a));
  let length = root(mul(normal.x,normal.x)+mul(normal.y,normal.y)+mul(normal.z,normal.z));
  let n = Dual3(divide(normal.x,length),divide(normal.y,length),divide(normal.z,length));
  let k = constant(1-p.eta*p.eta) + mul(n.z,n.z)*(p.eta*p.eta);
  if !(k.x > 1e-8) { return result; }
  let factor = root(k)-n.z*p.eta;
  var d = Dual3(mul(factor,n.x),mul(factor,n.y),constant(p.eta)+mul(factor,n.z));
  if p.eta == 1.0 { d = Dual3(constant(0),constant(0),constant(1)); }
  if !(d.z.x > 1e-6) || !(max(a.z.x,max(b.z.x,c.z.x)) < p.receiverZ) { return result; }
  let flux = p.irradiance * (p.aperture.x/f32(p.grid.x-1u)) * (p.aperture.y/f32(p.grid.y-1u)) * 0.5;
  result.beam = vec4f(d.x.x,d.y.x,d.z.x,flux);
  result.corners = array<Dual2,3>(project(a,d),project(b,d),project(c,d));
  for (var i=0u; i<3u; i++) {
    let q = result.corners[i];
    if !(abs(q.x.x)<1e12 && abs(q.y.x)<1e12) {
      result.beam = vec4f(0,0,1,0);
      return result;
    }
  }
  var area = cross2(difference(result.corners[1],result.corners[0]),difference(result.corners[2],result.corners[0]))*0.5;
  if area.x < 0 { area = -area; }
  result.area = area;
  return result;
}
struct Polygon { points: array<Dual2,8>, count: u32 }
fn clip(poly: Polygon, axis: u32, bound: f32, greater: bool) -> Polygon {
  var output: Polygon;
  if poly.count == 0u { return output; }
  var previous = poly.points[poly.count-1u];
  var prevCoord = select(previous.x,previous.y,axis==1u);
  var prevInside = select(prevCoord.x <= bound,prevCoord.x >= bound,greater);
  for (var i=0u; i<poly.count; i++) {
    let current = poly.points[i];
    let coord = select(current.x,current.y,axis==1u);
    let inside = select(coord.x <= bound,coord.x >= bound,greater);
    if inside != prevInside {
      let t = divide(constant(bound)-prevCoord,coord-prevCoord);
      output.points[output.count] = Dual2(previous.x+mul(t,current.x-previous.x),previous.y+mul(t,current.y-previous.y));
      output.count++;
    }
    if inside {
      output.points[output.count] = current;
      output.count++;
    }
    previous = current;
    prevCoord = coord;
    prevInside = inside;
  }
  return output;
}
fn overlap(f: Footprint, pixel: vec2u) -> vec4f {
  var poly: Polygon;
  poly.count = 3u;
  for (var i=0u; i<3u; i++) { poly.points[i] = f.corners[i]; }
  poly = clip(poly,0u,f32(pixel.x),true);
  poly = clip(poly,0u,f32(pixel.x+1u),false);
  poly = clip(poly,1u,f32(pixel.y),true);
  poly = clip(poly,1u,f32(pixel.y+1u),false);
  var area = vec4f(0);
  for (var i=1u; i+1u<poly.count; i++) {
    area += cross2(difference(poly.points[i],poly.points[0]),difference(poly.points[i+1u],poly.points[0]))*0.5;
  }
  if area.x < 0 { area = -area; }
  return area;
}
fn bounds(f: Footprint) -> vec4u {
  let a = vec2f(f.corners[0].x.x,f.corners[0].y.x);
  let b = vec2f(f.corners[1].x.x,f.corners[1].y.x);
  let c = vec2f(f.corners[2].x.x,f.corners[2].y.x);
  // Include touching pixels for the piecewise derivative at a coverage boundary.
  let lo = vec2u(clamp(floor(min(a,min(b,c))),vec2f(0),vec2f(p.image)));
  let hi = vec2u(clamp(floor(max(a,max(b,c)))+1,vec2f(0),vec2f(p.image)));
  return vec4u(lo,hi);
}
fn addEnergy(index: u32, value: f32) {
  if !(value > 0) { return; }
  var old = atomicLoad(&energy[index]);
  loop {
    let updated = bitcast<u32>(bitcast<f32>(old)+value);
    let attempt = atomicCompareExchangeWeak(&energy[index],old,updated);
    if attempt.exchanged { break; }
    old = attempt.old_value;
  }
}
@compute @workgroup_size(64)
fn forward(@builtin(global_invocation_id) id: vec3u) {
  let face = id.x;
  if face >= 2u*(p.grid.x-1u)*(p.grid.y-1u) { return; }
  let f = footprint(face);
  beams[face] = f.beam;
  if cacheFootprints { footprints[face] = f; }
  // Degenerate receiver footprints have no finite area density in this model.
  if f.beam.w == 0 || !(f.area.x > 1e-8) { return; }
  let box = bounds(f);
  let pixelArea = (p.receiverSize.x/f32(p.image.x))*(p.receiverSize.y/f32(p.image.y));
  for (var y=box.y; y<box.w; y++) {
    for (var x=box.x; x<box.z; x++) {
      let value = f.beam.w/pixelArea * overlap(f,vec2u(x,y)).x/f.area.x;
      addEnergy(y*p.image.x+x,value);
    }
  }
}
@compute @workgroup_size(64)
fn backward(@builtin(global_invocation_id) id: vec3u) {
  let face = id.x;
  if face >= 2u*(p.grid.x-1u)*(p.grid.y-1u) { return; }
  var f: Footprint;
  if cacheFootprints { f = footprints[face]; } else { f = footprint(face); }
  var gradient = vec3f(0);
  if f.beam.w > 0 && f.area.x > 1e-8 {
    let box = bounds(f);
    let pixelArea = (p.receiverSize.x/f32(p.image.x))*(p.receiverSize.y/f32(p.image.y));
    for (var y=box.y; y<box.w; y++) {
      for (var x=box.x; x<box.z; x++) {
        let contribution = divide(overlap(f,vec2u(x,y)),f.area)*(f.beam.w/pixelArea);
        gradient += contribution.yzw*pixelGradient[y*p.image.x+x];
      }
    }
  }
  if !all(abs(gradient)<vec3f(3e38)) { gradient = vec3f(0); }
  triangleGradient[face] = vec4f(gradient,0);
}
@compute @workgroup_size(64)
fn gather(@builtin(global_invocation_id) id: vec3u) {
  let v = id.x;
  if v >= p.grid.x*p.grid.y { return; }
  let col = i32(v%p.grid.x);
  let row = i32(v/p.grid.x);
  var sum = 0.0;
  for (var y=max(row-1,0); y<=min(row,i32(p.grid.y)-2); y++) {
    for (var x=max(col-1,0); x<=min(col,i32(p.grid.x)-2); x++) {
      let first = 2u*(u32(y)*(p.grid.x-1u)+u32(x));
      for (var face=first; face<first+2u; face++) {
        let ids = indices(face);
        let g = triangleGradient[face];
        for (var j=0u; j<3u; j++) { if ids[j]==v { sum += g[j]; } }
      }
    }
  }
  heightGradient[v] = sum;
}
