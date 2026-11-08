// A fixed-size lane team shares each triangle's pixel work.
override lanesPerTriangle: u32 = 64u;
@compute @workgroup_size(64)
fn prepareFootprints(@builtin(global_invocation_id) id: vec3u) {
  if id.x >= 2u*(p.grid.x-1u)*(p.grid.y-1u) { return; }
  let f = footprint(id.x);
  footprints[id.x] = f;
  beams[id.x] = f.beam;
}

@compute @workgroup_size(64)
fn forwardCooperative(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) lane: u32) {
  let face = group.x*(64u/lanesPerTriangle)+lane/lanesPerTriangle;
  let localLane = lane%lanesPerTriangle;
  var f: Footprint;
  if face < 2u*(p.grid.x-1u)*(p.grid.y-1u) { f = footprints[face]; }
  if f.beam.w == 0 || !(f.area.x > 1e-8) { return; }
  let box = bounds(f);
  let width = box.z-box.x;
  let count = width*(box.w-box.y);
  let pixelArea = (p.receiverSize.x/f32(p.image.x))*(p.receiverSize.y/f32(p.image.y));
  for (var i=localLane; i<count; i+=lanesPerTriangle) {
    let pixel = box.xy + vec2u(i%width,i/width);
    let value = f.beam.w/pixelArea * overlap(f,pixel).x/f.area.x;
    addEnergy(pixel.y*p.image.x+pixel.x,value);
  }
}

var<workgroup> partialGradient: array<vec3f,64>;

@compute @workgroup_size(64)
fn backwardCooperative(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) lane: u32) {
  let face = group.x*(64u/lanesPerTriangle)+lane/lanesPerTriangle;
  let localLane = lane%lanesPerTriangle;
  var f: Footprint;
  if face < 2u*(p.grid.x-1u)*(p.grid.y-1u) { f = footprints[face]; }
  var gradient = vec3f(0);
  if f.beam.w > 0 && f.area.x > 1e-8 {
    let box = bounds(f);
    let width = box.z-box.x;
    let count = width*(box.w-box.y);
    let pixelArea = (p.receiverSize.x/f32(p.image.x))*(p.receiverSize.y/f32(p.image.y));
    for (var i=localLane; i<count; i+=lanesPerTriangle) {
      let pixel = box.xy + vec2u(i%width,i/width);
      let contribution = divide(overlap(f,pixel),f.area)*(f.beam.w/pixelArea);
      gradient += contribution.yzw*pixelGradient[pixel.y*p.image.x+pixel.x];
    }
  }
  partialGradient[lane] = gradient;
  workgroupBarrier();
  for (var stride=lanesPerTriangle/2u; stride>0u; stride/=2u) {
    if localLane<stride { partialGradient[lane] += partialGradient[lane+stride]; }
    workgroupBarrier();
  }
  if localLane==0u && face < 2u*(p.grid.x-1u)*(p.grid.y-1u) {
    var sum = partialGradient[lane];
    if !all(abs(sum)<vec3f(3e38)) { sum = vec3f(0); }
    triangleGradient[face] = vec4f(sum,0);
  }
}
