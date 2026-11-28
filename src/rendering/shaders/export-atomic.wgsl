// WebGPU guarantees u32 atomics on baseline devices. Accumulate linear f32
// radiance with compare-and-exchange when the adapter lacks f32 blending.
@group(2) @binding(3) var<storage, read_write> pixelSums: array<atomic<u32>>;

fn accumulateChannel(index: u32, term: f32) {
  if (!(term > 0.0)) { return; }
  var old = atomicLoad(&pixelSums[index]);
  // Helper fragments may get indeterminate atomic results. Always bound
  // retries; a real fragment that exhausts them requests a smaller draw.
  for (var attempt = 0u; attempt < 128u; attempt++) {
    let sum = bitcast<u32>(bitcast<f32>(old) + term);
    if (sum == old) { return; }
    let result = atomicCompareExchangeWeak(&pixelSums[index], old, sum);
    if (result.exchanged) { return; }
    old = result.old_value;
  }
  atomicStore(&pixelSums[arrayLength(&pixelSums) - 1u], 1u);
}

fn sumOffset(p: vec2f) -> u32 {
  return 3u * (u32(p.y) * u32(frame.targetSize.x) + u32(p.x));
}

@fragment
fn exportFragment(input: ExportSplat) -> @location(0) vec4f {
  let radiance = exportRadiance(input);
  let first = sumOffset(input.position.xy);
  accumulateChannel(first, radiance.x);
  accumulateChannel(first + 1u, radiance.y);
  accumulateChannel(first + 2u, radiance.z);
  return vec4f(0);
}

@vertex
fn resolveSumVertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let xy = array(vec2f(-1,-1), vec2f(3,-1), vec2f(-1,3));
  return vec4f(xy[i], 0, 1);
}

@fragment
fn resolveSumFragment(@builtin(position) p: vec4f) -> @location(0) vec4f {
  let i = sumOffset(p.xy);
  return vec4f(bitcast<f32>(atomicLoad(&pixelSums[i])), bitcast<f32>(atomicLoad(&pixelSums[i+1u])),
               bitcast<f32>(atomicLoad(&pixelSums[i+2u])), 0.0);
}
