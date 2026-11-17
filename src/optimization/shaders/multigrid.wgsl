// Transfers between the fine lens grid and a coarser training grid whose cell
// count divides the fine one. Restrict samples the shared vertices; prolong
// interpolates the coarse heights with a Catmull-Rom spline (C1, interpolating).

struct Transfer {
  fineGrid: vec2<u32>,
  coarseGrid: vec2<u32>,
  // Fine cells per coarse cell along x and y.
  ratio: vec2<u32>,
}

@group(0) @binding(0) var<uniform> transfer: Transfer;
@group(0) @binding(1) var<storage, read_write> fine: array<f32>;
@group(0) @binding(2) var<storage, read_write> coarse: array<f32>;

fn linearIndex(id: vec3u, groups: vec3u) -> u32 {
  return id.x + id.y * groups.x * 64u;
}

@compute @workgroup_size(64)
fn restrictHeights(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let v = linearIndex(id, groups);
  if (v >= transfer.coarseGrid.x * transfer.coarseGrid.y) { return; }
  let c = vec2<u32>(v % transfer.coarseGrid.x, v / transfer.coarseGrid.x) * transfer.ratio;
  coarse[v] = fine[c.y * transfer.fineGrid.x + c.x];
}

fn coarseAt(x: i32, y: i32) -> f32 {
  let size = vec2<i32>(transfer.coarseGrid);
  // Linear extrapolation keeps boundary slopes instead of flattening the rim.
  let cx = clamp(x, 0, size.x - 1);
  let cy = clamp(y, 0, size.y - 1);
  var h = coarse[u32(cy) * transfer.coarseGrid.x + u32(cx)];
  if (x != cx) {
    let inner = clamp(cx + select(1, -1, x > cx), 0, size.x - 1);
    h += f32(abs(x - cx)) * (h - coarse[u32(cy) * transfer.coarseGrid.x + u32(inner)]);
  }
  if (y != cy) {
    let inner = clamp(cy + select(1, -1, y > cy), 0, size.y - 1);
    h += f32(abs(y - cy)) * (h - coarse[u32(inner) * transfer.coarseGrid.x + u32(cx)]);
  }
  return h;
}

fn catmullRom(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
  let t2 = t * t;
  let t3 = t2 * t;
  return 0.5 * (2.0 * p1 + (p2 - p0) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
    (3.0 * p1 - p0 - 3.0 * p2 + p3) * t3);
}

@compute @workgroup_size(64)
fn prolongHeights(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let v = linearIndex(id, groups);
  if (v >= transfer.fineGrid.x * transfer.fineGrid.y) { return; }
  let f = vec2<u32>(v % transfer.fineGrid.x, v / transfer.fineGrid.x);
  let cell = vec2<i32>(min(f / transfer.ratio, transfer.coarseGrid - vec2<u32>(2u)));
  let t = (vec2<f32>(f) - vec2<f32>(cell) * vec2<f32>(transfer.ratio)) / vec2<f32>(transfer.ratio);
  var rows: array<f32, 4>;
  for (var j = 0; j < 4; j++) {
    let y = cell.y - 1 + j;
    rows[j] = catmullRom(coarseAt(cell.x - 1, y), coarseAt(cell.x, y),
      coarseAt(cell.x + 1, y), coarseAt(cell.x + 2, y), t.x);
  }
  fine[v] = catmullRom(rows[0], rows[1], rows[2], rows[3], t.y);
}
