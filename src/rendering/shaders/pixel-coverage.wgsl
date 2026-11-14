// Integrate the constant irradiance of a triangular footprint over a unit
// pixel. Clipping in pixel-local coordinates improves precision.
struct Polygon { points: array<vec2f, 8>, count: u32 }

fn cross2(a: vec2f, b: vec2f) -> f32 { return a.x * b.y - a.y * b.x; }

fn clipEdge(poly: Polygon, axis: u32, bound: f32, keepGreater: bool) -> Polygon {
  var result: Polygon;
  if (poly.count == 0u) { return result; }
  var prev = poly.points[poly.count - 1u];
  for (var i = 0u; i < poly.count; i++) {
    let curr = poly.points[i];
    let prevIn = select(prev[axis] <= bound, prev[axis] >= bound, keepGreater);
    let currIn = select(curr[axis] <= bound, curr[axis] >= bound, keepGreater);
    if (prevIn != currIn) {
      let t = (bound - prev[axis]) / (curr[axis] - prev[axis]);
      result.points[result.count] = mix(prev, curr, t);
      result.count++;
    }
    if (currIn) {
      result.points[result.count] = curr;
      result.count++;
    }
    prev = curr;
  }
  return result;
}

fn pixelCoverage(a: vec2f, b: vec2f, c: vec2f) -> f32 {
  var poly: Polygon;
  poly.count = 3u;
  poly.points[0] = a;
  poly.points[1] = b;
  poly.points[2] = c;
  poly = clipEdge(poly, 0u, 0.0, true);
  poly = clipEdge(poly, 0u, 1.0, false);
  poly = clipEdge(poly, 1u, 0.0, true);
  poly = clipEdge(poly, 1u, 1.0, false);
  var twiceArea = 0.0;
  for (var i = 1u; i + 1u < poly.count; i++) {
    twiceArea += cross2(poly.points[i] - poly.points[0], poly.points[i+1u] - poly.points[0]);
  }
  return clamp(0.5 * abs(twiceArea), 0.0, 1.0);
}
