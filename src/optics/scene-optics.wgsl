struct DesignPixel { trianglePlusOne: u32, depth: f32 }

@group(0) @binding(8) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(9) var<storage, read> nodes: array<Node>;
@group(0) @binding(10) var<storage, read_write> sceneHits: array<Hit>;
@group(0) @binding(11) var<storage, read> designPixels: array<DesignPixel>;

fn values(v: Dual3) -> vec3f { return vec3f(v.x.x, v.y.x, v.z.x); }
fn dotNormal(n: vec3f, v: Dual3) -> vec4f { return n.x*v.x + n.y*v.y + n.z*v.z; }

@compute @workgroup_size(64)
fn sceneTrace(@builtin(global_invocation_id) id: vec3u) {
  let face = id.x;
  if face >= 2u*(p.grid.x-1u)*(p.grid.y-1u) { return; }
  let ids = indices(face);
  let a = vertex(ids.x,vec3f(0));
  let b = vertex(ids.y,vec3f(0));
  let c = vertex(ids.z,vec3f(0));
  // Refraction and visibility share the exact convention used by the orbit renderer.
  let ray = refractFace(a,b,c).beam;
  beams[face] = ray;
  var hit = Hit();
  if ray.w > 0 { hit = closestHit((values(a)+values(b)+values(c))/3.0, ray.xyz); }
  sceneHits[face] = hit;
}

// Intersect one corner ray with the plane through the hit triangle.
fn surfaceCorner(v: Dual3, d: Dual3, n: vec3f, a: vec3f, incidence: vec4f) -> Dual2 {
  let t = divide(dotNormal(n,v) - constant(dot(n,a)), incidence);
  return Dual2((v.x+mul(t,d.x)+constant(0.5*p.receiverSize.x))*(f32(p.image.x)/p.receiverSize.x),
               (v.y+mul(t,d.y)+constant(0.5*p.receiverSize.y))*(f32(p.image.y)/p.receiverSize.y));
}

struct SceneSpot { polygon: Footprint, density: vec4f, normal: vec3f }
fn sceneSpot(face: u32, tri: Triangle, derivatives: bool) -> SceneSpot {
  var spot: SceneSpot;
  let n = normalize(cross(tri.b-tri.a,tri.c-tri.a));
  if !(n.z < 0) { return spot; }
  let ids = indices(face);
  let a = vertex(ids.x, vec3f(1,0,0));
  let b = vertex(ids.y, vec3f(0,1,0));
  let c = vertex(ids.z, vec3f(0,0,1));
  let ray = beams[face];
  var d = Dual3(constant(ray.x),constant(ray.y),constant(ray.z));
  if derivatives { d = refractFace(a,b,c).direction; }
  let incidence = -dotNormal(n,d);
  if !(incidence.x > 1e-7) { return spot; }
  let sourceNormal = crossDual(subtract(b,a),subtract(c,a));
  var perpendicularArea = 0.5*(mul(sourceNormal.x,d.x)+mul(sourceNormal.y,d.y)+mul(sourceNormal.z,d.z));
  if perpendicularArea.x < 0 { perpendicularArea = -perpendicularArea; }
  if !(perpendicularArea.x > 0) { return spot; }
  spot.polygon.corners = array<Dual2,3>(surfaceCorner(a,d,n,tri.a,incidence),
    surfaceCorner(b,d,n,tri.a,incidence),surfaceCorner(c,d,n,tri.a,incidence));
  // Density is power per square meter on the inclined receiving surface.
  spot.normal = n;
  spot.density = divide(incidence*ray.w, perpendicularArea);
  return spot;
}

fn sameSurface(xy: vec2u, facePlusOne: u32, tri: Triangle, n: vec3f) -> bool {
  let visible = designPixels[xy.y*p.image.x+xy.x];
  if visible.trianglePlusOne == facePlusOne { return true; }
  if visible.trianglePlusOne == 0u { return false; }
  let other = triangles[visible.trianglePlusOne-1u];
  if other.objectId != tri.objectId { return false; }
  let otherNormal = normalize(cross(other.b-other.a,other.c-other.a));
  if dot(n,otherNormal) < 0.85 { return false; }
  let step = p.receiverSize/vec2f(p.image);
  let pos = vec3f((vec2f(xy)+0.5)*step-0.5*p.receiverSize, visible.depth);
  return abs(dot(n,pos-tri.a)) <= 2e-5+0.25*max(step.x,step.y);
}

@compute @workgroup_size(64)
fn sceneForward(@builtin(global_invocation_id) id: vec3u) {
  let face = id.x;
  if face >= arrayLength(&sceneHits) { return; }
  let hit = sceneHits[face];
  if hit.trianglePlusOne == 0u { return; }
  let tri = triangles[hit.trianglePlusOne-1u];
  let spot = sceneSpot(face,tri,false);
  if !(spot.density.x > 0) { return; }
  let box = bounds(spot.polygon);
  for (var y=box.y; y<box.w; y++) {
    for (var x=box.x; x<box.z; x++) {
      if sameSurface(vec2u(x,y),hit.trianglePlusOne,tri,spot.normal) {
        addEnergy(y*p.image.x+x, spot.density.x*overlap(spot.polygon,vec2u(x,y)).x);
      }
    }
  }
}

@compute @workgroup_size(64)
fn sceneBackward(@builtin(global_invocation_id) id: vec3u) {
  let face = id.x;
  if face >= arrayLength(&sceneHits) { return; }
  triangleGradient[face] = vec4f(0);
  let hit = sceneHits[face];
  if hit.trianglePlusOne == 0u { return; }
  let tri = triangles[hit.trianglePlusOne-1u];
  // Differentiate on the surface selected by the matching forward evaluation.
  let spot = sceneSpot(face,tri,true);
  if !(spot.density.x > 0) { return; }
  let box = bounds(spot.polygon);
  var gradient = vec3f(0);
  for (var y=box.y; y<box.w; y++) {
    for (var x=box.x; x<box.z; x++) {
      if sameSurface(vec2u(x,y),hit.trianglePlusOne,tri,spot.normal) {
        let value = mul(spot.density,overlap(spot.polygon,vec2u(x,y)));
        gradient += value.yzw*pixelGradient[y*p.image.x+x];
      }
    }
  }
  if all(abs(gradient)<vec3f(3e38)) { triangleGradient[face] = vec4f(gradient,0); }
}
