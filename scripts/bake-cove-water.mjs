// Bake wind ripples in world space onto the shoreline-aligned normal atlas.
// Run before bake-harbor-detail.mjs, which derives water roughness from this map.
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const width = 768, height = 384
const halfWidth = 7.6, front = 7.4
const rgb = Buffer.alloc(width * height * 3)
const output = fileURLToPath(new URL('../scene/smugglers-cove/tide-normal.png', import.meta.url))
let state = 0x53a67e
function random() {
  state = (Math.imul(1664525, state) + 1013904223) >>> 0
  return state / 4294967296
}

// Same shoreline and atlas parameterization as cove-shore.ts.
function coastline(x) {
  return 0.86 - 1.05 * Math.exp(-(((x + 0.6) / 2.75) ** 2)) +
    0.14 * Math.sin(1.4 * x) + 0.10 * Math.sin(0.5 * x + 0.8)
}

// Each group has a finite area, bowed crests, and a separate phase. At the
// boundary both the displacement and its slope vanish. Overlap leaves areas
// of shelter, interference and occasional sharper crests in the same bay.
const packets = []
function addPackets(cellX, cellZ, fine) {
  for (let row = -1; row <= Math.ceil(front / cellZ); row++) {
    for (let col = -1; col <= Math.ceil(2 * halfWidth / cellX); col++) {
      const x = -halfWidth + (col + 0.1 + 0.80 * random()) * cellX
      const z = (row + 0.1 + 0.80 * random()) * cellZ
      const wind = -0.25 + 0.52 * Math.sin(0.61 * x + 0.37 * z)
      const angle = wind + (random() - 0.5) * (fine ? 2.4 : 1.25)
      const span = cellX * (0.77 + random() * 0.57)
      const depth = cellZ * (0.79 + random() * 0.54)
      const wavelength = fine ? 0.17 + 0.17 * random() : 0.36 + 0.31 * random()
      const k = 2 * Math.PI / wavelength
      const slope = fine ? 0.039 + 0.064 * random() : 0.12 + 0.135 * random()
      packets.push({ x, z, cos: Math.cos(angle), sin: Math.sin(angle),
        invSpan2: 1 / (span * span), invDepth2: 1 / (depth * depth),
        radius: Math.max(span, depth), curve: (random() - 0.5) * 0.76 / span,
        phase: random() * 2 * Math.PI, k, amplitude: slope / k })
    }
  }
}
addPackets(2.12, 1.28, false)
addPackets(1.36, 0.94, true)

const byte = n => Math.round(127.5 * (1 + n))
for (let i = 0; i < width; i++) {
  const x = halfWidth * (2 * (i + 0.5) / width - 1)
  const line = coastline(x)
  const coastlineSlope = (coastline(x + 0.001) - coastline(x - 0.001)) / 0.002
  const nearby = packets.filter(p => Math.abs(x - p.x) < p.radius)
  for (let j = 0; j < height; j++) {
    const t = (j + 0.5) / height
    const z = line + (front - line) * t
    let slopeX = 0, slopeZ = 0
    for (const p of nearby) {
      const dx = x - p.x, dz = z - p.z
      const u = dx * p.cos + dz * p.sin, v = -dx * p.sin + dz * p.cos
      const mask = 1 - u * u * p.invSpan2 - v * v * p.invDepth2
      if (mask <= 0) continue
      const phase = p.k * (v + p.curve * u * u) + p.phase
      const edge = -4 * p.amplitude * mask * Math.sin(phase)
      const crest = p.amplitude * mask * mask * p.k * Math.cos(phase)
      const du = edge * u * p.invSpan2 + crest * 2 * p.curve * u
      const dv = edge * v * p.invDepth2 + crest
      slopeX += du * p.cos - dv * p.sin
      slopeZ += du * p.sin + dv * p.cos
    }
    // Fade the ripple slopes at the wet-sand seam. The mesh's U tangents
    // follow the shoreline in the shallows and straighten offshore.
    const edgeT = Math.min(1, (z - line) / 0.57)
    const fade = edgeT * edgeT * (3 - 2 * edgeT)
    const shear = (1 - t) * coastlineSlope
    const basisScale = fade / Math.hypot(1, shear)
    const nx = -(slopeX + shear * slopeZ) * basisScale
    const ny = (slopeZ - shear * slopeX) * basisScale
    const length = Math.hypot(nx, ny, 1), index = 3 * (j * width + i)
    rgb[index] = byte(nx / length)
    rgb[index + 1] = byte(ny / length)
    rgb[index + 2] = byte(1 / length)
  }
}
const result = spawnSync(process.env.FFMPEG || 'ffmpeg', ['-v', 'error', '-y',
  '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', `${width}x${height}`,
  '-i', 'pipe:0', '-frames:v', '1', '-compression_level', '9', '-pred', 'mixed', output], { input: rgb })
if (result.error || result.status !== 0) throw new Error(result.error?.message || result.stderr.toString())
