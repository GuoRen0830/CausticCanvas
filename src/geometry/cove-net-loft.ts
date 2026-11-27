import { HarborMesh } from './cove-quayside'

export function makeCoveNetLoft() {
  const loft = new HarborMesh()
  const left = -1.24, right = 1.24, back = -0.69, front = 0.64
  const ridge = 1.85, pitch = 0.59, roofY = (z: number) => ridge - Math.abs(z) * pitch
  const eave = roofY(back)
  const siding = ['ochre boards', 'sawn cedar', 'ochre boards', 'ochre boards']
  const doorLeft = -0.11, doorRight = 0.54, windowLeft = -1.06, windowRight = -0.48
  loft.box('weathered limestone', 0, 0.055, 0, 2.50, 0.11, 1.42)
  loft.box('dock oak', 0, 0.128, 0, 2.46, 0.04, 1.36)

  // The door opens into a furnished interior. The opening and recessed
  // window both pass through the siding, including its shadow-casting edges.
  const rows = 10, sill = 0.10, rowHeight = (eave - sill) / rows
  for (let row = 0; row < rows; row++) {
    const y = sill + (row + 0.5) * rowHeight
    const mat = siding[row % siding.length], lip = (row % 2) * 0.004
    const board = (x0: number, x1: number) => loft.box(mat, (x0+x1)/2, y, front+lip,
      x1-x0, rowHeight, 0.057)
    if (row < 8) {
      if (row >= 4 && row < 7) {
        board(left, windowLeft)
        board(windowRight, doorLeft)
      } else board(left, doorLeft)
      board(doorRight, right)
    } else board(left, right)
    loft.box(mat, 0, y, back-lip, right-left, rowHeight, 0.058)
    for (const x of [left, right]) {
      loft.box(siding[(row+1)%siding.length], x, y, (front+back)/2, 0.06+lip, rowHeight, front-back)
    }
  }
  for (const x of [left,right]) {
    loft.face('sawn cedar', [[x,eave,back],[x,ridge-0.02,0],[x,eave,front]], [x,0,0])
    for (const z of [front,back]) {
      loft.box('dock oak', x, (eave+0.10)/2, z, 0.087, eave-0.10, 0.082)
      loft.beam('dock oak', [x,eave-0.032,z],[x,ridge-0.045,0], 0.061,0.064)
    }
  }

  const windowBottom = sill+4*rowHeight, windowTop = sill+7*rowHeight
  loft.box('weathered zinc', (windowLeft+windowRight)/2, (windowTop+windowBottom)/2, front-0.042,
    windowRight-windowLeft, windowTop-windowBottom, 0.015)
  for (const x of [windowLeft, (windowLeft+windowRight)/2, windowRight]) {
    loft.box('harbor blue', x, (windowBottom+windowTop)/2, front+0.018, 0.034, windowTop-windowBottom+0.04, 0.086)
  }
  for (const y of [windowBottom, (windowBottom+windowTop)/2, windowTop]) {
    loft.box('harbor blue', (windowLeft+windowRight)/2, y, front+0.030, windowRight-windowLeft+0.08, 0.038, 0.086)
  }
  loft.box('sawn cedar', (windowLeft+windowRight)/2, windowBottom-0.010, front+0.05, 0.71, 0.045, 0.15)
  const doorTop = sill+8*rowHeight
  for (const x of [doorLeft, doorRight]) {
    loft.box('dock oak', x, (doorTop+sill)/2, front+0.016, 0.060, doorTop-sill, 0.10)
  }
  loft.box('dock oak', (doorLeft+doorRight)/2, doorTop+0.018, front+0.018, 0.78,0.086,0.11)
  loft.box('wrought iron', 0.54, doorTop+0.093, front+0.063, 1.44,0.028,0.035)
  for (let p = 0; p < 7; p++) {
    loft.box('harbor blue', 0.589+(p+0.5)*0.088, (sill+doorTop)/2+0.006,
      front+0.08, 0.085, doorTop-sill, 0.039)
  }
  for (const y of [sill+0.12,doorTop-0.11]) {
    loft.box('sawn cedar', 0.897, y, front+0.115, 0.61, 0.059, 0.033)
  }
  loft.beam('sawn cedar', [0.627,sill+0.15,front+0.116],[1.167,doorTop-0.142,front+0.116], 0.038,0.053)
  for (const x of [0.68,1.10]) {
    loft.box('wrought iron', x, doorTop+0.039, front+0.089, 0.027,0.13,0.025)
  }

  // A laced hemp net hangs from two pins against the open door. Alternating
  // knot rows give the diamonds real openings and the slack a catenary-like sag.
  const netWidth = 0.52, knots = 5
  function knot(i: number, j: number): [number,number,number] {
    const t = (i + (j % 2) * 0.5) / knots
    return [0.63+t*netWidth, 0.94 - j*0.065 - 0.16*t*(1-t), front+0.150+0.009*Math.sin(j*0.65+t*2.5)]
  }
  for (let j = 0; j < 7; j++) {
    for (let i = 0; i < knots; i++) {
      const offset = j % 2
      loft.beam('hemp rope', knot(i, j+offset), knot(i,j+1-offset), 0.009,0.009)
      loft.beam('hemp rope', knot(i+1,j+offset), knot(i,j+1-offset), 0.009,0.009)
    }
  }
  loft.box('sawn cedar', 0.23, 0.51, -0.30, 0.87,0.058,0.35)
  for (const x of [-0.13,0.58]) loft.box('dock oak', x,0.31,-0.30,0.055,0.36,0.30)
  loft.column('sawn cedar', 0.23,-0.30, [[0.54,0.124],[0.74,0.148]], 12)
  loft.column('wrought iron', 0.23,-0.30, [[0.696,0.145],[0.718,0.149]], 12)

  // Wide folded zinc sheets run from ridge to eaves. Raised standing seams,
  // drip edges and ridge flashing remain visible from the painting view.
  const roofHalf = 1.40, slopeLength = 0.827, panels = 10, width = 2*roofHalf/panels
  for (const side of [-1,1]) {
    const z = side*slopeLength
    loft.face('dock oak', [[-roofHalf,ridge-0.040,0],[roofHalf,ridge-0.040,0],
      [roofHalf,roofY(z)-0.040,z],[-roofHalf,roofY(z)-0.040,z]], [0,-1,-side])
    for (let i = 0; i < panels; i++) {
      const a = -roofHalf+i*width, b = a+width
      loft.face((i*3+side)%5 === 0 ? 'weathered zinc' : 'aged zinc',
        [[a,ridge,0],[b,ridge,0],[b,roofY(z),z],[a,roofY(z),z]], [0,1,side])
    }
    for (let i = 0; i <= panels; i++) {
      const x = -roofHalf+i*width
      loft.beam('zinc flashing', [x,ridge+0.004,0],[x,roofY(z)+0.004,z], 0.012, 0.018)
    }
    loft.box('dock oak', 0,roofY(z)-0.025,z,2*roofHalf,0.048,0.032)
    for (const x of [-roofHalf,roofHalf]) {
      loft.beam('sawn cedar', [x,ridge-0.025,0],[x,roofY(z)-0.025,z], 0.036, 0.055)
    }
    loft.face('zinc flashing', [[-roofHalf,ridge+0.017,0],[roofHalf,ridge+0.017,0],
      [roofHalf,roofY(0.05)+0.017,side*0.05],[-roofHalf,roofY(0.05)+0.017,side*0.05]], [0,1,side])
  }
  return loft.finish()
}
