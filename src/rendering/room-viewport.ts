// The artist composes the Room image in the space beside the drawing pad on
// desktop, or above it in the stacked layout. All inputs are CSS pixels.
export function roomViewport(width: number, height: number, panelRight: number, panelTop: number, barBottom: number) {
  const wide = width > 760
  const left = wide ? Math.max(0, Math.min(width - 1, panelRight)) : 0
  const bottom = wide ? height : Math.max(1, Math.min(height, panelTop))
  const top = Math.max(0, Math.min(bottom - 1, barBottom))
  return { left, top, width: Math.max(1, width - left), height: bottom - top }
}
