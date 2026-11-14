// Pack a linear stream into legal X/Y dimensions with a short final row.
export function workgroupGrid(count: number, limit: number): [number, number] {
  const rows = Math.ceil(count / limit)
  if (!Number.isSafeInteger(count) || count < 1 || rows > limit) {
    throw new Error(`Cannot dispatch ${count} workgroups within a ${limit} x ${limit} grid.`)
  }
  return [Math.ceil(count / rows), rows]
}
