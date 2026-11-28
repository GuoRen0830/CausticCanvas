function normalize(path: string) {
  const parts: string[] = []
  for (const part of path.replaceAll('\\', '/').split('/')) {
    if (part === '..') parts.pop()
    else if (part && part !== '.') parts.push(part)
  }
  return parts.join('/')
}

// Folder imports resolve references relative to the owning file. Flattened
// selections can use a basename only if it identifies exactly one file.
export function localFiles(files: File[]) {
  const paths = new Map<string, File | null>(), folded = new Map<string, File | null>(), names = new Map<string, File | null>()
  function add(index: Map<string, File | null>, key: string, file: File) {
    index.set(key, index.has(key) ? null : file)
  }
  const filePath = (f: File) => (f.webkitRelativePath || f.name).replaceAll('\\', '/')
  for (const file of files) {
    const path = filePath(file)
    add(paths, path, file)
    add(folded, path.toLowerCase(), file)
    add(names, file.name.toLowerCase(), file)
  }
  return (owner: File, ref: string) => {
    const path = ref.replace(/^"(.*)"$/, '$1').replaceAll('\\', '/')
    const dir = filePath(owner).split('/').slice(0, -1)
    const key = normalize([...dir, path].join('/'))
    if (!/^(?:[a-z]:\/|\/)/i.test(path)) {
      const exact = paths.get(key) ?? folded.get(key.toLowerCase())
      if (exact) return exact
      if (owner.webkitRelativePath) return undefined
    }
    return names.get(path.split('/').at(-1)!.toLowerCase()) ?? undefined
  }
}
