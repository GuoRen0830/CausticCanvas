import { localFiles } from './local-asset-files'
import { readPreviewGltf } from './preview-gltf'

// Read only files selected by the artist. A GLB may also refer to external
// images; resolve its companions exactly like those of a JSON glTF.
export async function readGltfAsset(file: File, files: File[], signal?: AbortSignal) {
  signal?.throwIfAborted()
  const find = localFiles(files)
  const data = await file.arrayBuffer()
  signal?.throwIfAborted()
  return readPreviewGltf({ data, base: new URL('file:///') }, signal, {
    pbr: true,
    async readUri(uri) {
      signal?.throwIfAborted()
      if (/^data:/i.test(uri)) {
        const response = await fetch(uri, { signal })
        return response.blob()
      }
      if (/^(?:[a-z][\w+.\-]*:|\/\/)/i.test(uri)) throw new Error('Select a local copy of ' + uri)
      const path = decodeURIComponent(uri.split(/[?#]/, 1)[0])
      const companion = find(file, path)
      if (!companion) throw new Error('Missing companion file ' + path + '. Select the model folder or its files together.')
      return companion
    },
  })
}
