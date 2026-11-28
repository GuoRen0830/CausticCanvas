import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { setImmediate } from 'node:timers/promises'
import { test } from 'node:test'
import { createImageExport } from '../src/ui/image-export.ts'

// Drive the shipped controls through rapid cancel/retry without a GPU or an actual download.
test('export animates until download, supports cancel and retry, and names both material comparisons', async () => {
  const page = await readFile(new URL('../index.html', import.meta.url), 'utf8')
  const controls = new Map()
  for (const [, id] of page.matchAll(/id="(export-[\w-]+)"/g)) controls.set('#' + id, new EventTarget())
  const control = name => controls.get('#export-' + name)
  const dialog = control('dialog'), size = control('size'), spinner = control('spinner')
  dialog.showModal = () => { dialog.open = true }
  dialog.close = () => { dialog.open = false }
  size.options = [1024,2048,4096].map(value => ({ value: String(value) }))
  size.value = '2048'
  control('cancel').focus = () => {}
  const saves = []
  let finishEncoding
  const canvas = { getContext: () => ({ putImageData() {} }), toBlob: callback => { finishEncoding = callback } }
  const previous = { document: globalThis.document, window: globalThis.window }
  const lifetime = new AbortController()
  let current = { name: 'Harbor.glb', aspect: 2, materialMode: 'basecolor' }, request, jobs = 0, complete, fail
  const click = name => control(name).dispatchEvent(new Event('click'))
  const submit = () => control('form').dispatchEvent(new Event('submit', { cancelable: true }))
  try {
    globalThis.document = { querySelector: selector => controls.get(selector), createElement(tag) {
      if (tag === 'canvas') return canvas
      assert.equal(tag, 'a')
      return { click() { saves.push({ href: this.href, name: this.download }) } }
    } }
    globalThis.window = Object.assign(new EventTarget(), { setTimeout: () => 1, clearTimeout() {} })
    const ui = createImageExport({
      signal: lifetime.signal, getScene: () => current,
      render: job => {
        jobs++; request = job
        return new Promise((resolve,reject) => { complete = resolve; fail = reject })
      },
    })
    assert.equal(size.options[1].textContent, '2048 × 1024 px')
    assert.equal(spinner.hidden, true)
    click('image'); submit(); submit()
    assert.equal(jobs, 1)
    assert.equal(dialog.open, true)
    assert.equal(spinner.hidden, false)
    assert.equal(size.disabled, true)

    const image = { width: 2048, height: 1024 }
    complete(image)
    await setImmediate()
    assert.equal(spinner.hidden, false)
    dialog.dispatchEvent(new Event('cancel', { cancelable: true }))
    assert.equal(request.signal.aborted, true)
    finishEncoding(new Blob(['canceled'], { type: 'image/png' }))
    await setImmediate()
    assert.equal(dialog.open, false)
    assert.equal(saves.length, 0)
    assert.equal(spinner.hidden, true)

    click('image'); submit()
    fail(new Error('GPU allocation failed'))
    await setImmediate()
    assert.equal(control('error').textContent, 'GPU allocation failed')
    assert.equal(dialog.open, true)
    assert.equal(control('submit').disabled, false)
    submit()
    complete(image)
    await setImmediate()
    finishEncoding(new Blob(['finished'], { type: 'image/png' }))
    await setImmediate()
    assert.equal(jobs, 3)
    assert.equal(saves.length, 1)
    assert.equal(saves[0].name, 'Harbor-basecolor-2048x1024.png')
    assert.match(saves[0].href, /^blob:/)
    assert.equal(dialog.open, false)
    assert.equal(spinner.hidden, true)

    current = { ...current, materialMode: 'pbr' }
    click('image'); submit()
    complete(image)
    await setImmediate()
    finishEncoding(new Blob(['pbr'], { type: 'image/png' }))
    await setImmediate()
    assert.equal(saves[1].name, 'Harbor-pbr-2048x1024.png')
    current = undefined
    ui.refresh()
    assert.equal(dialog.open, false)
    assert.equal(control('image').disabled, true)
  } finally {
    lifetime.abort()
    for (const [key, value] of Object.entries(previous)) {
      if (value === undefined) delete globalThis[key]
      else globalThis[key] = value
    }
  }
})
