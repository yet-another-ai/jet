#!/usr/bin/env node
import { createHash } from 'node:crypto'
import { cp, mkdir, readFile, readdir, writeFile } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const source = join(repo, 'playground/generated/wllama')
const destination = process.argv[2]

if (!destination) {
  console.error('Usage: node scripts/export-playground-runtime.mjs <Space checkout>')
  process.exitCode = 1
} else {
  const space = resolve(destination)
  const pkg = JSON.parse(await readFile(join(space, 'package.json'), 'utf8'))
  if (pkg.name !== 'jet-playground') {
    throw new Error('Destination must be the jet-playground Space checkout')
  }
  const manifest = JSON.parse(await readFile(join(source, 'runtime-manifest.json'), 'utf8'))
  // Verify the build manifest before changing the Space checkout.
  for (const [name, expected] of Object.entries(manifest.files)) {
    const digest = createHash('sha256').update(await readFile(join(source, name))).digest('hex')
    if (digest !== expected) throw new Error(`Runtime checksum mismatch: ${name}`)
  }
  await mkdir(join(space, 'public/playground-runtime'), { recursive: true })
  for (const entry of await readdir(source)) {
    if (entry === 'runtime-manifest.json') continue
    await cp(join(source, entry), join(space, 'public/playground-runtime', entry), {
      recursive: true,
    })
  }
  await writeFile(join(space, 'public/playground-runtime/ready.json'), '{"ok":true}\n')
  await writeFile(join(space, 'runtime-manifest.json'), JSON.stringify(manifest, null, 2) + '\n')
  console.log(`Exported verified runtime to ${space}`)
}
