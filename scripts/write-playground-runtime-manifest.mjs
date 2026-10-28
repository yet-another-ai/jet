#!/usr/bin/env node
import { createHash } from 'node:crypto'
import { execFileSync } from 'node:child_process'
import { readFile, readdir, writeFile } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const runtime = join(repo, 'playground/generated/wllama')
const files = {}
async function collect(relative = '') {
  for (const entry of await readdir(join(runtime, relative), { withFileTypes: true })) {
    const name = relative ? `${relative}/${entry.name}` : entry.name
    if (name === 'runtime-manifest.json') continue
    if (entry.isDirectory()) await collect(name)
    else files[name] = createHash('sha256').update(await readFile(join(runtime, name))).digest('hex')
  }
}
await collect()
for (const name of ['index.js', 'wasm/wllama.wasm']) {
  if (!files[name]) throw new Error(`Runtime build is incomplete: ${name}`)
}
const git = (...args) => execFileSync('git', ['-C', repo, ...args], { encoding: 'utf8' }).trim()
await writeFile(join(runtime, 'runtime-manifest.json'), JSON.stringify({
  source_repository: 'https://github.com/yet-another-ai/jet',
  source_revision: git('rev-parse', 'HEAD'),
  source_worktree_dirty: git('status', '--porcelain', '--untracked-files=no') !== '',
  llama_cpp_revision: git('rev-parse', 'HEAD:vendor/llama.cpp'),
  upstream_wllama_revision: 'dbc42c093dfd911360be12064ac8575fc629faa7',
  files,
}, null, 2) + '\n')
