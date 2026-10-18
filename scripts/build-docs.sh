#!/bin/sh
# Build the VitePress site and embed the WebGPU playground at /playground/.
# When the Wasm module is missing locally, write a stub page unless
# JET_DOCS_REQUIRE_PLAYGROUND=1 (CI sets this).
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

dist_dir="$repo_root/docs/.vitepress/dist"
playground_out="$dist_dir/playground"
generated="$repo_root/playground/generated/wllama"

pnpm exec vitepress build docs

embed_playground() {
  echo "Embedding WebGPU playground into $playground_out"
  (
    cd "$repo_root/playground"
    JET_PLAYGROUND_BASE=/playground/ \
      JET_PLAYGROUND_OUT_DIR="$playground_out" \
      pnpm build
  )
}

write_stub() {
  mkdir -p "$playground_out"
  cat >"$playground_out/index.html" <<'EOF'
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Jet WebGPU Playground</title>
    <style>
      :root {
        color-scheme: light dark;
        font-family: ui-sans-serif, system-ui, sans-serif;
        line-height: 1.5;
      }
      body {
        margin: 0;
        min-height: 100vh;
        display: grid;
        place-items: center;
        background: #0b1220;
        color: #e2e8f0;
      }
      main {
        max-width: 36rem;
        padding: 2rem;
      }
      a {
        color: #5eead4;
      }
      code {
        font-family: ui-monospace, monospace;
        font-size: 0.95em;
      }
    </style>
  </head>
  <body>
    <main>
      <h1>Playground Wasm not built</h1>
      <p>
        This docs build did not include the WebGPU Wasm module. Build it locally,
        then rebuild the docs:
      </p>
      <p><code>mise run playground:wllama && mise exec -- pnpm docs:build</code></p>
      <p>
        See the
        <a href="/guide/playground">playground guide</a>
        for requirements (Docker, Emscripten, nightly Rust).
      </p>
    </main>
  </body>
</html>
EOF
  echo "warning: playground Wasm missing at $generated; wrote stub at /playground/" >&2
}

if [ -d "$generated" ]; then
  embed_playground
elif [ "${JET_DOCS_REQUIRE_PLAYGROUND:-0}" = 1 ]; then
  echo "error: JET_DOCS_REQUIRE_PLAYGROUND=1 but $generated is missing" >&2
  exit 1
else
  write_stub
fi
