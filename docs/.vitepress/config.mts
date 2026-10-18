import { defineConfig } from 'vitepress'

const repository = 'https://github.com/yet-another-ai/jet'

const isolationHeaders = {
  'Cross-Origin-Embedder-Policy': 'require-corp',
  'Cross-Origin-Opener-Policy': 'same-origin'
}

export default defineConfig({
  title: 'Jet',
  titleTemplate: ':title · Jet',
  description: 'Local structured decisions with language models',
  base: '/',
  cleanUrls: true,
  lastUpdated: true,
  // /playground/ is the WebGPU SPA merged after the VitePress build (CI or
  // scripts/build-docs.sh). Ignore it during the VitePress link check.
  ignoreDeadLinks: [/^\/playground/],
  sitemap: {
    hostname: 'https://jet.yetanother.ai'
  },
  head: [
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:site_name', content: 'Jet' }],
    ['meta', { property: 'og:title', content: 'Jet documentation' }],
    ['meta', {
      property: 'og:description',
      content: 'Score fixed candidate answers with a local language model.'
    }],
    ['meta', { name: 'twitter:card', content: 'summary' }]
  ],
  vite: {
    // Keep Vue SFC resolution quiet for the custom landing theme.
    // COOP/COEP let the embedded /playground/ assets use SharedArrayBuffer
    // during local docs:dev / docs:preview. GitHub Pages uses the
    // coi-serviceworker shim instead (see playground/index.html).
    server: {
      fs: {
        allow: ['..']
      },
      headers: isolationHeaders
    },
    preview: {
      headers: isolationHeaders
    }
  },
  themeConfig: {
    nav: [
      { text: 'Guide', link: '/guide/getting-started' },
      // Full-page load: /playground/ is a separate SPA merged into dist, not a
      // VitePress route. Without target, the client router 404s until refresh.
      { text: 'Playground', link: '/playground/', target: '_self' },
      { text: 'Results', link: '/results' },
      { text: 'Advanced', link: '/advanced' },
      { text: 'Get Jet', link: '/#get-jet' },
      { text: 'GitHub', link: repository }
    ],
    sidebar: [
      {
        text: 'Guide',
        items: [
          { text: 'Getting started', link: '/guide/getting-started' },
          { text: 'Requests and responses', link: '/guide/requests-and-responses' },
          { text: 'HTTP server', link: '/guide/http-server' },
          { text: 'Rust API', link: '/guide/rust-api' },
          { text: 'WebGPU playground', link: '/guide/playground' }
        ]
      },
      {
        text: 'Reference',
        items: [
          { text: 'Model comparison', link: '/results' },
          { text: 'Build and advanced usage', link: '/advanced' },
          { text: 'Continuous integration', link: '/ci' }
        ]
      }
    ],
    search: {
      provider: 'local'
    },
    socialLinks: [
      { icon: 'github', link: repository }
    ],
    editLink: {
      pattern: `${repository}/edit/main/docs/:path`,
      text: 'Edit this page on GitHub'
    },
    outline: {
      level: [2, 3]
    },
    footer: {
      message: 'Released under the Apache-2.0 License.',
      copyright: 'Copyright © 2026 the Jet contributors'
    }
  }
})
