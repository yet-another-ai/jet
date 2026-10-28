import { defineConfig } from 'vitepress'

const repository = 'https://github.com/yet-another-ai/jet'

export default defineConfig({
  title: 'Jet',
  titleTemplate: ':title · Jet',
  description: 'Local structured decisions with language models',
  base: '/',
  cleanUrls: true,
  lastUpdated: true,
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
  themeConfig: {
    nav: [
      { text: 'Guide', link: '/guide/getting-started' },
      { text: 'Playground', link: 'https://huggingface.co/spaces/yet-another-ai/jet-playground' },
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
