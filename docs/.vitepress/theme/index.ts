import DefaultTheme from 'vitepress/theme-without-fonts'
import type { Theme } from 'vitepress'
import HomePage from './components/home/HomePage.vue'
import '@fontsource-variable/inter/wght.css'
import '@fontsource-variable/jetbrains-mono/wght.css'
import './custom.css'

function isPlaygroundPath(to: string): boolean {
  const path = new URL(to, 'http://vitepress.local').pathname
  return path === '/playground' || path.startsWith('/playground/')
}

export default {
  extends: DefaultTheme,
  enhanceApp({ app, router }) {
    app.component('HomePage', HomePage)
    // Safety net for markdown/nav links that omit target="_self": the playground
    // SPA is not a VitePress page, so client-side routing would 404.
    const previous = router.onBeforeRouteChange
    router.onBeforeRouteChange = async (to) => {
      if (isPlaygroundPath(to)) {
        window.location.assign(to)
        return false
      }
      return previous?.(to)
    }
  }
} satisfies Theme
