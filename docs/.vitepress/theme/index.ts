import DefaultTheme from 'vitepress/theme-without-fonts'
import type { Theme } from 'vitepress'
import HomePage from './components/home/HomePage.vue'
import '@fontsource-variable/inter/wght.css'
import '@fontsource-variable/jetbrains-mono/wght.css'
import './custom.css'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('HomePage', HomePage)
  }
} satisfies Theme
