import DefaultTheme from 'vitepress/theme-without-fonts'
import { defineClientComponent } from 'vitepress'
import type { Theme } from 'vitepress'
import HomePage from './components/home/HomePage.vue'
import '@fontsource-variable/inter/wght.css'
import '@fontsource-variable/jetbrains-mono/wght.css'
import './custom.css'

const PlaygroundApp = defineClientComponent(
  () => import('./components/playground/PlaygroundApp.vue')
)

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('HomePage', HomePage)
    app.component('PlaygroundApp', PlaygroundApp)
  }
} satisfies Theme
