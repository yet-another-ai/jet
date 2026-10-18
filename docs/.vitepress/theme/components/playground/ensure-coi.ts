/**
 * GitHub Pages cannot set COOP/COEP. Load the scoped service worker shim so
 * SharedArrayBuffer works for the playground page only (`/playground/` scope).
 * When Vite already sends the headers (docs:dev / docs:preview), this no-ops.
 */
export function ensurePlaygroundCoi(): void {
  if (typeof window === 'undefined') return
  if (window.crossOriginIsolated !== false) return
  if (!window.isSecureContext) return
  if (!('serviceWorker' in navigator)) return
  if (document.querySelector('script[data-jet-coi]')) return

  const script = document.createElement('script')
  script.src = '/playground/coi-serviceworker.js'
  script.dataset.jetCoi = '1'
  document.head.appendChild(script)
}
