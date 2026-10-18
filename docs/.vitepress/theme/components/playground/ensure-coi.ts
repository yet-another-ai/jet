/**
 * GitHub Pages cannot set COOP/COEP. Register the scoped service worker so
 * SharedArrayBuffer works for the playground page (`/playground/` scope).
 * When Vite already sends the headers (docs:dev / docs:preview), this no-ops.
 *
 * Registers by absolute URL (does not inject the stock coi-serviceworker.js
 * window bootstrap, which relies on document.currentScript).
 */

const RELOAD_FLAG = 'jet-playground-coi-reload'
const WORKER_URL = '/playground/coi-serviceworker.js'

export type CoiStatus = 'ok' | 'reloading' | 'unavailable'

function coepCredentialless(): boolean {
  // Match coi-serviceworker default: credentialless outside Chromium/Firefox.
  return !(
    typeof window !== 'undefined' &&
    ((window as Window & { chrome?: unknown }).chrome ||
      (window as Window & { netscape?: unknown }).netscape)
  )
}

function waitForWorker(worker: ServiceWorker): Promise<void> {
  if (worker.state === 'activated') return Promise.resolve()
  return new Promise((resolve) => {
    worker.addEventListener('statechange', () => {
      if (worker.state === 'activated') resolve()
    })
  })
}

export async function ensurePlaygroundCoi(): Promise<CoiStatus> {
  if (typeof window === 'undefined') return 'unavailable'

  if (window.crossOriginIsolated) {
    try {
      sessionStorage.removeItem(RELOAD_FLAG)
    } catch {
      /* ignore */
    }
    return 'ok'
  }

  // undefined means the browser has no notion of isolation.
  if (window.crossOriginIsolated !== false) return 'unavailable'
  if (!window.isSecureContext) return 'unavailable'
  if (!('serviceWorker' in navigator)) return 'unavailable'

  try {
    const registration = await navigator.serviceWorker.register(WORKER_URL)
    if (navigator.serviceWorker.controller) {
      navigator.serviceWorker.controller.postMessage({
        type: 'coepCredentialless',
        value: coepCredentialless()
      })
    }

    const worker =
      registration.installing || registration.waiting || registration.active
    if (worker) await waitForWorker(worker)

    // Already reloaded once for this tab and still not isolated — give up.
    try {
      if (sessionStorage.getItem(RELOAD_FLAG) === '1') return 'unavailable'
      sessionStorage.setItem(RELOAD_FLAG, '1')
    } catch {
      /* private mode: still attempt one reload */
    }

    // Document must be fetched under the SW to receive COOP/COEP.
    window.location.reload()
    return 'reloading'
  } catch {
    return 'unavailable'
  }
}
