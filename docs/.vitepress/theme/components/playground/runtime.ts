export type Progress = {
  loaded: number
  total: number
}

type WllamaModule = {
  LoggerWithoutDebug: unknown
  Wllama: new (
    paths: { default: string },
    options: { logger: unknown; suppressNativeLog: boolean }
  ) => {
    loadModelFromUrl: (
      url: string,
      options: Record<string, unknown>
    ) => Promise<void>
    exit: () => Promise<void>
    isModelLoaded: () => boolean
    jetDecide: (request: unknown) => Promise<unknown>
  }
}

const RUNTIME_BASE = '/playground-runtime'

let model: InstanceType<WllamaModule['Wllama']> | undefined
let loadedModelUrl: string | undefined
let runtimeModule: WllamaModule | undefined
let runtimeMissing = false

export function supportsWebGpu(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator
}

export async function probeRuntime(): Promise<boolean> {
  if (runtimeMissing) return false
  if (runtimeModule) return true
  try {
    // Prefer the small ready.json sentinel written by prepare-playground-runtime.sh;
    // fall back to index.js so a plain copy of generated/wllama still works.
    for (const path of ['ready.json', 'index.js'] as const) {
      const response = await fetch(`${RUNTIME_BASE}/${path}`, { cache: 'no-store' })
      if (response.ok) {
        if (path === 'index.js') {
          await response.arrayBuffer()
        }
        return true
      }
    }
    runtimeMissing = true
    return false
  } catch {
    runtimeMissing = true
    return false
  }
}

async function loadRuntime(): Promise<WllamaModule> {
  if (runtimeModule) return runtimeModule
  if (runtimeMissing) {
    throw new Error(
      'Playground Wasm runtime is not available. Build it with `mise run playground:wllama`.'
    )
  }
  try {
    runtimeModule = (await import(
      /* @vite-ignore */ `${RUNTIME_BASE}/index.js`
    )) as WllamaModule
    return runtimeModule
  } catch (error) {
    runtimeMissing = true
    throw error instanceof Error
      ? error
      : new Error('Failed to load the playground Wasm runtime.')
  }
}

export async function loadModel(
  modelUrl: string,
  onProgress: (progress: Progress) => void
): Promise<void> {
  if (!supportsWebGpu()) {
    throw new Error('This browser does not expose WebGPU.')
  }
  if (model && loadedModelUrl === modelUrl) {
    return
  }
  if (model) {
    await model.exit()
    model = undefined
    loadedModelUrl = undefined
  }

  const { Wllama, LoggerWithoutDebug } = await loadRuntime()
  const wllamaWasmUrl = `${RUNTIME_BASE}/wasm/wllama.wasm`
  const nextModel = new Wllama(
    { default: wllamaWasmUrl },
    { logger: LoggerWithoutDebug, suppressNativeLog: true }
  )
  const modelId = decodeURIComponent(modelUrl.split('/').pop() || modelUrl)
  try {
    await nextModel.loadModelFromUrl(modelUrl, {
      jet: {
        model_id: modelId,
        context_tokens_per_sequence: 1024,
        max_sequences: 9,
        max_output_rows: 256,
        preparation_pipeline: true,
        execution_mode: 'batched',
        thinking: {
          mode: 'disabled',
          max_tokens: 256
        },
        collect_timings: true
      },
      progressCallback: onProgress
    })
  } catch (error) {
    await nextModel.exit()
    throw error
  }
  model = nextModel
  loadedModelUrl = modelUrl
}

export function isModelLoaded(modelUrl: string): boolean {
  return model?.isModelLoaded() === true && loadedModelUrl === modelUrl
}

export async function runDecision(
  modelUrl: string,
  request: unknown
): Promise<unknown> {
  if (!model || !isModelLoaded(modelUrl)) {
    throw new Error('Load this model before running Jet.')
  }
  return model.jetDecide(request)
}
