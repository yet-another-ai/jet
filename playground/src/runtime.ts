import { LoggerWithoutDebug, Wllama } from '../generated/wllama/index.js';

const wllamaWasmUrl = new URL(
  '../generated/wllama/wasm/wllama.wasm',
  import.meta.url,
).href;

let model: Wllama | undefined;
let loadedModelUrl: string | undefined;

export type Progress = {
  loaded: number;
  total: number;
};

export function supportsWebGpu(): boolean {
  return 'gpu' in navigator;
}

export async function loadModel(
  modelUrl: string,
  onProgress: (progress: Progress) => void,
): Promise<void> {
  if (!supportsWebGpu()) {
    throw new Error('This browser does not expose WebGPU.');
  }
  if (model && loadedModelUrl === modelUrl) {
    return;
  }
  if (model) {
    await model.exit();
    model = undefined;
    loadedModelUrl = undefined;
  }

  const nextModel = new Wllama(
    { default: wllamaWasmUrl },
    { logger: LoggerWithoutDebug, suppressNativeLog: true },
  );
  const modelId = decodeURIComponent(modelUrl.split('/').pop() || modelUrl);
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
          max_tokens: 256,
        },
        collect_timings: true,
      },
      progressCallback: onProgress,
    });
  } catch (error) {
    await nextModel.exit();
    throw error;
  }
  model = nextModel;
  loadedModelUrl = modelUrl;
}

export function isModelLoaded(modelUrl: string): boolean {
  return model?.isModelLoaded() === true && loadedModelUrl === modelUrl;
}

export async function runDecision(
  modelUrl: string,
  request: unknown,
): Promise<unknown> {
  if (!model || !isModelLoaded(modelUrl)) {
    throw new Error('Load this model before running Jet.');
  }
  return model.jetDecide(request);
}
