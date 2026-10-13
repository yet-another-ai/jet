export type CapabilityKey = "decide" | "local" | "embed"
export type InstallPlatform = "macos" | "windows" | "linux"
export type InstallBackend = "cpu" | "cuda" | "metal" | "vulkan"

export interface HomeLink {
  label: string
  href: string
}

export interface CapabilityCopy {
  key: CapabilityKey
  tab: string
  kicker: string
  title: string
  body: string
  links: HomeLink[]
}

export interface ValueCopy {
  title: string
  body: string
}

export interface InstallTarget {
  command: {
    label: string
    lines: string[]
  }
  note: string
}

export interface HomeCopy {
  deskLabel: string
  heroBrand: string
  heroEyebrow: string
  heroTitle: string
  heroLead: string
  heroPrimary: string
  heroSecondary: string
  capabilitiesKicker: string
  capabilitiesTitle: string
  capabilitiesAria: string
  capabilities: [CapabilityCopy, CapabilityCopy, CapabilityCopy]
  valuesKicker: string
  valuesHeadline: string
  valuesTitle: string
  values: [ValueCopy, ValueCopy, ValueCopy, ValueCopy]
  installKicker: string
  installTitle: string
  installLead: string
  installPlatforms: Record<InstallPlatform, string>
  installBackends: Record<InstallBackend, string>
  /** Backends documented for each OS in README / advanced.md. */
  installBackendsByPlatform: Record<InstallPlatform, InstallBackend[]>
  copy: string
  copied: string
  install: Record<InstallPlatform, Partial<Record<InstallBackend, InstallTarget>>>
  installLinks: HomeLink[]
}

export const homeCopy: HomeCopy = {
  deskLabel: "Jet request, candidate scores, and structured result panels working side by side",
  heroBrand: "Jet",
  heroEyebrow: "Local structured decisions",
  heroTitle: "Local decisions, structured answers",
  heroLead:
    "Give a local language model context and fixed candidates. Jet scores them and returns a result you can use directly.",
  heroPrimary: "Get started",
  heroSecondary: "See what it does",
  capabilitiesKicker: "What Jet does",
  capabilitiesTitle: "Score candidates. Keep the schema. Stay local.",
  capabilitiesAria: "Jet capabilities",
  capabilities: [
    {
      key: "decide",
      tab: "Decide",
      kicker: "Fixed candidates",
      title: "Choose, confirm, or rate.",
      body: "Ask a choice, yes/no, or score question over a fixed answer set. Candidate keys and score indexes come from your schema, so downstream code does not need an output parser.",
      links: [
        { label: "Requests and responses", href: "/guide/requests-and-responses" },
        { label: "Getting started", href: "/guide/getting-started" }
      ]
    },
    {
      key: "local",
      tab: "Local",
      kicker: "Local inference",
      title: "Run where the data already is.",
      body: "Use GGUF models on CPU by default, or offload to CUDA, Metal, or Vulkan. With vision enabled, score the same structured questions against PNG or JPEG inputs—including the Doom demo’s screenshot-driven controls.",
      links: [
        { label: "Build and backends", href: "/advanced" },
        { label: "Doom multimodal demo", href: "/#doom" }
      ]
    },
    {
      key: "embed",
      tab: "Embed",
      kicker: "CLI and Rust API",
      title: "Stream JSONL or call from Rust.",
      body: "Send one request per line and receive ordered, machine-readable results from the CLI, or embed the decision engine with typed requests, responses, configuration, and error handling.",
      links: [
        { label: "Rust API", href: "/guide/rust-api" },
        { label: "Continuous integration", href: "/ci" }
      ]
    }
  ],
  valuesKicker: "Why Jet",
  valuesHeadline: "Structured answers.",
  valuesTitle: "Fixed candidates in. Predictable JSON out.",
  values: [
    {
      title: "Fixed candidate scoring",
      body: "Choose an answer, make a yes/no decision, or rate an item without parsing free-form model output."
    },
    {
      title: "Local inference",
      body: "Run GGUF models on CPU, CUDA, Metal, or Vulkan without standing up an HTTP service."
    },
    {
      title: "Streaming JSONL",
      body: "Process one request per line and receive ordered, machine-readable results from the CLI."
    },
    {
      title: "Rust API",
      body: "Embed the decision engine directly with typed requests, responses, configuration, and error handling."
    }
  ],
  installKicker: "Get Jet",
  installTitle: "Build it once. Run it locally.",
  installLead:
    "Pick your OS and backend. Commands below match the repository getting-started and advanced guides—clone with submodules first, then install mise 2026.10.0 or newer.",
  installPlatforms: {
    macos: "macOS",
    windows: "Windows",
    linux: "Linux"
  },
  installBackends: {
    cpu: "CPU",
    cuda: "CUDA",
    metal: "Metal",
    vulkan: "Vulkan"
  },
  installBackendsByPlatform: {
    macos: ["cpu", "metal"],
    windows: ["cpu", "cuda", "vulkan"],
    linux: ["cpu", "cuda", "vulkan"]
  },
  copy: "Copy",
  copied: "Copied",
  install: {
    macos: {
      cpu: {
        command: {
          label: "Build and run the CPU CLI",
          lines: [
            "mise install",
            "./scripts/download-test-model.sh",
            "mise exec -- cargo build --release -p jet-cli",
            "./target/release/jet judge --model-path models/Qwen3-0.6B-Q8_0.gguf --input tests/fixtures/decisions.jsonl --output -"
          ]
        },
        note: "CPU is the default backend. Requires mise and a toolchain that can build Rust plus the bundled llama.cpp."
      },
      metal: {
        command: {
          label: "Build and run with Metal",
          lines: [
            "mise exec -- cargo build --release -p jet-cli --features metal",
            "./target/release/jet judge --backend metal --model-path models/Qwen3-0.6B-Q8_0.gguf --input tests/fixtures/decisions.jsonl --output -"
          ]
        },
        note: "Apple Silicon with Xcode Command Line Tools. Download the example model first if you have not already. mise run package:metal produces dist/jet-metal-macos-arm64.tar.gz."
      }
    },
    windows: {
      cpu: {
        command: {
          label: "Build and run the CPU CLI",
          lines: [
            "mise install",
            "./scripts/download-test-model.sh",
            "mise exec -- cargo build --release -p jet-cli",
            "./target/release/jet judge --model-path models/Qwen3-0.6B-Q8_0.gguf --input tests/fixtures/decisions.jsonl --output -"
          ]
        },
        note: "CPU is the default backend. The model download script is a POSIX shell script; run it from Git Bash, WSL, or another sh-compatible shell. mise run package:cpu produces a Windows x64 archive."
      },
      cuda: {
        command: {
          label: "Build and run with CUDA",
          lines: [
            "mise exec -- cargo build --release -p jet-cli --features cuda",
            "./target/release/jet judge --backend cuda --model-path models/Qwen3-0.6B-Q8_0.gguf --input tests/fixtures/decisions.jsonl --output -"
          ]
        },
        note: "Needs an NVIDIA driver and CUDA Toolkit with nvcc. To package CUDA runtime DLLs for distribution, see the Windows PowerShell steps in the advanced guide (mise run package:cuda)."
      },
      vulkan: {
        command: {
          label: "Build and run with Vulkan",
          lines: [
            "mise exec -- cargo build --release -p jet-cli --features vulkan",
            "./target/release/jet judge --backend vulkan --model-path models/Qwen3-0.6B-Q8_0.gguf --input tests/fixtures/decisions.jsonl --output -"
          ]
        },
        note: "Install the Vulkan SDK (loader, headers, SPIR-V headers, and glslc). An explicit --backend vulkan fails if the feature or device is unavailable."
      }
    },
    linux: {
      cpu: {
        command: {
          label: "Build and run the CPU CLI",
          lines: [
            "mise install",
            "./scripts/download-test-model.sh",
            "mise exec -- cargo build --release -p jet-cli",
            "./target/release/jet judge --model-path models/Qwen3-0.6B-Q8_0.gguf --input tests/fixtures/decisions.jsonl --output -"
          ]
        },
        note: "CPU is the default backend. Clone with --recurse-submodules so the pinned llama.cpp submodule is present."
      },
      cuda: {
        command: {
          label: "Build and run with CUDA",
          lines: [
            "mise exec -- cargo build --release -p jet-cli --features cuda",
            "./target/release/jet judge --backend cuda --model-path models/Qwen3-0.6B-Q8_0.gguf --input tests/fixtures/decisions.jsonl --output -"
          ]
        },
        note: "Needs an NVIDIA driver and CUDA Toolkit with nvcc. Set JET_CUDA_ARCHITECTURES when building for a different GPU or without a local GPU."
      },
      vulkan: {
        command: {
          label: "Build and run with Vulkan",
          lines: [
            "mise exec -- cargo build --release -p jet-cli --features vulkan",
            "./target/release/jet judge --backend vulkan --model-path models/Qwen3-0.6B-Q8_0.gguf --input tests/fixtures/decisions.jsonl --output -"
          ]
        },
        note: "Install a Vulkan loader and development headers, SPIR-V headers, and glslc (from shaderc or the Vulkan SDK)."
      }
    }
  },
  installLinks: [
    { label: "Getting started guide", href: "/guide/getting-started" },
    { label: "Build and backends", href: "/advanced" },
    { label: "View on GitHub", href: "https://github.com/yet-another-ai/jet" }
  ]
}
