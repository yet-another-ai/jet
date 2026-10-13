export type CapabilityKey = "decide" | "local" | "embed"
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
  installBackends: Record<InstallBackend, string>
  copy: string
  copied: string
  install: Record<InstallBackend, InstallTarget>
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
    "Install the pinned toolchain, download the example model, and build the CLI for the backend you need.",
  installBackends: {
    cpu: "CPU",
    cuda: "CUDA",
    metal: "Metal",
    vulkan: "Vulkan"
  },
  copy: "Copy",
  copied: "Copied",
  install: {
    cpu: {
      command: {
        label: "Build the CPU CLI",
        lines: [
          "mise install",
          "./scripts/download-test-model.sh",
          "mise exec -- cargo build --release -p jet-cli"
        ]
      },
      note: "CPU is the default backend. No GPU toolchain is required."
    },
    cuda: {
      command: {
        label: "Build with CUDA",
        lines: ["mise exec -- cargo build --release -p jet-cli --features cuda"]
      },
      note: "Needs an NVIDIA driver and CUDA Toolkit. Pass --backend cuda when you run Jet."
    },
    metal: {
      command: {
        label: "Build with Metal",
        lines: ["mise exec -- cargo build --release -p jet-cli --features metal"]
      },
      note: "Requires Apple Silicon and Xcode Command Line Tools. Pass --backend metal when you run Jet."
    },
    vulkan: {
      command: {
        label: "Build with Vulkan",
        lines: ["mise exec -- cargo build --release -p jet-cli --features vulkan"]
      },
      note: "Needs a Vulkan loader, development headers, SPIR-V headers, and glslc."
    }
  },
  installLinks: [
    { label: "Getting started guide", href: "/guide/getting-started" },
    { label: "View on GitHub", href: "https://github.com/yet-another-ai/jet" }
  ]
}
