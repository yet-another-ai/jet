use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[cfg(feature = "vision")]
use base64::Engine as _;
use clap::{Parser, Subcommand, ValueEnum};
use jet_core::{DecisionRequest, ErrorCode, JetError, Result};
use jet_engine::{Engine, EngineConfig, ExecutionMode, ThinkingMode};
#[cfg(feature = "vision")]
use jet_engine::{ImageInput, MultimodalDecisionRequest, VisionConfig};
#[cfg(feature = "vision")]
use serde::Deserialize;
use serde::Serialize;

#[derive(Parser)]
#[command(name = "jet", version, about = "Local teacher-forced decision scoring")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Judge(JudgeArgs),
    #[cfg(feature = "vision")]
    JudgeMultimodal(JudgeMultimodalArgs),
}

#[cfg(feature = "vision")]
#[derive(clap::Args)]
struct JudgeMultimodalArgs {
    #[arg(long)]
    model_path: PathBuf,
    #[arg(long, default_value = "qwen/qwen3.6-35b-a3b-q4_k_m")]
    model_id: String,
    #[arg(long, default_value = "-")]
    input: PathBuf,
    #[arg(long, default_value = "-")]
    output: PathBuf,
    #[arg(long, value_name = "PATH", value_parser = parse_timings_path)]
    timings: Option<PathBuf>,
    #[arg(long)]
    mmproj_path: PathBuf,
    #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
    backend: BackendArg,
    #[arg(long)]
    gpu_layers: Option<u32>,
    #[arg(long, default_value_t = 0)]
    cpu_moe_layers: u32,
    #[arg(long)]
    no_mmap: bool,
    #[arg(long, default_value_t = 4096)]
    context_tokens: u32,
    #[arg(long, default_value_t = 2048)]
    token_batch: u32,
    #[arg(long, default_value_t = 512)]
    micro_batch: u32,
    #[arg(long, default_value_t = 2)]
    max_sequences: u32,
    #[arg(long, default_value_t = 256)]
    max_output_rows: u32,
    #[arg(long, value_enum, default_value_t = ExecutionArg::Batched)]
    execution: ExecutionArg,
    #[arg(long)]
    threads: Option<i32>,
    #[arg(long, default_value_t = 4)]
    max_images: usize,
    #[arg(long, default_value_t = 10 * 1024 * 1024)]
    max_image_bytes: usize,
    #[arg(long, default_value_t = 16 * 1024 * 1024)]
    max_image_pixels: usize,
    #[arg(long, default_value_t = 1024)]
    image_max_tokens: i32,
}

#[derive(clap::Args)]
struct JudgeArgs {
    #[arg(long)]
    model_path: PathBuf,
    #[arg(long, default_value = "qwen/qwen3-0.6b-q8_0")]
    model_id: String,
    #[arg(long, default_value = "-")]
    input: PathBuf,
    #[arg(long, default_value = "-")]
    output: PathBuf,
    /// Write cumulative engine stage timings to a separate JSON file.
    #[arg(long, value_name = "PATH", value_parser = parse_timings_path)]
    timings: Option<PathBuf>,
    #[arg(long, default_value_t = 8)]
    batch_requests: usize,
    #[arg(long, default_value_t = 2_048)]
    context_tokens: u32,
    #[arg(long, default_value_t = 2_048)]
    token_batch: u32,
    #[arg(long, default_value_t = 512)]
    micro_batch: u32,
    #[arg(long, default_value_t = 9)]
    max_sequences: u32,
    /// Disable overlapping CPU prompt preparation with GPU hybrid execution.
    #[arg(long)]
    no_preparation_pipeline: bool,
    #[arg(long)]
    max_output_rows: Option<u32>,
    #[arg(long)]
    threads: Option<i32>,
    #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
    backend: BackendArg,
    /// Number of model layers to place on GPU; GPU backends default to all layers.
    #[arg(long)]
    gpu_layers: Option<u32>,
    /// Keep routed MoE experts in the first N layers on CPU (GPU backends only).
    #[arg(long, default_value_t = 0)]
    cpu_moe_layers: u32,
    /// Load weights into allocated memory instead of mapping the model file.
    #[arg(long)]
    no_mmap: bool,
    #[arg(long, value_enum, default_value_t = ExecutionArg::Batched)]
    execution: ExecutionArg,
    #[arg(long, value_enum, default_value_t = ThinkingArg::Disabled)]
    thinking: ThinkingArg,
    #[arg(long, default_value_t = 256)]
    thinking_tokens: u32,
    #[arg(long, default_value_t = 0.6)]
    thinking_temperature: f32,
    #[arg(long, default_value_t = 20)]
    thinking_top_k: i32,
    #[arg(long, default_value_t = 0.95)]
    thinking_top_p: f32,
    #[arg(long, default_value_t = 0)]
    thinking_seed: u32,
}

#[derive(Clone, Copy, ValueEnum)]
enum ExecutionArg {
    Reference,
    Batched,
}

#[derive(Clone, Copy, ValueEnum)]
enum BackendArg {
    Auto,
    Cpu,
    Vulkan,
    Cuda,
}

#[derive(Clone, Copy, ValueEnum)]
enum ThinkingArg {
    Disabled,
    Auto,
    Required,
}

#[derive(Serialize)]
struct ErrorEnvelope {
    error: ErrorBody,
}

#[derive(Serialize)]
struct ErrorBody {
    code: ErrorCode,
    message: String,
}

enum InputLine {
    Request(DecisionRequest),
    Error(ErrorEnvelope),
}

#[cfg(feature = "vision")]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MultimodalInputLine {
    #[serde(default)]
    request_id: Option<String>,
    #[serde(default)]
    state: serde_json::Value,
    images: Vec<WireImage>,
    questions: std::collections::BTreeMap<String, jet_core::Question>,
}

#[cfg(feature = "vision")]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireImage {
    id: String,
    source: ImageSource,
}

#[cfg(feature = "vision")]
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum ImageSource {
    Base64 { media_type: String, data: String },
    Path { media_type: String, path: PathBuf },
}

#[cfg(feature = "vision")]
#[derive(Serialize)]
struct MultimodalErrorEnvelope {
    #[serde(skip_serializing_if = "Option::is_none")]
    request_id: Option<String>,
    error: ErrorBody,
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(io::stderr)
        .init();

    match Cli::parse().command {
        Command::Judge(args) => match run_judge(args) {
            Ok(had_errors) if had_errors => ExitCode::FAILURE,
            Ok(_) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("jet: {error}");
                ExitCode::FAILURE
            }
        },
        #[cfg(feature = "vision")]
        Command::JudgeMultimodal(args) => match run_judge_multimodal(args) {
            Ok(had_errors) if had_errors => ExitCode::FAILURE,
            Ok(_) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("jet: {error}");
                ExitCode::FAILURE
            }
        },
    }
}

fn run_judge(args: JudgeArgs) -> Result<bool> {
    if args.batch_requests == 0 {
        return Err(JetError::InvalidRequest(
            "--batch-requests must be positive".to_owned(),
        ));
    }
    if let Some(timings) = &args.timings {
        validate_timings_path(timings, &args.input, &args.output)?;
    }

    let config = engine_config(&args);
    let mut engine = Engine::load(config)?;
    let input = open_input(&args.input)?;
    let mut output = open_output(&args.output)?;
    let mut chunk = Vec::with_capacity(args.batch_requests);
    let mut had_errors = false;
    for line in input.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<DecisionRequest>(&line) {
            Ok(request) => chunk.push(InputLine::Request(request)),
            Err(error) => chunk.push(InputLine::Error(error_envelope(JetError::Json(error)))),
        }
        if chunk.len() == args.batch_requests {
            had_errors |= process_chunk(&mut engine, &chunk, &mut *output)?;
            chunk.clear();
        }
    }
    if !chunk.is_empty() {
        had_errors |= process_chunk(&mut engine, &chunk, &mut *output)?;
    }
    output.flush()?;
    if let Some(path) = args.timings {
        let mut timings_output = BufWriter::new(File::create(path)?);
        serde_json::to_writer_pretty(&mut timings_output, engine.timings())?;
        timings_output.write_all(b"\n")?;
        timings_output.flush()?;
    }
    Ok(had_errors)
}

fn engine_config(args: &JudgeArgs) -> EngineConfig {
    let mut config = match args.backend {
        BackendArg::Auto => EngineConfig::auto(&args.model_path, &args.model_id),
        BackendArg::Cpu => EngineConfig::cpu(&args.model_path, &args.model_id),
        BackendArg::Vulkan => EngineConfig::vulkan(&args.model_path, &args.model_id),
        BackendArg::Cuda => EngineConfig::cuda(&args.model_path, &args.model_id),
    };
    config.gpu_layers = args.gpu_layers;
    config.cpu_moe_layers = args.cpu_moe_layers;
    config.use_mmap = !args.no_mmap;
    config.context_tokens_per_sequence = args.context_tokens;
    config.token_batch = args.token_batch;
    config.micro_batch = args.micro_batch;
    config.max_sequences = args.max_sequences;
    config.preparation_pipeline = !args.no_preparation_pipeline;
    if let Some(max_output_rows) = args.max_output_rows {
        config.max_output_rows = max_output_rows;
    }
    if let Some(threads) = args.threads {
        config.threads = threads;
    }
    config.execution_mode = match args.execution {
        ExecutionArg::Reference => ExecutionMode::Reference,
        ExecutionArg::Batched => ExecutionMode::Batched,
    };
    config.thinking.mode = match args.thinking {
        ThinkingArg::Disabled => ThinkingMode::Disabled,
        ThinkingArg::Auto => ThinkingMode::Auto,
        ThinkingArg::Required => ThinkingMode::Required,
    };
    config.thinking.max_tokens = args.thinking_tokens;
    config.thinking.temperature = args.thinking_temperature;
    config.thinking.top_k = args.thinking_top_k;
    config.thinking.top_p = args.thinking_top_p;
    config.thinking.seed = args.thinking_seed;
    config.collect_timings = args.timings.is_some();
    config
}

#[cfg(feature = "vision")]
fn run_judge_multimodal(args: JudgeMultimodalArgs) -> Result<bool> {
    if let Some(timings) = &args.timings {
        validate_timings_path(timings, &args.input, &args.output)?;
    }
    let mut config = match args.backend {
        BackendArg::Auto => EngineConfig::auto(&args.model_path, args.model_id),
        BackendArg::Cpu => EngineConfig::cpu(&args.model_path, args.model_id),
        BackendArg::Vulkan => EngineConfig::vulkan(&args.model_path, args.model_id),
        BackendArg::Cuda => EngineConfig::cuda(&args.model_path, args.model_id),
    };
    config.gpu_layers = args.gpu_layers;
    config.cpu_moe_layers = args.cpu_moe_layers;
    config.use_mmap = !args.no_mmap;
    config.context_tokens_per_sequence = args.context_tokens;
    config.token_batch = args.token_batch;
    config.micro_batch = args.micro_batch;
    config.max_sequences = args.max_sequences;
    config.max_output_rows = args.max_output_rows;
    config.execution_mode = match args.execution {
        ExecutionArg::Reference => ExecutionMode::Reference,
        ExecutionArg::Batched => ExecutionMode::Batched,
    };
    if let Some(threads) = args.threads {
        config.threads = threads;
    }
    let mut vision = VisionConfig::new(args.mmproj_path);
    vision.max_images = args.max_images;
    vision.max_image_bytes = args.max_image_bytes;
    vision.max_image_pixels = args.max_image_pixels;
    vision.image_max_tokens = args.image_max_tokens;
    config.vision = Some(vision);
    config.collect_timings = args.timings.is_some();
    let mut engine = Engine::load(config)?;
    let input = open_input(&args.input)?;
    let mut output = open_output(&args.output)?;
    let base_dir = if args.input == Path::new("-") {
        std::env::current_dir()?
    } else {
        args.input
            .canonicalize()?
            .parent()
            .ok_or_else(|| JetError::InvalidRequest("input path has no parent".to_owned()))?
            .to_path_buf()
    };
    let mut had_errors = false;
    for line in input.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let parsed = serde_json::from_str::<MultimodalInputLine>(&line);
        let request_id = parsed
            .as_ref()
            .ok()
            .and_then(|input| input.request_id.clone());
        let result = parsed
            .map_err(JetError::from)
            .and_then(|input| multimodal_request(input, &base_dir, args.max_image_bytes))
            .and_then(|request| engine.decide_multimodal(request));
        match result {
            Ok(response) => write_json_line(&mut *output, &response)?,
            Err(error) => {
                had_errors = true;
                write_json_line(
                    &mut *output,
                    &MultimodalErrorEnvelope {
                        request_id,
                        error: ErrorBody {
                            code: error.code(),
                            message: error.to_string(),
                        },
                    },
                )?;
            }
        }
        output.flush()?;
    }
    if let Some(path) = args.timings {
        let mut timings_output = BufWriter::new(File::create(path)?);
        serde_json::to_writer_pretty(&mut timings_output, engine.timings())?;
        timings_output.write_all(b"\n")?;
        timings_output.flush()?;
    }
    Ok(had_errors)
}

#[cfg(feature = "vision")]
fn multimodal_request(
    input: MultimodalInputLine,
    base_dir: &Path,
    max_image_bytes: usize,
) -> Result<MultimodalDecisionRequest> {
    let mut images = Vec::with_capacity(input.images.len());
    for image in input.images {
        let (media_type, data) = match image.source {
            ImageSource::Base64 { media_type, data } => {
                if data.len() > (max_image_bytes.saturating_mul(4) / 3).saturating_add(8) {
                    return Err(JetError::InvalidRequest(format!(
                        "image {:?} exceeds byte limit",
                        image.id
                    )));
                }
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(data)
                    .map_err(|error| {
                        JetError::InvalidRequest(format!(
                            "image {:?} has invalid base64: {error}",
                            image.id
                        ))
                    })?;
                (media_type, bytes)
            }
            ImageSource::Path { media_type, path } => {
                let resolved = if path.is_absolute() {
                    path
                } else {
                    base_dir.join(path)
                };
                let length = std::fs::metadata(&resolved)?.len();
                if length > u64::try_from(max_image_bytes).unwrap_or(u64::MAX) {
                    return Err(JetError::InvalidRequest(format!(
                        "image {:?} exceeds byte limit",
                        image.id
                    )));
                }
                (media_type, std::fs::read(resolved)?)
            }
        };
        if data.len() > max_image_bytes {
            return Err(JetError::InvalidRequest(format!(
                "image {:?} exceeds byte limit",
                image.id
            )));
        }
        images.push(ImageInput {
            id: image.id,
            media_type,
            data,
        });
    }
    Ok(MultimodalDecisionRequest {
        request_id: input.request_id,
        state: input.state,
        images,
        questions: input.questions,
    })
}

fn parse_timings_path(value: &str) -> std::result::Result<PathBuf, String> {
    if value == "-" {
        Err("--timings requires a file path; '-' is reserved for JSONL input/output".to_owned())
    } else {
        Ok(PathBuf::from(value))
    }
}

fn validate_timings_path(timings: &Path, input: &Path, output: &Path) -> Result<()> {
    let timings = normalized_file_path(timings)?;
    for path in [input, output] {
        if path == Path::new("-") {
            continue;
        }
        let path = normalized_file_path(path)?;
        let same_file = timings == path;
        #[cfg(unix)]
        let same_file = same_file || {
            use std::os::unix::fs::MetadataExt;

            match (std::fs::metadata(&timings), std::fs::metadata(&path)) {
                (Ok(left), Ok(right)) => left.dev() == right.dev() && left.ino() == right.ino(),
                _ => false,
            }
        };
        if same_file {
            return Err(JetError::InvalidRequest(
                "--timings must use a separate file from --input and --output".to_owned(),
            ));
        }
    }
    Ok(())
}

fn normalized_file_path(path: &Path) -> io::Result<PathBuf> {
    match path.canonicalize() {
        Ok(path) => Ok(path),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            let parent = path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
                .unwrap_or(Path::new("."));
            // A symlink can alias an output that will only be created after this check.
            if let Ok(target) = std::fs::read_link(path) {
                return normalized_file_path(&if target.is_absolute() {
                    target
                } else {
                    parent.join(target)
                });
            }
            let name = path.file_name().ok_or(error)?;
            Ok(parent.canonicalize()?.join(name))
        }
        Err(error) => Err(error),
    }
}

fn process_chunk(engine: &mut Engine, lines: &[InputLine], output: &mut dyn Write) -> Result<bool> {
    let mut had_errors = false;
    let mut requests = Vec::new();
    for line in lines {
        if let InputLine::Request(request) = line {
            requests.push(request.clone());
        }
    }
    let mut results = engine.decide_batch(&requests).into_iter();
    for line in lines {
        match line {
            InputLine::Error(error) => {
                had_errors = true;
                write_json_line(output, error)?;
            }
            InputLine::Request(_) => match results.next() {
                Some(Ok(response)) => write_json_line(output, &response)?,
                Some(Err(error)) => {
                    had_errors = true;
                    write_json_line(output, &error_envelope(error))?;
                }
                None => {
                    had_errors = true;
                    write_json_line(
                        output,
                        &error_envelope(JetError::NativeRuntime(
                            "batch evaluator omitted a result".to_owned(),
                        )),
                    )?;
                }
            },
        }
    }
    Ok(had_errors)
}

fn open_input(path: &Path) -> Result<Box<dyn BufRead>> {
    if path == Path::new("-") {
        Ok(Box::new(BufReader::new(io::stdin())))
    } else {
        Ok(Box::new(BufReader::new(File::open(path)?)))
    }
}

fn open_output(path: &Path) -> Result<Box<dyn Write>> {
    if path == Path::new("-") {
        Ok(Box::new(BufWriter::new(io::stdout())))
    } else {
        Ok(Box::new(BufWriter::new(File::create(path)?)))
    }
}

fn write_json_line(writer: &mut dyn Write, value: &impl Serialize) -> Result<()> {
    serde_json::to_writer(&mut *writer, value)?;
    writer.write_all(b"\n")?;
    Ok(())
}

fn error_envelope(error: JetError) -> ErrorEnvelope {
    ErrorEnvelope {
        error: ErrorBody {
            code: error.code(),
            message: error.to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_defaults_to_auto_and_accepts_cuda() -> std::result::Result<(), clap::Error> {
        let default = parsed_judge(Cli::try_parse_from([
            "jet",
            "judge",
            "--model-path",
            "model.gguf",
        ])?)?;
        assert!(matches!(default.backend, BackendArg::Auto));
        let cuda = parsed_judge(Cli::try_parse_from([
            "jet",
            "judge",
            "--model-path",
            "model.gguf",
            "--backend",
            "cuda",
        ])?)?;
        assert!(matches!(cuda.backend, BackendArg::Cuda));
        Ok(())
    }

    fn parsed_judge(cli: Cli) -> std::result::Result<JudgeArgs, clap::Error> {
        match cli.command {
            Command::Judge(args) => Ok(args),
            #[cfg(feature = "vision")]
            Command::JudgeMultimodal(_) => Err(clap::Error::raw(
                clap::error::ErrorKind::InvalidSubcommand,
                "expected judge command",
            )),
        }
    }

    #[cfg(feature = "vision")]
    #[test]
    fn multimodal_json_accepts_bytes_and_paths_with_the_same_image() -> Result<()> {
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/vision/red.png");
        let expected = std::fs::read(&path)?;
        let base64 = base64::engine::general_purpose::STANDARD.encode(&expected);
        let image_directory = path.parent().ok_or_else(|| {
            JetError::InvalidRequest("fixture image has no parent directory".to_owned())
        })?;
        let filename = path
            .file_name()
            .ok_or_else(|| JetError::InvalidRequest("fixture image has no filename".to_owned()))?;
        for source in [
            serde_json::json!({"type":"base64", "media_type":"image/png", "data":base64}),
            serde_json::json!({"type":"path", "media_type":"image/png", "path":filename.to_string_lossy()}),
        ] {
            let value = serde_json::json!({
                "request_id":"frame-1", "state":"x",
                "images":[{"id":"screen", "source":source}],
                "questions":{"action":{"type":"choice", "instructions":"choose",
                    "criteria":{"a":"A", "b":"B"}}}
            });
            let parsed: MultimodalInputLine = serde_json::from_value(value)?;
            let request = multimodal_request(parsed, image_directory, 1024)?;
            assert_eq!(request.request_id.as_deref(), Some("frame-1"));
            assert_eq!(request.images[0].data, expected);
        }
        let without_state: MultimodalInputLine = serde_json::from_value(serde_json::json!({
            "images": [],
            "questions": {}
        }))?;
        assert!(without_state.state.is_null());
        Ok(())
    }

    #[test]
    fn preparation_pipeline_defaults_to_enabled() -> std::result::Result<(), clap::Error> {
        let cli = Cli::try_parse_from(["jet", "judge", "--model-path", "model.gguf"])?;
        let args = parsed_judge(cli)?;
        assert!(!args.no_preparation_pipeline);
        Ok(())
    }

    #[test]
    fn preparation_options_accept_explicit_overrides_and_reject_negative_counts()
    -> std::result::Result<(), clap::Error> {
        let cli = Cli::try_parse_from([
            "jet",
            "judge",
            "--model-path",
            "model.gguf",
            "--max-sequences",
            "4",
            "--no-preparation-pipeline",
        ])?;
        let args = parsed_judge(cli)?;
        assert_eq!(args.max_sequences, 4);
        assert!(args.no_preparation_pipeline);
        assert!(
            Cli::try_parse_from([
                "jet",
                "judge",
                "--model-path",
                "model.gguf",
                "--max-sequences",
                "-1"
            ])
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn offload_options_accept_explicit_counts_and_reject_negative_values()
    -> std::result::Result<(), clap::Error> {
        let cli = Cli::try_parse_from([
            "jet",
            "judge",
            "--model-path",
            "model.gguf",
            "--backend",
            "vulkan",
            "--gpu-layers",
            "12",
            "--cpu-moe-layers",
            "30",
        ])?;
        let args = parsed_judge(cli)?;
        assert_eq!(args.gpu_layers, Some(12));
        assert_eq!(args.cpu_moe_layers, 30);
        for option in ["--gpu-layers", "--cpu-moe-layers"] {
            assert!(
                Cli::try_parse_from(["jet", "judge", "--model-path", "model.gguf", option, "-1",])
                    .is_err()
            );
        }
        Ok(())
    }

    #[test]
    fn serializes_machine_readable_error() -> Result<()> {
        let value = serde_json::to_value(error_envelope(JetError::InvalidRequest("bad".into())))?;
        assert_eq!(value["error"]["code"], "invalid_request");
        assert_eq!(value["error"]["message"], "invalid request: bad");
        Ok(())
    }

    #[test]
    fn timings_cannot_share_standard_output_with_responses() {
        let result = Cli::try_parse_from([
            "jet",
            "judge",
            "--model-path",
            "model.gguf",
            "--timings",
            "-",
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn timings_cannot_overwrite_responses() -> std::result::Result<(), clap::Error> {
        let cli = Cli::try_parse_from([
            "jet",
            "judge",
            "--model-path",
            "model.gguf",
            "--output",
            "results.jsonl",
            "--timings",
            "results.jsonl",
        ])?;
        let args = parsed_judge(cli)?;
        assert!(matches!(run_judge(args), Err(JetError::InvalidRequest(_))));
        Ok(())
    }

    #[test]
    fn timings_cannot_overwrite_aliased_input_or_output()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        use std::ffi::OsStr;
        use std::time::{SystemTime, UNIX_EPOCH};

        let name = format!(
            "jet-timings-path-test-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
        );
        let directory = std::env::current_dir()?.join(&name);
        std::fs::create_dir(&directory)?;
        let path = directory.join("responses.jsonl");
        std::fs::write(&path, b"preserve existing content\n")?;
        let aliases = vec![PathBuf::from(&name).join("responses.jsonl")];
        #[cfg(unix)]
        let aliases = {
            let mut aliases = aliases;
            let symlink = directory.join("symlink.jsonl");
            std::os::unix::fs::symlink(&path, &symlink)?;
            aliases.push(symlink);
            let hardlink = directory.join("hardlink.jsonl");
            std::fs::hard_link(&path, &hardlink)?;
            aliases.push(hardlink);
            aliases
        };
        for alias in aliases {
            for argument in ["--input", "--output"] {
                let cli = Cli::try_parse_from([
                    OsStr::new("jet"),
                    OsStr::new("judge"),
                    OsStr::new("--model-path"),
                    OsStr::new("model.gguf"),
                    OsStr::new(argument),
                    path.as_os_str(),
                    OsStr::new("--timings"),
                    alias.as_os_str(),
                ])?;
                let args = parsed_judge(cli)?;
                assert!(matches!(run_judge(args), Err(JetError::InvalidRequest(_))));
                assert_eq!(std::fs::read(&path)?, b"preserve existing content\n");
            }
        }
        let missing = directory.join("new.jsonl");
        assert!(matches!(
            validate_timings_path(
                &PathBuf::from(&name).join("new.jsonl"),
                Path::new("-"),
                &missing,
            ),
            Err(JetError::InvalidRequest(_))
        ));
        assert!(!missing.exists());
        #[cfg(unix)]
        {
            let symlink = directory.join("new-link.jsonl");
            std::os::unix::fs::symlink("new.jsonl", &symlink)?;
            assert!(matches!(
                validate_timings_path(&symlink, Path::new("-"), &missing),
                Err(JetError::InvalidRequest(_))
            ));
            assert!(!missing.exists());
        }
        std::fs::remove_dir_all(directory)?;
        Ok(())
    }
}
