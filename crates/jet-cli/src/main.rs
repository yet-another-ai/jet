use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use jet_core::{DecisionRequest, ErrorCode, JetError, Result};
use jet_engine::{Engine, EngineConfig, ExecutionMode, ThinkingMode};
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
    #[arg(long)]
    max_output_rows: Option<u32>,
    #[arg(long)]
    threads: Option<i32>,
    #[arg(long, value_enum, default_value_t = BackendArg::Cpu)]
    backend: BackendArg,
    /// Number of model layers to place on GPU; Vulkan defaults to all layers.
    #[arg(long)]
    gpu_layers: Option<u32>,
    /// Keep routed MoE experts in the first N layers on CPU (Vulkan only).
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
    Cpu,
    Vulkan,
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

    let mut config = match args.backend {
        BackendArg::Cpu => EngineConfig::cpu(&args.model_path, args.model_id),
        BackendArg::Vulkan => EngineConfig::vulkan(&args.model_path, args.model_id),
    };
    config.gpu_layers = args.gpu_layers;
    config.cpu_moe_layers = args.cpu_moe_layers;
    config.use_mmap = !args.no_mmap;
    config.context_tokens_per_sequence = args.context_tokens;
    config.token_batch = args.token_batch;
    config.micro_batch = args.micro_batch;
    config.max_sequences = args.max_sequences;
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
        let Command::Judge(args) = cli.command;
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
        let Command::Judge(args) = cli.command;
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
                let Command::Judge(args) = cli.command;
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
