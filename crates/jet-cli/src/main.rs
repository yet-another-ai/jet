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

    let mut config = match args.backend {
        BackendArg::Cpu => EngineConfig::cpu(&args.model_path, args.model_id),
        BackendArg::Vulkan => EngineConfig::vulkan(&args.model_path, args.model_id),
    };
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
    Ok(had_errors)
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
    fn serializes_machine_readable_error() -> Result<()> {
        let value = serde_json::to_value(error_envelope(JetError::InvalidRequest("bad".into())))?;
        assert_eq!(value["error"]["code"], "invalid_request");
        assert_eq!(value["error"]["message"], "invalid request: bad");
        Ok(())
    }
}
