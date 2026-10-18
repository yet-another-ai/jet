use std::net::SocketAddr;
use std::num::{NonZeroU64, NonZeroUsize};
use std::time::Duration;

use jet_core::{JetError, Result};
use jet_server::ServerConfig;

use crate::{EngineArgs, engine_config};

#[derive(clap::Args)]
pub(crate) struct ServeArgs {
    #[command(flatten)]
    engine: EngineArgs,
    /// HTTP listen address. Set JET_API_KEY to require a Bearer token.
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: SocketAddr,
    /// Maximum waiting requests, excluding the request being evaluated.
    #[arg(long, default_value = "1")]
    queue_capacity: NonZeroUsize,
    /// Queue plus inference deadline; running native inference cannot be interrupted.
    #[arg(long, default_value = "30000")]
    request_timeout_ms: NonZeroU64,
    /// HTTP workers; model inference runs on a separate single owner thread.
    #[arg(long, default_value = "2")]
    http_workers: NonZeroUsize,
    /// Maximum JSON request body size in bytes.
    #[arg(long, default_value = "1048576")]
    max_body_bytes: NonZeroUsize,
}

pub(crate) fn run(args: ServeArgs) -> Result<()> {
    let api_key = match std::env::var("JET_API_KEY") {
        Ok(key) if key.is_empty() => {
            return Err(JetError::InvalidRequest(
                "JET_API_KEY must be nonempty when set".to_owned(),
            ));
        }
        Ok(key) => Some(key),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(JetError::InvalidRequest(
                "JET_API_KEY must be valid Unicode".to_owned(),
            ));
        }
    };
    let config = ServerConfig {
        bind: args.bind,
        queue_capacity: args.queue_capacity.get(),
        request_timeout: Duration::from_millis(args.request_timeout_ms.get()),
        http_workers: args.http_workers.get(),
        max_body_bytes: args.max_body_bytes.get(),
        api_key,
    };
    actix_web::rt::System::new().block_on(jet_server::serve(config, engine_config(&args.engine)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cli, Command};
    use clap::Parser;

    #[test]
    fn serving_keeps_engine_options_and_latency_defaults()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let cli = Cli::try_parse_from([
            "jet",
            "serve",
            "--model-path",
            "model.gguf",
            "--model-id",
            "local/jet",
            "--backend",
            "cpu",
            "--threads",
            "3",
            "--gpu-layers",
            "0",
        ])?;
        let Command::Serve(args) = cli.command else {
            return Err("expected serve arguments".into());
        };
        let engine = engine_config(&args.engine);
        assert_eq!(engine.model_id, "local/jet");
        assert_eq!(engine.backend, jet_engine::Backend::Cpu);
        assert_eq!(engine.threads, 3);
        assert_eq!(engine.gpu_layers, Some(0));
        assert_eq!(engine.thinking.mode, jet_engine::ThinkingMode::Disabled);
        assert_eq!(args.bind, "127.0.0.1:8080".parse::<SocketAddr>()?);
        assert_eq!(args.queue_capacity.get(), 1);
        assert_eq!(args.request_timeout_ms.get(), 30_000);
        assert_eq!(args.http_workers.get(), 2);
        Ok(())
    }

    #[test]
    fn serving_rejects_zero_limits_and_jsonl_only_options() {
        for option in [
            "--queue-capacity",
            "--request-timeout-ms",
            "--http-workers",
            "--max-body-bytes",
        ] {
            assert!(
                Cli::try_parse_from(["jet", "serve", "--model-path", "model.gguf", option, "0"])
                    .is_err()
            );
        }
        for option in ["--batch-requests", "--input", "--output", "--timings"] {
            assert!(
                Cli::try_parse_from(["jet", "serve", "--model-path", "model.gguf", option, "1"])
                    .is_err()
            );
        }
    }
}
