use std::net::SocketAddr;
use std::num::{NonZeroU64, NonZeroUsize};
#[cfg(feature = "vision")]
use std::path::PathBuf;
use std::time::Duration;

use jet_core::{JetError, Result};
use jet_engine::EngineConfig;
#[cfg(feature = "vision")]
use jet_engine::{ThinkingMode, VisionConfig};
use jet_server::ServerConfig;

use crate::{EngineArgs, engine_config};

#[derive(clap::Args)]
pub(crate) struct ServeArgs {
    #[command(flatten)]
    engine: EngineArgs,
    /// HTTP listen address. Set JET_API_KEY to require a Bearer token.
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: SocketAddr,
    /// Reject requests immediately while the engine is busy instead of queueing them.
    #[arg(long)]
    no_queue: bool,
    /// Maximum waiting requests, excluding the request being evaluated.
    #[arg(long, default_value = "1", conflicts_with = "no_queue")]
    queue_capacity: NonZeroUsize,
    /// Queue plus inference deadline; running native inference cannot be interrupted.
    #[arg(long, default_value = "30000")]
    request_timeout_ms: NonZeroU64,
    /// HTTP workers; model inference runs on a separate single owner thread.
    #[arg(long, default_value = "2")]
    http_workers: NonZeroUsize,
    /// Maximum JSON body bytes; defaults to 1 MiB, or 16 MiB with --mmproj-path.
    #[arg(long)]
    max_body_bytes: Option<NonZeroUsize>,
    /// Matching vision projector; enables PNG/JPEG data URL inputs.
    #[cfg(feature = "vision")]
    #[arg(long)]
    mmproj_path: Option<PathBuf>,
    /// Maximum images per request.
    #[cfg(feature = "vision")]
    #[arg(long, default_value = "4", requires = "mmproj_path")]
    max_images: NonZeroUsize,
    /// Maximum decoded file bytes per image, before pixel decoding.
    #[cfg(feature = "vision")]
    #[arg(long, default_value = "10485760", requires = "mmproj_path")]
    max_image_bytes: NonZeroUsize,
    /// Maximum decoded pixels per image.
    #[cfg(feature = "vision")]
    #[arg(long, default_value = "16777216", requires = "mmproj_path")]
    max_image_pixels: NonZeroUsize,
    /// Vision token budget per image.
    #[cfg(feature = "vision")]
    #[arg(long, default_value = "1024", requires = "mmproj_path", value_parser = clap::value_parser!(i32).range(1..))]
    image_max_tokens: i32,
}

impl ServeArgs {
    fn max_body_bytes(&self) -> usize {
        if let Some(limit) = self.max_body_bytes {
            return limit.get();
        }
        #[cfg(feature = "vision")]
        if self.mmproj_path.is_some() {
            return 16 * 1024 * 1024;
        }
        1024 * 1024
    }
}

fn build_engine_config(args: &ServeArgs) -> Result<EngineConfig> {
    let config = engine_config(&args.engine);
    #[cfg(feature = "vision")]
    {
        let mut config = config;
        if let Some(path) = &args.mmproj_path {
            if config.thinking.mode != ThinkingMode::Disabled {
                return Err(JetError::InvalidRequest(
                    "--mmproj-path requires --thinking disabled".to_owned(),
                ));
            }
            let mut vision = VisionConfig::new(path);
            vision.max_images = args.max_images.get();
            vision.max_image_bytes = args.max_image_bytes.get();
            vision.max_image_pixels = args.max_image_pixels.get();
            vision.image_max_tokens = args.image_max_tokens;
            config.vision = Some(vision);
        }
        Ok(config)
    }
    #[cfg(not(feature = "vision"))]
    Ok(config)
}

pub(crate) fn run(args: ServeArgs) -> Result<()> {
    let engine_config = build_engine_config(&args)?;
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
        queue_enabled: !args.no_queue,
        queue_capacity: args.queue_capacity.get(),
        request_timeout: Duration::from_millis(args.request_timeout_ms.get()),
        http_workers: args.http_workers.get(),
        max_body_bytes: args.max_body_bytes(),
        api_key,
    };
    actix_web::rt::System::new().block_on(jet_server::serve(config, engine_config))
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
        let engine = build_engine_config(&args)?;
        assert_eq!(engine.model_id, "local/jet");
        assert_eq!(engine.backend, jet_engine::Backend::Cpu);
        assert_eq!(engine.threads, 3);
        assert_eq!(engine.gpu_layers, Some(0));
        assert_eq!(engine.thinking.mode, jet_engine::ThinkingMode::Disabled);
        assert_eq!(args.bind, "127.0.0.1:8080".parse::<SocketAddr>()?);
        assert_eq!(args.queue_capacity.get(), 1);
        assert!(!args.no_queue);
        assert_eq!(args.request_timeout_ms.get(), 30_000);
        assert_eq!(args.http_workers.get(), 2);
        assert_eq!(args.max_body_bytes(), 1024 * 1024);
        #[cfg(feature = "vision")]
        assert!(engine.vision.is_none());
        Ok(())
    }

    #[test]
    fn serving_can_disable_queueing() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let cli =
            Cli::try_parse_from(["jet", "serve", "--model-path", "model.gguf", "--no-queue"])?;
        let Command::Serve(args) = cli.command else {
            return Err("expected serve arguments".into());
        };
        assert!(args.no_queue);
        assert!(
            Cli::try_parse_from([
                "jet",
                "serve",
                "--model-path",
                "model.gguf",
                "--no-queue",
                "--queue-capacity",
                "2",
            ])
            .is_err()
        );
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

    #[test]
    fn serving_honors_explicit_body_limit() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let cli = Cli::try_parse_from([
            "jet",
            "serve",
            "--model-path",
            "model.gguf",
            "--max-body-bytes",
            "1234",
        ])?;
        let Command::Serve(args) = cli.command else {
            return Err("expected serve arguments".into());
        };
        assert_eq!(args.max_body_bytes(), 1234);
        Ok(())
    }

    #[cfg(feature = "vision")]
    #[test]
    fn serving_configures_vision_without_loading_a_model()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let cli = Cli::try_parse_from([
            "jet",
            "serve",
            "--model-path",
            "model.gguf",
            "--mmproj-path",
            "mmproj.gguf",
        ])?;
        let Command::Serve(args) = cli.command else {
            return Err("expected serve arguments".into());
        };
        let config = build_engine_config(&args)?;
        let vision = config.vision.ok_or("expected vision configuration")?;
        assert_eq!(vision.mmproj_path, PathBuf::from("mmproj.gguf"));
        assert_eq!(vision.max_images, 4);
        assert_eq!(vision.max_image_bytes, 10 * 1024 * 1024);
        assert_eq!(vision.max_image_pixels, 16 * 1024 * 1024);
        assert_eq!(vision.image_max_tokens, 1024);
        assert_eq!(args.max_body_bytes(), 16 * 1024 * 1024);

        let cli = Cli::try_parse_from([
            "jet",
            "serve",
            "--model-path",
            "model.gguf",
            "--mmproj-path",
            "mmproj.gguf",
            "--max-images",
            "2",
            "--max-image-bytes",
            "1234",
            "--max-image-pixels",
            "4321",
            "--image-max-tokens",
            "128",
            "--max-body-bytes",
            "5678",
        ])?;
        let Command::Serve(args) = cli.command else {
            return Err("expected serve arguments".into());
        };
        let vision = build_engine_config(&args)?
            .vision
            .ok_or("expected vision configuration")?;
        assert_eq!(vision.max_images, 2);
        assert_eq!(vision.max_image_bytes, 1234);
        assert_eq!(vision.max_image_pixels, 4321);
        assert_eq!(vision.image_max_tokens, 128);
        assert_eq!(args.max_body_bytes(), 5678);
        Ok(())
    }

    #[cfg(feature = "vision")]
    #[test]
    fn serving_rejects_vision_with_thinking() -> std::result::Result<(), Box<dyn std::error::Error>>
    {
        for mode in ["auto", "required"] {
            let cli = Cli::try_parse_from([
                "jet",
                "serve",
                "--model-path",
                "model.gguf",
                "--mmproj-path",
                "mmproj.gguf",
                "--thinking",
                mode,
            ])?;
            let Command::Serve(args) = cli.command else {
                return Err("expected serve arguments".into());
            };
            let error = build_engine_config(&args)
                .err()
                .ok_or("expected configuration error")?;
            assert!(error.to_string().contains("--thinking disabled"));
        }
        Ok(())
    }

    #[cfg(feature = "vision")]
    #[test]
    fn serving_requires_projector_and_positive_image_limits() {
        for option in [
            "--max-images",
            "--max-image-bytes",
            "--max-image-pixels",
            "--image-max-tokens",
        ] {
            assert!(
                Cli::try_parse_from(["jet", "serve", "--model-path", "model.gguf", option, "1",])
                    .is_err()
            );
            assert!(
                Cli::try_parse_from([
                    "jet",
                    "serve",
                    "--model-path",
                    "model.gguf",
                    "--mmproj-path",
                    "mmproj.gguf",
                    option,
                    "0",
                ])
                .is_err()
            );
        }
        assert!(
            Cli::try_parse_from([
                "jet",
                "serve",
                "--model-path",
                "model.gguf",
                "--mmproj-path",
                "mmproj.gguf",
                "--image-max-tokens=-1",
            ])
            .is_err()
        );
    }

    #[cfg(not(feature = "vision"))]
    #[test]
    fn serving_without_vision_feature_rejects_projector() {
        assert!(
            Cli::try_parse_from([
                "jet",
                "serve",
                "--model-path",
                "model.gguf",
                "--mmproj-path",
                "mmproj.gguf",
            ])
            .is_err()
        );
    }
}
