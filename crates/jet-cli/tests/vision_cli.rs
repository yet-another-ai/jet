#![cfg(feature = "vision")]

use std::error::Error;
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

use serde_json::{Value, json};

fn image_path(name: &str) -> String {
    format!(
        "{}/../../tests/fixtures/vision/{name}.png",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn request(frame: &str, image: &str) -> String {
    json!({
        "request_id": frame,
        "state": "Choose the color shown in the image.",
        "images": [{
            "id": "screen",
            "source": {"type": "path", "media_type": "image/png", "path": image_path(image)}
        }],
        "questions": {"color": {
            "type": "choice",
            "instructions": "Which color fills the image?",
            "criteria": {"blue": "Blue", "red": "Red"}
        }}
    })
    .to_string()
}

fn backend() -> (String, &'static str) {
    let value = std::env::var("JET_VISION_BACKEND").unwrap_or_else(|_| "cpu".to_owned());
    let context_tokens = if value == "vulkan" || value == "cuda" || value == "auto" {
        "2048"
    } else {
        "4096"
    };
    (value, context_tokens)
}

#[test]
#[ignore = "requires JET_VISION_MODEL_PATH and JET_VISION_MMPROJ_PATH"]
fn stdin_returns_each_frame_before_eof_and_uses_the_image() -> Result<(), Box<dyn Error>> {
    let model = std::env::var("JET_VISION_MODEL_PATH")?;
    let mmproj = std::env::var("JET_VISION_MMPROJ_PATH")?;
    let (backend, context_tokens) = backend();
    let mut command = Command::new(env!("CARGO_BIN_EXE_jet"));
    command.args([
        "judge-multimodal",
        "--model-path",
        &model,
        "--mmproj-path",
        &mmproj,
        "--backend",
        &backend,
        "--context-tokens",
        context_tokens,
        "--image-max-tokens",
        "512",
        "--threads",
        "8",
    ]);
    if backend != "cpu" {
        command.args(["--no-mmap", "--micro-batch", "256"]);
    }
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;
    let mut input = child.stdin.take().ok_or("missing child stdin")?;
    let output = child.stdout.take().ok_or("missing child stdout")?;
    let (sender, receiver) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        for line in BufReader::new(output).lines() {
            if sender.send(line).is_err() {
                break;
            }
        }
    });
    let result = (|| -> Result<(), Box<dyn Error>> {
        for (frame, image, answer) in [("one", "red", "red"), ("two", "blue", "blue")] {
            writeln!(input, "{}", request(frame, image))?;
            input.flush()?;
            let line = receiver.recv_timeout(Duration::from_secs(240))??;
            let response: Value = serde_json::from_str(&line)?;
            if response["request_id"] != frame
                || response["answers"]["color"]["choice"] != answer
                || response["usage"]["input_image_tokens"]
                    .as_u64()
                    .unwrap_or(0)
                    == 0
            {
                return Err(format!("unexpected response for {frame}: {line}").into());
            }
        }
        writeln!(input, "{{bad-json}}")?;
        input.flush()?;
        let line = receiver.recv_timeout(Duration::from_secs(30))??;
        let response: Value = serde_json::from_str(&line)?;
        if response["error"]["code"] != "json" {
            return Err(format!("malformed input was not reported: {line}").into());
        }
        Ok(())
    })();
    drop(input);
    if result.is_err() {
        child.kill()?;
    }
    let status = child.wait()?;
    reader.join().map_err(|_| "stdout reader panicked")?;
    result?;
    assert!(!status.success()); // The malformed line is reported without stopping earlier frames.
    Ok(())
}

#[test]
#[ignore = "requires JET_VISION_MODEL_PATH and JET_VISION_MMPROJ_PATH"]
fn prefix_reuse_matches_independent_vision_scoring() -> Result<(), Box<dyn Error>> {
    let model = std::env::var("JET_VISION_MODEL_PATH")?;
    let mmproj = std::env::var("JET_VISION_MMPROJ_PATH")?;
    let (backend, context_tokens) = backend();
    let mut answers = Vec::new();
    for mode in ["batched", "reference"] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_jet"));
        command.args([
            "judge-multimodal",
            "--model-path",
            &model,
            "--mmproj-path",
            &mmproj,
            "--backend",
            &backend,
            "--execution",
            mode,
            "--context-tokens",
            context_tokens,
            "--image-max-tokens",
            "512",
            "--threads",
            "8",
        ]);
        if backend != "cpu" {
            command.args(["--no-mmap", "--micro-batch", "256"]);
        }
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        {
            let mut input = child.stdin.take().ok_or("missing child stdin")?;
            writeln!(input, "{}", request("ref", "red"))?;
        }
        let result = child.wait_with_output()?;
        if !result.status.success() {
            return Err(format!(
                "vision {mode} failed with status {}: {}",
                result.status,
                String::from_utf8_lossy(&result.stdout)
            )
            .into());
        }
        let response: Value = serde_json::from_slice(&result.stdout)?;
        answers.push(response);
    }
    for key in ["red", "blue"] {
        let optimized = answers[0]["answers"]["color"]["probabilities"][key]
            .as_f64()
            .ok_or("missing optimized probability")?;
        let reference = answers[1]["answers"]["color"]["probabilities"][key]
            .as_f64()
            .ok_or("missing reference probability")?;
        assert!(
            (optimized - reference).abs() < 1e-4,
            "{key}: {optimized} != {reference}"
        );
    }
    Ok(())
}
