use std::error::Error;
use std::ffi::OsStr;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const MODEL_ID: &str = "fixture-model";
const API_KEY: &str = "jet-serve-integration-test";

struct ChildGuard(Child);

impl ChildGuard {
    fn wait_for_exit(&mut self, timeout: Duration) -> TestResult<ExitStatus> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(status) = self.0.try_wait()? {
                return Ok(status);
            }
            if Instant::now() >= deadline {
                return Err("jet process did not exit before its test deadline".into());
            }
            thread::sleep(Duration::from_millis(20));
        }
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new(name: &str) -> TestResult<Self> {
        let suffix = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let path =
            std::env::temp_dir().join(format!("jet-serve-{name}-{}-{suffix}", std::process::id()));
        fs::create_dir(&path)?;
        Ok(Self(path))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn jet_command() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_jet"));
    command
        .env_remove("JET_API_KEY")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    command
}

fn engine_options(command: &mut Command, model_path: &OsStr) {
    command.arg("--model-path").arg(model_path).args([
        "--model-id",
        MODEL_ID,
        "--backend",
        "cpu",
        "--context-tokens",
        "256",
        "--token-batch",
        "256",
        "--micro-batch",
        "128",
        "--max-sequences",
        "4",
        "--max-output-rows",
        "256",
        "--threads",
        "2",
    ]);
}

#[test]
fn serve_help_is_available_without_loading_a_model() -> TestResult {
    let directory = TestDirectory::new("help")?;
    let output_path = directory.path().join("help.txt");
    let mut command = jet_command();
    command
        .args(["serve", "--help"])
        .stdout(File::create(&output_path)?);
    let mut child = ChildGuard(command.spawn()?);
    assert!(child.wait_for_exit(Duration::from_secs(10))?.success());
    let help = fs::read_to_string(output_path)?;
    assert!(help.contains("--model-path"));
    assert!(help.contains("--bind"));
    assert!(help.contains("--queue-capacity"));
    Ok(())
}

#[test]
fn serve_exits_when_the_model_file_is_missing() -> TestResult {
    let directory = TestDirectory::new("missing-model")?;
    let mut command = jet_command();
    command.args(["serve", "--bind", "127.0.0.1:0"]);
    engine_options(
        &mut command,
        directory.path().join("missing.gguf").as_os_str(),
    );
    let mut child = ChildGuard(command.spawn()?);
    assert!(!child.wait_for_exit(Duration::from_secs(10))?.success());
    Ok(())
}

struct HttpResponse {
    status: u16,
    body: Vec<u8>,
}

fn http_json(
    address: SocketAddr,
    method: &str,
    path: &str,
    api_key: Option<&str>,
    body: Option<&Value>,
    timeout: Duration,
) -> TestResult<HttpResponse> {
    let mut stream = TcpStream::connect_timeout(&address, timeout)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    let body = body
        .map(serde_json::to_vec)
        .transpose()?
        .unwrap_or_default();
    let authorization = api_key
        .map(|key| format!("Authorization: Bearer {key}\r\n"))
        .unwrap_or_default();
    let headers = format!(
        "{method} {path} HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{authorization}\r\n",
        body.len()
    );
    stream.write_all(headers.as_bytes())?;
    stream.write_all(&body)?;
    let mut response = Vec::new();
    stream.read_to_end(&mut response)?;
    let separator = response
        .windows(4)
        .position(|bytes| bytes == b"\r\n\r\n")
        .ok_or("HTTP response is missing its header terminator")?;
    let headers = std::str::from_utf8(&response[..separator])?;
    let status = headers
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .ok_or("HTTP response is missing its status code")?
        .parse()?;
    let body = response[separator + 4..].to_vec();
    Ok(HttpResponse { status, body })
}

fn wait_until_ready(child: &mut ChildGuard, address: SocketAddr, timeout: Duration) -> TestResult {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child.0.try_wait()? {
            return Err(format!("jet serve exited before readiness: {status}").into());
        }
        if let Ok(response) = http_json(
            address,
            "GET",
            "/readyz",
            Some(API_KEY),
            None,
            Duration::from_secs(1),
        ) && response.status == 200
        {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(format!("jet serve did not become ready within {timeout:?}").into());
        }
        thread::sleep(Duration::from_millis(50));
    }
}

#[test]
#[ignore = "requires JET_MODEL_PATH pointing to the pinned Qwen3 GGUF"]
fn served_decisions_match_single_request_jsonl_results() -> TestResult {
    let model_path = std::env::var_os("JET_MODEL_PATH")
        .ok_or("JET_MODEL_PATH is required for the HTTP model test")?;
    let directory = TestDirectory::new("model")?;
    let input_path = directory.path().join("input.jsonl");
    let output_path = directory.path().join("output.jsonl");
    let requests = [
        json!({"state": "Alpha", "questions": {"ok": {"type": "noul", "instructions": "Choose"}}}),
        json!({"state": "Alpha", "questions": {"pick": {"type": "choice", "instructions": "Choose", "criteria": {"a": "A", "b": "B"}}}}),
        json!({"state": "Alpha", "questions": {"level": {"type": "score", "instructions": "Rate", "criteria": ["Low", "High"]}}}),
    ];
    let jsonl = requests
        .iter()
        .map(serde_json::to_string)
        .collect::<serde_json::Result<Vec<_>>>()?
        .join("\n");
    fs::write(&input_path, format!("{jsonl}\n"))?;

    let mut judge_command = jet_command();
    judge_command.arg("judge");
    engine_options(&mut judge_command, &model_path);
    judge_command
        .args(["--batch-requests", "1", "--input"])
        .arg(&input_path)
        .arg("--output")
        .arg(&output_path);
    let mut judge = ChildGuard(judge_command.spawn()?);
    assert!(judge.wait_for_exit(Duration::from_secs(120))?.success());
    let baseline = fs::read_to_string(output_path)?
        .lines()
        .map(serde_json::from_str::<Value>)
        .collect::<serde_json::Result<Vec<_>>>()?;
    assert_eq!(baseline.len(), requests.len());

    // Reserve an available loopback port only for discovery; the server owns its listener.
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let address = listener.local_addr()?;
    drop(listener);

    let mut serve_command = jet_command();
    serve_command
        .args(["serve", "--bind"])
        .arg(address.to_string())
        .env("JET_API_KEY", API_KEY);
    engine_options(&mut serve_command, &model_path);
    let mut server = ChildGuard(serve_command.spawn()?);
    wait_until_ready(&mut server, address, Duration::from_secs(120))?;

    let timeout = Duration::from_secs(30);
    assert_eq!(
        http_json(address, "GET", "/healthz", None, None, timeout)?.status,
        200
    );
    let models = http_json(
        address,
        "GET",
        "/api/v1/models",
        Some(API_KEY),
        None,
        timeout,
    )?;
    assert_eq!(models.status, 200);
    let models: Value = serde_json::from_slice(&models.body)?;
    assert_eq!(models["data"][0]["id"], MODEL_ID);
    assert!(
        models["data"][0]["architecture"]["output_modalities"]
            .as_array()
            .is_some_and(|modalities| modalities.contains(&json!("decisions")))
    );

    let mut unauthorized_request = requests[0].clone();
    unauthorized_request["model"] = json!(MODEL_ID);
    let unauthorized = http_json(
        address,
        "POST",
        "/api/alpha/decisions",
        None,
        Some(&unauthorized_request),
        timeout,
    )?;
    assert_eq!(unauthorized.status, 401);
    let unauthorized: Value = serde_json::from_slice(&unauthorized.body)?;
    assert_eq!(unauthorized["error"]["code"], 401);

    for (mut request, expected) in requests.into_iter().zip(baseline) {
        request["model"] = json!(MODEL_ID);
        let response = http_json(
            address,
            "POST",
            "/api/alpha/decisions",
            Some(API_KEY),
            Some(&request),
            timeout,
        )?;
        assert_eq!(
            response.status,
            200,
            "{}",
            String::from_utf8_lossy(&response.body)
        );
        let mut actual: Value = serde_json::from_slice(&response.body)?;
        assert!(
            actual["inference_ms"]
                .as_f64()
                .is_some_and(|milliseconds| milliseconds.is_finite() && milliseconds > 0.0)
        );
        assert!(
            actual["queue_ms"]
                .as_f64()
                .is_some_and(|milliseconds| milliseconds.is_finite() && milliseconds >= 0.0)
        );
        let fields = actual
            .as_object_mut()
            .ok_or("decision response must be an object")?;
        fields.remove("inference_ms");
        fields.remove("queue_ms");
        assert_eq!(actual, expected);
    }

    Ok(())
}

#[cfg(feature = "vision")]
fn inline_image(bytes: &[u8]) -> Value {
    use base64::{Engine as _, engine::general_purpose::STANDARD};

    json!({
        "type": "image_url",
        "image_url": {"url": format!("data:image/png;base64,{}", STANDARD.encode(bytes))}
    })
}

#[cfg(feature = "vision")]
fn color_question(instructions: &str) -> Value {
    json!({
        "type": "choice",
        "instructions": instructions,
        "criteria": {"blue": "Blue", "red": "Red"}
    })
}

#[cfg(feature = "vision")]
fn assert_decision_timings(response: &Value) {
    assert!(
        response["inference_ms"]
            .as_f64()
            .is_some_and(|milliseconds| milliseconds.is_finite() && milliseconds > 0.0),
        "missing positive inference_ms: {response}"
    );
    assert!(
        response["queue_ms"]
            .as_f64()
            .is_some_and(|milliseconds| milliseconds.is_finite() && milliseconds >= 0.0),
        "missing nonnegative queue_ms: {response}"
    );
}

#[cfg(feature = "vision")]
#[test]
#[ignore = "requires JET_VISION_MODEL_PATH and JET_VISION_MMPROJ_PATH; runs real CPU image inference"]
fn served_images_use_pixels_and_preserve_interleaved_order() -> TestResult {
    let model = std::env::var_os("JET_VISION_MODEL_PATH")
        .ok_or("JET_VISION_MODEL_PATH is required for the HTTP vision test")?;
    let mmproj = std::env::var_os("JET_VISION_MMPROJ_PATH")
        .ok_or("JET_VISION_MMPROJ_PATH is required for the HTTP vision test")?;
    let red = inline_image(include_bytes!("../../../tests/fixtures/vision/red.png"));
    let blue = inline_image(include_bytes!("../../../tests/fixtures/vision/blue.png"));

    let listener = TcpListener::bind("127.0.0.1:0")?;
    let address = listener.local_addr()?;
    drop(listener);

    // Load once and reuse the same process for all vision, rejection, and recovery checks.
    let mut command = jet_command();
    command
        .args(["serve", "--bind"])
        .arg(address.to_string())
        .arg("--model-path")
        .arg(model)
        .arg("--mmproj-path")
        .arg(mmproj)
        .env("JET_API_KEY", API_KEY)
        .args([
            "--model-id",
            MODEL_ID,
            "--backend",
            "cpu",
            "--thinking",
            "disabled",
            "--threads",
            "8",
            "--context-tokens",
            "4096",
            "--token-batch",
            "512",
            "--micro-batch",
            "128",
            "--max-sequences",
            "2",
            "--max-images",
            "2",
            "--max-image-bytes",
            "1024",
            "--image-max-tokens",
            "64",
            "--request-timeout-ms",
            "300000",
        ]);
    let mut server = ChildGuard(command.spawn()?);
    wait_until_ready(&mut server, address, Duration::from_secs(90))?;

    let timeout = Duration::from_secs(305);
    let models = http_json(
        address,
        "GET",
        "/api/v1/models",
        Some(API_KEY),
        None,
        timeout,
    )?;
    assert_eq!(models.status, 200);
    let models: Value = serde_json::from_slice(&models.body)?;
    assert_eq!(models["data"][0]["id"], MODEL_ID);
    assert_eq!(
        models["data"][0]["architecture"]["input_modalities"],
        json!(["text", "image"])
    );

    let post = |request: Value, status: u16| -> TestResult<Value> {
        let started = Instant::now();
        let response = http_json(
            address,
            "POST",
            "/api/alpha/decisions",
            Some(API_KEY),
            Some(&request),
            timeout,
        )?;
        assert_eq!(
            response.status,
            status,
            "{}",
            String::from_utf8_lossy(&response.body)
        );
        let response: Value = serde_json::from_slice(&response.body)?;
        eprintln!(
            "HTTP vision fixture: status={status}, elapsed={:?}, answers={}, inference_ms={}",
            started.elapsed(),
            response["answers"],
            response["inference_ms"]
        );
        if status == 200 {
            assert_decision_timings(&response);
        } else {
            assert_eq!(response["error"]["code"], status);
        }
        Ok(response)
    };

    let mut oversized_png = b"\x89PNG\r\n\x1a\n".to_vec();
    oversized_png.resize(1025, 0);
    for state in [
        // Valid signature, invalid raster: exercises native decode failure and recovery.
        json!([inline_image(b"\x89PNG\r\n\x1a\n")]),
        json!([{"type": "image_url", "image_url": {"url": "data:image/png;base64,%%%="}}]),
        json!([red.clone(), blue.clone(), red.clone()]),
        json!([inline_image(&oversized_png)]),
    ] {
        post(
            json!({
                "model": MODEL_ID,
                "state": state,
                "questions": {"color": color_question("Which color fills the image?")}
            }),
            400,
        )?;
    }

    let text = post(
        json!({
            "model": MODEL_ID,
            "state": "The signal is red.",
            "questions": {"color": color_question("What color is the signal?")}
        }),
        200,
    )?;
    assert_eq!(text["answers"]["color"]["choice"], "red", "{text}");

    for (image, expected) in [(red.clone(), "red"), (blue.clone(), "blue")] {
        let mut request = json!({
            "model": MODEL_ID,
            "state": ["Inspect the following image.", image],
            "questions": {"color": color_question("Which color fills the image?")}
        });
        if expected == "red" {
            request["questions"]["is_red"] = json!({
                "type": "noul",
                "instructions": "Is the image red?"
            });
            request["questions"]["redness"] = json!({
                "type": "score",
                "instructions": "How much of the image is red?",
                "criteria": ["None of it is red", "All of it is red"]
            });
        }
        let response = post(request, 200)?;
        assert_eq!(
            response["answers"]["color"]["choice"], expected,
            "{response}"
        );
        if expected == "red" {
            for (question, field) in [("is_red", "noul"), ("redness", "score")] {
                assert!(
                    response["answers"][question][field]
                        .as_f64()
                        .is_some_and(|value| value.is_finite() && (0.0..=1.0).contains(&value)),
                    "missing {question} {field}: {response}"
                );
            }
        }
    }

    for (first, second, first_color, second_color) in [
        (red.clone(), blue.clone(), "red", "blue"),
        (blue, red, "blue", "red"),
    ] {
        let response = post(
            json!({
                "model": MODEL_ID,
                "state": [
                    "Inspect these two images in the given order. First image:",
                    first,
                    "Second image:",
                    second,
                    "Answer each question using the indicated image position."
                ],
                "questions": {
                    "first_color": color_question("Which color fills the first image?"),
                    "second_color": color_question("Which color fills the second image?")
                }
            }),
            200,
        )?;
        assert_eq!(
            response["answers"]["first_color"]["choice"], first_color,
            "{response}"
        );
        assert_eq!(
            response["answers"]["second_color"]["choice"], second_color,
            "{response}"
        );
    }
    Ok(())
}
