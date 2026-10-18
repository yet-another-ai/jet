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

fn wait_until_ready(child: &mut ChildGuard, address: SocketAddr) -> TestResult {
    let deadline = Instant::now() + Duration::from_secs(120);
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
            return Err("jet serve did not become ready within 120 seconds".into());
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
    wait_until_ready(&mut server, address)?;

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
