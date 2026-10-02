use std::env;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use clap::{Parser, Subcommand};
use flate2::{Compression, read::GzDecoder, write::GzEncoder};
use zip::{ZipArchive, ZipWriter, write::SimpleFileOptions};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    task: Task,
}

#[derive(Subcommand)]
enum Task {
    /// Build a native CPU CLI archive and smoke-test the extracted executable.
    Package {
        /// Expected native host, e.g. windows-x64; defaults to the current host.
        #[arg(long)]
        platform: Option<String>,
    },
}

fn main() -> Result<()> {
    match Args::parse().task {
        Task::Package { platform } => package(platform.as_deref()),
    }
}

fn native_platform() -> Result<&'static str> {
    match (env::consts::OS, env::consts::ARCH) {
        ("windows", "x86_64") => Ok("windows-x64"),
        ("linux", "x86_64") => Ok("linux-x64"),
        ("linux", "aarch64") => Ok("linux-arm64"),
        ("macos", "aarch64") => Ok("macos-arm64"),
        host => Err(format!("Unsupported native CPU packaging host: {host:?}").into()),
    }
}

fn package(expected: Option<&str>) -> Result<()> {
    let platform = native_platform()?;
    if let Some(expected) = expected
        && expected != platform
    {
        return Err(
            format!("Native packaging requires {expected}; current host is {platform}").into(),
        );
    }
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or("xtask must be inside the workspace")?;
    let mut build = Command::new(env::var_os("CARGO").unwrap_or_else(|| "cargo".into()));
    build.current_dir(repo).env("JET_CPU_PORTABLE", "1").args([
        "build",
        "--locked",
        "--release",
        "-p",
        "jet-cli",
        "--no-default-features",
    ]);
    if cfg!(windows) {
        if env::var_os("CARGO_ENCODED_RUSTFLAGS").is_some() {
            return Err("Unset CARGO_ENCODED_RUSTFLAGS for static Windows CPU packaging".into());
        }
        let flags = env::var("RUSTFLAGS").unwrap_or_default();
        if !flags
            .split_whitespace()
            .any(|flag| flag == "target-feature=+crt-static")
        {
            build.env(
                "RUSTFLAGS",
                format!("{flags} -C target-feature=+crt-static"),
            );
        }
    }
    if !build.status()?.success() {
        return Err("CPU release build failed".into());
    }

    let target = env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo.join("target"));
    let target = repo.join(target);
    let binary_name = if cfg!(windows) { "jet.exe" } else { "jet" };
    let files = [
        (target.join("release").join(binary_name), binary_name),
        (repo.join("LICENSE"), "LICENSE"),
        (repo.join("README.md"), "README.md"),
        (repo.join("vendor/llama.cpp/LICENSE"), "LLAMA-LICENSE"),
        (
            repo.join("tests/fixtures/decisions.jsonl"),
            "decisions.jsonl",
        ),
    ];
    let name = format!("jet-cpu-{platform}");
    let dist = repo.join("dist");
    fs::create_dir_all(&dist)?;
    let is_zip = cfg!(windows);
    let extension = if is_zip { "zip" } else { "tar.gz" };
    let archive = dist.join(format!("{name}.{extension}"));
    write_archive(&archive, &name, &files, is_zip)?;

    // Test the actual artifact in a fresh directory, without models or repo cwd.
    let extracted = tempfile::Builder::new()
        .prefix(".cpu-smoke-")
        .tempdir_in(&dist)?;
    extract_archive(&archive, extracted.path(), is_zip)?;
    smoke(&extracted.path().join(&name), binary_name)?;
    println!("Packaged and verified {}", archive.display());
    Ok(())
}

fn write_archive(
    archive: &Path,
    name: &str,
    files: &[(PathBuf, &str)],
    is_zip: bool,
) -> Result<()> {
    let file = File::create(archive)?;
    if is_zip {
        let mut zip = ZipWriter::new(file);
        for (source, filename) in files {
            zip.start_file(format!("{name}/{filename}"), SimpleFileOptions::default())?;
            io::copy(&mut File::open(source)?, &mut zip)?;
        }
        zip.finish()?;
    } else {
        let mut tar = tar::Builder::new(GzEncoder::new(file, Compression::default()));
        for (source, filename) in files {
            tar.append_file(format!("{name}/{filename}"), &mut File::open(source)?)?;
        }
        tar.into_inner()?.finish()?;
    }
    Ok(())
}

fn extract_archive(archive: &Path, destination: &Path, is_zip: bool) -> Result<()> {
    let file = File::open(archive)?;
    if is_zip {
        ZipArchive::new(file)?.extract(destination)?;
    } else {
        tar::Archive::new(GzDecoder::new(file)).unpack(destination)?;
    }
    Ok(())
}

fn check_output(output: Output, action: &str) -> Result<Output> {
    if !output.status.success() {
        return Err(format!(
            "{action} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(output)
}

fn smoke(payload: &Path, binary_name: &str) -> Result<()> {
    let cli = payload.join(binary_name);
    let version = check_output(
        Command::new(&cli)
            .current_dir(payload)
            .arg("--version")
            .output()?,
        "version smoke",
    )?;
    print!("{}", String::from_utf8_lossy(&version.stdout));
    check_output(
        Command::new(&cli)
            .current_dir(payload)
            .arg("--help")
            .output()?,
        "help smoke",
    )?;
    let requests = fs::read_to_string(payload.join("decisions.jsonl"))?;
    let mut child = Command::new(&cli)
        .current_dir(payload)
        .arg("export-prompts")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .take()
        .ok_or("Missing exporter stdin")?
        .write_all(requests.as_bytes())?;
    let output = check_output(child.wait_with_output()?, "prompt export smoke")?;
    let responses = String::from_utf8(output.stdout)?;
    validate_responses(&requests, &responses)
}

fn validate_responses(requests: &str, responses: &str) -> Result<()> {
    let requests = requests
        .lines()
        .filter(|line| !line.trim().is_empty())
        .count();
    let responses: Vec<_> = responses.lines().collect();
    if responses.len() != requests {
        return Err("Packaged exporter lost JSONL requests".into());
    }
    for response in responses {
        let value: serde_json::Value = serde_json::from_str(response)?;
        if value.get("error").is_some()
            || value
                .get("questions")
                .and_then(serde_json::Value::as_array)
                .is_none_or(|q| q.is_empty())
        {
            return Err("Packaged exporter produced an invalid prompt response".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_archive_formats_preserve_payload() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let source = dir.path().join("fixture");
        fs::write(&source, "JSONL fixture with UTF-8: 中文 β 🙂\n")?;
        for is_zip in [true, false] {
            let archive = dir
                .path()
                .join(if is_zip { "test.zip" } else { "test.tar.gz" });
            write_archive(
                &archive,
                "jet-cpu-test",
                &[(source.clone(), "decisions.jsonl")],
                is_zip,
            )?;
            let extracted = tempfile::tempdir_in(dir.path())?;
            extract_archive(&archive, extracted.path(), is_zip)?;
            assert_eq!(
                fs::read(extracted.path().join("jet-cpu-test/decisions.jsonl"))?,
                fs::read(&source)?
            );
        }
        Ok(())
    }

    #[test]
    fn smoke_checks_export_responses() {
        let valid = "{\"questions\":[{\"question_id\":\"q\"}]}\n";
        assert!(validate_responses("request\n", valid).is_ok());
        let requests = "request\nrequest\n";
        assert!(validate_responses(requests, valid).is_err());
        assert!(validate_responses("request\n", "{\"error\":\"bad request\"}\n").is_err());
        assert!(validate_responses("request\n", "{\"questions\":[]}\n").is_err());
    }
}
