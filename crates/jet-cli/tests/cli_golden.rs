use std::error::Error;
use std::fs;
use std::process::Command;

use serde_json::Value;

#[test]
#[ignore = "requires JET_MODEL_PATH pointing to the pinned Qwen3 GGUF"]
fn jsonl_order_shape_and_error_exit_are_stable() -> Result<(), Box<dyn Error>> {
    let model_path = std::env::var_os("JET_MODEL_PATH")
        .ok_or("JET_MODEL_PATH is required for the CLI model test")?;
    let directory = std::env::temp_dir().join(format!("jet-cli-test-{}", std::process::id()));
    fs::create_dir_all(&directory)?;
    let input_path = directory.join("input.jsonl");
    let output_path = directory.join("output.jsonl");
    let timings_path = directory.join("timings.json");
    fs::write(
        &input_path,
        concat!(
            "{\"state\":\"first\",\"questions\":{\"ok\":{\"type\":\"noul\",\"instructions\":\"Choose\"}}}\n",
            "{not-json}\n",
            "{\"state\":\"third\",\"questions\":{\"pick\":{\"type\":\"choice\",\"instructions\":\"Choose\",\"criteria\":{\"a\":\"A\",\"b\":\"B\"}}}}\n",
        ),
    )?;

    let output = Command::new(env!("CARGO_BIN_EXE_jet"))
        .args([
            "judge",
            "--backend",
            "cpu",
            "--model-path",
            &model_path.to_string_lossy(),
            "--model-id",
            "fixture-model",
            "--input",
            &input_path.to_string_lossy(),
            "--output",
            &output_path.to_string_lossy(),
            "--timings",
            &timings_path.to_string_lossy(),
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
        ])
        .output()?;
    assert!(!output.status.success());

    let contents = fs::read_to_string(&output_path)?;
    let lines = contents
        .lines()
        .map(serde_json::from_str::<Value>)
        .collect::<serde_json::Result<Vec<_>>>()?;
    assert_eq!(lines.len(), 3);
    assert_eq!(lines[0]["model"], "fixture-model");
    assert_eq!(lines[0]["answers"]["ok"]["type"], "noul");
    assert_eq!(lines[1]["error"]["code"], "json");
    assert_eq!(lines[2]["answers"]["pick"]["type"], "choice");
    let keys = lines[0]
        .as_object()
        .ok_or("success response must be an object")?
        .keys()
        .map(String::as_str)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        keys,
        std::collections::BTreeSet::from(["answers", "model", "usage"])
    );

    let timings: Value = serde_json::from_str(&fs::read_to_string(&timings_path)?)?;
    assert!(timings["load_ms"].as_f64().is_some_and(|ms| ms > 0.0));
    assert!(timings["batch_ms"].as_f64().is_some_and(|ms| ms > 0.0));
    assert!(timings["prefill_tokens"].as_u64().is_some_and(|n| n > 0));

    fs::remove_dir_all(directory)?;
    Ok(())
}
