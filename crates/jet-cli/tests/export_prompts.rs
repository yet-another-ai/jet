use std::error::Error;
use std::io::Write;
use std::process::{Command, Output, Stdio};

use jet_core::DecisionRequest;
use serde_json::Value;

fn export(input: &str) -> Result<Output, Box<dyn Error>> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_jet"))
        .arg("export-prompts")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .take()
        .ok_or("missing child stdin")?
        .write_all(input.as_bytes())?;
    Ok(child.wait_with_output()?)
}

#[test]
fn exports_exact_prompt_contract_without_model_arguments() -> Result<(), Box<dyn Error>> {
    let request = serde_json::json!({
        "state": {"passage": "A < B & 中文"},
        "questions": {"answer": {
            "type": "choice", "instructions": "Choose the owner",
            "criteria": {"B": "Platform team", "A": "Application team"}
        }}
    });
    let output = export(&format!("\n{request}\n\n"))?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let lines = String::from_utf8(output.stdout)?;
    assert_eq!(lines.lines().count(), 1);
    let exported: Value = serde_json::from_str(&lines)?;
    let typed_request: DecisionRequest = serde_json::from_value(request)?;
    let expected = serde_json::to_value(jet_engine::export_prompts(&typed_request)?)?;
    assert_eq!(exported, expected);
    assert_eq!(exported["questions"][0]["question_id"], "answer");
    assert_eq!(
        exported["questions"][0]["candidates"][0]["target"],
        "\"Application team\""
    );
    assert_eq!(exported["questions"][0]["messages"][0]["role"], "system");
    Ok(())
}

#[test]
fn reports_bad_lines_and_validation_errors_while_preserving_order() -> Result<(), Box<dyn Error>> {
    let output = export(concat!(
        "{not-json}\n",
        "{\"state\":\"x\",\"questions\":{}}\n",
        "{\"state\":\"last\",\"questions\":{\"answer\":{\"type\":\"noul\",\"instructions\":\"True?\"}}}\n",
    ))?;
    assert!(!output.status.success());
    let lines = String::from_utf8(output.stdout)?;
    let results = lines
        .lines()
        .map(serde_json::from_str::<Value>)
        .collect::<serde_json::Result<Vec<_>>>()?;
    assert_eq!(results.len(), 3);
    assert_eq!(results[0]["error"]["code"], "json");
    assert_eq!(results[1]["error"]["code"], "invalid_request");
    assert_eq!(results[2]["questions"][0]["question_id"], "answer");
    assert_eq!(
        results[2]["questions"][0]["candidates"][1]["target"],
        "\"Yes\""
    );
    Ok(())
}
