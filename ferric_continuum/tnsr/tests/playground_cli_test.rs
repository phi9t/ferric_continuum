use std::path::PathBuf;
use std::process::Command;

use tnsr::playground::{DemoOptions, DemoOutput};

#[test]
fn parses_caller_selected_artifact_paths() {
    let opts = DemoOptions::parse_from([
        "tnsr_demo",
        "--dot",
        "/tmp/tnsr-playground/block.dot",
        "--trace-json",
        "/tmp/tnsr-playground/trace.json",
    ])
    .expect("demo options should parse");

    assert_eq!(
        opts.outputs,
        vec![
            DemoOutput::Dot(PathBuf::from("/tmp/tnsr-playground/block.dot")),
            DemoOutput::TraceJson(PathBuf::from("/tmp/tnsr-playground/trace.json")),
        ]
    );
}

#[test]
fn rejects_unknown_cli_flag() {
    let err = DemoOptions::parse_from(["tnsr_demo", "--unknown"])
        .expect_err("unknown flags should be rejected");

    assert!(err.contains("unknown argument: --unknown"));
}

#[test]
fn writes_requested_dot_and_trace_json_artifacts() {
    let out_dir = PathBuf::from(std::env::var("TEST_TMPDIR").expect("TEST_TMPDIR"))
        .join("tnsr-playground-artifacts");
    std::fs::create_dir_all(&out_dir).expect("create output dir");
    let dot_path = out_dir.join("block.dot");
    let trace_path = out_dir.join("trace.json");

    let output = Command::new(tnsr_demo_runfile())
        .arg("--dot")
        .arg(&dot_path)
        .arg("--trace-json")
        .arg(&trace_path)
        .output()
        .expect("run tnsr_demo");

    assert!(
        output.status.success(),
        "tnsr_demo should write requested artifacts; stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let dot = std::fs::read_to_string(&dot_path).expect("read dot artifact");
    assert!(dot.contains("digraph Autograd"));

    let trace_text = std::fs::read_to_string(&trace_path).expect("read trace artifact");
    let trace: serde_json::Value =
        serde_json::from_str(&trace_text).expect("trace artifact should be valid JSON");
    assert_eq!(trace["schema"], "tnsr.debug_trace");
    assert_eq!(trace["schema_version"], 1);
    assert!(
        !trace["forward_ops"]
            .as_array()
            .expect("forward_ops array")
            .is_empty()
    );
}

#[test]
fn dot_output_path_that_is_directory_exits_nonzero() {
    let bad_dot_path = PathBuf::from(std::env::var("TEST_TMPDIR").expect("TEST_TMPDIR"))
        .join("dot-output-is-directory");
    std::fs::create_dir_all(&bad_dot_path).expect("create bad dot output dir");

    let output = Command::new(tnsr_demo_runfile())
        .arg("--dot")
        .arg(&bad_dot_path)
        .output()
        .expect("run tnsr_demo");

    assert!(
        !output.status.success(),
        "tnsr_demo should reject directory output paths; stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{stdout}\n{stderr}");
    assert!(stderr.contains("write DOT"));
    assert!(!combined.contains(&format!("Written {}", bad_dot_path.display())));
}

fn tnsr_demo_runfile() -> PathBuf {
    let runfiles = PathBuf::from(std::env::var("TEST_SRCDIR").expect("TEST_SRCDIR"));
    let workspace = std::env::var("TEST_WORKSPACE").expect("TEST_WORKSPACE");
    runfiles
        .join(workspace)
        .join("ferric_continuum/tnsr/tnsr_demo")
}
