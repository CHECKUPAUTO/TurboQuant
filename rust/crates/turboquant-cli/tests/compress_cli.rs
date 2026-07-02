//! End-to-end tests for the `turboquant` binary: compress, verify,
//! overwrite refusal, and honest backend errors.

// Test-data generation casts are intentional.
#![allow(clippy::cast_precision_loss)]

use std::path::Path;
use std::process::Command;
use turboquant_gguf::turbo;
use turboquant_gguf::{GgmlType, GgufParser, GgufValue, GgufWriter};

fn turboquant() -> Command {
    Command::new(env!("CARGO_BIN_EXE_turboquant"))
}

fn write_synthetic_model(path: &Path) -> Vec<f32> {
    let mut w = GgufWriter::new();
    w.add_metadata("general.name", GgufValue::String("cli-test".into()));
    let values: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.02).sin() * 2.0).collect();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    w.add_tensor("blk.0.weight", vec![1024], GgmlType::F32, bytes)
        .unwrap();
    // Non-float tensor must pass through unchanged.
    w.add_tensor("ids", vec![4], GgmlType::I32, vec![0u8; 16])
        .unwrap();
    w.write_to_file(path).unwrap();
    values
}

#[test]
fn compress_produces_decompressible_output_with_plausible_snr() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("model.gguf");
    let values = write_synthetic_model(&input);

    let out = turboquant()
        .args(["compress", input.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "compress failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // Default output path is <stem>-turbo3.gguf next to the input.
    let output = dir.path().join("model-turbo3.gguf");
    assert!(output.exists(), "expected {} to exist", output.display());

    let parsed = GgufParser::parse_file(&output).unwrap();
    assert!(turbo::is_turbo_compressed(&parsed));
    let recon = turbo::decompress_tensor(&parsed, "blk.0.weight").unwrap();
    assert_eq!(recon.len(), values.len());
    let snr = turbo::snr_db(&values, &recon);
    assert!(snr > 10.0, "SNR too low: {snr:.2} dB");

    // Passthrough tensor kept its type and bytes.
    let ids = parsed.tensor("ids").unwrap();
    assert_eq!(ids.ggml_type, GgmlType::I32);
    assert_eq!(parsed.tensor_data(ids).unwrap(), &[0u8; 16]);

    // Rerunning without --in-place must refuse to overwrite.
    let rerun = turboquant()
        .args(["compress", input.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!rerun.status.success());
    let stderr = String::from_utf8_lossy(&rerun.stderr);
    assert!(
        stderr.contains("refusing to overwrite"),
        "unexpected stderr: {stderr}"
    );

    // verify --original reports success (real SNR check).
    let verify = turboquant()
        .args([
            "verify",
            output.to_str().unwrap(),
            "--original",
            input.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        verify.status.success(),
        "verify failed: {}",
        String::from_utf8_lossy(&verify.stderr)
    );
    let stdout = String::from_utf8_lossy(&verify.stdout);
    assert!(
        stdout.contains("SNR"),
        "verify output missing SNR: {stdout}"
    );
}

#[test]
fn in_place_overwrites_input() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("model.gguf");
    let values = write_synthetic_model(&input);
    let original_size = std::fs::metadata(&input).unwrap().len();

    let out = turboquant()
        .args(["compress", input.to_str().unwrap(), "--in-place"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "compress --in-place failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let parsed = GgufParser::parse_file(&input).unwrap();
    assert!(turbo::is_turbo_compressed(&parsed));
    let recon = turbo::decompress_tensor(&parsed, "blk.0.weight").unwrap();
    assert!(turbo::snr_db(&values, &recon) > 10.0);
    assert!(std::fs::metadata(&input).unwrap().len() < original_size);
}

#[test]
fn cuda_backend_errors_clearly() {
    let out = turboquant()
        .args(["--backend", "cuda", "info"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("CUDA backend not implemented"),
        "unexpected stderr: {stderr}"
    );
}

#[test]
fn unsupported_bits_error() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("model.gguf");
    write_synthetic_model(&input);
    let out = turboquant()
        .args(["compress", input.to_str().unwrap(), "--bits", "4"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("only 3-bit quantization is implemented"),
        "unexpected stderr: {stderr}"
    );
}
