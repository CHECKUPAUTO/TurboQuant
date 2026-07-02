//! Compress command implementation.

use colored::Colorize;
use std::path::{Path, PathBuf};
use turboquant_core::qjl::ScaleMode;
use turboquant_gguf::turbo::{self, TurboOptions};
use turboquant_gguf::GgufParser;

/// Run the compress command.
///
/// # Errors
///
/// Returns an error on invalid options, or if any input file fails to
/// compress.
pub fn run(
    files: Vec<String>,
    output: Option<String>,
    in_place: bool,
    bits: u8,
    block_size: usize,
    scale_mode: String,
) -> Result<(), Box<dyn std::error::Error>> {
    if files.is_empty() {
        eprintln!("{} No input files specified", "error:".red().bold());
        eprintln!("Usage: turboquant compress <FILE>... [OPTIONS]");
        return Err("no input files".into());
    }

    if bits == 0 || bits > 7 {
        return Err(format!("bits must be 1-7, got {bits}").into());
    }
    if bits != 3 {
        return Err(format!(
            "only 3-bit quantization is implemented in this build (requested {bits})"
        )
        .into());
    }

    if !block_size.is_power_of_two() || block_size < 8 {
        return Err(format!("block-size must be a power of two >= 8, got {block_size}").into());
    }

    let scale_mode = parse_scale_mode(&scale_mode)?;

    if in_place && output.is_some() {
        return Err("--in-place and --output are mutually exclusive".into());
    }
    if output.is_some() && files.len() > 1 {
        return Err("--output can only be used with a single input file".into());
    }

    let opts = TurboOptions {
        bits,
        block_size,
        scale_mode,
        ..Default::default()
    };

    let mut failures = 0usize;
    for file in &files {
        if let Err(e) = compress_one(file, output.as_deref(), in_place, &opts) {
            eprintln!("{} {}: {}", "warning:".yellow().bold(), file, e);
            failures += 1;
        }
    }

    if failures > 0 {
        return Err(format!("{failures} of {} file(s) failed to compress", files.len()).into());
    }
    Ok(())
}

fn parse_scale_mode(name: &str) -> Result<ScaleMode, Box<dyn std::error::Error>> {
    match name {
        "absmax" => Ok(ScaleMode::PerBlockAbsMax),
        "percentile" => Ok(ScaleMode::PerBlockPercentile(0.99)),
        "adaptive" => Ok(ScaleMode::Adaptive),
        other => Err(format!(
            "unknown scale mode '{other}' (expected: absmax, percentile, adaptive)"
        )
        .into()),
    }
}

/// Compress a single file, honoring `--output`/`--in-place` semantics.
fn compress_one(
    file: &str,
    output: Option<&str>,
    in_place: bool,
    opts: &TurboOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new(file);
    if !path.exists() {
        return Err("file not found".into());
    }
    if path.is_dir() {
        return Err("is a directory".into());
    }

    let parsed = GgufParser::parse_file(path).map_err(|e| format!("not a valid GGUF file: {e}"))?;
    if turbo::is_turbo_compressed(&parsed) {
        return Err("already TurboQuant-compressed, skipping".into());
    }

    let input_size = std::fs::metadata(path)?.len();

    let (dest, tmp): (PathBuf, PathBuf) = if in_place {
        let tmp = path.with_extension("gguf.turbo3-tmp");
        (path.to_path_buf(), tmp)
    } else {
        let dest = match output {
            Some(o) => PathBuf::from(o),
            None => default_output_path(path),
        };
        if dest == path {
            return Err(
                "output path equals input path; use --in-place to overwrite the input".into(),
            );
        }
        if dest.exists() {
            return Err(format!(
                "refusing to overwrite existing file {} (choose another --output or remove it)",
                dest.display()
            )
            .into());
        }
        (dest.clone(), dest)
    };

    let (writer, stats) = turbo::compress(&parsed, opts)?;
    writer.write_to_file(&tmp)?;
    if in_place {
        std::fs::rename(&tmp, &dest)?;
    }

    let output_size = std::fs::metadata(&dest)?.len();
    let ratio = if output_size > 0 {
        input_size as f64 / output_size as f64
    } else {
        0.0
    };
    println!(
        "{} Compressed {} -> {}",
        "OK".green().bold(),
        file.cyan(),
        dest.display().to_string().cyan()
    );
    println!(
        "   {} tensor(s) quantized ({} bits, block_size={}), {} passed through",
        stats.tensors_compressed.to_string().cyan(),
        opts.bits,
        opts.block_size,
        stats.tensors_passthrough
    );
    println!(
        "   {} -> {} bytes ({:.2}x)",
        input_size,
        output_size,
        ratio.to_string().bright_green().bold()
    );
    Ok(())
}

/// Derive `<stem>-turbo3.gguf` next to the input file.
fn default_output_path(input: &Path) -> PathBuf {
    let stem = input
        .file_stem()
        .map_or_else(|| "model".to_string(), |s| s.to_string_lossy().into_owned());
    input.with_file_name(format!("{stem}-turbo3.gguf"))
}
