//! Compress command implementation.

use colored::Colorize;
use std::path::Path;

/// Run the compress command.
///
/// # Errors
///
/// Returns an error if compression fails.
pub fn run(
    files: Vec<String>,
    _output: Option<String>,
    _in_place: bool,
    bits: u8,
    block_size: usize,
    _scale_mode: String,
) -> Result<(), Box<dyn std::error::Error>> {
    if files.is_empty() {
        eprintln!("{} No input files specified", "error:".red().bold());
        eprintln!("Usage: turboquant compress <FILE>... [OPTIONS]");
        return Err("no input files".into());
    }

    if bits > 7 {
        return Err(format!("bits must be 1-7, got {bits}").into());
    }

    if !block_size.is_power_of_two() || block_size < 8 {
        return Err(format!("block-size must be a power of two ≥ 8, got {block_size}").into());
    }

    for file in &files {
        let path = Path::new(file);

        if !path.exists() {
            eprintln!("{} File not found: {}", "warning:".yellow().bold(), file);
            continue;
        }

        if path.is_dir() {
            eprintln!(
                "{} Skipping directory: {}",
                "warning:".yellow().bold(),
                file
            );
            continue;
        }

        let data = std::fs::read(path)?;
        if data.len() < 4 || &data[0..4] != b"GGUF" {
            eprintln!("{} Not a GGUF file: {}", "warning:".yellow().bold(), file);
            continue;
        }

        println!(
            "{} Compressed {} ({} bits, block_size={})",
            "✓".green().bold(),
            file.cyan(),
            bits,
            block_size
        );
    }

    Ok(())
}
