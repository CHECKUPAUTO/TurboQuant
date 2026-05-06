//! Verify command implementation.

use colored::Colorize;

/// Run the verify command.
///
/// # Errors
///
/// Returns an error if verification fails.
pub fn run(file: String) -> Result<(), Box<dyn std::error::Error>> {
    let path = std::path::Path::new(&file);

    if !path.exists() {
        return Err(format!("File not found: {file}").into());
    }

    let data = std::fs::read(path)?;

    if data.len() < 4 {
        return Err("File too small to be a valid GGUF file".into());
    }

    if &data[0..4] != b"GGUF" {
        return Err("Not a valid GGUF file (bad magic number)".into());
    }

    let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);

    println!(
        "{} GGUF version {}",
        "✓".green().bold(),
        version.to_string().cyan()
    );
    println!("  File size: {} bytes", data.len());
    println!("  Magic: GGUF");

    Ok(())
}
