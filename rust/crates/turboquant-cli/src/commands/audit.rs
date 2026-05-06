//! Audit command implementation.

use colored::Colorize;
use std::path::Path;

/// Run the audit command.
///
/// # Errors
///
/// Returns an error if audit fails.
pub fn run(path: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    let dir = path.unwrap_or_else(|| {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
        format!("{home}/.ollama/models")
    });

    let dir_path = Path::new(&dir);

    println!("{} Auditing: {}", "🔍".cyan().bold(), dir.cyan());

    if !dir_path.exists() {
        println!(
            "{} Directory not found: {}",
            "⚠".yellow().bold(),
            dir.yellow()
        );
        return Ok(());
    }

    let mut total_size = 0u64;
    let mut gguf_count = 0u64;

    // Walk the directory
    if let Ok(entries) = std::fs::read_dir(dir_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Ok(meta) = path.metadata() {
                if meta.is_file() {
                    // Quick check for GGUF files
                    if let Ok(data) = std::fs::read(&path) {
                        if data.len() >= 4 && &data[0..4] == b"GGUF" {
                            gguf_count += 1;
                            total_size += meta.len();
                        }
                    }
                }
            }
        }
    }

    let estimated_compressed = (total_size as f64 * 3.0 / 16.0) as u64;

    println!();
    println!("{}", "Audit Results:".bold().green());
    println!(
        "  GGUF files found:      {}",
        gguf_count.to_string().cyan()
    );
    println!(
        "  Total size:            {} MB",
        (total_size / (1024 * 1024)).to_string().yellow()
    );
    println!(
        "  Est. compressed size:  {} MB",
        (estimated_compressed / (1024 * 1024)).to_string().green()
    );
    println!(
        "  Estimated savings:     {} MB ({}×)",
        ((total_size - estimated_compressed) / (1024 * 1024)).to_string().bright_green().bold(),
        format!("{:.1}", total_size as f64 / estimated_compressed.max(1) as f64).bright_green().bold()
    );

    Ok(())
}
