//! Info command implementation.

use colored::Colorize;

/// Run the info command.
///
/// # Errors
///
/// Always returns Ok.
pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "TurboQuant System Information".bold().cyan());
    println!("{}", "════════════════════════════".bold());
    println!();
    println!(
        "  {} {}",
        "Version:".bold(),
        env!("CARGO_PKG_VERSION").cyan()
    );
    println!("  {} {}", "Rust:".bold(), rustc_version().cyan());
    println!(
        "  {} {}",
        "CPU threads:".bold(),
        num_cpus::get().to_string().cyan()
    );
    // This build compiles in the CPU backend only. The CUDA backend is
    // not implemented (turboquant-cuda is a stub).
    println!(
        "  {} {}, {}",
        "Backends:".bold(),
        "cpu (available)".green(),
        "cuda (not implemented)".yellow()
    );

    // OS info
    println!("  {} {}", "OS:".bold(), std::env::consts::OS.cyan());
    println!("  {} {}", "Arch:".bold(), std::env::consts::ARCH.cyan());

    Ok(())
}

fn rustc_version() -> String {
    if let Ok(output) = std::process::Command::new("rustc")
        .arg("--version")
        .output()
    {
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    } else {
        "unknown".to_string()
    }
}
