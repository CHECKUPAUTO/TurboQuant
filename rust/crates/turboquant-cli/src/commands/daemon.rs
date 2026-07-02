//! Daemon command implementation.

use colored::Colorize;
use std::path::Path;

/// Run the daemon command.
///
/// # Errors
///
/// Returns an error if the config cannot be loaded or the daemon fails
/// to start.
pub fn run(config: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    let config = match config {
        Some(path) => turboquant_daemon::config::DaemonConfig::load(Path::new(&path))
            .map_err(|e| format!("failed to load config {path}: {e}"))?,
        None => turboquant_daemon::config::DaemonConfig::default(),
    };

    println!(
        "{} Starting TurboQuant daemon (watching {:?})",
        ">>".cyan().bold(),
        config.watch_dirs
    );

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        turboquant_daemon::runtime::run(config)
            .await
            .map_err(|e| -> Box<dyn std::error::Error> {
                Box::new(std::io::Error::other(format!("{e}")))
            })
    })?;

    Ok(())
}
