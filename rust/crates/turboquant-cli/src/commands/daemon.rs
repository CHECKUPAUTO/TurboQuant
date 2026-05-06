//! Daemon command implementation.

use colored::Colorize;

/// Run the daemon command.
///
/// # Errors
///
/// Returns an error if daemon fails to start.
pub fn run(_config: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "{} Starting TurboQuant daemon...",
        "🚀".cyan().bold()
    );

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let config = turboquant_daemon::config::DaemonConfig::default();
        turboquant_daemon::runtime::run(config).await
            .map_err(|e| -> Box<dyn std::error::Error> { Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("{e}"))) })
    })?;

    Ok(())
}
