//! Daemon runtime.

use crate::config::DaemonConfig;
use tracing::info;

/// Run the daemon with the given configuration.
///
/// # Errors
///
/// Returns an error if the daemon fails to start.
pub async fn run(config: DaemonConfig) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    info!("Starting TurboQuant daemon");
    info!("Watching directories: {:?}", config.watch_dirs);
    info!("Listening on {}", config.listen_addr);

    // Notify systemd we're ready
    sd_notify::notify(true, &[sd_notify::NotifyState::Ready])?;

    // Keep running
    loop {
        tokio::time::sleep(std::time::Duration::from_secs(config.interval_secs)).await;
    }
}
