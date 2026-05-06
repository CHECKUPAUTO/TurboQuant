//! Daemon configuration.

use serde::{Deserialize, Serialize};

/// Configuration for the TurboQuant daemon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonConfig {
    /// Directories to watch for new models.
    pub watch_dirs: Vec<String>,
    /// HTTP listen address.
    pub listen_addr: String,
    /// Watch interval in seconds (for polling fallback).
    pub interval_secs: u64,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            watch_dirs: vec!["~/.ollama/models".to_string()],
            listen_addr: "127.0.0.1:7460".to_string(),
            interval_secs: 30,
        }
    }
}
