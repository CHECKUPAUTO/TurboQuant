#![deny(missing_docs)]

//! `TurboQuant` daemon binary entry point.
//!
//! Watches configured directories for `.gguf` files and compresses them
//! into an output directory; serves a `GET /healthz` endpoint. See
//! [`turboquant_daemon::config::DaemonConfig`] for the JSON config
//! format.

use std::path::PathBuf;
use turboquant_daemon::config::DaemonConfig;

const USAGE: &str = "Usage: turboquant-daemon [-c|--config <config.json>]

Options:
  -c, --config <FILE>  JSON config file (defaults apply for missing keys)
  -h, --help           Show this help";

fn parse_args() -> Result<Option<PathBuf>, String> {
    let mut config = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-c" | "--config" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --config".to_string())?;
                config = Some(PathBuf::from(value));
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument '{other}'\n{USAGE}")),
        }
    }
    Ok(config)
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let result = parse_args()
        .and_then(|config_path| match config_path {
            Some(path) => DaemonConfig::load(&path)
                .map_err(|e| format!("failed to load config {}: {e}", path.display())),
            None => Ok(DaemonConfig::default()),
        })
        .and_then(|config| {
            let runtime = tokio::runtime::Runtime::new()
                .map_err(|e| format!("failed to start tokio runtime: {e}"))?;
            runtime
                .block_on(turboquant_daemon::runtime::run(config))
                .map_err(|e| e.to_string())
        });

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
