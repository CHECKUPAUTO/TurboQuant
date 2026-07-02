//! Daemon runtime: filesystem watcher, compression, health endpoint,
//! systemd watchdog keepalives, and graceful shutdown.

use crate::config::DaemonConfig;
use axum::extract::State;
use axum::routing::get;
use axum::{Json, Router};
use notify::{Event, EventKind, RecursiveMode, Watcher};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};
use turboquant_gguf::turbo::{self, TurboOptions};
use turboquant_gguf::GgufParser;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Floor for the watchdog keepalive interval, guarding against an
/// absurdly small `WATCHDOG_USEC` producing a busy ping loop.
const MIN_WATCHDOG_PING: Duration = Duration::from_secs(1);

/// Derive the keepalive ping interval from systemd's `WATCHDOG_USEC`
/// value: half the watchdog timeout (so a missed tick still leaves one
/// more chance before systemd fires), clamped to [`MIN_WATCHDOG_PING`].
fn watchdog_ping_interval(watchdog_usec: u64) -> Duration {
    Duration::from_micros(watchdog_usec / 2).max(MIN_WATCHDOG_PING)
}

/// Parse systemd watchdog configuration as passed via the environment
/// (`sd_watchdog_enabled(3)` semantics). Returns the keepalive interval
/// if a watchdog is armed for this process:
///
/// - `usec` (`WATCHDOG_USEC`) must be a positive integer.
/// - `pid` (`WATCHDOG_PID`), when set, must equal `own_pid`; otherwise
///   the watchdog is meant for another process and we must not ping.
fn watchdog_interval(usec: Option<&str>, pid: Option<&str>, own_pid: u32) -> Option<Duration> {
    let usec: u64 = usec?.parse().ok()?;
    if usec == 0 {
        return None;
    }
    if let Some(pid) = pid {
        if pid.parse::<u32>().ok()? != own_pid {
            return None;
        }
    }
    Some(watchdog_ping_interval(usec))
}

/// Read the watchdog configuration from the process environment.
fn watchdog_interval_from_env() -> Option<Duration> {
    watchdog_interval(
        std::env::var("WATCHDOG_USEC").ok().as_deref(),
        std::env::var("WATCHDOG_PID").ok().as_deref(),
        std::process::id(),
    )
}

/// Shared daemon state, exposed via the `/healthz` endpoint.
#[derive(Debug, Default)]
pub struct DaemonState {
    /// Files compressed since startup.
    pub files_compressed: AtomicU64,
    /// Failed compression attempts since startup.
    pub failures: AtomicU64,
}

/// Outcome of a single compression attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompressOutcome {
    /// The file was compressed; contains the output path.
    Compressed(PathBuf),
    /// Skipped: the input is already TurboQuant-compressed.
    AlreadyCompressed,
    /// Skipped: an up-to-date compressed copy already exists.
    UpToDate(PathBuf),
}

/// Compress a single GGUF file into `output_dir` as
/// `<stem>-turbo3.gguf`, using the same library path as the CLI.
///
/// Skips inputs that are already compressed and inputs whose output is
/// newer than the source. Writes atomically (temp file + rename).
///
/// # Errors
///
/// Returns an error on I/O failure or malformed GGUF input.
pub fn compress_once(
    input: &Path,
    output_dir: &Path,
    block_size: usize,
) -> Result<CompressOutcome, BoxError> {
    let parsed = GgufParser::parse_file(input)?;
    if turbo::is_turbo_compressed(&parsed) {
        return Ok(CompressOutcome::AlreadyCompressed);
    }

    let stem = input
        .file_stem()
        .map_or_else(|| "model".to_string(), |s| s.to_string_lossy().into_owned());
    let output = output_dir.join(format!("{stem}-turbo3.gguf"));

    if output.exists() {
        let input_mtime = input.metadata()?.modified()?;
        let output_mtime = output.metadata()?.modified()?;
        if output_mtime >= input_mtime {
            return Ok(CompressOutcome::UpToDate(output));
        }
    }

    let opts = TurboOptions {
        block_size,
        ..Default::default()
    };
    let (writer, stats) = turbo::compress(&parsed, &opts)?;
    let tmp = output_dir.join(format!(".{stem}-turbo3.gguf.tmp"));
    writer.write_to_file(&tmp)?;
    std::fs::rename(&tmp, &output)?;
    info!(
        "compressed {} -> {} ({} tensor(s) quantized, {} passed through, {:.2}x data ratio)",
        input.display(),
        output.display(),
        stats.tensors_compressed,
        stats.tensors_passthrough,
        stats.data_ratio()
    );
    Ok(CompressOutcome::Compressed(output))
}

/// Run the daemon with the given configuration: serve `/healthz`, watch
/// the configured directories for `.gguf` files, and compress each into
/// the output directory. Shuts down gracefully on SIGTERM or ctrl-c.
///
/// # Errors
///
/// Returns an error if the config is invalid, the listen address cannot
/// be bound, or the watcher cannot be created.
pub async fn run(config: DaemonConfig) -> Result<(), BoxError> {
    config.validate()?;
    info!("Starting TurboQuant daemon");

    let state = Arc::new(DaemonState::default());
    let output_dir = config.expanded_output_dir();
    std::fs::create_dir_all(&output_dir)?;

    // Health endpoint.
    let app = Router::new()
        .route("/healthz", get(healthz))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind(&config.listen_addr).await?;
    info!("Health endpoint on http://{}/healthz", config.listen_addr);

    // Filesystem watcher. notify runs its own thread; bridge events into
    // the tokio runtime through an unbounded channel.
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<PathBuf>();
    let mut watcher = notify::recommended_watcher(move |res: notify::Result<Event>| match res {
        Ok(event) => {
            if matches!(event.kind, EventKind::Create(_) | EventKind::Modify(_)) {
                for path in event.paths {
                    if path.extension().is_some_and(|e| e == "gguf") {
                        let _ = tx.send(path);
                    }
                }
            }
        }
        Err(e) => warn!("watch error: {e}"),
    })?;
    let mut n_watched = 0usize;
    for dir in config.expanded_watch_dirs() {
        if dir.is_dir() {
            watcher.watch(&dir, RecursiveMode::Recursive)?;
            info!("Watching {}", dir.display());
            n_watched += 1;
        } else {
            warn!("watch directory {} does not exist, skipping", dir.display());
        }
    }
    if n_watched == 0 {
        warn!("no watch directories exist; only the health endpoint is active");
    }

    // Tell systemd we are ready (no-op outside systemd). Keep
    // NOTIFY_SOCKET set (unset_env = false): the watchdog task below
    // sends keepalives over the same socket.
    if let Err(e) = sd_notify::notify(false, &[sd_notify::NotifyState::Ready]) {
        debug!("sd_notify failed (not running under systemd?): {e}");
    }

    // systemd watchdog keepalives. Only spawned when systemd armed a
    // watchdog for this process; normal runs pay zero overhead.
    let watchdog = watchdog_interval_from_env().map(|interval| {
        debug!("systemd watchdog enabled, pinging every {interval:?}");
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            loop {
                ticker.tick().await;
                if let Err(e) = sd_notify::notify(false, &[sd_notify::NotifyState::Watchdog]) {
                    warn!("watchdog keepalive failed: {e}");
                }
            }
        })
    });

    // HTTP server with graceful shutdown.
    let (http_shutdown_tx, http_shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = http_shutdown_rx.await;
            })
            .await
    });

    let debounce = Duration::from_secs(config.interval_secs.max(1));
    let mut last_handled: HashMap<PathBuf, Instant> = HashMap::new();
    let block_size = config.block_size;

    let shutdown = shutdown_signal();
    tokio::pin!(shutdown);

    loop {
        tokio::select! {
            () = &mut shutdown => {
                info!("shutdown signal received");
                break;
            }
            maybe_path = rx.recv() => {
                let Some(path) = maybe_path else { break };
                // Never re-compress our own output.
                if path.starts_with(&output_dir) {
                    continue;
                }
                let now = Instant::now();
                if last_handled
                    .get(&path)
                    .is_some_and(|t| now.duration_since(*t) < debounce)
                {
                    continue;
                }
                last_handled.insert(path.clone(), now);

                // Give the writer a moment to finish the file.
                tokio::time::sleep(Duration::from_millis(200)).await;

                let out_dir = output_dir.clone();
                let task_state = state.clone();
                tokio::task::spawn_blocking(move || {
                    match compress_once(&path, &out_dir, block_size) {
                        Ok(CompressOutcome::Compressed(_)) => {
                            task_state.files_compressed.fetch_add(1, Ordering::Relaxed);
                        }
                        Ok(CompressOutcome::AlreadyCompressed) => {
                            debug!("{} already compressed, skipping", path.display());
                        }
                        Ok(CompressOutcome::UpToDate(out)) => {
                            debug!("{} up to date at {}", path.display(), out.display());
                        }
                        Err(e) => {
                            task_state.failures.fetch_add(1, Ordering::Relaxed);
                            warn!("failed to compress {}: {e}", path.display());
                        }
                    }
                });
            }
        }
    }

    // Stop the keepalives and the watcher, then drain the HTTP server.
    if let Some(task) = watchdog {
        task.abort();
    }
    drop(watcher);
    let _ = http_shutdown_tx.send(());
    match server.await {
        Ok(result) => result?,
        Err(e) => warn!("HTTP server task failed: {e}"),
    }
    info!("daemon stopped");
    Ok(())
}

/// `GET /healthz` — JSON liveness/status report.
async fn healthz(State(state): State<Arc<DaemonState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "files_compressed": state.files_compressed.load(Ordering::Relaxed),
        "failures": state.failures.load(Ordering::Relaxed),
    }))
}

/// Resolves when SIGTERM (unix) or ctrl-c is received.
async fn shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        match signal(SignalKind::terminate()) {
            Ok(mut term) => {
                tokio::select! {
                    _ = ctrl_c => {}
                    _ = term.recv() => {}
                }
            }
            Err(e) => {
                warn!("cannot install SIGTERM handler: {e}");
                let _ = ctrl_c.await;
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = ctrl_c.await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use turboquant_gguf::{GgmlType, GgufValue, GgufWriter};

    fn write_test_model(path: &Path) {
        let mut w = GgufWriter::new();
        w.add_metadata("general.name", GgufValue::String("daemon-test".into()));
        let values: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        w.add_tensor("w", vec![256], GgmlType::F32, bytes).unwrap();
        w.write_to_file(path).unwrap();
    }

    #[test]
    fn compress_once_compresses_then_skips() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("model.gguf");
        let out_dir = dir.path().join("out");
        std::fs::create_dir_all(&out_dir).unwrap();
        write_test_model(&input);

        // First run compresses.
        let outcome = compress_once(&input, &out_dir, 64).unwrap();
        let CompressOutcome::Compressed(output) = outcome else {
            panic!("expected Compressed, got {outcome:?}");
        };
        assert!(output.exists());
        let parsed = GgufParser::parse_file(&output).unwrap();
        assert!(turbo::is_turbo_compressed(&parsed));
        assert_eq!(turbo::decompress_tensor(&parsed, "w").unwrap().len(), 256);

        // Second run is a no-op (output newer than input).
        assert_eq!(
            compress_once(&input, &out_dir, 64).unwrap(),
            CompressOutcome::UpToDate(output.clone())
        );

        // Compressing an already-compressed file is refused.
        assert_eq!(
            compress_once(&output, &out_dir, 64).unwrap(),
            CompressOutcome::AlreadyCompressed
        );
    }

    #[test]
    fn watchdog_ping_interval_halves_and_clamps() {
        // WatchdogSec=60 -> ping every 30s.
        assert_eq!(watchdog_ping_interval(60_000_000), Duration::from_secs(30));
        // 4s -> 2s (the smoke-test configuration).
        assert_eq!(watchdog_ping_interval(4_000_000), Duration::from_secs(2));
        // Tiny or zero values clamp to the 1s floor instead of busy-looping.
        assert_eq!(watchdog_ping_interval(100_000), MIN_WATCHDOG_PING);
        assert_eq!(watchdog_ping_interval(0), MIN_WATCHDOG_PING);
        // Exactly 2s -> exactly the floor.
        assert_eq!(watchdog_ping_interval(2_000_000), MIN_WATCHDOG_PING);
    }

    #[test]
    fn watchdog_interval_parses_environment_values() {
        let pid = 4242;
        // Enabled: usec set, no pid restriction.
        assert_eq!(
            watchdog_interval(Some("60000000"), None, pid),
            Some(Duration::from_secs(30))
        );
        // Enabled: pid restriction matches us.
        assert_eq!(
            watchdog_interval(Some("4000000"), Some("4242"), pid),
            Some(Duration::from_secs(2))
        );
        // Disabled: watchdog armed for a different process.
        assert_eq!(watchdog_interval(Some("4000000"), Some("1"), pid), None);
        // Disabled: unset, zero, or garbage usec; garbage pid.
        assert_eq!(watchdog_interval(None, None, pid), None);
        assert_eq!(watchdog_interval(Some("0"), None, pid), None);
        assert_eq!(watchdog_interval(Some("soon"), None, pid), None);
        assert_eq!(watchdog_interval(Some("-1"), None, pid), None);
        assert_eq!(watchdog_interval(Some("4000000"), Some("pid"), pid), None);
    }

    #[test]
    fn compress_once_rejects_non_gguf() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("junk.gguf");
        std::fs::write(&input, b"not a gguf file").unwrap();
        assert!(compress_once(&input, dir.path(), 64).is_err());
    }
}
