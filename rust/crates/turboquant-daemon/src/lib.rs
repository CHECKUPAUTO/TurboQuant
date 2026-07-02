#![deny(missing_docs)]
// Pedantic allows: run() exceeds the line lint because startup wiring
// (health endpoint, watcher, shutdown) reads best as one sequence; casts
// are test-data generation.
#![allow(clippy::too_many_lines)]
#![allow(clippy::cast_precision_loss)]

//! `TurboQuant` daemon library.
//!
//! Provides the daemon runtime: watches configured directories for
//! `.gguf` model files (via `notify`), compresses them into an output
//! directory using the same library path as the CLI
//! (`turboquant_gguf::turbo`), serves a `GET /healthz` JSON endpoint,
//! sends systemd readiness and watchdog keepalives (`sd_notify`), and
//! shuts down gracefully on SIGTERM/ctrl-c.

/// Daemon configuration.
pub mod config;
/// Daemon runtime.
pub mod runtime;
