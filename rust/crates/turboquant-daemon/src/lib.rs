#![deny(missing_docs)]

//! `TurboQuant` daemon library.
//!
//! Provides the daemon runtime: filesystem watching, Ollama model audit,
//! and HTTP API server.

/// Daemon configuration.
pub mod config;
/// Daemon runtime.
pub mod runtime;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
