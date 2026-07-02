#![deny(missing_docs)]

//! CUDA backend for `TurboQuant` — **NOT IMPLEMENTED**.
//!
//! This crate is a placeholder that reserves the API surface for a
//! future GPU backend. It contains **no GPU code**: no kernels, no
//! device management, no CUDA calls (the optional `cudarc` dependency
//! behind the `cuda` feature is never used). Every constructor fails
//! with a clear error so callers cannot mistake it for a working
//! backend. Use the CPU backend (`turboquant-cpu`) instead.

use std::fmt;

/// Error returned by every entry point of this crate: the CUDA backend
/// does not exist yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaNotImplemented;

impl fmt::Display for CudaNotImplemented {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CUDA backend not implemented (turboquant-cuda is a stub; use the CPU backend)"
        )
    }
}

impl std::error::Error for CudaNotImplemented {}

/// Placeholder for a future CUDA backend. **Not implemented** — it
/// cannot be constructed; [`CudaBackend::new`] always returns an error.
#[derive(Debug)]
pub struct CudaBackend {
    _private: (),
}

impl CudaBackend {
    /// Attempt to create the CUDA backend.
    ///
    /// # Errors
    ///
    /// Always returns [`CudaNotImplemented`]; no GPU code exists in this
    /// crate yet (with or without the `cuda` feature enabled).
    pub fn new() -> Result<Self, CudaNotImplemented> {
        Err(CudaNotImplemented)
    }

    /// Whether a usable CUDA backend is compiled in. Always `false`.
    #[must_use]
    pub const fn is_available() -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_is_honest_about_not_existing() {
        assert!(!CudaBackend::is_available());
        let err = CudaBackend::new().unwrap_err();
        assert!(err.to_string().contains("CUDA backend not implemented"));
    }
}
