#![deny(missing_docs)]

//! CUDA backend for `TurboQuant`.
//!
//! Feature-gated via the `cuda` feature. Uses `cudarc` for GPU kernels.

/// CUDA backend using cudarc.
///
/// Only available when the `cuda` feature is enabled.
#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub struct CudaBackend;

/// Stub backend when CUDA is not available.
#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
pub struct CudaBackend;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
