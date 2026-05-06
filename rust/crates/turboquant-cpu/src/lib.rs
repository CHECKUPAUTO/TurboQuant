#![deny(missing_docs)]

//! CPU backend for TurboQuant.
//!
//! Provides a CPU implementation of the `Backend` trait using
//! rayon for parallelism and the `wide` crate for SIMD acceleration.

/// CPU backend using rayon + SIMD (wide).
#[allow(dead_code)]
pub struct CpuBackend;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
