//! Error types for TurboQuant operations.

use thiserror::Error;

/// Errors that can occur during `TurboQuant` operations.
#[derive(Error, Debug)]
pub enum TurboQuantError {
    /// Generic I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Invalid dimension parameter.
    #[error("Invalid dimension: {0}")]
    InvalidDimension(String),

    /// Invalid bit width (must be 1-7).
    #[error("Invalid bit width: {0} (must be 1-7)")]
    InvalidBitWidth(u8),

    /// Block size must be a power of two and at least 8.
    #[error("Invalid block size: {0}")]
    InvalidBlockSize(usize),

    /// Data not aligned to 8-value boundary for 3-bit packing.
    #[error("Data length {len} is not aligned to 8-value boundary")]
    UnalignedData {
        /// Length of the unaligned data.
        len: usize,
    },

    /// Compression/decompression failure.
    #[error("Compression failed: {0}")]
    CompressionError(String),

    /// Decompression failure.
    #[error("Decompression failed: {0}")]
    DecompressionError(String),

    /// Invalid GGUF format.
    #[error("Invalid GGUF: {0}")]
    InvalidGguf(String),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(String),
}
