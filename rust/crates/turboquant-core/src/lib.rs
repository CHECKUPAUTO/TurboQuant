#![deny(missing_docs)]
#![doc = include_str!("../README.md")]

//! TurboQuant Core: fundamental algorithms for 3-bit KV cache compression.
//!
//! This crate provides:
//! - Polar rotation (QR, Householder, Hadamard)
//! - QJL 3-bit quantization with 1-bit residual correction
//! - 3-bit bit packing/unpacking
//! - Compressed KV block storage

pub mod bitpack;
pub mod kv_block;
pub mod polar;
pub mod qjl;
pub mod quantize;
pub mod rotation;

/// Common error type for TurboQuant operations.
pub mod error;

/// Result type alias for TurboQuant operations.
pub type Result<T> = std::result::Result<T, error::TurboQuantError>;
