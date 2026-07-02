#![deny(missing_docs)]
// Clippy allows for practical numerical code (same set as turboquant-core).
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_errors_doc)]

//! Python bindings for `TurboQuant` via pyo3.
//!
//! Exposes a `turboquant` Python module with:
//!
//! - `Quantizer(bits=3, block_size=64, scale_mode='absmax', percentile=None,
//!   correction=True, correction_scale=None, fixed_scale=None)` with
//!   `quantize(...)` and `dequantize(...)` methods,
//! - `pack_3bit(values)` / `unpack_3bit(packed, n)`,
//! - `hadamard_rotate(data, seed, inverse=False)`,
//! - `__version__`.
//!
//! The crate is split in two layers:
//!
//! - `logic`: pure-Rust implementation (no pyo3 types) that is unit
//!   tested with plain `cargo test`,
//! - `bindings`: the thin pyo3/numpy layer. It is compiled out of the test
//!   harness (`#[cfg(not(test))]`) because pyo3's `extension-module`
//!   feature leaves the Python C API symbols unresolved, which would fail
//!   to link in a test executable.

mod logic;

#[cfg(not(test))]
mod bindings;
