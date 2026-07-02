//! Thin pyo3/numpy layer over [`crate::logic`].
//!
//! Compiled only outside the test harness (see `lib.rs`): with pyo3's
//! `extension-module` feature the Python C API symbols stay unresolved,
//! which is fine for the cdylib but would fail to link in a test binary.

// Doc comments here become Python __doc__ strings (help() output), so they
// use Python docstring conventions, not rustdoc markdown.
#![allow(clippy::doc_markdown)]
// pyo3's #[pyfunction]/#[pymethods] extraction needs PyReadonlyArray guards
// by value; taking references breaks FromPyObject.
#![allow(clippy::needless_pass_by_value)]

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::logic;

/// Maps a logic-layer error message onto a Python `ValueError`.
fn value_error(message: String) -> PyErr {
    PyValueError::new_err(message)
}

/// Python-level result of `Quantizer.quantize`:
/// `(packed uint8 array, scale, optional correction uint8 array)`.
type PyQuantizeOutput<'py> = (
    Bound<'py, PyArray1<u8>>,
    f32,
    Option<Bound<'py, PyArray1<u8>>>,
);

/// Borrows a numpy array as a contiguous slice or raises `ValueError`.
fn contiguous<'a, T: numpy::Element>(
    array: &'a PyReadonlyArray1<'_, T>,
    name: &str,
) -> PyResult<&'a [T]> {
    array.as_slice().map_err(|_| {
        value_error(format!(
            "{name} must be C-contiguous (use np.ascontiguousarray)"
        ))
    })
}

/// 3-bit QJL quantizer with optional 1-bit residual correction.
#[pyclass(module = "turboquant")]
struct Quantizer {
    inner: logic::QuantizerImpl,
}

#[pymethods]
impl Quantizer {
    /// Creates a quantizer.
    ///
    /// Args:
    ///     bits: bits per value; only 3 is supported.
    ///     block_size: nominal block size; each `quantize` call treats its
    ///         whole input as one block with one scale.
    ///     scale_mode: 'absmax', 'percentile', 'adaptive' or 'fixed'.
    ///     percentile: required for scale_mode='percentile', in [0, 1].
    ///     correction: enable 1-bit residual correction.
    ///     correction_scale: correction magnitude relative to the block
    ///         scale (default 0.25/3.5 ≈ 0.0714, the MSE-optimal quarter step).
    ///     fixed_scale: required for scale_mode='fixed', > 0.
    #[new]
    #[pyo3(signature = (
        bits = 3,
        block_size = 64,
        scale_mode = "absmax",
        percentile = None,
        correction = true,
        correction_scale = None,
        fixed_scale = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        bits: u8,
        block_size: usize,
        scale_mode: &str,
        percentile: Option<f32>,
        correction: bool,
        correction_scale: Option<f32>,
        fixed_scale: Option<f32>,
    ) -> PyResult<Self> {
        logic::QuantizerImpl::new(
            bits,
            block_size,
            scale_mode,
            percentile,
            correction,
            correction_scale,
            fixed_scale,
        )
        .map(|inner| Self { inner })
        .map_err(value_error)
    }

    /// Bits per value.
    #[getter]
    fn bits(&self) -> u8 {
        self.inner.bits()
    }

    /// Nominal block size.
    #[getter]
    fn block_size(&self) -> usize {
        self.inner.block_size()
    }

    /// Scale mode name.
    #[getter]
    fn scale_mode(&self) -> &'static str {
        self.inner.scale_mode_name()
    }

    /// Whether 1-bit residual correction is enabled.
    #[getter]
    fn correction(&self) -> bool {
        self.inner.correction_enabled()
    }

    fn __repr__(&self) -> String {
        format!(
            "Quantizer(bits={}, block_size={}, scale_mode='{}', correction={})",
            self.inner.bits(),
            self.inner.block_size(),
            self.inner.scale_mode_name(),
            if self.inner.correction_enabled() {
                "True"
            } else {
                "False"
            },
        )
    }

    /// Quantizes a 1-D float32 array as a single block.
    ///
    /// Returns `(packed, scale, correction)` where `packed` is a uint8
    /// array of packed 3-bit values, `scale` is the block scale (float),
    /// and `correction` is a uint8 array of packed 1-bit residual signs or
    /// `None` when correction is disabled.
    fn quantize<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<PyQuantizeOutput<'py>> {
        let slice = contiguous(&data, "data")?;
        let (packed, scale, correction) = self.inner.quantize(slice).map_err(value_error)?;
        Ok((
            packed.into_pyarray(py),
            scale,
            correction.map(|c| c.into_pyarray(py)),
        ))
    }

    /// Dequantizes `packed` (uint8, from `quantize`) back to `n` float32
    /// values using `scale` and the optional `correction` array.
    #[pyo3(signature = (packed, n, scale, correction = None))]
    fn dequantize<'py>(
        &self,
        py: Python<'py>,
        packed: PyReadonlyArray1<'py, u8>,
        n: usize,
        scale: f32,
        correction: Option<PyReadonlyArray1<'py, u8>>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let packed_slice = contiguous(&packed, "packed")?;
        let corr_slice = match &correction {
            Some(corr) => Some(contiguous(corr, "correction")?),
            None => None,
        };
        self.inner
            .dequantize(packed_slice, n, scale, corr_slice)
            .map(|values| values.into_pyarray(py))
            .map_err(value_error)
    }
}

/// Packs 3-bit values (uint8 in 0..=7, length a multiple of 8) into a
/// uint8 array of `3 * len(values) / 8` bytes.
#[pyfunction]
fn pack_3bit<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let slice = contiguous(&values, "values")?;
    logic::pack_3bit(slice)
        .map(|packed| packed.into_pyarray(py))
        .map_err(value_error)
}

/// Unpacks `n` 3-bit values (`n` a multiple of 8) from a packed uint8
/// array produced by `pack_3bit`.
#[pyfunction]
fn unpack_3bit<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray1<'py, u8>,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let slice = contiguous(&packed, "packed")?;
    logic::unpack_3bit(slice, n)
        .map(|values| values.into_pyarray(py))
        .map_err(value_error)
}

/// Applies the seeded fast Hadamard rotation to a 1-D float32 array whose
/// length is a power of two. `inverse=True` applies the inverse rotation.
#[pyfunction]
#[pyo3(signature = (data, seed, inverse = false))]
fn hadamard_rotate<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f32>,
    seed: u64,
    inverse: bool,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let slice = contiguous(&data, "data")?;
    logic::hadamard_rotate(slice, seed, inverse)
        .map(|rotated| rotated.into_pyarray(py))
        .map_err(value_error)
}

/// TurboQuant: 3-bit KV-cache quantization.
#[pymodule]
fn turboquant(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<Quantizer>()?;
    m.add_function(wrap_pyfunction!(pack_3bit, m)?)?;
    m.add_function(wrap_pyfunction!(unpack_3bit, m)?)?;
    m.add_function(wrap_pyfunction!(hadamard_rotate, m)?)?;
    Ok(())
}
