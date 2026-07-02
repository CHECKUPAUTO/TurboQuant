//! Pure-Rust implementation behind the Python bindings.
//!
//! Everything in this module is plain Rust (no pyo3 types) so it can be
//! unit tested with `cargo test` even though pyo3 is built with the
//! `extension-module` feature. All fallible functions return
//! `Result<_, String>`; the bindings layer maps the `String` onto a Python
//! `ValueError`.

use half::f16;
use ndarray::Array2;
use turboquant_core::bitpack;
use turboquant_core::qjl::{CompressedBlock, CorrectionMode, QjlConfig, QjlQuantizer, ScaleMode};
use turboquant_core::rotation::{FastHadamardRotation, Rotation};

/// Number of packed bytes produced when quantizing `n` values
/// (padded up to a multiple of 8 values, 3 bits per value).
pub(crate) fn packed_size(n: usize) -> usize {
    n.div_ceil(8) * 3
}

/// Number of correction bytes produced when quantizing `n` values with
/// correction enabled (1 bit per value).
pub(crate) const fn corr_size(n: usize) -> usize {
    n.div_ceil(8)
}

/// Result of quantizing one block: `(packed bytes, scale, optional
/// correction bytes)`.
pub(crate) type QuantizeOutput = (Vec<u8>, f32, Option<Vec<u8>>);

/// Pure-Rust core of the Python `Quantizer` class.
pub(crate) struct QuantizerImpl {
    quantizer: QjlQuantizer,
    scale_mode_name: &'static str,
}

impl QuantizerImpl {
    /// Validates the Python-level configuration and builds the core
    /// quantizer.
    pub(crate) fn new(
        bits: u8,
        block_size: usize,
        scale_mode: &str,
        percentile: Option<f32>,
        correction: bool,
        correction_scale: Option<f32>,
        fixed_scale: Option<f32>,
    ) -> Result<Self, String> {
        if bits != 3 {
            return Err(format!(
                "bits must be 3 (got {bits}); only 3-bit quantization is supported"
            ));
        }
        if block_size == 0 {
            return Err("block_size must be positive".to_string());
        }
        if percentile.is_some() && scale_mode != "percentile" {
            return Err("percentile is only valid with scale_mode='percentile'".to_string());
        }
        if fixed_scale.is_some() && scale_mode != "fixed" {
            return Err("fixed_scale is only valid with scale_mode='fixed'".to_string());
        }
        let (scale_mode, scale_mode_name) = match scale_mode {
            "absmax" => (ScaleMode::PerBlockAbsMax, "absmax"),
            "percentile" => {
                let p = percentile
                    .ok_or_else(|| "scale_mode='percentile' requires percentile=".to_string())?;
                if !p.is_finite() || !(0.0..=1.0).contains(&p) {
                    return Err(format!("percentile must be in [0.0, 1.0] (got {p})"));
                }
                (ScaleMode::PerBlockPercentile(p), "percentile")
            }
            "adaptive" => (ScaleMode::Adaptive, "adaptive"),
            "fixed" => {
                let s = fixed_scale
                    .ok_or_else(|| "scale_mode='fixed' requires fixed_scale=".to_string())?;
                if !s.is_finite() || s <= 0.0 {
                    return Err(format!("fixed_scale must be finite and > 0 (got {s})"));
                }
                (ScaleMode::Fixed(s), "fixed")
            }
            other => {
                return Err(format!(
                    "unknown scale_mode {other:?}; expected 'absmax', 'percentile', 'adaptive' or 'fixed'"
                ))
            }
        };
        let correction_mode = if correction {
            let learned_scale =
                correction_scale.unwrap_or(turboquant_core::qjl::DEFAULT_CORRECTION_SCALE);
            if !learned_scale.is_finite() || learned_scale < 0.0 {
                return Err(format!(
                    "correction_scale must be finite and >= 0 (got {learned_scale})"
                ));
            }
            CorrectionMode::OneBitResidual { learned_scale }
        } else {
            CorrectionMode::None
        };

        Ok(Self {
            quantizer: QjlQuantizer::new(QjlConfig {
                bits,
                block_size,
                scale_mode,
                correction: correction_mode,
            }),
            scale_mode_name,
        })
    }

    /// Bits per value (always 3 for now).
    pub(crate) const fn bits(&self) -> u8 {
        self.quantizer.config().bits
    }

    /// Nominal block size from the configuration.
    pub(crate) const fn block_size(&self) -> usize {
        self.quantizer.config().block_size
    }

    /// Scale-mode name as passed to the constructor.
    pub(crate) const fn scale_mode_name(&self) -> &'static str {
        self.scale_mode_name
    }

    /// Whether 1-bit residual correction is enabled.
    pub(crate) fn correction_enabled(&self) -> bool {
        matches!(
            self.quantizer.config().correction,
            CorrectionMode::OneBitResidual { .. }
        )
    }

    /// Quantizes `data` as a single block. Returns
    /// `(packed bytes, scale, optional correction bytes)`.
    pub(crate) fn quantize(&self, data: &[f32]) -> Result<QuantizeOutput, String> {
        if data.is_empty() {
            return Err("input array must not be empty".to_string());
        }
        let block = self.quantizer.quantize_block(data);
        if block.packed.len() != packed_size(data.len()) {
            return Err("internal error: unexpected packed length".to_string());
        }
        Ok((block.packed, block.scale.to_f32(), block.correction_bits))
    }

    /// Dequantizes `packed` (with optional `correction` bits) back to `n`
    /// float values using `scale`.
    pub(crate) fn dequantize(
        &self,
        packed: &[u8],
        n: usize,
        scale: f32,
        correction: Option<&[u8]>,
    ) -> Result<Vec<f32>, String> {
        if n == 0 {
            return Err("n must be positive".to_string());
        }
        if !scale.is_finite() || scale <= 0.0 {
            return Err(format!("scale must be finite and > 0 (got {scale})"));
        }
        let needed = packed_size(n);
        if packed.len() < needed {
            return Err(format!(
                "packed buffer too small: got {} bytes, need {needed} for n={n}",
                packed.len()
            ));
        }
        if let Some(corr) = correction {
            let needed_corr = corr_size(n);
            if corr.len() < needed_corr {
                return Err(format!(
                    "correction buffer too small: got {} bytes, need {needed_corr} for n={n}",
                    corr.len()
                ));
            }
        }
        let block = CompressedBlock {
            packed: packed[..needed].to_vec(),
            scale: f16::from_f32(scale),
            correction_bits: correction.map(|c| c[..corr_size(n)].to_vec()),
        };
        let values = self.quantizer.dequantize_block(&block, n);
        if values.len() != n {
            return Err("internal error: unexpected output length".to_string());
        }
        Ok(values)
    }
}

/// Packs 3-bit values (each in `0..=7`, length a multiple of 8) into bytes.
pub(crate) fn pack_3bit(values: &[u8]) -> Result<Vec<u8>, String> {
    if let Some(bad) = values.iter().find(|&&v| v > 7) {
        return Err(format!("values must be 3-bit (0..=7), got {bad}"));
    }
    bitpack::pack_3bit_slice(values).map_err(|e| e.to_string())
}

/// Unpacks `n` 3-bit values (`n` a multiple of 8) from packed bytes.
pub(crate) fn unpack_3bit(packed: &[u8], n: usize) -> Result<Vec<u8>, String> {
    bitpack::unpack_3bit_slice(packed, n).map_err(|e| e.to_string())
}

/// Applies the seeded fast Hadamard rotation (forward or inverse) to a
/// 1-D vector whose length must be a power of two.
pub(crate) fn hadamard_rotate(data: &[f32], seed: u64, inverse: bool) -> Result<Vec<f32>, String> {
    let n = data.len();
    if !n.is_power_of_two() {
        return Err(format!("data length must be a power of two (got {n})"));
    }
    let rotation = FastHadamardRotation::new(n, Some(seed));
    let mut matrix = Array2::from_shape_vec((1, n), data.to_vec())
        .map_err(|e| format!("internal error: {e}"))?;
    if inverse {
        rotation.inverse(&mut matrix.view_mut());
    } else {
        rotation.forward(&mut matrix.view_mut());
    }
    Ok(matrix.into_raw_vec_and_offset().0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_quantizer() -> QuantizerImpl {
        QuantizerImpl::new(3, 64, "absmax", None, true, None, None).unwrap()
    }

    #[test]
    fn accessors_report_config() {
        let q = QuantizerImpl::new(3, 32, "percentile", Some(0.95), false, None, None).unwrap();
        assert_eq!(q.bits(), 3);
        assert_eq!(q.block_size(), 32);
        assert_eq!(q.scale_mode_name(), "percentile");
        assert!(!q.correction_enabled());
        assert!(default_quantizer().correction_enabled());
        assert_eq!(default_quantizer().scale_mode_name(), "absmax");
    }

    #[test]
    fn sizes() {
        assert_eq!(packed_size(64), 24);
        assert_eq!(packed_size(1), 3);
        assert_eq!(corr_size(64), 8);
        assert_eq!(corr_size(1), 1);
    }

    #[test]
    fn config_validation_errors() {
        // Wrong bit width.
        assert!(QuantizerImpl::new(4, 64, "absmax", None, true, None, None).is_err());
        // Zero block size.
        assert!(QuantizerImpl::new(3, 0, "absmax", None, true, None, None).is_err());
        // Unknown scale mode.
        assert!(QuantizerImpl::new(3, 64, "bogus", None, true, None, None).is_err());
        // Percentile mode without / with bad percentile.
        assert!(QuantizerImpl::new(3, 64, "percentile", None, true, None, None).is_err());
        assert!(QuantizerImpl::new(3, 64, "percentile", Some(1.5), true, None, None).is_err());
        // Percentile passed to the wrong mode.
        assert!(QuantizerImpl::new(3, 64, "absmax", Some(0.9), true, None, None).is_err());
        // Fixed mode without / with bad fixed_scale.
        assert!(QuantizerImpl::new(3, 64, "fixed", None, true, None, None).is_err());
        assert!(QuantizerImpl::new(3, 64, "fixed", None, true, None, Some(-1.0)).is_err());
        // fixed_scale passed to the wrong mode.
        assert!(QuantizerImpl::new(3, 64, "adaptive", None, true, None, Some(1.0)).is_err());
        // Bad correction scale.
        assert!(QuantizerImpl::new(3, 64, "absmax", None, true, Some(f32::NAN), None).is_err());
        // Valid configs for every mode.
        for (mode, p, f) in [
            ("absmax", None, None),
            ("percentile", Some(0.99), None),
            ("adaptive", None, None),
            ("fixed", None, Some(2.0)),
        ] {
            assert!(
                QuantizerImpl::new(3, 64, mode, p, true, Some(0.02), f).is_ok(),
                "mode {mode}"
            );
        }
    }

    #[test]
    fn quantize_roundtrip_matches_core() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 10.0).collect();
        let q = default_quantizer();
        let (packed, scale, corr) = q.quantize(&data).unwrap();
        assert_eq!(packed.len(), packed_size(64));
        let corr = corr.expect("correction enabled");
        assert_eq!(corr.len(), corr_size(64));

        let ours = q.dequantize(&packed, 64, scale, Some(&corr)).unwrap();

        // Reference straight through turboquant-core, with the same
        // defaults the Python-facing Quantizer uses.
        let reference = QjlQuantizer::new(QjlConfig::default());
        let block = reference.quantize_block(&data);
        assert_eq!(scale.to_bits(), block.scale.to_f32().to_bits());
        assert_eq!(ours, reference.dequantize_block(&block, 64));
    }

    #[test]
    fn quantize_no_correction_unaligned_len() {
        let data: Vec<f32> = (0..50).map(|i| ((i * 7) % 13) as f32 - 6.0).collect();
        let q = QuantizerImpl::new(3, 64, "absmax", None, false, None, None).unwrap();
        let (packed, scale, corr) = q.quantize(&data).unwrap();
        assert!(corr.is_none());
        let out = q.dequantize(&packed, 50, scale, None).unwrap();
        assert_eq!(out.len(), 50);
        let mse: f32 = data
            .iter()
            .zip(&out)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / data.len() as f32;
        assert!(mse < 2.0, "mse too high: {mse}");
    }

    #[test]
    fn quantize_dequantize_errors() {
        let q = default_quantizer();
        assert!(q.quantize(&[]).is_err());

        let (packed, scale, corr) = q.quantize(&[1.0f32; 16]).unwrap();
        let corr = corr.unwrap();
        // n = 0.
        assert!(q.dequantize(&packed, 0, scale, Some(&corr)).is_err());
        // Bad scales.
        for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
            assert!(q.dequantize(&packed, 16, bad, Some(&corr)).is_err());
        }
        // Packed too short.
        assert!(q
            .dequantize(&packed[..packed.len() - 1], 16, scale, Some(&corr))
            .is_err());
        // Correction too short.
        assert!(q.dequantize(&packed, 16, scale, Some(&corr[..1])).is_err());
        // Missing correction is allowed (correction is skipped).
        assert!(q.dequantize(&packed, 16, scale, None).is_ok());
    }

    #[test]
    fn pack_unpack_roundtrip_and_errors() {
        let values: Vec<u8> = (0..64u8).map(|x| x % 8).collect();
        let packed = pack_3bit(&values).unwrap();
        assert_eq!(packed.len(), 24);
        assert_eq!(unpack_3bit(&packed, 64).unwrap(), values);

        // Length not a multiple of 8.
        assert!(pack_3bit(&[0u8; 7]).is_err());
        assert!(unpack_3bit(&packed, 63).is_err());
        // Packed buffer too short.
        assert!(unpack_3bit(&packed[..3], 64).is_err());
        // Out-of-range value.
        assert!(pack_3bit(&[8u8; 8]).is_err());
    }

    #[test]
    fn hadamard_roundtrip_and_errors() {
        let data: Vec<f32> = (0..128).map(|i| (i as f32).sin()).collect();
        let rotated = hadamard_rotate(&data, 42, false).unwrap();
        assert_eq!(rotated.len(), data.len());
        // The rotation must actually change the data.
        assert!(rotated.iter().zip(&data).any(|(a, b)| (a - b).abs() > 1e-3));
        // Deterministic for a fixed seed.
        assert_eq!(rotated, hadamard_rotate(&data, 42, false).unwrap());

        let restored = hadamard_rotate(&rotated, 42, true).unwrap();
        for (a, b) in data.iter().zip(&restored) {
            assert!((a - b).abs() < 1e-4, "roundtrip mismatch: {a} vs {b}");
        }

        // Non-power-of-two lengths (including 0) are rejected.
        assert!(hadamard_rotate(&[1.0; 3], 42, false).is_err());
        assert!(hadamard_rotate(&[], 42, false).is_err());
    }
}
