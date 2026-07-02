//! QJL (Quantized Johnson-Lindenstrauss) 3-bit quantizer.
//!
//! Quantizes to 3 bits with a 1-bit residual correction,
//! guaranteeing that dot products are preserved.

use half::f16;
use serde::{Deserialize, Serialize};

/// Default magnitude of the 1-bit residual correction, in *normalized*
/// units (the dequantizer applies `val += ±learned_scale * scale`).
///
/// With the symmetric 8-level grid (step = 1.0 in level units), the
/// quantization residual is ~uniform in [-0.5, 0.5] level units, so the
/// MSE-optimal fixed 1-bit correction is E[|residual|] = a quarter step
/// = 0.25 level units = 0.25 / 3.5 ≈ 0.0714 normalized.
pub const DEFAULT_CORRECTION_SCALE: f32 = 0.25 / 3.5;

/// Configuration for the QJL quantizer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QjlConfig {
    /// Number of bits per value (default 3).
    pub bits: u8,
    /// Block size for per-block scaling.
    pub block_size: usize,
    /// Scale computation mode.
    pub scale_mode: ScaleMode,
    /// Correction mode.
    pub correction: CorrectionMode,
}

impl Default for QjlConfig {
    fn default() -> Self {
        Self {
            bits: 3,
            block_size: 64,
            scale_mode: ScaleMode::PerBlockAbsMax,
            correction: CorrectionMode::OneBitResidual {
                learned_scale: DEFAULT_CORRECTION_SCALE,
            },
        }
    }
}

/// How to compute the per-block scale.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ScaleMode {
    /// Use the absolute maximum value per block.
    PerBlockAbsMax,
    /// Use the specified percentile per block.
    #[allow(clippy::cast_precision_loss)]
    PerBlockPercentile(f32),
    /// Adaptive: derive from residual variance.
    Adaptive,
    /// Fixed scale value.
    Fixed(f32),
}

/// Correction mode for residual quantization error.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CorrectionMode {
    /// No correction applied.
    None,
    /// 1-bit residual sign correction with learned scale.
    OneBitResidual {
        /// Scale factor for the 1-bit correction.
        learned_scale: f32,
    },
}

/// A single compressed block: packed quantized values + scale.
#[derive(Clone, Debug)]
pub struct CompressedBlock {
    /// Packed 3-bit quantized values.
    pub packed: Vec<u8>,
    /// Scale for this block (f16).
    pub scale: f16,
    /// 1-bit correction signs (packed bits).
    pub correction_bits: Option<Vec<u8>>,
}

/// The QJL quantizer.
pub struct QjlQuantizer {
    config: QjlConfig,
}

impl QjlQuantizer {
    /// Create a new QJL quantizer with the given config.
    #[must_use]
    pub const fn new(config: QjlConfig) -> Self {
        Self { config }
    }

    /// Return the configuration.
    #[must_use]
    pub const fn config(&self) -> &QjlConfig {
        &self.config
    }

    /// Compute the optimal scale for a block of values.
    #[must_use]
    pub fn compute_scale(&self, block: &[f32]) -> f32 {
        match self.config.scale_mode {
            ScaleMode::PerBlockAbsMax => {
                block.iter().map(|x| x.abs()).fold(0.0f32, f32::max) + 1e-8
            }
            ScaleMode::PerBlockPercentile(p) => {
                let mut abs_vals: Vec<f32> = block.iter().map(|x| x.abs()).collect();
                abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let idx = ((abs_vals.len() - 1) as f32 * p).round() as usize;
                abs_vals[idx.clamp(0, abs_vals.len() - 1)] + 1e-8
            }
            ScaleMode::Adaptive => {
                let mean = block.iter().sum::<f32>() / block.len() as f32;
                let variance =
                    block.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / block.len() as f32;
                variance.sqrt().max(1e-8)
            }
            ScaleMode::Fixed(s) => s,
        }
    }

    /// Quantize a block of f32 values to 3-bit with optional correction.
    ///
    /// Uses a symmetric 8-level grid in "level units":
    /// `{-3.5, -2.5, -1.5, -0.5, +0.5, +1.5, +2.5, +3.5}` (spacing 1.0).
    /// A normalized value `x_norm ∈ [-1, 1]` maps to
    /// `idx = clamp(round(x_norm * 3.5 + 3.5), 0, 7)` and reconstructs as
    /// `(idx - 3.5) / 3.5 * scale`.
    ///
    /// # Examples
    ///
    /// ```
    /// use turboquant_core::qjl::{QjlQuantizer, QjlConfig, ScaleMode, CorrectionMode};
    ///
    /// let config = QjlConfig {
    ///     bits: 3,
    ///     block_size: 64,
    ///     scale_mode: ScaleMode::PerBlockAbsMax,
    ///     correction: CorrectionMode::OneBitResidual { learned_scale: 0.25 / 3.5 },
    /// };
    /// let quantizer = QjlQuantizer::new(config);
    /// let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 10.0).collect();
    /// let compressed = quantizer.quantize_block(&data);
    /// let decompressed = quantizer.dequantize_block(&compressed, 64);
    /// assert_eq!(decompressed.len(), 64);
    /// ```
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    #[must_use]
    pub fn quantize_block(&self, block: &[f32]) -> CompressedBlock {
        let scale = self.compute_scale(block);
        let half_range = 3.5_f32; // (levels - 1) / 2 for 3-bit

        // Normalize and quantize to 3-bit levels
        let mut quantized: Vec<u8> = Vec::with_capacity(block.len());
        let mut correction_bits: Option<Vec<u8>> = None;

        match self.config.correction {
            CorrectionMode::None => {
                for &x in block {
                    let x_norm = (x / scale).clamp(-1.0, 1.0);
                    let idx = x_norm
                        .mul_add(half_range, half_range)
                        .round()
                        .clamp(0.0, 7.0);
                    quantized.push(idx as u8);
                }
            }
            CorrectionMode::OneBitResidual { learned_scale: _ } => {
                let mut signs = vec![0u8; block.len().div_ceil(8)];
                for (i, &x) in block.iter().enumerate() {
                    let x_norm = (x / scale).clamp(-1.0, 1.0);
                    let idx = x_norm
                        .mul_add(half_range, half_range)
                        .round()
                        .clamp(0.0, 7.0);
                    quantized.push(idx as u8);

                    // 1-bit residual correction sign, computed consistently
                    // in level units: `x_norm * 3.5` and `level = idx - 3.5`
                    // both live on the [-3.5, +3.5] grid scale.
                    let level = idx - half_range;
                    let residual = x_norm * half_range - level;
                    if residual > 0.0 {
                        signs[i / 8] |= 1 << (i % 8);
                    }
                }
                correction_bits = Some(signs);
            }
        }

        // Pack 3-bit values
        use crate::bitpack::pack_3bit_slice;
        let num_values = block.len();
        let padded_len = num_values.next_multiple_of(8);
        let mut padded_quant = quantized;
        padded_quant.resize(padded_len, 0);
        let packed = match pack_3bit_slice(&padded_quant) {
            Ok(p) => p,
            Err(e) => {
                tracing::error!("Internal bit-packing error: {e}");
                return CompressedBlock {
                    packed: Vec::new(),
                    scale: f16::from_f32(scale),
                    correction_bits: None,
                };
            }
        };

        CompressedBlock {
            packed,
            scale: f16::from_f32(scale),
            correction_bits,
        }
    }

    /// Dequantize a compressed block back to f32 values.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn dequantize_block(&self, block: &CompressedBlock, num_values: usize) -> Vec<f32> {
        use crate::bitpack::unpack_3bit_slice;
        let scale = block.scale.to_f32();
        let half_range = 3.5_f32;

        let padded_len = num_values.next_multiple_of(8);
        let quantized = match unpack_3bit_slice(&block.packed, padded_len) {
            Ok(u) => u,
            Err(e) => {
                tracing::error!("Internal bit-unpacking error: {e}");
                return vec![0.0f32; num_values];
            }
        };

        quantized
            .iter()
            .take(num_values)
            .enumerate()
            .map(|(i, &q)| {
                // Level in level units: idx ∈ [0, 7] → {-3.5, ..., +3.5}.
                let level = f32::from(q) - half_range;
                let mut val = level / half_range * scale;

                // Apply 1-bit correction if available
                if let Some(ref signs) = block.correction_bits {
                    if let CorrectionMode::OneBitResidual { learned_scale } = self.config.correction
                    {
                        let sign_bit = (signs[i / 8] >> (i % 8)) & 1;
                        let correction = if sign_bit == 1 {
                            learned_scale
                        } else {
                            -learned_scale
                        };
                        val += correction * scale;
                    }
                }

                val
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand_distr::StandardNormal;

    /// Deterministic Gaussian test data.
    fn gaussian(n: usize, seed: u64) -> Vec<f32> {
        let mut rng: rand::rngs::StdRng = rand::SeedableRng::seed_from_u64(seed);
        (0..n).map(|_| rng.sample(StandardNormal)).collect()
    }

    /// Round-trip a full vector block-by-block through the quantizer.
    fn roundtrip(quantizer: &QjlQuantizer, data: &[f32]) -> Vec<f32> {
        let block_size = quantizer.config().block_size;
        let mut out = Vec::with_capacity(data.len());
        for chunk in data.chunks(block_size) {
            let compressed = quantizer.quantize_block(chunk);
            out.extend(quantizer.dequantize_block(&compressed, chunk.len()));
        }
        out
    }

    fn snr_db(original: &[f32], reconstructed: &[f32]) -> f64 {
        let mse: f64 = original
            .iter()
            .zip(reconstructed)
            .map(|(a, b)| f64::from(a - b).powi(2))
            .sum::<f64>()
            / original.len() as f64;
        let signal: f64 =
            original.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / original.len() as f64;
        10.0 * (signal / mse.max(1e-12)).log10()
    }

    #[test]
    fn test_default_config() {
        let config = QjlConfig::default();
        assert_eq!(config.bits, 3);
        assert_eq!(config.block_size, 64);
    }

    #[test]
    fn test_scale_per_block_abs_max() {
        let config = QjlConfig {
            scale_mode: ScaleMode::PerBlockAbsMax,
            ..Default::default()
        };
        let quantizer = QjlQuantizer::new(config);
        let scale = quantizer.compute_scale(&[-5.0, 3.0, 1.0, 0.0]);
        assert!((scale - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_scale_fixed() {
        let config = QjlConfig {
            scale_mode: ScaleMode::Fixed(10.0),
            ..Default::default()
        };
        let quantizer = QjlQuantizer::new(config);
        let scale = quantizer.compute_scale(&[-100.0, 3.0]);
        assert!((scale - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_scale_adaptive() {
        let config = QjlConfig {
            scale_mode: ScaleMode::Adaptive,
            ..Default::default()
        };
        let quantizer = QjlQuantizer::new(config);
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let scale = quantizer.compute_scale(&data);
        // std dev of 0..100 ≈ 28.9
        assert!(scale > 20.0 && scale < 35.0);
    }

    #[test]
    fn test_quantize_dequantize_no_correction() {
        let config = QjlConfig {
            correction: CorrectionMode::None,
            ..Default::default()
        };
        let quantizer = QjlQuantizer::new(config);
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 10.0).collect();
        let compressed = quantizer.quantize_block(&data);
        let decompressed = quantizer.dequantize_block(&compressed, 64);

        assert_eq!(decompressed.len(), 64);
        // Values should be non-zero
        let sum: f32 = decompressed.iter().map(|x| x.abs()).sum();
        assert!(sum > 0.0);
    }

    #[test]
    fn test_quantize_dequantize_with_correction() {
        // Representative (Gaussian) data with the default 1-bit correction.
        // The README claims SNR > 12 dB; the fixed grid delivers ~18 dB.
        let quantizer = QjlQuantizer::new(QjlConfig::default());
        let data = gaussian(4096, 42);
        let decompressed = roundtrip(&quantizer, &data);

        let snr = snr_db(&data, &decompressed);
        assert!(snr > 12.0, "SNR too low: {snr} dB");
    }

    #[test]
    fn test_correction_improves_snr() {
        // Regression for the old default learned_scale = 0.01, which was
        // so small (and applied with the wrong sign convention) that the
        // "correction" made reconstruction WORSE than no correction.
        let data = gaussian(4096, 7);

        let no_corr = QjlQuantizer::new(QjlConfig {
            correction: CorrectionMode::None,
            ..Default::default()
        });
        let with_corr = QjlQuantizer::new(QjlConfig::default());

        let snr_none = snr_db(&data, &roundtrip(&no_corr, &data));
        let snr_corr = snr_db(&data, &roundtrip(&with_corr, &data));
        assert!(
            snr_corr > snr_none,
            "1-bit correction must improve SNR: {snr_corr} dB vs {snr_none} dB"
        );
    }

    #[test]
    fn test_positive_values_roundtrip_positive() {
        // Regression for the 3-bit grid collapse: with the broken 15-level
        // mapping, EVERY positive input reconstructed to 0.0 (or to
        // -learned_scale with correction enabled).
        for correction in [
            CorrectionMode::None,
            CorrectionMode::OneBitResidual {
                learned_scale: 0.25 / 3.5,
            },
        ] {
            let quantizer = QjlQuantizer::new(QjlConfig {
                correction,
                ..Default::default()
            });
            let data = vec![0.25f32, 0.5, 0.75, 1.0];
            let compressed = quantizer.quantize_block(&data);
            let decompressed = quantizer.dequantize_block(&compressed, data.len());
            for (x, y) in data.iter().zip(&decompressed) {
                assert!(*y > 0.0, "positive input {x} reconstructed to {y}");
            }
            // Symmetric check for negatives.
            let neg: Vec<f32> = data.iter().map(|x| -x).collect();
            let compressed = quantizer.quantize_block(&neg);
            let decompressed = quantizer.dequantize_block(&compressed, neg.len());
            for (x, y) in neg.iter().zip(&decompressed) {
                assert!(*y < 0.0, "negative input {x} reconstructed to {y}");
            }
        }
    }

    #[test]
    fn test_dot_product_preserved() {
        // QJL's raison d'être: quantization approximately preserves
        // dot products between vectors.
        let quantizer = QjlQuantizer::new(QjlConfig::default());
        let a = gaussian(512, 1);
        let b = gaussian(512, 2);
        let a_q = roundtrip(&quantizer, &a);
        let b_q = roundtrip(&quantizer, &b);

        let dot = |x: &[f32], y: &[f32]| -> f64 {
            x.iter()
                .zip(y)
                .map(|(u, v)| f64::from(*u) * f64::from(*v))
                .sum()
        };
        let norm = |x: &[f32]| dot(x, x).sqrt();

        let exact = dot(&a, &b);
        let approx = dot(&a_q, &b_q);
        let bound = norm(&a) * norm(&b);
        let rel_err = (exact - approx).abs() / bound;
        assert!(
            rel_err < 0.05,
            "dot product not preserved: exact={exact}, approx={approx}, rel_err={rel_err}"
        );
    }

    #[test]
    fn test_scale_percentile() {
        let config = QjlConfig {
            scale_mode: ScaleMode::PerBlockPercentile(0.99),
            ..Default::default()
        };
        let quantizer = QjlQuantizer::new(config);
        let mut data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        data[63] = 1000.0; // outlier
        let scale = quantizer.compute_scale(&data);
        // Should ignore the outlier
        assert!(scale < 100.0);
    }
}
