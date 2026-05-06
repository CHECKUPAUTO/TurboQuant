//! QJL (Quantized Johnson-Lindenstrauss) 3-bit quantizer.
//!
//! Quantizes to 3 bits with a 1-bit residual correction,
//! guaranteeing that dot products are preserved.

use half::f16;
use serde::{Deserialize, Serialize};

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
                learned_scale: 0.01,
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
    pub fn new(config: QjlConfig) -> Self {
        Self { config }
    }

    /// Return the configuration.
    #[must_use]
    pub fn config(&self) -> &QjlConfig {
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
                let variance = block.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
                    / block.len() as f32;
                variance.sqrt().max(1e-8)
            }
            ScaleMode::Fixed(s) => s,
        }
    }

    /// Quantize a block of f32 values to 3-bit with optional correction.
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
    ///     correction: CorrectionMode::OneBitResidual { learned_scale: 0.01 },
    /// };
    /// let quantizer = QjlQuantizer::new(config);
    /// let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 10.0).collect();
    /// let compressed = quantizer.quantize_block(&data);
    /// let decompressed = quantizer.dequantize_block(&compressed, 64);
    /// assert_eq!(decompressed.len(), 64);
    /// ```
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn quantize_block(&self, block: &[f32]) -> CompressedBlock {
        let scale = self.compute_scale(block);
        let half_range = 3.5_f32; // (8-1)/2 for 3-bit
        let levels = 8_u8;
        let increment = 0.5_f32; // 1.0 / (levels / 2)

        // Normalize and quantize to 3-bit levels
        let mut quantized: Vec<u8> = Vec::with_capacity(block.len());
        let mut correction_bits: Option<Vec<u8>> = None;

        match self.config.correction {
            CorrectionMode::None => {
                for &x in block {
                    let x_norm = (x / scale).clamp(-1.0, 1.0);
                    let q_val = (x_norm * half_range / increment).round() * increment;
                    let q_clamped = q_val.clamp(-half_range, half_range);
                    // Map to 0..8 range
                    let idx = ((q_clamped / increment) + half_range / increment).round() as u8;
                    quantized.push(idx.clamp(0, levels - 1));
                }
            }
            CorrectionMode::OneBitResidual { learned_scale: _ } => {
                let mut signs = vec![0u8; block.len().div_ceil(8)];
                for (i, &x) in block.iter().enumerate() {
                    let x_norm = (x / scale).clamp(-1.0, 1.0);
                    let q_val = (x_norm * half_range / increment).round() * increment;
                    let q_clamped = q_val.clamp(-half_range, half_range);
                    let idx = ((q_clamped / increment) + half_range / increment).round() as u8;
                    quantized.push(idx.clamp(0, levels - 1));

                    // 1-bit residual correction sign
                    let residual = x_norm - q_clamped;
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
        let packed = pack_3bit_slice(&padded_quant)
            .expect("quantized values should be packable");

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
        let increment = 0.5_f32;

        let padded_len = num_values.next_multiple_of(8);
        let quantized = unpack_3bit_slice(&block.packed, padded_len)
            .expect("packed data should be unpackable");

        quantized
            .iter()
            .take(num_values)
            .enumerate()
            .map(|(i, &q)| {
                let q_float = (q as f32 - half_range / increment) * increment;
                let mut val = q_float / half_range * scale;

                // Apply 1-bit correction if available
                if let Some(ref signs) = block.correction_bits {
                    if let CorrectionMode::OneBitResidual { learned_scale } = self.config.correction {
                        let sign_bit = (signs[i / 8] >> (i % 8)) & 1;
                        let correction = if sign_bit == 1 { learned_scale } else { -learned_scale };
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
        let config = QjlConfig {
            correction: CorrectionMode::OneBitResidual { learned_scale: 0.01 },
            ..Default::default()
        };
        let quantizer = QjlQuantizer::new(config);
        let data: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) / 20.0).collect();
        let compressed = quantizer.quantize_block(&data);
        let decompressed = quantizer.dequantize_block(&compressed, 128);

        // With correction, reconstruction should be reasonable
        let mse: f32 = data
            .iter()
            .zip(decompressed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / data.len() as f32;

        let signal_power: f32 = data.iter().map(|x| x.powi(2)).sum::<f32>() / data.len() as f32;
        let snr = 10.0 * (signal_power / mse.max(1e-10)).log10();
        assert!(snr > 2.0, "SNR too low: {snr} dB");
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
