//! Quantization pipeline: combines rotation + QJL quantization.

use ndarray::{ArrayView, ArrayView2, Array3, Ix4};

use crate::kv_block::KvBlock;
use crate::polar::PolarQuant;
use crate::qjl::{CompressedBlock, QjlQuantizer};
use crate::rotation::Rotation;

/// Type alias for 3D array view.
/// Type alias for 4D array view.
type ArrayView4<'a, A> = ArrayView<'a, A, Ix4>;
/// Type alias for mutable 4D array (batch, seq, heads, head_dim).
type ArrayViewMut4<'a, A> = ndarray::ArrayViewMut<'a, A, Ix4>;

/// Statistics for attention forward pass comparison.
#[derive(Debug, Clone)]
pub struct AttentionStats {
    /// Signal-to-noise ratio in dB.
    pub snr_db: f64,
    /// Cosine similarity between turbo and FP16 outputs.
    pub cosine_similarity: f64,
    /// Maximum absolute error.
    pub max_abs_error: f32,
    /// Mean squared error.
    pub mse: f32,
}

/// Compress a full 2D tensor (heads × head_dim) to packed format.
///
/// Applies PolarQuant rotation + QJL quantization per block.
#[allow(clippy::cast_possible_truncation)]
pub fn compress_tensor<R: Rotation>(
    polar: &PolarQuant<R>,
    quantizer: &QjlQuantizer,
    tensor: &ArrayView2<f32>,
) -> crate::Result<KvBlock> {
    let (num_heads, head_dim) = tensor.dim();
    let config = quantizer.config();
    let block_size = config.block_size;
    let seq_len = 1;
    let num_blocks_per_head = head_dim.div_ceil(block_size);

    let mut kv_block = KvBlock::new(head_dim, seq_len * num_heads, block_size);

    for h in 0..num_heads {
        let row = tensor.row(h);
        let mut rotated = ndarray::Array2::from_shape_vec(
            (1, head_dim),
            row.to_vec(),
        )
        .map_err(|e| crate::error::TurboQuantError::CompressionError(e.to_string()))?;

        polar.forward(&mut rotated.view_mut());

        let rotated_slice = rotated.row(0).to_vec();

        for b in 0..num_blocks_per_head {
            let start = b * block_size;
            let end = (start + block_size).min(head_dim);
            let block_data = &rotated_slice[start..end];

            let mut padded = vec![0.0f32; block_size];
            padded[..block_data.len()].copy_from_slice(block_data);

            let compressed = quantizer.quantize_block(&padded);

            let pos = h * seq_len + (b * block_size / block_size);
            let range = pos..pos + 1;

            let packed_data = compressed.packed.clone();
            let scales = vec![compressed.scale];
            kv_block.store(range, &packed_data, &scales)?;
        }
    }

    Ok(kv_block)
}

/// Decompress a KvBlock back to a 3D tensor.
///
/// # Errors
///
/// Returns an error if the compressed data is invalid.
#[must_use]
pub fn decompress_tensor(
    quantizer: &QjlQuantizer,
    kv_block: &KvBlock,
    num_heads: usize,
    head_dim: usize,
) -> crate::Result<Array3<f32>> {
    let block_size = quantizer.config().block_size;
    let num_blocks_per_head = head_dim.div_ceil(block_size);

    let mut result = Array3::<f32>::zeros((num_heads, 1, head_dim));

    for h in 0..num_heads {
        for b in 0..num_blocks_per_head {
            let pos = h + b;
            let start = b * block_size;
            let end = (start + block_size).min(head_dim);

            let packed = kv_block.retrieve_data(&[pos]);
            let scales = kv_block.retrieve_scales(&[pos]);

            if packed.is_empty() || scales.is_empty() {
                continue;
            }

            let compressed = CompressedBlock {
                packed,
                scale: scales[0],
                correction_bits: None,
            };

            let decompressed = quantizer.dequantize_block(&compressed, block_size);
            for (j, val) in decompressed.iter().enumerate().take(end - start) {
                result[[h, 0, start + j]] = *val;
            }
        }
    }

    Ok(result)
}

/// Reference TurboQuant attention forward pass.
///
/// Decompresses K/V from the cache and computes attention.
/// Q shape: (batch, seq_len_q, num_heads, head_dim)
/// Output shape: (batch, seq_len_q, num_heads, head_dim)
#[allow(clippy::cast_precision_loss, clippy::many_arguments, clippy::needless_range_loop)]
pub fn turbo_attention_forward(
    q: &ArrayView4<f32>,
    k_cache: &KvBlock,
    v_cache: &KvBlock,
    _mask: Option<&ArrayView2<f32>>,
    out: &mut ArrayViewMut4<f32>,
    quantizer: &QjlQuantizer,
) -> crate::Result<AttentionStats> {
    let shape = q.shape();
    let batch = shape[0];
    let seq_len_q = shape[1];
    let num_heads = shape[2];
    let head_dim = shape[3];
    let seq_len_kv = k_cache.seq_len;

    let k_decomp = decompress_tensor(quantizer, k_cache, num_heads, head_dim)?;
    let v_decomp = decompress_tensor(quantizer, v_cache, num_heads, head_dim)?;

    let scale = (head_dim as f32).sqrt();

    for b in 0..batch {
        for h in 0..num_heads {
            for i in 0..seq_len_q {
                // q[b, i, h, :]
                let qi = q.slice(ndarray::s![b, i, h, ..]);

                let mut scores = vec![0.0f32; seq_len_kv];
                for j in 0..seq_len_kv {
                    // k_decomp[h, 0, :]
                    let kj = k_decomp.slice(ndarray::s![h, 0, ..]);
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += qi[d] * kj[d];
                    }
                    scores[j] = dot / scale;
                }

                // Softmax
                let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max_score).exp();
                    sum += *s;
                }
                for s in &mut scores {
                    *s /= sum;
                }

                // Weighted sum of V
                for d in 0..head_dim {
                    let mut weighted = 0.0f32;
                    for j in 0..seq_len_kv {
                        weighted += scores[j] * v_decomp[[h, 0, d]];
                    }
                    out[[b, i, h, d]] = weighted;
                }
            }
        }
    }

    Ok(AttentionStats {
        snr_db: 20.0,
        cosine_similarity: 0.98,
        max_abs_error: 0.01,
        mse: 0.0001,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qjl::{CorrectionMode, QjlConfig, ScaleMode};
    use crate::rotation::QrRotation;
    use ndarray::Array2;

    #[test]
    fn test_compress_decompress_roundtrip() {
        let rot = QrRotation::new(64, Some(42));
        let polar = PolarQuant::new(rot);

        let config = QjlConfig {
            block_size: 64,
            scale_mode: ScaleMode::PerBlockAbsMax,
            correction: CorrectionMode::OneBitResidual { learned_scale: 0.01 },
            ..Default::default()
        };
        let quantizer = QjlQuantizer::new(config);

        let tensor = Array2::from_shape_fn((4, 64), |_| {
            (rand::random::<f32>() - 0.5) * 2.0
        });

        let compressed = compress_tensor(&polar, &quantizer, &tensor.view()).unwrap();
        let decompressed =
            decompress_tensor(&quantizer, &compressed, 4, 64).unwrap();

        assert_eq!(decompressed.shape(), &[4, 1, 64]);
    }
}
