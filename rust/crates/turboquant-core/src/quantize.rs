//! Quantization pipeline: combines rotation + QJL quantization.

use half::f16;
use ndarray::{Array3, ArrayView, ArrayView2, Ix4};

use crate::kv_block::KvBlock;
use crate::polar::PolarQuant;
use crate::qjl::{CompressedBlock, QjlQuantizer};
use crate::rotation::Rotation;

/// Type alias for 4D array view.
type ArrayView4<'a, A> = ArrayView<'a, A, Ix4>;
/// Type alias for mutable 4D array (batch, seq, heads, `head_dim`).
type ArrayViewMut4<'a, A> = ndarray::ArrayViewMut<'a, A, Ix4>;

/// Statistics for attention forward pass comparison.
///
/// These are **measured** by [`turbo_attention_forward`], not assumed:
/// the fully-quantized attention path (Q round-tripped through the
/// quantizer, K/V from the compressed cache) is compared element-wise
/// against the float-Q path (float Q against the same dequantized K/V,
/// which is what is written to `out`).
#[derive(Debug, Clone)]
pub struct AttentionStats {
    /// Signal-to-noise ratio in dB.
    pub snr_db: f64,
    /// Cosine similarity between the two attention outputs.
    pub cosine_similarity: f64,
    /// Maximum absolute error.
    pub max_abs_error: f32,
    /// Mean squared error.
    pub mse: f32,
}

/// Compress a full 2D tensor (rows × `head_dim`) to packed format.
///
/// Applies `PolarQuant` rotation + QJL quantization per block. Each row
/// of the input (a head, or a KV position) is stored at its own position
/// in the returned [`KvBlock`]: position `r` holds the packed 3-bit codes
/// and the `head_dim.div_ceil(block_size)` per-block scales for row `r`.
/// When the quantizer is configured with a 1-bit residual correction, the
/// correction signs are persisted in the `KvBlock` as well.
///
/// # Errors
///
/// Returns an error if `head_dim` or the configured block size is not a
/// multiple of 8 (required for 3-bit packing alignment).
#[allow(clippy::cast_possible_truncation)]
pub fn compress_tensor<R: Rotation>(
    polar: &PolarQuant<R>,
    quantizer: &QjlQuantizer,
    tensor: &ArrayView2<f32>,
) -> crate::Result<KvBlock> {
    let (num_rows, head_dim) = tensor.dim();
    let block_size = quantizer.config().block_size;
    if head_dim % 8 != 0 {
        return Err(crate::error::TurboQuantError::UnalignedData { len: head_dim });
    }
    if block_size % 8 != 0 {
        return Err(crate::error::TurboQuantError::InvalidBlockSize(block_size));
    }
    let num_blocks_per_row = head_dim.div_ceil(block_size);

    let mut kv_block = KvBlock::new(head_dim, num_rows, block_size);

    for r in 0..num_rows {
        let row = tensor.row(r);
        let mut rotated = ndarray::Array2::from_shape_vec((1, head_dim), row.to_vec())
            .map_err(|e| crate::error::TurboQuantError::CompressionError(e.to_string()))?;

        polar.forward(&mut rotated.view_mut());

        let rotated_slice = rotated.row(0).to_vec();

        let mut packed_row: Vec<u8> = Vec::with_capacity(head_dim * 3 / 8);
        let mut scales: Vec<f16> = Vec::with_capacity(num_blocks_per_row);
        let mut correction_row: Vec<u8> = Vec::new();

        for b in 0..num_blocks_per_row {
            let start = b * block_size;
            let end = (start + block_size).min(head_dim);
            let block_data = &rotated_slice[start..end];

            let mut padded = vec![0.0f32; block_size];
            padded[..block_data.len()].copy_from_slice(block_data);

            let compressed = quantizer.quantize_block(&padded);

            // Keep only the bytes covering values that belong to this row
            // (the tail block may have been padded up to `block_size`).
            let bytes_used = (end - start) * 3 / 8;
            packed_row.extend_from_slice(&compressed.packed[..bytes_used]);
            scales.push(compressed.scale);
            if let Some(bits) = &compressed.correction_bits {
                correction_row.extend_from_slice(&bits[..(end - start) / 8]);
            }
        }

        kv_block.store(r..r + 1, &packed_row, &scales)?;
        if !correction_row.is_empty() {
            kv_block.store_correction(r..r + 1, &correction_row)?;
        }
    }

    Ok(kv_block)
}

/// Decompress a `KvBlock` back to a 3D tensor of shape
/// (`num_rows`, 1, `head_dim`).
///
/// Position `r` of the `KvBlock` maps to row `r` of the output, mirroring
/// [`compress_tensor`]. The inverse rotation is **not** applied; the
/// returned values live in the rotated domain. This is the right form for
/// consumers that operate in the rotated domain (e.g. attention, where the
/// shared orthogonal rotation cancels in Q·Kᵀ). Callers that need
/// original-domain values should use [`decompress_tensor_unrotated`].
///
/// # Errors
///
/// Returns an error if the compressed data is invalid.
pub fn decompress_tensor(
    quantizer: &QjlQuantizer,
    kv_block: &KvBlock,
    num_rows: usize,
    head_dim: usize,
) -> crate::Result<Array3<f32>> {
    let block_size = quantizer.config().block_size;
    let num_blocks_per_row = head_dim.div_ceil(block_size);

    let mut result = Array3::<f32>::zeros((num_rows, 1, head_dim));

    for r in 0..num_rows {
        let packed = kv_block.retrieve_data(&[r]);
        let scales = kv_block.retrieve_scales(&[r]);
        let correction = kv_block.retrieve_correction(&[r]);

        if packed.is_empty() || scales.is_empty() {
            continue;
        }

        for b in 0..num_blocks_per_row {
            let start = b * block_size;
            let end = (start + block_size).min(head_dim);
            let byte_start = start * 3 / 8;
            let byte_end = end * 3 / 8;

            if byte_end > packed.len() || b >= scales.len() {
                return Err(crate::error::TurboQuantError::DecompressionError(format!(
                    "KvBlock too small for row {r}, block {b}"
                )));
            }

            // Correction bits for this block: 1 bit per value, and block
            // boundaries are 8-value aligned.
            let correction_bits = correction
                .as_ref()
                .map(|bits| bits[start / 8..end / 8].to_vec());

            let compressed = CompressedBlock {
                packed: packed[byte_start..byte_end].to_vec(),
                scale: scales[b],
                correction_bits,
            };

            let decompressed = quantizer.dequantize_block(&compressed, end - start);
            for (j, val) in decompressed.iter().enumerate() {
                result[[r, 0, start + j]] = *val;
            }
        }
    }

    Ok(result)
}

/// Decompress a `KvBlock` back to the **original** (un-rotated) domain.
///
/// Dequantizes exactly like [`decompress_tensor`], then applies `polar`'s
/// inverse rotation to each full `head_dim`-sized row — mirroring
/// [`compress_tensor`], which applies the forward rotation to each row as
/// a whole before block-wise quantization. Output shape is
/// (`num_heads`, 1, `head_dim`), with row `r` of the compressed input at
/// position `r`, exactly like [`decompress_tensor`].
///
/// Use [`decompress_tensor`] when the consumer stays in the rotated
/// domain; use this function when the caller needs values comparable to
/// the tensor originally passed to [`compress_tensor`].
///
/// # Errors
///
/// Returns [`TurboQuantError::InvalidDimension`] if `head_dim` does not
/// match the rotation dimension `polar.dim()`, plus any error
/// [`decompress_tensor`] reports for invalid compressed data.
///
/// [`TurboQuantError::InvalidDimension`]: crate::error::TurboQuantError::InvalidDimension
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use turboquant_core::polar::PolarQuant;
/// use turboquant_core::qjl::{QjlConfig, QjlQuantizer};
/// use turboquant_core::quantize::{compress_tensor, decompress_tensor_unrotated};
/// use turboquant_core::rotation::QrRotation;
///
/// let polar = PolarQuant::new(QrRotation::new(64, Some(42)));
/// let quantizer = QjlQuantizer::new(QjlConfig::default());
/// let tensor = Array2::from_shape_fn((4, 64), |(i, j)| ((i * 64 + j) as f32).sin());
///
/// let block = compress_tensor(&polar, &quantizer, &tensor.view()).unwrap();
/// let restored = decompress_tensor_unrotated(&polar, &quantizer, &block, 4, 64).unwrap();
/// assert_eq!(restored.shape(), &[4, 1, 64]);
/// ```
pub fn decompress_tensor_unrotated<R: Rotation>(
    polar: &PolarQuant<R>,
    quantizer: &QjlQuantizer,
    kv_block: &KvBlock,
    num_heads: usize,
    head_dim: usize,
) -> crate::Result<Array3<f32>> {
    if polar.dim() != head_dim {
        return Err(crate::error::TurboQuantError::InvalidDimension(format!(
            "head_dim {head_dim} does not match rotation dim {}",
            polar.dim()
        )));
    }

    let mut result = decompress_tensor(quantizer, kv_block, num_heads, head_dim)?;

    // Invert the rotation row-by-row, on the same (1, head_dim) row shape
    // compress_tensor used when applying the forward rotation.
    for r in 0..num_heads {
        let mut row = result.slice_mut(ndarray::s![r, .., ..]);
        polar.inverse(&mut row);
    }

    Ok(result)
}

/// Quantize + dequantize a vector block-by-block (no rotation, no storage).
fn quantize_roundtrip(quantizer: &QjlQuantizer, values: &[f32]) -> Vec<f32> {
    let block_size = quantizer.config().block_size;
    let mut out = Vec::with_capacity(values.len());
    for chunk in values.chunks(block_size) {
        let compressed = quantizer.quantize_block(chunk);
        out.extend(quantizer.dequantize_block(&compressed, chunk.len()));
    }
    out
}

/// Single-query attention over decompressed K/V: softmax(q·Kᵀ/√d)·V.
fn attend_single(
    qi: &[f32],
    k: &Array3<f32>,
    v: &Array3<f32>,
    mask_row: Option<&[f32]>,
    scale: f32,
) -> Vec<f32> {
    let seq_len_kv = k.dim().0;
    let head_dim = qi.len();

    let mut scores = vec![0.0f32; seq_len_kv];
    for (j, score) in scores.iter_mut().enumerate() {
        let mut dot = 0.0f32;
        for (d, q_d) in qi.iter().enumerate() {
            dot += q_d * k[[j, 0, d]];
        }
        *score = dot / scale;
        if let Some(m) = mask_row {
            *score += m[j];
        }
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
    let mut out = vec![0.0f32; head_dim];
    for (j, w) in scores.iter().enumerate() {
        for (d, o) in out.iter_mut().enumerate() {
            *o += w * v[[j, 0, d]];
        }
    }
    out
}

/// Compute measured comparison statistics between a reference signal and
/// an approximation of it.
#[allow(clippy::cast_precision_loss)]
fn compute_stats(reference: &[f32], approx: &[f32]) -> AttentionStats {
    debug_assert_eq!(reference.len(), approx.len());
    if reference.is_empty() {
        return AttentionStats {
            snr_db: f64::INFINITY,
            cosine_similarity: 1.0,
            max_abs_error: 0.0,
            mse: 0.0,
        };
    }

    let n = reference.len() as f64;
    let mut sq_err = 0.0f64;
    let mut sq_ref = 0.0f64;
    let mut sq_app = 0.0f64;
    let mut dot = 0.0f64;
    let mut max_abs_error = 0.0f32;

    for (&r, &a) in reference.iter().zip(approx) {
        let (rf, af) = (f64::from(r), f64::from(a));
        sq_err += (rf - af).powi(2);
        sq_ref += rf * rf;
        sq_app += af * af;
        dot += rf * af;
        max_abs_error = max_abs_error.max((r - a).abs());
    }

    let mse = sq_err / n;
    let signal_power = sq_ref / n;
    let snr_db = if mse > 0.0 {
        10.0 * (signal_power / mse).log10()
    } else {
        f64::INFINITY
    };
    let denom = (sq_ref * sq_app).sqrt();
    let cosine_similarity = if denom > 0.0 { dot / denom } else { 1.0 };

    #[allow(clippy::cast_possible_truncation)]
    AttentionStats {
        snr_db,
        cosine_similarity,
        max_abs_error,
        mse: mse as f32,
    }
}

/// Reference `TurboQuant` attention forward pass.
///
/// Decompresses K/V from the compressed caches and computes attention.
/// The caches are interpreted as `seq_len_kv = k_cache.seq_len` KV
/// positions of dimension `head_dim`, shared by all query heads
/// (multi-query-style reference implementation).
///
/// Q shape: (batch, `seq_len_q`, `num_heads`, `head_dim`)
/// Output shape: (batch, `seq_len_q`, `num_heads`, `head_dim`)
///
/// `mask`, if provided, is an additive (`seq_len_q` × `seq_len_kv`) score
/// mask applied before softmax.
///
/// `out` receives the float-Q attention output (float Q against the
/// dequantized K/V). The returned [`AttentionStats`] are **computed from
/// the actual data**: the same attention is re-run with Q round-tripped
/// through `quantizer` (the fully-quantized path) and compared against
/// `out`. Since the original uncompressed K/V are not available inside
/// this function, end-to-end parity against FP32 attention is asserted
/// separately in `tests/attention_parity.rs`.
///
/// # Errors
///
/// Returns an error if the cache dimensions do not match `q`.
#[allow(clippy::cast_precision_loss)]
pub fn turbo_attention_forward(
    q: &ArrayView4<f32>,
    k_cache: &KvBlock,
    v_cache: &KvBlock,
    mask: Option<&ArrayView2<f32>>,
    out: &mut ArrayViewMut4<f32>,
    quantizer: &QjlQuantizer,
) -> crate::Result<AttentionStats> {
    let shape = q.shape();
    let batch = shape[0];
    let seq_len_q = shape[1];
    let num_heads = shape[2];
    let head_dim = shape[3];
    let seq_len_kv = k_cache.seq_len;

    if k_cache.head_dim != head_dim || v_cache.head_dim != head_dim {
        return Err(crate::error::TurboQuantError::InvalidDimension(format!(
            "q head_dim {head_dim} does not match caches (k: {}, v: {})",
            k_cache.head_dim, v_cache.head_dim
        )));
    }
    if v_cache.seq_len != seq_len_kv {
        return Err(crate::error::TurboQuantError::InvalidDimension(format!(
            "K cache seq_len {seq_len_kv} does not match V cache seq_len {}",
            v_cache.seq_len
        )));
    }

    let k_decomp = decompress_tensor(quantizer, k_cache, seq_len_kv, head_dim)?;
    let v_decomp = decompress_tensor(quantizer, v_cache, seq_len_kv, head_dim)?;

    let scale = (head_dim as f32).sqrt();

    let total = batch * seq_len_q * num_heads * head_dim;
    let mut float_q_out: Vec<f32> = Vec::with_capacity(total);
    let mut quant_q_out: Vec<f32> = Vec::with_capacity(total);

    for b in 0..batch {
        for i in 0..seq_len_q {
            let mask_row: Option<Vec<f32>> =
                mask.map(|m| (0..seq_len_kv).map(|j| m[[i, j]]).collect());
            for h in 0..num_heads {
                let qi: Vec<f32> = (0..head_dim).map(|d| q[[b, i, h, d]]).collect();

                // Float-Q path (written to `out`).
                let o_ref = attend_single(&qi, &k_decomp, &v_decomp, mask_row.as_deref(), scale);
                // Fully-quantized path (Q quantized like K/V).
                let qi_quant = quantize_roundtrip(quantizer, &qi);
                let o_turbo =
                    attend_single(&qi_quant, &k_decomp, &v_decomp, mask_row.as_deref(), scale);

                for (d, val) in o_ref.iter().enumerate() {
                    out[[b, i, h, d]] = *val;
                }
                float_q_out.extend_from_slice(&o_ref);
                quant_q_out.extend_from_slice(&o_turbo);
            }
        }
    }

    Ok(compute_stats(&float_q_out, &quant_q_out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qjl::{CorrectionMode, QjlConfig, ScaleMode};
    use crate::rotation::{FastHadamardRotation, HouseholderRotation, QrRotation};
    use ndarray::Array2;
    use rand::Rng;
    use rand_distr::StandardNormal;

    fn gaussian_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
        let mut rng: rand::rngs::StdRng = rand::SeedableRng::seed_from_u64(seed);
        Array2::from_shape_fn((rows, cols), |_| rng.sample(StandardNormal))
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
    fn test_compress_decompress_roundtrip() {
        let rot = QrRotation::new(64, Some(42));
        let polar = PolarQuant::new(rot);

        let config = QjlConfig {
            block_size: 64,
            scale_mode: ScaleMode::PerBlockAbsMax,
            correction: CorrectionMode::None,
            ..Default::default()
        };
        let quantizer = QjlQuantizer::new(config);

        let tensor = gaussian_matrix(4, 64, 42);

        let compressed = compress_tensor(&polar, &quantizer, &tensor.view()).unwrap();
        let decompressed = decompress_tensor(&quantizer, &compressed, 4, 64).unwrap();

        assert_eq!(decompressed.shape(), &[4, 1, 64]);
    }

    #[test]
    fn test_multi_block_positions_do_not_collide() {
        // Regression for the position-collision bug: with head_dim = 2 ×
        // block_size, block b of head h was stored at position h + b, so
        // head 1's first block overwrote head 0's second block (and the
        // per-position packed length/scale count never matched).
        let head_dim = 128;
        let block_size = 64;
        let num_heads = 4;

        let rot = QrRotation::new(head_dim, Some(7));
        let polar = PolarQuant::new(rot);
        let quantizer = QjlQuantizer::new(QjlConfig {
            block_size,
            ..Default::default()
        });

        let tensor = gaussian_matrix(num_heads, head_dim, 7);

        let compressed = compress_tensor(&polar, &quantizer, &tensor.view()).unwrap();
        let decompressed = decompress_tensor(&quantizer, &compressed, num_heads, head_dim).unwrap();
        assert_eq!(decompressed.shape(), &[num_heads, 1, head_dim]);

        // The decompressed values live in the rotated domain; compare
        // against the rotated original, head by head.
        let mut rotated = tensor.clone();
        polar.forward(&mut rotated.view_mut());

        for h in 0..num_heads {
            let expected: Vec<f32> = (0..head_dim).map(|d| rotated[[h, d]]).collect();
            let actual: Vec<f32> = (0..head_dim).map(|d| decompressed[[h, 0, d]]).collect();
            let snr = snr_db(&expected, &actual);
            assert!(
                snr > 10.0,
                "head {h} round-trip SNR too low ({snr:.2} dB) — position collision?"
            );
        }
    }

    /// Compress Gaussian data with the given rotation, decompress via
    /// `decompress_tensor_unrotated`, and assert the round-trip SNR vs the
    /// ORIGINAL (un-rotated) input clears 12 dB.
    fn assert_unrotated_roundtrip<R: Rotation>(rot: R, label: &str) {
        let num_rows = 8;
        let head_dim = rot.dim();
        let polar = PolarQuant::new(rot);
        // Default config: block_size 64, 1-bit residual correction.
        let quantizer = QjlQuantizer::new(QjlConfig::default());
        let tensor = gaussian_matrix(num_rows, head_dim, 42);

        let compressed = compress_tensor(&polar, &quantizer, &tensor.view()).unwrap();
        let restored =
            decompress_tensor_unrotated(&polar, &quantizer, &compressed, num_rows, head_dim)
                .unwrap();
        assert_eq!(restored.shape(), &[num_rows, 1, head_dim]);

        let original: Vec<f32> = tensor.iter().copied().collect();
        let restored_flat: Vec<f32> = restored.iter().copied().collect();
        let snr = snr_db(&original, &restored_flat);
        println!("{label}: unrotated round-trip SNR vs original = {snr:.2} dB");
        assert!(
            snr > 12.0,
            "{label}: SNR vs original too low ({snr:.2} dB), inverse rotation missing or wrong?"
        );
    }

    #[test]
    fn test_unrotated_roundtrip_qr() {
        assert_unrotated_roundtrip(QrRotation::new(64, Some(42)), "QrRotation");
    }

    #[test]
    fn test_unrotated_roundtrip_householder() {
        assert_unrotated_roundtrip(
            HouseholderRotation::new(64, 16, Some(42)),
            "HouseholderRotation",
        );
    }

    #[test]
    fn test_unrotated_roundtrip_hadamard() {
        assert_unrotated_roundtrip(
            FastHadamardRotation::new(64, Some(42)),
            "FastHadamardRotation",
        );
    }

    #[test]
    fn test_decompress_tensor_stays_rotated() {
        // Pins the semantic difference between the two decompression APIs:
        // decompress_tensor returns ROTATED-domain values (far from the
        // original), and only after the inverse rotation do they match the
        // original — bit-for-bit the same as decompress_tensor_unrotated.
        let num_rows = 8;
        let head_dim = 64;
        let polar = PolarQuant::new(QrRotation::new(head_dim, Some(21)));
        let quantizer = QjlQuantizer::new(QjlConfig::default());
        let tensor = gaussian_matrix(num_rows, head_dim, 21);

        let compressed = compress_tensor(&polar, &quantizer, &tensor.view()).unwrap();
        let rotated = decompress_tensor(&quantizer, &compressed, num_rows, head_dim).unwrap();

        let original: Vec<f32> = tensor.iter().copied().collect();
        let rotated_flat: Vec<f32> = rotated.iter().copied().collect();
        let snr_rotated = snr_db(&original, &rotated_flat);
        println!("rotated-domain output vs original: {snr_rotated:.2} dB");
        assert!(
            snr_rotated < 6.0,
            "decompress_tensor output should differ from the original \
             (rotated domain), but SNR is {snr_rotated:.2} dB"
        );

        // Manual inverse rotation, row by row, recovers the original...
        let mut manual = rotated.clone();
        for r in 0..num_rows {
            let mut row = manual.slice_mut(ndarray::s![r, .., ..]);
            polar.inverse(&mut row);
        }
        let manual_flat: Vec<f32> = manual.iter().copied().collect();
        let snr_manual = snr_db(&original, &manual_flat);
        println!("manually un-rotated output vs original: {snr_manual:.2} dB");
        assert!(
            snr_manual > 12.0,
            "manual inverse rotation SNR too low: {snr_manual:.2} dB"
        );

        // ...and agrees with decompress_tensor_unrotated.
        let unrotated =
            decompress_tensor_unrotated(&polar, &quantizer, &compressed, num_rows, head_dim)
                .unwrap();
        for (m, u) in manual.iter().zip(unrotated.iter()) {
            assert!(
                (m - u).abs() < 1e-5,
                "manual inverse ({m}) disagrees with decompress_tensor_unrotated ({u})"
            );
        }
    }

    #[test]
    fn test_unrotated_rejects_rotation_dim_mismatch() {
        let head_dim = 64;
        let polar = PolarQuant::new(QrRotation::new(head_dim, Some(5)));
        let quantizer = QjlQuantizer::new(QjlConfig::default());
        let tensor = gaussian_matrix(2, head_dim, 5);
        let compressed = compress_tensor(&polar, &quantizer, &tensor.view()).unwrap();

        let wrong_polar = PolarQuant::new(QrRotation::new(32, Some(5)));
        let err = decompress_tensor_unrotated(&wrong_polar, &quantizer, &compressed, 2, head_dim)
            .unwrap_err();
        assert!(
            matches!(err, crate::error::TurboQuantError::InvalidDimension(_)),
            "expected InvalidDimension, got {err:?}"
        );
    }

    #[test]
    fn test_various_shapes_roundtrip() {
        // Any (num_heads, head_dim, block_size) combination with 8-aligned
        // dimensions must round-trip.
        for &(num_heads, head_dim, block_size) in &[
            (1usize, 64usize, 64usize),
            (2, 128, 64),
            (3, 64, 32),
            (2, 96, 64), // ragged tail block
            (5, 256, 64),
        ] {
            let rot = QrRotation::new(head_dim, Some(3));
            let polar = PolarQuant::new(rot);
            let quantizer = QjlQuantizer::new(QjlConfig {
                block_size,
                ..Default::default()
            });
            let tensor = gaussian_matrix(num_heads, head_dim, 11);

            let compressed = compress_tensor(&polar, &quantizer, &tensor.view()).unwrap();
            let decompressed =
                decompress_tensor(&quantizer, &compressed, num_heads, head_dim).unwrap();

            let mut rotated = tensor.clone();
            polar.forward(&mut rotated.view_mut());

            let expected: Vec<f32> = rotated.iter().copied().collect();
            let actual: Vec<f32> = (0..num_heads)
                .flat_map(|h| (0..head_dim).map(move |d| (h, d)))
                .map(|(h, d)| decompressed[[h, 0, d]])
                .collect();
            let snr = snr_db(&expected, &actual);
            assert!(
                snr > 10.0,
                "shape ({num_heads}, {head_dim}, block {block_size}) SNR too low: {snr:.2} dB"
            );
        }
    }
}
