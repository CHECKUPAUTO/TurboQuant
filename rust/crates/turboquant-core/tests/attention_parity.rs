//! End-to-end parity tests for `turbo_attention_forward`.
//!
//! Verifies (a) that attention computed from the compressed KV cache
//! matches a float reference computed from the same (rotated) K/V, and
//! (b) that the returned `AttentionStats` are actually measured from the
//! data — not hardcoded.
//!
//! Referenced from `BUGS_FIXED.md` (Bug 7 of the Python port and the
//! Rust-port fake-stats bug).

// Test-only: small-size casts are exact; exact float comparisons against
// the old hardcoded constants are the whole point of the anti-hardcoding
// test.
#![allow(clippy::cast_precision_loss, clippy::float_cmp)]

use ndarray::{Array2, Array4};
use rand::Rng;
use rand_distr::StandardNormal;
use turboquant_core::polar::PolarQuant;
use turboquant_core::qjl::{QjlConfig, QjlQuantizer};
use turboquant_core::quantize::{compress_tensor, turbo_attention_forward, AttentionStats};
use turboquant_core::rotation::QrRotation;

const HEAD_DIM: usize = 64;
const SEQ_KV: usize = 32;
const SEQ_Q: usize = 2;
const HEADS: usize = 2;

fn gaussian2(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut rng: rand::rngs::StdRng = rand::SeedableRng::seed_from_u64(seed);
    Array2::from_shape_fn((rows, cols), |_| rng.sample(StandardNormal))
}

fn gaussian4(shape: (usize, usize, usize, usize), seed: u64) -> Array4<f32> {
    let mut rng: rand::rngs::StdRng = rand::SeedableRng::seed_from_u64(seed);
    Array4::from_shape_fn(shape, |_| rng.sample(StandardNormal))
}

fn snr_db(reference: &[f32], approx: &[f32]) -> f64 {
    let mse: f64 = reference
        .iter()
        .zip(approx)
        .map(|(a, b)| f64::from(a - b).powi(2))
        .sum::<f64>()
        / reference.len() as f64;
    let signal: f64 =
        reference.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / reference.len() as f64;
    10.0 * (signal / mse.max(1e-15)).log10()
}

/// Float reference attention: softmax(q·Kᵀ/√d)·V over (`seq_kv`, `head_dim`)
/// K/V shared by all query heads (same convention as the turbo path).
fn reference_attention(q: &Array4<f32>, k: &Array2<f32>, v: &Array2<f32>) -> Array4<f32> {
    let (batch, seq_q, heads, head_dim) = q.dim();
    let seq_kv = k.dim().0;
    let scale = (head_dim as f32).sqrt();
    let mut out = Array4::<f32>::zeros((batch, seq_q, heads, head_dim));

    for b in 0..batch {
        for i in 0..seq_q {
            for h in 0..heads {
                let mut scores = vec![0.0f32; seq_kv];
                for (j, s) in scores.iter_mut().enumerate() {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[[b, i, h, d]] * k[[j, d]];
                    }
                    *s = dot / scale;
                }
                let max = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                for s in &mut scores {
                    *s /= sum;
                }
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for (j, w) in scores.iter().enumerate() {
                        acc += w * v[[j, d]];
                    }
                    out[[b, i, h, d]] = acc;
                }
            }
        }
    }
    out
}

/// Run the full pipeline for one seed and return (output SNR vs float
/// reference, stats reported by `turbo_attention_forward`).
fn run_pipeline(seed: u64) -> (f64, AttentionStats) {
    let rot = QrRotation::new(HEAD_DIM, Some(42));
    let polar = PolarQuant::new(rot);
    let quantizer = QjlQuantizer::new(QjlConfig::default());

    let k = gaussian2(SEQ_KV, HEAD_DIM, seed);
    let v = gaussian2(SEQ_KV, HEAD_DIM, seed + 1);
    let q = gaussian4((1, SEQ_Q, HEADS, HEAD_DIM), seed + 2);

    let k_cache = compress_tensor(&polar, &quantizer, &k.view()).unwrap();
    let v_cache = compress_tensor(&polar, &quantizer, &v.view()).unwrap();

    let mut out = Array4::<f32>::zeros((1, SEQ_Q, HEADS, HEAD_DIM));
    let stats = turbo_attention_forward(
        &q.view(),
        &k_cache,
        &v_cache,
        None,
        &mut out.view_mut(),
        &quantizer,
    )
    .unwrap();

    // The cache holds K/V in the rotated domain, so the float reference
    // uses the same rotation applied in f32.
    let mut k_rot = k.clone();
    let mut v_rot = v.clone();
    polar.forward(&mut k_rot.view_mut());
    polar.forward(&mut v_rot.view_mut());
    let reference = reference_attention(&q, &k_rot, &v_rot);

    let ref_flat: Vec<f32> = reference.iter().copied().collect();
    let out_flat: Vec<f32> = out.iter().copied().collect();
    (snr_db(&ref_flat, &out_flat), stats)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

#[test]
fn attention_output_matches_float_reference() {
    let rot = QrRotation::new(HEAD_DIM, Some(42));
    let polar = PolarQuant::new(rot);
    let quantizer = QjlQuantizer::new(QjlConfig::default());

    let k = gaussian2(SEQ_KV, HEAD_DIM, 1000);
    let v = gaussian2(SEQ_KV, HEAD_DIM, 1001);
    let q = gaussian4((1, SEQ_Q, HEADS, HEAD_DIM), 1002);

    let k_cache = compress_tensor(&polar, &quantizer, &k.view()).unwrap();
    let v_cache = compress_tensor(&polar, &quantizer, &v.view()).unwrap();

    let mut out = Array4::<f32>::zeros((1, SEQ_Q, HEADS, HEAD_DIM));
    turbo_attention_forward(
        &q.view(),
        &k_cache,
        &v_cache,
        None,
        &mut out.view_mut(),
        &quantizer,
    )
    .unwrap();

    let mut k_rot = k.clone();
    let mut v_rot = v.clone();
    polar.forward(&mut k_rot.view_mut());
    polar.forward(&mut v_rot.view_mut());
    let reference = reference_attention(&q, &k_rot, &v_rot);

    let ref_flat: Vec<f32> = reference.iter().copied().collect();
    let out_flat: Vec<f32> = out.iter().copied().collect();

    let out_snr = snr_db(&ref_flat, &out_flat);
    assert!(
        out_snr > 12.0,
        "attention output vs float reference SNR too low: {out_snr:.2} dB"
    );
    let cos = cosine(&ref_flat, &out_flat);
    assert!(
        cos > 0.96,
        "attention output vs float reference cosine too low: {cos:.4}"
    );
}

#[test]
fn stats_are_real_and_well_conditioned() {
    let (_, stats) = run_pipeline(2000);
    assert!(
        stats.snr_db.is_finite() && stats.snr_db > 12.0,
        "measured stats SNR too low: {} dB",
        stats.snr_db
    );
    assert!(
        stats.cosine_similarity > 0.96 && stats.cosine_similarity <= 1.0,
        "cosine similarity out of range: {}",
        stats.cosine_similarity
    );
    assert!(stats.mse > 0.0, "MSE must be measured, not zero");
    assert!(stats.max_abs_error > 0.0);
}

#[test]
fn stats_change_when_input_changes() {
    // Anti-hardcoding regression: the old implementation returned
    // { snr_db: 20.0, cosine_similarity: 0.98, max_abs_error: 0.01,
    //   mse: 0.0001 } regardless of input.
    let (_, s1) = run_pipeline(3000);
    let (_, s2) = run_pipeline(4000);

    assert!(
        (s1.snr_db - s2.snr_db).abs() > f64::EPSILON,
        "snr_db identical across different inputs: {} — hardcoded?",
        s1.snr_db
    );
    assert!(
        (s1.mse - s2.mse).abs() > 0.0,
        "mse identical across different inputs: {} — hardcoded?",
        s1.mse
    );

    for s in [&s1, &s2] {
        assert!(
            !(s.snr_db == 20.0
                && s.cosine_similarity == 0.98
                && s.max_abs_error == 0.01
                && s.mse == 0.0001),
            "stats exactly match the old hardcoded constants"
        );
    }
}
