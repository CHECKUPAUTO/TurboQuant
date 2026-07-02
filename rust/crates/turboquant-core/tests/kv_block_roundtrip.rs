//! Store/retrieve round-trip tests for `KvBlock`, including partial-range
//! writes and multi-block-per-position layouts.
//!
//! Referenced from `BUGS_FIXED.md` (Bugs 3-5 of the Python port and the
//! Rust-port position-collision bug).

// Test-only: small-size casts are exact and `vec!` literals keep the
// assertions readable.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::useless_vec
)]

use half::f16;
use turboquant_core::kv_block::KvBlock;

/// Distinct byte pattern for a given position, `len` bytes long.
fn pattern(pos: usize, len: usize) -> Vec<u8> {
    (0..len).map(|i| ((pos * 31 + i) % 251) as u8).collect()
}

#[test]
fn full_store_retrieve_roundtrip() {
    let head_dim = 64;
    let seq_len = 256;
    let mut block = KvBlock::new(head_dim, seq_len, 64);
    let bytes_per_pos = head_dim * 3 / 8;

    let mut all_data = Vec::new();
    let mut all_scales = Vec::new();
    for pos in 0..seq_len {
        all_data.extend(pattern(pos, bytes_per_pos));
        all_scales.push(f16::from_f32(pos as f32 + 1.0));
    }
    block.store(0..seq_len, &all_data, &all_scales).unwrap();

    for pos in [0usize, 1, 10, 100, 255] {
        assert_eq!(block.retrieve_data(&[pos]), pattern(pos, bytes_per_pos));
        assert_eq!(
            block.retrieve_scales(&[pos]),
            vec![f16::from_f32(pos as f32 + 1.0)]
        );
    }
}

#[test]
fn partial_range_stores_do_not_overlap() {
    let head_dim = 64;
    let mut block = KvBlock::new(head_dim, 256, 64);
    let bytes_per_pos = head_dim * 3 / 8;

    // Store three disjoint ranges, each with its own pattern and scales.
    for (start, end) in [(0usize, 64usize), (64, 128), (128, 192)] {
        let mut data = Vec::new();
        let mut scales = Vec::new();
        for pos in start..end {
            data.extend(pattern(pos, bytes_per_pos));
            scales.push(f16::from_f32(pos as f32));
        }
        block.store(start..end, &data, &scales).unwrap();
    }

    // Every written position must read back its own bytes and scale.
    for pos in [0usize, 63, 64, 100, 127, 128, 191] {
        assert_eq!(
            block.retrieve_data(&[pos]),
            pattern(pos, bytes_per_pos),
            "position {pos} corrupted by partial store"
        );
        assert_eq!(
            block.retrieve_scales(&[pos]),
            vec![f16::from_f32(pos as f32)]
        );
    }

    // Unwritten tail is still zeroed.
    assert!(block.retrieve_data(&[200]).iter().all(|&b| b == 0));
}

#[test]
fn multi_scale_per_position_roundtrip() {
    // head_dim = 2 × block_size → 2 scales per position. Regression for
    // the scale-indexing bug where position p's scales started at
    // p * head_dim / block_size instead of p * scales_per_position.
    let head_dim = 128;
    let block_size = 64;
    let seq_len = 8;
    let mut block = KvBlock::new(head_dim, seq_len, block_size);
    assert_eq!(block.scales_per_position(), 2);
    let bytes_per_pos = head_dim * 3 / 8;

    for pos in 0..seq_len {
        let scales = vec![
            f16::from_f32(pos as f32 * 2.0),
            f16::from_f32(pos as f32 * 2.0 + 1.0),
        ];
        block
            .store(pos..pos + 1, &pattern(pos, bytes_per_pos), &scales)
            .unwrap();
    }

    for pos in 0..seq_len {
        assert_eq!(block.retrieve_data(&[pos]), pattern(pos, bytes_per_pos));
        assert_eq!(
            block.retrieve_scales(&[pos]),
            vec![
                f16::from_f32(pos as f32 * 2.0),
                f16::from_f32(pos as f32 * 2.0 + 1.0)
            ],
            "scales for position {pos} collided with a neighbour"
        );
    }
}

#[test]
fn out_of_range_store_errors() {
    let mut block = KvBlock::new(64, 256, 64);
    let data = vec![0u8; 64 * 3 / 8 * 10];
    let scales = vec![f16::ZERO; 10];
    assert!(block.store(250..260, &data, &scales).is_err());
}

#[test]
fn short_data_or_wrong_scale_count_errors() {
    let mut block = KvBlock::new(64, 16, 64);
    // Too few packed bytes for the range.
    assert!(block
        .store(0..2, &vec![0u8; 24], &vec![f16::ZERO; 2])
        .is_err());
    // Wrong number of scales for the range.
    assert!(block
        .store(0..1, &vec![0u8; 24], &vec![f16::ZERO; 2])
        .is_err());
}

#[test]
fn out_of_range_retrieve_is_empty() {
    let block = KvBlock::new(64, 16, 64);
    assert!(block.retrieve_data(&[16]).is_empty());
    assert!(block.retrieve_scales(&[99]).is_empty());
}
