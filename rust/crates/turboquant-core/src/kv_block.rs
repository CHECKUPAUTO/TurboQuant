//! Compressed KV block storage.
//!
//! Stores packed 3-bit K/V data with per-block scales.

use half::f16;
use serde::{Deserialize, Serialize};
use std::ops::Range;

/// A compressed block of KV cache data.
///
/// Stores packed 3-bit values with one `f16` scale per block.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KvBlock {
    /// Packed 3-bit data.
    pub data: Vec<u8>,
    /// Per-block scales (f16).
    pub scales: Vec<f16>,
    /// Block size (number of values per scale).
    pub block_size: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Sequence length stored.
    pub seq_len: usize,
}

impl KvBlock {
    /// Create a new empty KV block.
    ///
    /// # Examples
    ///
    /// ```
    /// use turboquant_core::kv_block::KvBlock;
    ///
    /// let block = KvBlock::new(128, 4096, 64);
    /// assert_eq!(block.head_dim, 128);
    /// assert_eq!(block.seq_len, 4096);
    /// ```
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn new(head_dim: usize, seq_len: usize, block_size: usize) -> Self {
        let num_values = head_dim * seq_len;
        let num_blocks = num_values.div_ceil(block_size);
        let packed_bytes = num_values * 3 / 8;

        Self {
            data: vec![0u8; packed_bytes],
            scales: vec![f16::ZERO; num_blocks],
            block_size,
            head_dim,
            seq_len,
        }
    }

    /// Store packed data at a specific positional range.
    ///
    /// # Errors
    ///
    /// Returns an error if the range is invalid or unaligned.
    pub fn store(
        &mut self,
        range: Range<usize>,
        packed_data: &[u8],
        scales: &[f16],
    ) -> crate::Result<()> {
        if range.end > self.seq_len {
            return Err(crate::error::TurboQuantError::CompressionError(format!(
                "Range {:?} exceeds seq_len {}",
                range, self.seq_len
            )));
        }

        let values_per_position = self.head_dim;
        let byte_offset = range.start * values_per_position * 3 / 8;
        let byte_len = (range.end - range.start) * values_per_position * 3 / 8;

        if byte_offset + byte_len > self.data.len() {
            return Err(crate::error::TurboQuantError::CompressionError(
                "Packed data overflow".into(),
            ));
        }

        self.data[byte_offset..byte_offset + byte_len].copy_from_slice(&packed_data[..byte_len]);

        let scale_start = range.start * self.head_dim / self.block_size;
        let scale_end = range.end * self.head_dim / self.block_size;
        self.scales[scale_start..scale_end].copy_from_slice(scales);

        Ok(())
    }

    /// Retrieve packed data for specific positions.
    #[must_use]
    pub fn retrieve_data(&self, positions: &[usize]) -> Vec<u8> {
        let values_per_position = self.head_dim;
        let mut result = Vec::with_capacity(positions.len() * values_per_position * 3 / 8);

        for &pos in positions {
            if pos >= self.seq_len {
                continue;
            }
            let byte_offset = pos * values_per_position * 3 / 8;
            let byte_len = values_per_position * 3 / 8;
            result.extend_from_slice(&self.data[byte_offset..byte_offset + byte_len]);
        }

        result
    }

    /// Retrieve scales for specific positions.
    #[must_use]
    pub fn retrieve_scales(&self, positions: &[usize]) -> Vec<f16> {
        let mut result = Vec::new();

        for &pos in positions {
            if pos >= self.seq_len {
                continue;
            }
            let scale_start = pos * self.head_dim / self.block_size;
            let num_scales = self.head_dim.div_ceil(self.block_size);
            result.extend_from_slice(&self.scales[scale_start..scale_start + num_scales]);
        }

        result
    }

    /// Compute the memory usage in bytes.
    #[must_use]
    pub fn memory_usage_bytes(&self) -> usize {
        self.data.len() + self.scales.len() * 2
    }

    /// Compute compression ratio vs FP16.
    ///
    /// Includes overhead from per-block scales.
    #[must_use]
    pub fn compression_ratio_vs_fp16(&self) -> f64 {
        let fp16_bits = (self.head_dim * self.seq_len * 16) as f64;
        let payload_bits = (self.data.len() * 8) as f64;
        let scale_bits = (self.scales.len() * 16) as f64;
        fp16_bits / (payload_bits + scale_bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_block_sizes() {
        let block = KvBlock::new(128, 4096, 64);
        // 128 * 4096 = 524288 values, 3 bits each → 196608 bytes
        assert_eq!(block.data.len(), 196_608);
        // 524288 / 64 = 8192 blocks, each with 1 f16 scale
        assert_eq!(block.scales.len(), 8192);
    }

    #[test]
    fn test_store_and_retrieve() {
        let mut block = KvBlock::new(64, 256, 64);
        let packed_data = vec![0u8; 64 * 256 * 3 / 8]; // full block
        let scales = vec![f16::from_f32(1.0); 256];

        block.store(0..256, &packed_data, &scales).unwrap();

        let retrieved_data = block.retrieve_data(&[0, 1, 10, 255]);
        let retrieved_scales = block.retrieve_scales(&[0, 1, 10, 255]);

        assert_eq!(retrieved_data.len(), 4 * 64 * 3 / 8);
        assert_eq!(retrieved_scales.len(), 4);
    }

    #[test]
    fn test_store_partial_range() {
        let mut block = KvBlock::new(64, 256, 64);
        let packed_chunk = vec![1u8; 64 * 64 * 3 / 8]; // 64 positions worth
        let chunk_scales = vec![f16::from_f32(2.0); 64];

        block.store(0..64, &packed_chunk, &chunk_scales).unwrap();
        block.store(64..128, &packed_chunk, &chunk_scales).unwrap();
        block.store(128..192, &packed_chunk, &chunk_scales).unwrap();

        // Check we can read back
        let data = block.retrieve_data(&[0, 64, 128]);
        assert_eq!(data.len(), 3 * 64 * 3 / 8);
    }

    #[test]
    fn test_store_out_of_range() {
        let mut block = KvBlock::new(64, 256, 64);
        let data = vec![0u8; 100];
        let scales = vec![f16::ZERO];

        assert!(block.store(250..260, &data, &scales).is_err());
    }

    #[test]
    fn test_compression_ratio_includes_overhead() {
        let block = KvBlock::new(128, 4096, 64);
        let ratio = block.compression_ratio_vs_fp16();
        // Should be less than naive 16/3 ≈ 5.33 due to scale overhead
        assert!(ratio < 5.33);
        // But still significant
        assert!(ratio > 4.5);
    }
}
