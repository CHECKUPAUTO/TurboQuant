"""
Tests unitaires pour TurboQuant.

Usage:
    python -m pytest tests/test_core.py -v
    # ou
    python tests/test_core.py
"""

import math
import unittest

import torch

from turboquant import (
    PolarQuant,
    QJLQuantizer,
    TurboQuantKVCache,
    TurboQuantAttention,
    benchmark_turboquant,
)


class TestPolarQuant(unittest.TestCase):
    """Tests for PolarQuant (Phase 1: Geometric Rotation)."""

    def setUp(self):
        torch.manual_seed(0)
        self.dim = 128
        self.pq = PolarQuant(dim=self.dim)

    def test_init_validates_dim(self):
        """dim <= 0 should raise ValueError."""
        with self.assertRaises(ValueError):
            PolarQuant(dim=0)
        with self.assertRaises(ValueError):
            PolarQuant(dim=-1)

    def test_matrix_is_orthogonal(self):
        """R^T @ R should be identity (orthogonal property)."""
        I = torch.matmul(self.pq.R_T, self.pq.R)
        expected = torch.eye(self.dim)
        self.assertTrue(torch.allclose(I, expected, atol=1e-5))

    def test_rotation_preserves_norm(self):
        """||R @ x|| should equal ||x||."""
        x = torch.randn(32, 128)
        rotated = self.pq.rotate(x)
        self.assertTrue(torch.allclose(
            x.norm(dim=-1), rotated.norm(dim=-1), atol=1e-4
        ))

    def test_roundtrip(self):
        """rotate then inverse_rotate should recover original."""
        x = torch.randn(16, 64, self.dim)
        rotated = self.pq.rotate(x)
        recovered = self.pq.inverse_rotate(rotated)
        self.assertTrue(torch.allclose(x, recovered, atol=1e-5))

    def test_batch_rotation(self):
        """Should handle (batch, seq, dim) tensors."""
        x = torch.randn(2, 16, self.dim)
        result = self.pq.rotate(x)
        self.assertEqual(result.shape, x.shape)


class TestQJLQuantizer(unittest.TestCase):
    """Tests for QJLQuantizer (Phase 2: Johnson-Lindenstrauss Correction)."""

    def setUp(self):
        torch.manual_seed(0)
        self.quant = QJLQuantizer(bits=3)

    def test_output_shape_preserved(self):
        """Quantize should not change tensor shape."""
        x = torch.randn(4, 16, 128)
        q = self.quant.quantize(x)
        self.assertEqual(q.shape, x.shape)

    def test_quantize_values_in_range(self):
        """Quantized values should be in expected range."""
        x = torch.randn(1000) * 10
        q = self.quant.quantize(x)
        half_range = (self.quant.levels - 1) / 2
        # Including correction, values should be roughly in [-half_range, half_range]
        self.assertTrue(q.abs().max() <= half_range + 0.1)

    def test_dequantize_approximates_original(self):
        """Round-trip error should be reasonable."""
        x = torch.randn(128)
        original_scale = x.abs().max().item() + 1e-8
        q = self.quant.quantize(x)
        d = self.quant.dequantize(q, original_scale)
        error = (x - d).norm().item() / x.norm().item()
        self.assertLess(error, 0.2)  # 3-bit should be within ~20% L2

    def test_correction_scale_exists(self):
        """QJL correction scale should be a valid parameter."""
        self.assertIsNotNone(self.quant.scale)

    def test_learnable_vs_fixed_scale(self):
        """Test both learnable and fixed scale modes."""
        ql = QJLQuantizer(learn_scale=True)
        qf = QJLQuantizer(learn_scale=False)
        x = torch.randn(64)
        rl = ql.quantize(x)
        rf = qf.quantize(x)
        self.assertEqual(rl.shape, rf.shape)


class TestTurboQuantKVCache(unittest.TestCase):
    """Integration tests for TurboQuantKVCache."""

    def setUp(self):
        torch.manual_seed(0)
        self.cache = TurboQuantKVCache(
            num_layers=4,
            max_seq_len=512,
            head_dim=64,
            num_heads=8,
            bits=3,
        )

    def test_memory_usage_is_positive(self):
        """Memory usage should be positive."""
        mem = self.cache.memory_usage_mb()
        self.assertGreater(mem, 0)

    def test_compression_ratio(self):
        """Compression ratio should be ~5.3x for 3-bit vs FP16."""
        ratio = self.cache.compression_ratio_vs_fp16()
        self.assertAlmostEqual(ratio, 16 / 3, delta=0.1)

    def test_compress_decompress_roundtrip(self):
        """Compress then decompress should approximate original."""
        k = torch.randn(1, 8, 16, 64)  # (batch, heads, seq, dim)
        compressed, scale = self.cache.compress(k, layer_idx=0)
        decompressed = self.cache.decompress(compressed, scale, layer_idx=0)
        self.assertEqual(decompressed.shape, k.shape)
        # Error should be bounded
        error = (k - decompressed).norm().item() / k.norm().item()
        self.assertLess(error, 0.3)


class TestTurboQuantAttention(unittest.TestCase):
    """Integration tests for TurboQuantAttention."""

    def setUp(self):
        torch.manual_seed(0)
        self.attn = TurboQuantAttention(
            embed_dim=256,
            num_heads=8,
            max_seq_len=128,
        )

    def test_forward_shape_no_cache(self):
        """Output shape should be correct without cache."""
        x = torch.randn(2, 16, 256)
        out, cache = self.attn(x, use_cache=False)
        self.assertEqual(out.shape, x.shape)
        self.assertIsNone(cache)

    def test_forward_shape_with_cache(self):
        """Output shape should be correct with cache."""
        x = torch.randn(2, 16, 256)
        out, cache = self.attn(x, use_cache=True)
        self.assertEqual(out.shape, x.shape)
        self.assertIsNotNone(cache)

    def test_forward_with_past_cache(self):
        """Forward pass should work with past_key_value."""
        x = torch.randn(1, 8, 256)
        # First pass: no past
        _, cache1 = self.attn(x, use_cache=True)
        # Second pass: with past
        out, cache2 = self.attn(x, use_cache=True, past_key_value=cache1)
        self.assertEqual(out.shape, x.shape)
        self.assertIsNotNone(cache2)


class TestBenchmark(unittest.TestCase):
    """Tests for benchmark function."""

    def test_benchmark_returns_dict(self):
        """Should return expected keys."""
        results = benchmark_turboquant(seq_len=256, num_layers=4)
        self.assertIsInstance(results, dict)
        for key in ['fp16_memory_mb', 'turboquant_memory_mb',
                     'compression_ratio', 'bits_per_value']:
            self.assertIn(key, results)

    def test_benchmark_on_cpu(self):
        """Should work on CPU."""
        results = benchmark_turboquant(seq_len=128, num_layers=1, device='cpu')
        self.assertGreater(results['compression_ratio'], 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
