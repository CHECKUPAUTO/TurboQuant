# TurboQuant Algorithm

## Mathematical Foundation

### Johnson-Lindenstrauss Lemma

For any set of `n` points in ℝᵈ and ε ∈ (0, 1), there exists a linear map
`f: ℝᵈ → ℝᵏ` with `k = O(log(n)/ε²)` such that for all x, y in the set:

```
(1 - ε)‖x - y‖² ≤ ‖f(x) - f(y)‖² ≤ (1 + ε)‖x - y‖²
```

### Polar Decomposition

Any matrix A can be decomposed as A = U · P where U is unitary (orthogonal
for real matrices) and P is positive semi-definite. The unitary part U is
used as the rotation matrix R.

### QJL Bound

For the quantized variant with 1-bit correction:

```
|⟨Q(x), Q(y)⟩ - ⟨x, y⟩| ≤ ε · ‖x‖ · ‖y‖
```

where ε depends on the quantization level. Empirically (Gaussian data,
default config) the relative dot-product error stays below 0.05; see
`test_dot_product_preserved` in `turboquant-core/src/qjl.rs`.

## Rotation Strategies

| Method | Init | Apply | Memory |
|--------|------|-------|--------|
| QR | O(d³) | O(d²) | O(d²) |
| Householder (k) | O(k·d) | O(k·d) | O(k·d) |
| Hadamard | O(d) | O(d log d) | O(d) |

## Quantization Pipeline

```
Input x ∈ ℝᵈ (FP16)
  ↓
PolarQuant: y = x · R (orthogonal rotation)
  ↓
Per-block scaling: y_norm = y / s_block ∈ [-1, 1] (s_block ∈ f16)
  ↓
3-bit quantize: idx = clamp(round(y_norm · 3.5 + 3.5), 0, 7)
  (symmetric 8-level grid {-3.5, -2.5, …, +2.5, +3.5}, spacing 1.0;
   reconstruction: (idx − 3.5) / 3.5 · s_block)
  ↓
1-bit residual: sign of (y_norm · 3.5 − (idx − 3.5)), one bit per value
  (dequantization adds ±learned_scale · s_block; the default
   learned_scale is the MSE-optimal quarter step,
   0.25 / 3.5 ≈ 0.0714 — `DEFAULT_CORRECTION_SCALE` in qjl.rs)
  ↓
Pack: 8 values × 3 bits → 3 bytes
  ↓
Output: packed bytes + per-block scales + correction bits
```

## Decompression APIs

There are two decompression entry points, differing only in which domain
the output lives in. `decompress_tensor` unpacks and dequantizes a
`KvBlock` but does **not** undo the polar rotation — its output stays in
the rotated domain, which is exactly what attention wants: Q and K share
the same orthogonal rotation, so it cancels in Q·Kᵀ and un-rotating would
be wasted work. `decompress_tensor_unrotated` performs the same
dequantization and then applies the inverse rotation to each
`head_dim`-sized row (mirroring how `compress_tensor` rotated each full
row before block-wise quantization), returning caller-facing values in
the original input domain; it fails with `InvalidDimension` if `head_dim`
does not match the rotation's dimension. Round-trip SNR against the
original input matches the rotated-domain figures above, since the
rotation is orthogonal (see the `test_unrotated_roundtrip_*` tests in
`turboquant-core/src/quantize.rs`).

## Compression Ratio

```
Without correction (CorrectionMode::None):
bits_per_value = 3 (payload) + 16/block_size (scale overhead)
              = 3 + 16/64 = 3.25 bits
ratio = 16 / 3.25 ≈ 4.92× for block_size=64

With the default 1-bit residual correction:
bits_per_value = 3 + 1 (correction) + 16/block_size
              = 4 + 16/64 = 4.25 bits
ratio = 16 / 4.25 ≈ 3.76× for block_size=64
```

Exact ratio includes alignment padding for values not divisible by 8.

## Measured Quality

Round-trip SNR on standard-normal data with block_size = 64 (measured by
the `turboquant-core` test suite): **~13 dB** without correction,
**~19 dB** with the default 1-bit correction.
