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

where ε depends on the quantization level (3 bits → ε ≈ 0.07).

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
Per-block scaling: y_norm = y / s_block (s_block ∈ f16)
  ↓
3-bit quantize: q = round(y_norm · 3.5 / 0.5) · 0.5
  ↓
1-bit residual: correction = sign(y_norm - q) · ε
  ↓
Pack: 8 values × 3 bits → 3 bytes
  ↓
Output: packed bytes + per-block scales
```

## Compression Ratio

```
bits_per_value = 3 (payload) + 16/block_size (scale overhead)
              = 3 + 16/64 = 3.25 bits

ratio = 16 / 3.25 ≈ 4.92× for block_size=64
ratio = 16 / 3.13 ≈ 5.11× for block_size=128
```

Exact ratio includes alignment padding for values not divisible by 8.
