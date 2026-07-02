# QJL — Quantized Johnson-Lindenstrauss

3-bit quantization with 1-bit residual correction, preserving dot
products within the JL bound.

## The 3-bit grid

Each value is normalized by its block scale (`x_norm = x / s_block ∈
[-1, 1]`) and snapped to a **symmetric 8-level grid**
`{-3.5, -2.5, -1.5, -0.5, +0.5, +1.5, +2.5, +3.5}` (spacing 1.0, in
level units):

```text
idx   = clamp(round(x_norm · 3.5 + 3.5), 0, 7)     // stored 3-bit code
x̂     = (idx − 3.5) / 3.5 · s_block                // reconstruction
```

There is no zero level: the grid is symmetric, so positive and negative
values round-trip with the correct sign.

## 1-bit residual correction

One extra bit per value stores the sign of the rounding residual
(computed in level units: `x_norm · 3.5 − (idx − 3.5)`). Dequantization
adds `±learned_scale · s_block`. The residual is ~uniform within ±half a
grid step, so the MSE-optimal fixed magnitude is a **quarter step**:
`DEFAULT_CORRECTION_SCALE = 0.25 / 3.5 ≈ 0.0714` (normalized units).

Measured round-trip SNR on Gaussian data (block size 64): **~13 dB**
without correction, **~19 dB** with it.

## Scale Modes

- **AbsMax**: Simple but outlier-sensitive
- **Percentile**: Robust to outliers (e.g. p99)
- **Adaptive**: Standard deviation of the block
- **Fixed**: User-specified
