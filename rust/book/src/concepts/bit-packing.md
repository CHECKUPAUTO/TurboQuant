# 3-bit Bit Packing

TurboQuant packs exactly 8 three-bit values into 3 bytes (24 bits).

## Efficiency

- 8 values → 24 bits = exactly 3 bits per value
- No padding waste within a block (tails are zero-padded to a multiple
  of 8 values)
- Scalar implementation, auto-vectorized by LLVM (no hand-written SIMD)

When the 1-bit residual correction is enabled (the default), the
correction signs are stored separately at 1 bit per value, so the
payload is 4 bits per value plus one f16 scale per block.
