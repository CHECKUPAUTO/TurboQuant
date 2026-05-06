# 3-bit Bit Packing

TurboQuant packs exactly 8 three-bit values into 3 bytes (24 bits).

## Efficiency

- 8 values → 24 bits = exactly 3 bits per value
- No padding waste within a block
- Optional SIMD acceleration for 2-4× throughput
