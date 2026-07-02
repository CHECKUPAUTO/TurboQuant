# Benchmarks

## How to measure

Numbers depend on your hardware — measure locally:

```bash
# End-to-end compression benchmark (CPU backend)
turboquant bench --head-dim 128 --seq-len 4096 --num-heads 32

# Criterion micro-benchmarks (compression, rotation, attention)
cargo bench -p turboquant-bench
# then open target/criterion/report/index.html
```

## Compression Ratio (measured)

Ratios vs FP16 include the per-block f16 scale (one per 64 values by
default) and, in the default mode, the persisted 1-bit correction signs:

| Mode | Bits/value | Ratio vs FP16 |
|------|------------|---------------|
| 3-bit, no correction | 3 + 16/64 = 3.25 | ~4.9× |
| 3-bit + 1-bit residual correction (default) | 4 + 16/64 = 4.25 | ~3.8× |

`turboquant bench` reports the same ~3.8× for the default configuration.

## Quality (measured by the test suite)

Round-trip SNR on standard-normal data, block size 64
(`cargo test -p turboquant-core`):

| Scenario | SNR (dB) |
|----------|----------|
| 3-bit grid only (no correction) | ~13 |
| 3-bit + default 1-bit correction | ~19 |

End-to-end attention parity (compressed K/V vs float reference,
`tests/attention_parity.rs`) asserts **SNR > 12 dB** and
**cosine similarity > 0.96**; the statistics are computed from the actual
tensors, not hardcoded.

## Throughput

Compression throughput is hardware-dependent and no reference numbers
are published yet — run `turboquant bench` or the criterion suite above
on your target machine. The CPU backend parallelizes over heads with
rayon; there is no hand-written SIMD (scalar code is auto-vectorized).

## Rotation Strategies (asymptotics)

| Method | Init | Apply/vec | Memory |
|--------|------|-----------|--------|
| QR (d×d matrix) | O(d³) | O(d²) | O(d²) |
| Householder (k reflectors) | O(k·d) | O(k·d) | O(k·d) |
| Hadamard | O(d) | O(d log d) | O(d) |

Relative wall-clock costs are measured by `cargo bench -p turboquant-bench`.
