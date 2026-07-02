# Benchmarks

- Compression ratio: **~3.8×** vs FP16 with the default 1-bit correction,
  **~4.9×** without correction (both include per-block scale overhead)
- Round-trip SNR: **~19 dB** with the default correction, **~13 dB**
  without (Gaussian data, block size 64 — measured by the test suite)
- Attention parity: **SNR > 12 dB** and **cosine similarity > 0.96** on
  attention outputs (asserted in `tests/attention_parity.rs`)
- Throughput: hardware-dependent — run `turboquant bench` or
  `cargo bench -p turboquant-bench` on your machine

See [docs/BENCHMARKS.md](../docs/BENCHMARKS.md) for details.
