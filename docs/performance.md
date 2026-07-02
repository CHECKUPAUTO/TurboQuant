# Performance Projections

> **Status: theoretical projections, not measurements.** The tables below
> are back-of-the-envelope estimates of what MLA + TurboQuant could
> deliver. No perplexity, latency, or tokens/s in this document has been
> measured with a real model. For numbers that *are* measured (compression
> ratio and round-trip SNR of the quantizer itself), see
> [rust/docs/BENCHMARKS.md](../rust/docs/BENCHMARKS.md) and the README.

## Memory Comparison (projected)

Idealized payload-only arithmetic (3 bits per value, overhead ignored):

| Context Length | FP16 (GB) | MLA (GB) | TurboQuant (GB) | MLA+TurboQuant (GB) |
|----------------|-----------|----------|------------------|---------------------|
| 4K | 0.5 | 0.125 | 0.094 | 0.023 |
| 8K | 1.0 | 0.25 | 0.188 | 0.047 |
| 16K | 2.0 | 0.5 | 0.375 | 0.094 |
| 32K | 4.0 | 1.0 | 0.75 | 0.188 |
| 64K | 8.0 | 2.0 | 1.5 | 0.375 |
| 128K | 16.0 | 4.0 | 3.0 | 0.75 |

**Calculation**:
- FP16: `seq_len × head_dim × num_heads × num_layers × 2 bytes`
- MLA: FP16 ÷ 4 (latent compression)
- TurboQuant: FP16 × (3/16) — payload only
- MLA+TurboQuant: FP16 ÷ 4 × (3/16)

**Real overhead**: the actual TurboQuant format also stores one f16 scale
per 64-value block and, in the default mode, 1 correction bit per value —
i.e. 3.25 bits/value (~4.9× vs FP16) without correction, 4.25 bits/value
(~3.8×) with the default 1-bit correction. Scale the TurboQuant columns by
3.25/3 or 4.25/3 accordingly.

## Latency Impact (projected)

| Model Size | Baseline (ms) | TurboQuant (ms) | Overhead |
|------------|---------------|------------------|----------|
| 7B | 45 | ~47 | +4% |
| 13B | 78 | ~81 | +4% |
| 70B | 420 | ~430 | +2% |

**Note**: Overhead is expected to be small because the rotation can be
pre-absorbed into the projection weights (see
[MLA_TurboQuant_Synergy.md](../MLA_TurboQuant_Synergy.md)). Not yet
measured in a real inference engine.

## Quality Impact

No model-level perplexity results are available yet — no inference engine
reads the turbo3 format today. Measured quantizer-level quality (Gaussian
data, block size 64): round-trip SNR ~13 dB without correction, ~19 dB
with the default 1-bit correction; attention-output parity SNR > 12 dB
and cosine similarity > 0.96 in the test suite.

## Throughput (projected)

Decode throughput should improve with context length as KV-cache memory
bandwidth becomes the bottleneck; the effect cannot be quantified until
an engine implements the turbo3 cache path. No tokens/s measurements
exist yet.
